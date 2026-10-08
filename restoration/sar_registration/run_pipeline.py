from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
from PIL import Image

from sar_registration.simclr import LearningConfig, train
from sar_registration.learned_matching import match_iteratively
from sar_registration.reporting import export_final, comparison, IterationRecorder
from sar_registration.geometry import project
from sar_registration.maritime_targets import MaritimeTargets,localized_sar_points,point_labels
from sar_registration.maritime_image_validation import MaritimeImageEvidence
from sar_registration.baseline_validation import input_fingerprint,load_comparable_baseline
from registration_metrics import calculate_registration_metrics
from sar_registration.simclr import valid_centers
from sar_registration.sift_fusion import FusionConfig, ThreeLayerSIFTFusion


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Same-grid Original/SR restoration layers -> SIFT consensus T0 -> "
            "SIFT union -> reference-only SimCLR v1"
        )
    )
    parser.add_argument(
        "--original",
        "--low_res",
        dest="original",
        type=Path,
        required=True,
        help="Original low-resolution/low-quality sensed image",
    )
    parser.add_argument(
        "--sr_candidates",
        type=Path,
        nargs="*",
        default=[],
        help="Same-size SR restoration results, for example alpha=0.25..1.00",
    )
    parser.add_argument(
        "--layer_labels",
        default=None,
        help="Comma-separated SR labels; Original is added automatically",
    )
    parser.add_argument(
        "--layer_confidences",
        default="1,1,1",
        help=(
            "Comma-separated confidence priors for Original and each "
            "restoration layer. For this three-representation pipeline pass "
            "Original,round1,round2, for example 1,1,1."
        ),
    )
    parser.add_argument(
        "--reference", type=Path, required=True, help="High-resolution reference image"
    )
    parser.add_argument(
        "--dataset", required=True, help="Experiment name for models and output folders"
    )

    parser.add_argument("--max_match_size", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument("--min_sift_matches", "--min_dfm_matches", dest="min_sift_matches", type=int, default=6)
    parser.add_argument("--sift_ratio_threshold", type=float, default=0.90)
    parser.add_argument("--ransac_threshold", type=float, default=5.0)
    parser.add_argument("--coarse_displacement_gate", type=float, default=24.0)
    parser.add_argument(
        "--min_phase_correlation_response", type=float, default=0.02
    )
    parser.add_argument(
        "--phase_transform_consistency_threshold", type=float, default=18.0
    )
    parser.add_argument("--transform_consensus_threshold", type=float, default=20.0)

    parser.add_argument("--sift_max_features_per_image", type=int, default=6000)
    parser.add_argument("--sift_cluster_radius", type=float, default=3.0)
    parser.add_argument("--min_candidate_confidence", type=float, default=0.35)
    parser.add_argument("--sift_edge_margin", type=int, default=8)
    parser.add_argument("--reference_patch_margin", type=int, default=64)
    parser.add_argument("--max_candidate_points", type=int, default=3000)
    parser.add_argument("--grid_rows", type=int, default=16)
    parser.add_argument("--grid_cols", type=int, default=16)
    parser.add_argument("--max_points_per_grid_cell", type=int, default=20)

    parser.add_argument("--output_patch_size", type=int, default=64)

    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Create T0, candidate coordinates and lazy patch manifest; no training",
    )
    parser.add_argument("--similarity_search_radius", type=float, default=80.0)
    parser.add_argument("--min_cosine", type=float, default=0.5)
    parser.add_argument("--min_margin", type=float, default=0.015)
    parser.add_argument("--similarity_batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--contrastive_lr", type=float, default=0.0003)
    parser.add_argument("--contrastive_epochs", type=int, default=100)
    parser.add_argument("--final_ransac_threshold", type=float, default=2.0)
    parser.add_argument("--max_final_matches", type=int, default=24)
    parser.add_argument("--no_image_guidance", action="store_true")
    parser.add_argument("--target_brightness_threshold",type=float,default=30.)
    parser.add_argument("--warm_start",type=Path,default=None)
    parser.add_argument("--ship_steps_per_epoch",type=int,default=12)
    parser.add_argument(
        "--skip_finalization",
        action="store_true",
        help="Skip final reports (normally leave this unset)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume epoch checkpoint with identical inputs/config")
    parser.add_argument("--matching_iterations", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.15)
    return parser.parse_args()


def _set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _labels(args: argparse.Namespace, sr_candidates: list[Path]) -> list[str]:
    if args.layer_labels:
        sr_labels = [
            item.strip() for item in args.layer_labels.split(",") if item.strip()
        ]
        if len(sr_labels) != len(sr_candidates):
            raise ValueError(
                "--layer_labels count must equal --sr_candidates count "
                "(do not include Original)"
            )
    else:
        sr_labels = [path.stem for path in sr_candidates]
    return ["Original", *sr_labels]


def _layer_confidences(
    args: argparse.Namespace, sr_candidates: list[Path]
) -> list[float] | None:
    if args.layer_confidences is None:
        return None
    sr_values = [
        float(item.strip())
        for item in args.layer_confidences.split(",")
        if item.strip()
    ]
    if len(sr_values) == len(sr_candidates) + 1:
        values = sr_values
    elif len(sr_values) == len(sr_candidates):
        # Backward compatibility with the old command-line convention where
        # only restoration confidences were supplied and Original was fixed
        # to 1.0.
        values = [1.0, *sr_values]
    else:
        raise ValueError(
            "--layer_confidences must contain either one value per "
            "representation (including Original) or one per restoration layer"
        )
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("--layer_confidences values must be within [0, 1]")
    return values


def _ensure_runtime_dirs() -> None:
    for name in ("fusion_runs", "saved_models", "registration_results"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if Path(args.dataset).name != args.dataset or args.dataset in (".", "..") or any(c in args.dataset for c in '/\\:'):
        raise ValueError("--dataset must be a simple directory name")
    for name in ('batch_size', 'similarity_batch_size', 'contrastive_epochs', 'matching_iterations',
                 'max_candidate_points', 'grid_rows', 'grid_cols', 'max_points_per_grid_cell'):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.batch_size < 2 or args.num_workers < 0 or args.temperature <= 0 or args.contrastive_lr <= 0:
        raise ValueError("Invalid SimCLR training configuration")
    if not 10 <= args.max_final_matches <= 30:
        raise ValueError("--max_final_matches must be in [10,30]")
    if args.output_patch_size != 64:
        raise ValueError("Paper sampling requires --output_patch_size 64")
    if args.similarity_search_radius <= 0 or args.final_ransac_threshold <= 0 or args.ransac_threshold <= 0:
        raise ValueError("Geometric thresholds must be positive")
    if not -1 <= args.min_cosine <= 1 or args.min_margin < 0:
        raise ValueError("Invalid descriptor thresholds")
    # Resolve inputs before changing directory, including direct runner use.
    args.original, args.reference = args.original.resolve(), args.reference.resolve()
    args.sr_candidates = [path.resolve() for path in args.sr_candidates]
    _set_reproducible_seed(args.seed)
    os.chdir(ROOT)
    _ensure_runtime_dirs()
    if len(args.sr_candidates) != 2:
        raise ValueError(
            "Provide exactly two existing same-grid restoration images with --sr_candidates."
        )
    sr_candidates = [path.resolve() for path in args.sr_candidates]
    representation_paths = [args.original.resolve(), *sr_candidates]
    labels = _labels(args, sr_candidates)
    if len(set(labels)) != 3:
        raise ValueError("Layer labels must be distinct to preserve all three reports")
    config_kwargs = {
        "max_match_size": args.max_match_size,
        "min_matches": args.min_sift_matches,
        "sift_ratio_threshold": args.sift_ratio_threshold,
        "ransac_threshold": args.ransac_threshold,
        "coarse_displacement_gate": args.coarse_displacement_gate,
        "min_phase_correlation_response": args.min_phase_correlation_response,
        "phase_transform_consistency_threshold": (
            args.phase_transform_consistency_threshold
        ),
        "transform_consensus_threshold": args.transform_consensus_threshold,
        "sift_max_features_per_image": args.sift_max_features_per_image,
        "sift_cluster_radius": args.sift_cluster_radius,
        "min_candidate_confidence": args.min_candidate_confidence,
        "sift_edge_margin": args.sift_edge_margin,
        "reference_patch_margin": max(64, args.reference_patch_margin),
        "max_candidate_points": args.max_candidate_points,
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "max_points_per_grid_cell": args.max_points_per_grid_cell,
    }
    if args.device:
        config_kwargs["device"] = args.device

    checkpoint = ROOT / "saved_models" / args.dataset / "simclr_v1" / "simclr_last.pt"
    if not args.prepare_only:
        if checkpoint.exists() and not args.resume:
            raise FileExistsError(f"Existing model {checkpoint}; use --resume or a new --dataset")
        if args.resume and not checkpoint.exists():
            raise FileNotFoundError(f"No checkpoint to resume: {checkpoint}")
    result = ThreeLayerSIFTFusion(FusionConfig(**config_kwargs)).run(
        args.reference.resolve(),
        representation_paths,
        labels,
        _layer_confidences(args, sr_candidates),
    )
    run_dir = ROOT / "fusion_runs" / args.dataset
    result.save(run_dir, reference_path=args.reference.resolve())
    (run_dir/"input_fingerprint.json").write_text(json.dumps(input_fingerprint([args.reference,args.original,*sr_candidates]),indent=2))
    (run_dir / "fusion_config.json").write_text(
        json.dumps(config_kwargs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    reference_rgb = np.asarray(Image.open(args.reference).convert("RGB"))
    sensed_rgb = np.asarray(Image.open(args.original).convert("RGB"))
    reference_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
    sensed_gray=cv2.cvtColor(sensed_rgb,cv2.COLOR_RGB2GRAY)
    targets=MaritimeTargets(reference_gray,sensed_gray,result.transform,args.target_brightness_threshold)
    ref_points,ref_provenance=localized_sar_points([reference_gray],targets.ref_labels)
    sensed_layers=[cv2.cvtColor(np.asarray(Image.open(path).convert("RGB")),cv2.COLOR_RGB2GRAY) for path in representation_paths]
    source_points,source_provenance=localized_sar_points(sensed_layers,targets.source_labels)
    valid=valid_centers(ref_points,reference_gray.shape)
    ref_points,ref_provenance=ref_points[valid],ref_provenance[valid]
    if len(ref_points)<10 or len(source_points)<10:
        raise RuntimeError("Too few genuine SAR-SIFT maritime target points; no background fallback")
    targets.save(run_dir/"maritime_targets",source_points,ref_points)
    np.savez_compressed(run_dir/"detector_provenance.npz",reference=ref_provenance,sensed=source_provenance)
    # Fixed, deduplicated SIFT anchors only validate iterative updates; they are
    # never used as labels or positive pairs for contrastive learning.
    selected = [result.representation_matches[i] for i in result.consensus_indices]
    anchor_s = np.concatenate([m.sensed_points[m.inlier_mask] for m in selected])
    anchor_r = np.concatenate([m.reference_points[m.inlier_mask] for m in selected])
    _, unique = np.unique(np.rint(np.column_stack((anchor_s,anchor_r))*2),axis=0,return_index=True)
    anchor_s, anchor_r = anchor_s[unique], anchor_r[unique]
    anchor_keep = np.linalg.norm(project(anchor_s,result.transform)-anchor_r,axis=1) <= args.ransac_threshold
    anchor_s, anchor_r = anchor_s[anchor_keep], anchor_r[anchor_keep]
    if len(anchor_s) < 4:
        raise RuntimeError("Fused T0 has fewer than four valid anchor correspondences")
    np.savez_compressed(run_dir / "patch_coordinates.npz", reference=ref_points,
        sensed=source_points, projected=project(source_points,result.transform),
        T0=result.transform, anchor_sensed=anchor_s, anchor_reference=anchor_r)
    if args.warm_start is not None:
        args.warm_start=args.warm_start.resolve()
    learning = LearningConfig(epochs=args.contrastive_epochs,batch_size=args.batch_size,
        workers=args.num_workers,lr=args.contrastive_lr,temperature=args.temperature,
        seed=args.seed,device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        resume=args.resume,steps_per_epoch=args.ship_steps_per_epoch,
        rotation_degrees=1.,
        warm_start=str(args.warm_start) if args.warm_start is not None else None)
    (run_dir / "sampling_manifest.json").write_text(json.dumps({
        "crop_image": str(args.reference), "transform_direction":"sensed_to_reference",
        "train_centers":"reference ship/shore SAR-SIFT keypoints",
        "target_instances":targets.active_ids, "train_sizes":list(range(56,129,8)),
        "test_centers":"reference keypoints and T(sensed keypoints)", "test_size":128,
        "network_input":64, "reference_points":len(ref_points), "sensed_points":len(source_points),
        "boundary_policy":"discard incomplete windows; no padding", "arguments":vars(args)
    }, default=str,ensure_ascii=False,indent=2),encoding="utf-8")
    if args.prepare_only:
        print(f"Prepared geometry and patch manifest at {run_dir}; no training requested")
        return
    model = train(reference_gray, ref_points, ROOT / "saved_models" / args.dataset / "simclr_v1", learning)
    evidence = None if args.no_image_guidance else MaritimeImageEvidence(reference_gray,sensed_gray,result.transform,targets)
    output_dir = ROOT / "registration_results" / args.dataset
    recorder = IterationRecorder(output_dir, reference_rgb, sensed_rgb)
    transform, pairs, scores, inliers, history = match_iteratively(model, reference_gray,
        source_points, ref_points, result.transform, anchor_s, anchor_r, run_dir,
        iterations=args.matching_iterations,radius=args.similarity_search_radius,
        min_cosine=args.min_cosine,min_margin=args.min_margin,
        batch_size=args.similarity_batch_size,threshold=args.final_ransac_threshold,
        iteration_callback=recorder,max_matches=args.max_final_matches,
        image_evidence=evidence,targets=targets)
    if len(pairs):
        source, target = source_points[pairs[:,0]], ref_points[pairs[:,1]]
        status = "learned_update_accepted"
    else:
        source,target=np.empty((0,2)),np.empty((0,2))
        inliers=np.empty(0,bool)
        status="target_matching_failed_no_background_fallback"
    np.savetxt(run_dir/"learned_matches.csv",np.column_stack((pairs,scores,inliers if len(pairs) else np.empty(0))),delimiter=",",
               header="sensed_index,reference_index,confidence,retained",comments="")
    if not args.skip_finalization:
        output_dir = ROOT / "registration_results" / args.dataset
        metrics = export_final(output_dir,reference_rgb,sensed_rgb,transform,source,target,inliers,status,scores=scores)
        rows = comparison(run_dir, output_dir, metrics)
        initial_anchor_rmse = float(np.sqrt(np.mean(np.sum(
            (project(anchor_s,result.transform)-anchor_r)**2,axis=1))))
        final_anchor_rmse = float(np.sqrt(np.mean(np.sum(
            (project(anchor_s,transform)-anchor_r)**2,axis=1))))
        initial_image = evidence.baseline if evidence else None
        final_image = evidence.score(transform) if evidence else None
        last_accepted = next((row for row in reversed(history) if row['accepted']),{})
        target_quality=targets.quality(source[inliers],target[inliers],transform)
        previous=load_comparable_baseline(ROOT,[args.reference,args.original,*sr_candidates])
        initial_metrics=calculate_registration_metrics(anchor_r,np.linalg.norm(project(anchor_s,result.transform)-anchor_r,axis=1),reference_gray.shape,source_points=anchor_s)
        baseline_h=np.asarray(previous["final_transform"]) if previous is not None else result.transform
        previous_target_image=evidence.score(baseline_h) if evidence is not None else None
        error_limit=previous["RMSEall"] if previous is not None else initial_metrics["RMSEall"]
        effective=bool(len(pairs) and 10<=metrics["Nred"]<=30
            and metrics["RMSEall"]<=error_limit
            and (metrics["Pquad"]<=previous["Pquad"] if previous else True)
            and metrics["RMSEloo"] is not None
            and (previous["RMSEloo"] if previous else initial_metrics["RMSEloo"]) is not None
            and metrics["RMSEloo"]<=(previous["RMSEloo"] if previous else initial_metrics["RMSEloo"])
            and target_quality["same_target_rate"]==1.
            and target_quality["spatially_valid"]
            and final_image and final_image["all_target_ncc"]>=previous_target_image["all_target_ncc"]-.002
            and final_image["holdout_ncc"]>=previous_target_image["holdout_ncc"]-.002)
        validation = dict(effective=effective,historical_baseline_comparable=previous is not None,initial_fixed_anchor_rmse=initial_anchor_rmse,
            final_fixed_anchor_rmse=final_anchor_rmse,initial_image=initial_image,final_image=final_image,
            predictive_loo_rmse=last_accepted.get('predictive_loo_rmse'),
            excluded_match_rmse=last_accepted.get('supporter_rmse'),
            excluded_match_count=last_accepted.get('supporter_count'),
            coverage=last_accepted.get('point_coverage'),target_quality=target_quality,
            previous_method_target_image=previous_target_image,
            note='Point metrics use different sets. Held-out image tiles and fixed anchors use the same data for before/after. No ground-truth control points.')
        (output_dir/'validation_report.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
        print(json.dumps(rows,ensure_ascii=False,indent=2))
        print('Independent validation:',json.dumps(validation),flush=True)


if __name__ == "__main__":
    main()
