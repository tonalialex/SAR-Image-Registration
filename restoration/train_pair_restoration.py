from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# Set this before importing torch so CUDA GEMM can use its deterministic mode.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from maritime_patch_filter import filter_pairs


ROOT = Path(__file__).resolve().parent
NAFNET_ROOT = ROOT / "third_party" / "NAFNet"
if str(NAFNET_ROOT) not in sys.path:
    sys.path.insert(0, str(NAFNET_ROOT))
from basicsr.models.archs.NAFNet_arch import NAFNet  # noqa: E402


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return image.astype(np.float32) / 255.0


def _points(path: str | Path) -> list[tuple[float, float, float, float]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    points = []
    for row in rows:
        if "inlier" in row and int(float(row["inlier"])) != 1:
            continue
        points.append((float(row["sensed_x"]), float(row["sensed_y"]), float(row["reference_x"]), float(row["reference_y"])))
    return points


def _crop(image: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    half = size // 2
    cx, cy = int(round(x)), int(round(y))
    # Jitter can move an inlier center outside the image. Clamp the center
    # before slicing so every training sample remains exactly size x size.
    height, width = image.shape[:2]
    cx = min(max(cx, 0), width - 1)
    cy = min(max(cy, 0), height - 1)
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_REFLECT_101)
    return padded[cy : cy + size, cx : cx + size]


def _sea_patch_statistics(
    patch: np.ndarray,
    black_threshold: float = 30.0,
    black_fraction: float = 0.95,
) -> dict:
    """Classify a restoration patch using low-intensity occupancy.

    Pixels below 30/255 are counted as black and a patch is sea when more than
    95% are black.  Otsu's threshold is recorded for diagnosis, while the
    fixed rule keeps the decision reproducible across the sensed/reference
    radiometric difference.
    """
    patch_u8 = np.rint(np.clip(patch, 0.0, 1.0) * 255.0).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(
        patch_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    ratio = float(np.mean(patch_u8 < float(black_threshold)))
    return {
        "black_ratio": ratio,
        "otsu_threshold": float(otsu_threshold),
        "is_sea": bool(ratio > float(black_fraction)),
    }


def _filter_sea_patch_pairs(
    sensed: np.ndarray,
    reference_aligned: np.ndarray,
    points: list[tuple[float, float, float, float]],
    output_dir: Path,
    patch_size: int,
    black_threshold: float = 30.0,
    black_fraction: float = 0.95,
) -> tuple[list[tuple[float, float, float, float]], dict]:
    """Remove restoration pairs when either base patch is classified as sea."""
    kept = []
    records = []
    for index, (sx, sy, rx, ry) in enumerate(points):
        source_patch = _crop(sensed, sx, sy, patch_size)
        target_patch = _crop(reference_aligned, sx, sy, patch_size)
        source_stats = _sea_patch_statistics(
            source_patch, black_threshold, black_fraction
        )
        target_stats = _sea_patch_statistics(
            target_patch, black_threshold, black_fraction
        )
        rejected = bool(source_stats["is_sea"] or target_stats["is_sea"])
        if not rejected:
            kept.append((sx, sy, rx, ry))
        records.append(
            {
                "input_index": int(index),
                "sensed_x": float(sx),
                "sensed_y": float(sy),
                "reference_x": float(rx),
                "reference_y": float(ry),
                "source_black_ratio": float(source_stats["black_ratio"]),
                "source_otsu_threshold": float(source_stats["otsu_threshold"]),
                "target_black_ratio": float(target_stats["black_ratio"]),
                "target_otsu_threshold": float(target_stats["otsu_threshold"]),
                "rejected_as_sea": rejected,
                "reject_reason": (
                    "source_and_target_sea"
                    if source_stats["is_sea"] and target_stats["is_sea"]
                    else "source_sea"
                    if source_stats["is_sea"]
                    else "target_sea"
                    if target_stats["is_sea"]
                    else ""
                ),
            }
        )
    report = {
        "rule": "pixel < black_threshold; reject pair when source or target black_ratio >= black_fraction",
        "black_threshold": float(black_threshold),
        "black_fraction": float(black_fraction),
        "input_pairs": int(len(points)),
        "removed_pairs": int(len(points) - len(kept)),
        "kept_pairs": int(len(kept)),
        "records": records,
    }
    report_path = output_dir / "sea_patch_filter.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["report_path"] = str(report_path)
    return kept, report


def _save_training_patch_pairs(
    sensed: np.ndarray,
    reference_aligned: np.ndarray,
    points: list[tuple[float, float, float, float]],
    output_dir: Path,
    patch_size: int,
) -> dict:
    """Save the deterministic zero-jitter patch pairs used as training bases."""
    input_dir = output_dir / "input_patches"
    target_dir = output_dir / "target_patches"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (sx, sy, rx, ry) in enumerate(points, start=1):
        # These are exactly the two domains used by PairPatchDataset.  The
        # target is already in the sensed frame after the inverse affine warp.
        source_patch = np.rint(_crop(sensed, sx, sy, patch_size) * 255.0).clip(0, 255).astype(np.uint8)
        target_patch = np.rint(_crop(reference_aligned, sx, sy, patch_size) * 255.0).clip(0, 255).astype(np.uint8)
        input_path = input_dir / f"{index:05d}.png"
        target_path = target_dir / f"{index:05d}.png"
        cv2.imwrite(str(input_path), source_patch)
        cv2.imwrite(str(target_path), target_patch)
        records.append({
            "index": index,
            "sensed_x": sx,
            "sensed_y": sy,
            "reference_x": rx,
            "reference_y": ry,
            "input_patch": str(input_path),
            "target_patch": str(target_path),
            "jitter": 0,
        })
    manifest = {
        "count": len(records),
        "patch_size": patch_size,
        "input_frame": "original_sensed_image",
        "target_frame": "reference_image_warped_to_sensed_frame",
        "note": "Saved zero-jitter base pairs; actual training additionally samples random jitter.",
        "pairs": records,
    }
    manifest_path = output_dir / "training_patch_pairs.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"directory": str(output_dir), "count": len(records), "manifest": str(manifest_path)}


def _load_affine(path: str | Path) -> np.ndarray:
    transform = np.asarray(np.loadtxt(str(path), dtype=np.float64))
    if transform.shape == (2, 3):
        transform = np.vstack([transform, [0.0, 0.0, 1.0]])
    if transform.shape != (3, 3):
        raise ValueError(f"期望2x3或3x3仿射矩阵，实际为 {transform.shape}")
    return transform


def _reference_in_sensed_frame(
    reference: np.ndarray,
    transform_path: str | Path | None,
) -> np.ndarray:
    """Warp the reference image into the sensed-image frame for patch supervision.

    T maps sensed coordinates to reference coordinates.  Patch losses are
    pixelwise, so the reference target must first be sampled back into the
    sensed frame; matching only the two patch centers is insufficient for a
    full-affine pair.
    """
    if not transform_path:
        return reference
    transform = _load_affine(transform_path)
    reference_to_sensed = np.linalg.inv(transform)
    height, width = reference.shape[:2]
    return cv2.warpAffine(
        reference,
        reference_to_sensed[:2],
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


class PairPatchDataset(Dataset):
    def __init__(self, sensed: np.ndarray, reference_aligned: np.ndarray, points, patch_size: int, jitter: int, length: int, patch_policy=None):
        self.sensed, self.reference_aligned, self.points = sensed, reference_aligned, points
        self.patch_size, self.jitter, self.length = patch_size, jitter, length
        self.patch_policy = patch_policy
        self.target_groups = patch_policy.groups(points) if patch_policy else None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.target_groups:
            index = random.choice(random.choice(self.target_groups))
        sx, sy, _rx, _ry = self.points[index % len(self.points)]
        dx = random.randint(-self.jitter, self.jitter) if self.jitter else 0
        dy = random.randint(-self.jitter, self.jitter) if self.jitter else 0
        cx, cy = self.patch_policy.jitter_center(sx, sy, dx, dy) if self.patch_policy else (sx+dx, sy+dy)
        source = _crop(self.sensed, cx, cy, self.patch_size)
        target = _crop(self.reference_aligned, cx, cy, self.patch_size)
        return torch.from_numpy(source[None].copy()), torch.from_numpy(target[None].copy())


def _calibrate_input_radiometry(
    sensed: np.ndarray,
    reference: np.ndarray,
    points: list[tuple[float, float, float, float]],
    patch_size: int,
) -> tuple[np.ndarray, dict]:
    """Estimate one global monotonic mapping from all matched signal patches."""
    threshold = 2.0 / 255.0
    source_samples, target_samples = [], []
    for sx, sy, rx, ry in points:
        source_patch = _crop(sensed, sx, sy, patch_size)
        target_patch = _crop(reference, rx, ry, patch_size)
        source_samples.append(source_patch[source_patch > threshold])
        target_samples.append(target_patch[target_patch > threshold])
    valid_source = [item for item in source_samples if item.size]
    valid_target = [item for item in target_samples if item.size]
    source = np.concatenate(valid_source) if valid_source else np.empty((0,), dtype=np.float32)
    target = np.concatenate(valid_target) if valid_target else np.empty((0,), dtype=np.float32)
    quantiles = np.asarray([0, 10, 25, 50, 75, 90, 95, 97, 99, 100], dtype=np.float64)
    if len(source) < 100 or len(target) < 100:
        source_knots = np.asarray([0.0, 1.0], dtype=np.float64)
        target_knots = source_knots.copy()
    else:
        source_knots = np.percentile(source, quantiles)
        target_knots = np.percentile(target, quantiles)
        keep = np.r_[True, np.diff(source_knots) > 1e-6]
        source_knots = np.maximum.accumulate(source_knots[keep])
        target_knots = np.maximum.accumulate(np.clip(target_knots[keep], 0.0, 1.0))
        if len(source_knots) < 3:
            source_knots = np.asarray([0.0, 1.0], dtype=np.float64)
            target_knots = source_knots.copy()
        else:
            source_knots = np.r_[0.0, source_knots]
            target_knots = np.r_[0.0, target_knots]
    calibrated = np.zeros_like(sensed, dtype=np.float32)
    signal = sensed > threshold
    calibrated[signal] = np.interp(
        sensed[signal], source_knots, target_knots,
    ).astype(np.float32)
    calibrated = np.clip(calibrated, 0.0, 1.0)
    report = {
        "method": "global_zero_preserving_monotonic_patch_quantile_mapping",
        "signal_threshold": threshold,
        "source_signal_pixels": int(len(source)),
        "target_signal_pixels": int(len(target)),
        "source_knots": source_knots.tolist(),
        "target_knots": target_knots.tolist(),
    }
    return calibrated, report


def _bounded_prediction(model, source: torch.Tensor, residual_limit: float) -> torch.Tensor:
    # Rollback baseline: let the pair-specific NAFNet learn the full mapping;
    # only the final image write-back is clamped to the valid intensity range.
    return model(source)


def _training_loss(
    prediction: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    args: argparse.Namespace,
    teacher_prediction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    pixel = F.l1_loss(prediction, target)
    total = pixel
    components = {
        "pixel": float(pixel.detach().cpu()),
    }
    if teacher_prediction is not None and args.teacher_weight > 0:
        teacher = F.l1_loss(prediction, teacher_prediction.detach())
        total = total + args.teacher_weight * teacher
        components["teacher_consistency"] = float(teacher.detach().cpu())
    return total, components


def _infer_tiled(
    model,
    image: np.ndarray,
    device: torch.device,
    tile_size: int = 384,
    tile_overlap: int = 64,
) -> np.ndarray:
    tensor = torch.from_numpy(image[None, None]).to(device)
    _, _, height, width = tensor.shape
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        with torch.no_grad():
            return torch.clamp(model(tensor), 0.0, 1.0)[0, 0].cpu().numpy().astype(np.float32)
    if tile_overlap < 0 or tile_overlap * 2 >= tile_size:
        raise ValueError("tile overlap must satisfy 0 <= overlap < tile_size / 2")
    core = tile_size - 2 * tile_overlap
    result = torch.empty_like(tensor)
    with torch.no_grad():
        for top in range(0, height, core):
            bottom = min(top + core, height)
            input_top = max(0, top - tile_overlap)
            input_bottom = min(height, bottom + tile_overlap)
            for left in range(0, width, core):
                right = min(left + core, width)
                input_left = max(0, left - tile_overlap)
                input_right = min(width, right + tile_overlap)
                tile = tensor[..., input_top:input_bottom, input_left:input_right]
                tile_output = torch.clamp(model(tile), 0.0, 1.0)
                result[..., top:bottom, left:right] = tile_output[
                    ...,
                    top - input_top : bottom - input_top,
                    left - input_left : right - input_left,
                ]
    return result[0, 0].cpu().numpy().astype(np.float32)


def train_pair_restoration(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sensed = _gray(args.sensed)
    reference = _gray(args.reference)
    if sensed.shape != reference.shape:
        raise ValueError("复原训练要求两幅图像尺寸相同")
    points = _points(args.matched_points)
    input_point_count = len(points)
    report = {
        "input_matched_points": input_point_count,
        "matched_points": input_point_count,
        "iterations": 0,
        "trained": False,
    }
    if not points:
        # The first SAR-SIFT stage can fail; identity is explicit and keeps the next stage runnable.
        cv2.imwrite(str(output_dir / "restored_sensed.png"), np.uint8(np.clip(sensed * 255.0, 0, 255)))
        report["reason"] = "第一轮没有 SAR-SIFT 内点，跳过 NAFNet 训练并保留原图"
        (output_dir / "restoration_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    reference_aligned = _reference_in_sensed_frame(reference, args.transform_file)
    legacy_report=None
    if getattr(args, 'target_filter', 'maritime') == 'legacy':
        points,legacy_report=_filter_sea_patch_pairs(sensed,reference_aligned,points,output_dir,args.patch_size,
            black_threshold=float(getattr(args,'sea_black_threshold',30.)),black_fraction=float(getattr(args,'sea_black_fraction',.95)))
    points,filter_report,patch_policy=filter_pairs(sensed,reference,points,output_dir,args.patch_size,
        _load_affine(args.transform_file),mode=getattr(args,'target_filter','maritime'),jitter=args.jitter,
        holdout_reference_path=getattr(args,'holdout_reference_mask',None),
        fixed_source_holdout_path=getattr(args,'fixed_source_holdout_mask',None))
    report['matched_points']=len(points)
    report['target_filter_report']={k:v for k,v in filter_report.items() if k!='records'}
    report['legacy_sea_patch_filter']={k:v for k,v in legacy_report.items() if k!='records'} if legacy_report else None
    print(f"[NAFNet] target filter {filter_report['mode']}: kept={len(points)}/{input_point_count}, ships={filter_report['ship_points']}, shores={filter_report['shore_points']}",flush=True)
    if not points:
        cv2.imwrite(
            str(output_dir / "restored_sensed.png"),
            np.uint8(np.clip(sensed * 255.0, 0, 255)),
        )
        report["reason"] = "全部 SAR-SIFT 内点 patch 均被海域先验过滤，跳过 NAFNet 训练"
        (output_dir / "restoration_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return report

    calibrated_sensed = sensed
    calibration = {"method": "disabled_for_rollback_baseline"}

    _seed_everything(args.seed)
    device = torch.device(args.device)
    model = NAFNet(img_channel=1, width=args.width, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 4], dec_blk_nums=[1, 1, 1, 1]).to(device)
    initialized_from = None
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        initialized_from = str(args.init_checkpoint)
    teacher_model = copy.deepcopy(model).eval() if args.init_checkpoint else None
    if teacher_model is not None:
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
    model.train()
    saved_patch_pairs = None
    if getattr(args, "save_training_patches", False):
        saved_patch_pairs = _save_training_patch_pairs(
            sensed,
            reference_aligned,
            points,
            output_dir / "training_patch_pairs",
            args.patch_size,
        )
        report["saved_training_patch_pairs"] = saved_patch_pairs
    dataset = PairPatchDataset(
        sensed,
        reference_aligned,
        points,
        args.patch_size,
        args.jitter,
        args.iterations * args.batch_size,
        patch_policy=patch_policy,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    losses = []
    last_components = {}
    for iteration, (source, target) in enumerate(loader, start=1):
        source, target = source.to(device), target.to(device)
        prediction = _bounded_prediction(model, source, args.residual_limit)
        with torch.no_grad() if teacher_model is not None else torch.enable_grad():
            teacher_prediction = teacher_model(source) if teacher_model is not None else None
        loss, last_components = _training_loss(
            prediction,
            source,
            target,
            args,
            teacher_prediction,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if iteration >= args.iterations:
            break
        if iteration % max(1, args.log_every) == 0:
            print(f"[NAFNet] iteration {iteration}/{args.iterations}, loss={losses[-1]:.6f}", flush=True)
    (output_dir/'loss_history.json').write_text(json.dumps(losses))
    model.eval()
    checkpoint_path = output_dir / "pair_specific_nafnet.pth"
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, checkpoint_path)
    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    restored = _infer_tiled(
        model,
        sensed,
        device,
        args.tile_size,
        args.tile_overlap,
    )
    cv2.imwrite(str(output_dir / "restored_sensed.png"), np.rint(np.clip(restored * 255.0, 0, 255)).astype(np.uint8))
    report.update({
        "iterations": len(losses),
        "trained": True,
        "final_loss": losses[-1],
        "device": str(device),
        "patch_size": args.patch_size,
        "checkpoint": str(checkpoint_path),
        "initialized_from": initialized_from,
        "input_radiometric_calibration": calibration,
        "transform_file": args.transform_file,
        "patch_target_frame": "sensed_frame_after_inverse_full_affine_warp",
        "teacher_consistency_weight": args.teacher_weight,
        "inference_mode": "overlap_tiled_baseline",
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "final_loss_components": last_components,
    })
    (output_dir / "restoration_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use first-round SAR-SIFT inliers to train pair-specific NAFNet")
    parser.add_argument("--sensed", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--matched-points", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--jitter", type=int, default=6)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--pad-multiple", type=int, default=16)
    parser.add_argument("--residual-limit", type=float, default=0.15)
    parser.add_argument("--signal-weight", type=float, default=4.0)
    parser.add_argument("--gradient-weight", type=float, default=0.5)
    parser.add_argument("--statistics-weight", type=float, default=0.2)
    parser.add_argument("--background-weight", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--transform-file", default=None)
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--sea-black-threshold", type=float, default=30.0)
    parser.add_argument("--sea-black-fraction", type=float, default=0.95)
    parser.add_argument('--target-filter',choices=['maritime','legacy'],default='maritime')
    parser.add_argument('--holdout-reference-mask',default=None)
    parser.add_argument('--fixed-source-holdout-mask',default=None)
    return parser


if __name__ == "__main__":
    train_pair_restoration(build_parser().parse_args())
