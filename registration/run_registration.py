#!/usr/bin/env python3
"""Run three-layer SAR-SIFT/T0, center-layout SimCLR and iterative registration.

The three sensed representations are the original image and two same-grid
restoration images. The SimCLR runner performs per-layer SAR-SIFT matching,
consensus T0 estimation, candidate-point filtering, and contrastive pretraining.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "sar_registration" / "run_pipeline.py"
DEFAULT_DATA_DIR = ROOT / "data" / "ship"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original",
        type=Path,
        default=DEFAULT_DATA_DIR / "sensed.jpg",
        help="Original sensed image (default: data/ship/sensed.jpg)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_DATA_DIR / "reference.jpg",
        help="Reference image (default: data/ship/reference.jpg)",
    )
    parser.add_argument(
        "--round1",
        type=Path,
        default=DEFAULT_DATA_DIR / "restored_1.png",
        help="First restored sensed image (default: data/ship/restored_1.png)",
    )
    parser.add_argument(
        "--round2",
        type=Path,
        default=DEFAULT_DATA_DIR / "restored_2.png",
        help="Second restored sensed image (default: data/ship/restored_2.png)",
    )
    parser.add_argument("--dataset", default="ship")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--contrastive-epochs", type=int, default=100)
    parser.add_argument("--similarity-batch-size", type=int, default=128)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--layer-confidences",
        default="1,1,1",
        help=(
            "Confidence priors for Original, round1 and round2, in that order; "
            "values must be between 0 and 1 (default: 1,1,1)."
        ),
    )
    parser.add_argument(
        "--min-candidate-confidence",
        type=float,
        default=0.35,
        help="Reject fused SIFT candidates below this score.",
    )
    parser.add_argument("--search-radius", type=float, default=40.0)
    parser.add_argument("--min-cosine", type=float, default=0.5)
    parser.add_argument("--min-margin", type=float, default=0.005)
    parser.add_argument("--sift-ratio-threshold", type=float, default=0.90)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--frontend-cache", type=Path, default=None,
                        help="Replay a verified identical-input frontend with --resume")
    parser.add_argument("--matching-iterations", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--max-candidate-points", type=int, default=3000)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--learned-ransac-threshold", type=float, default=1.3)
    parser.add_argument("--max-final-matches", type=int, default=24)
    parser.add_argument("--min-final-matches", type=int, choices=range(6,11), default=10)
    parser.add_argument("--relaxed-spatial", action="store_true")
    parser.add_argument("--target-brightness-threshold",type=float,default=30.)
    parser.add_argument("--warm-start",type=Path,default=None)
    parser.add_argument("--steps-per-epoch","--ship-steps-per-epoch",dest="ship_steps_per_epoch",type=int,default=12)
    parser.add_argument("--no-image-guidance", action="store_true",
                        help="Ablation: disable disjoint-tile whole-image guidance/validation")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0 < args.target_brightness_threshold <= 255:
        raise ValueError("Target brightness threshold must be in (0,255] on the 8-bit grayscale input")
    if args.ship_steps_per_epoch < 1:
        raise ValueError("Steps per epoch must be positive")
    inputs = [args.original, args.reference, args.round1, args.round2, RUNNER]
    if args.warm_start is not None:inputs.append(args.warm_start)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))
    with Image.open(args.original) as image:
        original_size = image.size
    for layer in (args.round1, args.round2):
        with Image.open(layer) as image:
            if image.size != original_size:
                raise ValueError(
                    f"Restoration layers must share the original pixel grid: "
                    f"{args.original}={original_size}, {layer}={image.size}"
                )
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot find a GPU")

    layer_confidences = [
        item.strip() for item in args.layer_confidences.split(",") if item.strip()
    ]
    if len(layer_confidences) != 3:
        raise ValueError(
            "--layer-confidences must contain exactly three values: "
            "Original,round1,round2"
        )

    command = [
        sys.executable,
        str(RUNNER),
        "--original", str(args.original.resolve()),
        "--reference", str(args.reference.resolve()),
        "--dataset", args.dataset,
        "--sr_candidates", str(args.round1.resolve()), str(args.round2.resolve()),
        "--layer_labels", "round1,round2",
        "--layer_confidences", ",".join(layer_confidences),
        "--min_candidate_confidence", str(args.min_candidate_confidence),
        "--sift_ratio_threshold", str(args.sift_ratio_threshold),
        "--similarity_search_radius", str(args.search_radius),
        "--min_cosine", str(args.min_cosine),
        "--min_margin", str(args.min_margin),
        "--seed", str(args.seed),
        "--temperature", str(args.temperature),
        "--matching_iterations", str(args.matching_iterations),
        "--max_candidate_points", str(args.max_candidate_points),
        "--ransac_threshold", str(args.ransac_threshold),
        "--final_ransac_threshold", str(args.learned_ransac_threshold),
        "--max_final_matches", str(args.max_final_matches),
        "--min_final_matches", str(args.min_final_matches),
        "--ship_steps_per_epoch", str(args.ship_steps_per_epoch),
        "--target_brightness_threshold",str(args.target_brightness_threshold),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--contrastive_lr", "0.0003",
        "--contrastive_epochs", str(args.contrastive_epochs),
        "--similarity_batch_size", str(args.similarity_batch_size),
        "--device", args.device,
    ]
    if args.output_root is not None:
        command.extend(["--output_root", str(args.output_root.resolve())])
    if args.relaxed_spatial:
        command.append("--relaxed_spatial")
    if args.resume:
        command.append("--resume")
    if args.frontend_cache is not None:
        command.extend(["--frontend_cache",str(args.frontend_cache.resolve())])
    if args.warm_start is not None:
        command.extend(["--warm_start",str(args.warm_start.resolve())])
    if args.no_image_guidance:
        command.append("--no_image_guidance")
    if args.prepare_only:
        command.append("--prepare_only")

    print("Three-layer SAR-SIFT/T0 -> center-layout SimCLR registration")
    print("  original =", args.original)
    print("  round1   =", args.round1)
    print("  round2   =", args.round2)
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
