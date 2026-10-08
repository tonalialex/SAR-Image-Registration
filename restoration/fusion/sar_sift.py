from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fusion.sar_sift_core import SARSIFT, SARSIFTConfig


@dataclass
class SarSiftConfig:
    ratio: float = 0.90
    ransac_threshold: float = 1.0
    fsc_threshold: float = 3.0
    max_features: int = 6000
    max_matches: int = 400
    processing_scale: float = 0.5


def _gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return image


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _draw_matches(sensed, reference, sensed_points, reference_points, path: Path) -> None:
    left = cv2.cvtColor(sensed, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    gap = max(8, int(round(reference.shape[1] * 0.01)))
    canvas = np.zeros((max(left.shape[0], right.shape[0]), left.shape[1] + gap + right.shape[1], 3), dtype=np.uint8)
    canvas[:left.shape[0], :left.shape[1]] = left
    canvas[:right.shape[0], left.shape[1] + gap:] = right
    canvas[:, left.shape[1]:left.shape[1] + gap] = 255
    for a, b in zip(sensed_points, reference_points):
        color = (0, 255, 0)
        p1 = tuple(np.round(a).astype(int))
        p2 = (int(round(b[0])) + left.shape[1] + gap, int(round(b[1])))
        cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 4, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def run_sar_sift(sensed_path, reference_path, output_dir, config=None, stage_name="stage") -> dict:
    config = config or SarSiftConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sensed = _gray(sensed_path)
    reference = _gray(reference_path)
    if sensed.shape != reference.shape:
        raise ValueError(f"两幅图像尺寸必须相同，当前为 {sensed.shape} 和 {reference.shape}")

    strict_config = SARSIFTConfig(
        ratio_test=config.ratio,
        # The supplied MATLAB SAR-SIFT uses a one-pixel FSC threshold. The
        # main pipeline's ransac_threshold is reserved for the later affine
        # refit and must not loosen the strict SAR-SIFT consensus stage.
        fsc_threshold=config.fsc_threshold,
        max_features=config.max_features,
        processing_scale=config.processing_scale,
    )
    matcher = SARSIFT(strict_config)
    inlier_reference, inlier_sensed, raw_reference, raw_sensed = matcher.match(
        reference, sensed, return_raw=True
    )
    if len(raw_sensed) > config.max_matches:
        raw_sensed = raw_sensed[:config.max_matches]
        raw_reference = raw_reference[:config.max_matches]

    if len(inlier_sensed) >= 3:
        # Keep the geometric model consistent with FSC: full affine in both
        # stages. The returned RANSAC mask is the actual final inlier set.
        transform, fit_mask = cv2.estimateAffine2D(
            inlier_sensed.astype(np.float32), inlier_reference.astype(np.float32),
            method=cv2.RANSAC, ransacReprojThreshold=max(1.0, config.ransac_threshold),
            maxIters=5000, confidence=0.999,
        )
        if transform is None or fit_mask is None:
            inlier_sensed = np.empty((0, 2), dtype=np.float64)
            inlier_reference = np.empty((0, 2), dtype=np.float64)
        else:
            keep = fit_mask.reshape(-1).astype(bool)
            if int(keep.sum()) < 3:
                inlier_sensed = np.empty((0, 2), dtype=np.float64)
                inlier_reference = np.empty((0, 2), dtype=np.float64)
                transform = None
            else:
                inlier_sensed = inlier_sensed[keep]
                inlier_reference = inlier_reference[keep]
    else:
        transform = None
    if transform is None:
        transform = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    if len(inlier_sensed):
        projected = cv2.transform(inlier_sensed.astype(np.float32)[None], transform)[0]
        errors = np.linalg.norm(projected - inlier_reference.astype(np.float32), axis=1)
    else:
        errors = np.empty((0,), dtype=np.float32)

    # Keep the complete ratio-test candidate set separate from the genuine
    # geometric inliers above. Restoration uses the latter, while multilayer
    # fusion and CroR must receive the former and perform their own screening.
    if len(raw_sensed):
        raw_projected = cv2.transform(raw_sensed.astype(np.float32)[None], transform)[0]
        raw_errors = np.linalg.norm(
            raw_projected - raw_reference.astype(np.float32), axis=1
        )
    else:
        raw_errors = np.empty((0,), dtype=np.float32)

    rows = []
    for index, (a, b, error) in enumerate(zip(inlier_sensed, inlier_reference, errors)):
        rows.append({
            "sensed_x": f"{a[0]:.6f}", "sensed_y": f"{a[1]:.6f}",
            "reference_x": f"{b[0]:.6f}", "reference_y": f"{b[1]:.6f}",
            "descriptor_distance": "0.0", "error": f"{float(error):.6f}", "inlier": 1,
        })
    _write_csv(output_dir / "sift_inlier_matches.csv", rows, ["sensed_x", "sensed_y", "reference_x", "reference_y", "descriptor_distance", "error", "inlier"])
    all_candidate_rows = []
    for sensed_point, reference_point, error in zip(
        raw_sensed, raw_reference, raw_errors
    ):
        all_candidate_rows.append({
            "sensed_x": f"{sensed_point[0]:.6f}",
            "sensed_y": f"{sensed_point[1]:.6f}",
            "reference_x": f"{reference_point[0]:.6f}",
            "reference_y": f"{reference_point[1]:.6f}",
            "descriptor_distance": "0.0",
            "error": f"{float(error):.6f}",
            "inlier": 0,
        })
    _write_csv(
        output_dir / "sift_all_matches.csv",
        all_candidate_rows,
        ["sensed_x", "sensed_y", "reference_x", "reference_y", "descriptor_distance", "error", "inlier"],
    )
    candidates = [{
        "x": r["sensed_x"], "y": r["sensed_y"], "reference_x": r["reference_x"], "reference_y": r["reference_y"],
        "confidence": f"{1.0 / (1.0 + float(r['error'])):.6f}", "support_count": 1,
        "restoration_prior": 1.0, "category": stage_name,
    } for r in all_candidate_rows]
    _write_csv(output_dir / "sift_candidate_details.csv", candidates, ["x", "y", "reference_x", "reference_y", "confidence", "support_count", "restoration_prior", "category"])
    np.savetxt(output_dir / "best_T0.txt", transform, fmt="%.10f")
    _draw_matches(sensed, reference, inlier_sensed, inlier_reference, output_dir / "sarsift_connecting_lines.png")
    report = {
        "stage": stage_name, "sensed": str(sensed_path), "reference": str(reference_path),
        "image_shape": list(sensed.shape), "raw_matches": int(len(raw_sensed)),
        "ratio_matches": int(len(raw_sensed)),
        "inlier_matches": int(len(inlier_sensed)), "all_candidate_matches": int(len(raw_sensed)), "algorithm": "strict_sar_sift_harris_logpolar_fsc",
        "processing_scale": config.processing_scale, "ratio_test": config.ratio, "fsc_threshold": strict_config.fsc_threshold,
        "transform": transform.tolist(),
    }
    (output_dir / "sarsift_stage_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
