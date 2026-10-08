from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _read_t0(path: str | Path) -> np.ndarray:
    matrix = np.loadtxt(str(path), dtype=np.float64)
    return np.asarray(matrix).reshape(2, 3)


def _project(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return cv2.transform(points.reshape(1, -1, 2).astype(np.float32), transform)[0]


def _write_csv(path: Path, rows: Iterable[dict], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _round_candidates(match_path: str | Path) -> list[dict]:
    rows = _read_csv(match_path)
    result = []
    for row in rows:
        result.append({
            "sensed_x": float(row["sensed_x"]),
            "sensed_y": float(row["sensed_y"]),
            "reference_x": float(row["reference_x"]),
            "reference_y": float(row["reference_y"]),
            "error": float(row.get("error", 0.0) or 0.0),
        })
    return result


def _choose_consistent_round(
    t0s: Sequence[np.ndarray],
    image_shape: tuple[int, int],
    threshold: float,
) -> tuple[int, list[float]]:
    """Choose the T0 medoid and retain all geometrically consistent rounds."""
    height, width = image_shape
    anchors = np.float32([
        [0, 0], [width - 1, 0], [0, height - 1],
        [width - 1, height - 1], [width / 2, height / 2],
    ])
    projected = [_project(anchors, transform) for transform in t0s]
    distances = []
    for index, current in enumerate(projected):
        pairwise = [
            float(np.median(np.linalg.norm(current - other, axis=1)))
            for other_index, other in enumerate(projected)
            if other_index != index
        ]
        distances.append(float(np.median(pairwise)) if pairwise else 0.0)
    chosen = int(np.argmin(distances)) if distances else 0
    return chosen, distances


def fuse_rounds_direct(
    round_match_paths: Sequence[str | Path],
    round_t0_paths: Sequence[str | Path],
    output_dir: str | Path,
    image_shape: tuple[int, int],
    t0_consistency_threshold: float = 40.0,
    point_merge_radius: float = 5.0,
    final_ransac_threshold: float = 5.0,
) -> dict:
    """Fuse any number of SAR-SIFT rounds without CroR."""
    if len(round_match_paths) != len(round_t0_paths) or not round_match_paths:
        raise ValueError("match files and T0 files must have the same nonzero length")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0s = [_read_t0(path) for path in round_t0_paths]
    chosen, t0_distances = _choose_consistent_round(
        t0s, image_shape, t0_consistency_threshold
    )
    consensus_transform = t0s[chosen].copy()
    accepted_rounds = [
        index for index, distance in enumerate(t0_distances)
        if distance <= t0_consistency_threshold
    ]
    if not accepted_rounds:
        accepted_rounds = [chosen]

    all_points = []
    round_counts = []
    for round_index in accepted_rounds:
        points = _round_candidates(round_match_paths[round_index])
        round_counts.append({
            "round": round_index + 1,
            "candidate_points": len(points),
        })
        for point in points:
            point["round"] = round_index + 1
            all_points.append(point)

    fused = []
    for point in sorted(all_points, key=lambda item: (item["error"], item["round"])):
        source = np.array([point["sensed_x"], point["sensed_y"]], dtype=np.float32)
        duplicate = None
        for index, existing in enumerate(fused):
            other = np.array(
                [existing["sensed_x"], existing["sensed_y"]], dtype=np.float32
            )
            if np.linalg.norm(source - other) <= point_merge_radius:
                duplicate = index
                break
        if duplicate is None:
            point["support_count"] = 1
            fused.append(point)
        else:
            fused[duplicate]["support_count"] += 1
            if point["error"] < fused[duplicate]["error"]:
                support = fused[duplicate]["support_count"]
                fused[duplicate].update(point)
                fused[duplicate]["support_count"] = max(2, support)

    sensed_points = np.float32(
        [[p["sensed_x"], p["sensed_y"]] for p in fused]
    ).reshape(-1, 2)
    reference_points = np.float32(
        [[p["reference_x"], p["reference_y"]] for p in fused]
    ).reshape(-1, 2)

    final_transform = t0s[chosen].copy()
    final_mask = np.ones(len(fused), dtype=np.uint8)
    if len(fused) >= 3:
        # Use the same full-affine model as SAR-SIFT/FSC for final fusion.
        estimated, mask = cv2.estimateAffine2D(
            sensed_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=final_ransac_threshold,
            maxIters=5000,
            confidence=0.999,
        )
        if estimated is not None:
            final_transform = estimated
            final_mask = mask.reshape(-1).astype(np.uint8)
    if len(fused) == 0:
        final_mask = np.empty((0,), dtype=np.uint8)

    rows = []
    for index, point in enumerate(fused):
        row = dict(point)
        row["final_inlier"] = int(final_mask[index])
        row["source_round"] = point["round"]
        rows.append(row)

    fields = [
        "sensed_x", "sensed_y", "reference_x", "reference_y", "error",
        "round", "support_count", "final_inlier", "source_round",
    ]
    _write_csv(output_dir / "fused_matches.csv", rows, fields)
    # Keep both transforms explicit: consensus_T0 is the transform selected by
    # the multilayer consistency stage; final_transform is the optional affine
    # refit used only for the direct-fusion audit result. CroR receives the
    # former so that CroR, rather than this RANSAC refit, performs final point
    # rejection.
    np.savetxt(output_dir / "consensus_T0.txt", consensus_transform, fmt="%.10f")
    np.savetxt(output_dir / "final_transform.txt", final_transform, fmt="%.10f")

    report = {
        "method": "multi_round_sarsift_candidate_fusion_before_cror",
        "candidate_source_policy": "all_sar_sift_ratio_matches_before_round_ransac",
        "accepted_rounds": [index + 1 for index in accepted_rounds],
        "consensus_round": chosen + 1,
        "t0_anchor_median_distance": t0_distances,
        "consensus_transform": consensus_transform.tolist(),
        "round_counts": round_counts,
        "pooled_points": len(all_points),
        "fused_points": len(fused),
        "final_inlier_points": int(final_mask.sum()),
        "transform": final_transform.tolist(),
    }
    (output_dir / "direct_fusion_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return {
        **report,
        "transform_array": final_transform,
        "consensus_transform_array": consensus_transform,
        "fused_rows": rows,
        "final_mask": final_mask,
    }


fuse_two_rounds_direct = fuse_rounds_direct
