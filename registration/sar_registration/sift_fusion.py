from __future__ import annotations

import json
import math
import itertools
from sar_registration.geometry import robust_affine, project, coverage
from scipy.spatial import cKDTree
from sar_registration.sar_sift_adapter import make_detector, detect_points
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from registration_metrics import calculate_registration_metrics


@dataclass
class FusionConfig:
    """Configuration for same-pixel-grid restoration-layer fusion."""

    max_match_size: int = 0
    min_matches: int = 8
    sift_ratio_threshold: float = 0.90
    ransac_threshold: float = 5.0
    coarse_displacement_gate: float = 24.0
    min_phase_correlation_response: float = 0.02
    phase_transform_consistency_threshold: float = 18.0
    transform_consensus_threshold: float = 10.0
    sift_max_features_per_image: int = 6000
    sift_contrast_threshold: float = 0.005
    sift_edge_margin: int = 8
    sift_cluster_radius: float = 3.0
    min_candidate_confidence: float = 0.35
    reference_patch_margin: int = 64
    max_candidate_points: int = 3000
    grid_rows: int = 16
    grid_cols: int = 16
    max_points_per_grid_cell: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RepresentationMatch:
    representation_index: int
    label: str
    path: str
    transform: np.ndarray
    sensed_points: np.ndarray
    reference_points: np.ndarray
    inlier_mask: np.ndarray
    quality: float
    median_error: float
    coverage: float
    inlier_ratio: float
    coarse_shift: tuple[float, float] | None = None
    coarse_response: float | None = None

    def summary(self) -> dict:
        return {
            "representation_index": self.representation_index,
            "label": self.label,
            "path": self.path,
            "matches": int(len(self.sensed_points)),
            "inliers": int(self.inlier_mask.sum()),
            "inlier_ratio": float(self.inlier_ratio),
            "median_error": float(self.median_error),
            "coverage": float(self.coverage),
            "quality": float(self.quality),
            "coarse_shift": (
                list(self.coarse_shift) if self.coarse_shift is not None else None
            ),
            "coarse_response": self.coarse_response,
            "transform": self.transform.tolist(),
        }


@dataclass
class CandidatePoint:
    point: np.ndarray
    projected_point: np.ndarray
    score: float
    support_count: int
    supports_original: bool
    category: str
    source_indices: tuple[int, ...]
    restoration_prior: float


@dataclass
class FusionResult:
    transform: np.ndarray
    sensed_candidate_points: np.ndarray
    projected_reference_points: np.ndarray
    candidate_scores: np.ndarray
    candidate_support_counts: np.ndarray
    candidate_categories: list[str]
    candidate_restoration_priors: np.ndarray
    layer_confidences: list[float]
    rejected_low_confidence_count: int
    representation_matches: list[RepresentationMatch]
    consensus_indices: list[int]
    representation_paths: list[Path]
    representation_labels: list[str]

    def save(
        self,
        output_dir: Path | str,
        reference_path: Path | str | None = None,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_dir / "best_T0.txt", self.transform, fmt="%.10f")
        np.savetxt(
            output_dir / "sift_candidates_sensed.csv",
            self.sensed_candidate_points,
            delimiter=",",
            fmt="%.4f",
            header="x,y",
            comments="",
        )
        np.savetxt(
            output_dir / "sift_candidates_projected_reference.csv",
            self.projected_reference_points,
            delimiter=",",
            fmt="%.4f",
            header="x,y",
            comments="",
        )
        with (output_dir / "sift_candidate_details.csv").open("w", encoding="utf-8") as handle:
            handle.write(
                "x,y,reference_x,reference_y,confidence,support_count,"
                "restoration_prior,category\n"
            )
            for point, projected, score, support, restoration_prior, category in zip(
                self.sensed_candidate_points,
                self.projected_reference_points,
                self.candidate_scores,
                self.candidate_support_counts,
                self.candidate_restoration_priors,
                self.candidate_categories,
            ):
                handle.write(
                    f"{point[0]:.4f},{point[1]:.4f},"
                    f"{projected[0]:.4f},{projected[1]:.4f},"
                    f"{score:.6f},{int(support)},{restoration_prior:.6f},{category}\n"
                )
        sift_visualizations = {}
        sift_metrics = {}
        if reference_path is not None:
            reference_image = _read_rgb(Path(reference_path))
            for match in self.representation_matches:
                sensed_image = _read_rgb(Path(match.path))
                safe_label = "".join(
                    char if char.isalnum() or char in "-_" else "_"
                    for char in match.label
                ).strip("_") or f"layer_{match.representation_index}"
                visualization_name = (
                    f"sift_matches_{match.representation_index:02d}_{safe_label}.png"
                )
                visualization = _draw_pair_visualization(
                    reference_image,
                    sensed_image,
                    match.reference_points,
                    match.sensed_points,
                    match.inlier_mask,
                    title=(
                        f"SAR-SIFT {match.label}: "
                        f"matches={len(match.sensed_points)}, "
                        f"inliers={int(match.inlier_mask.sum())}"
                    ),
                )
                Image.fromarray(visualization).save(output_dir / visualization_name)
                sift_visualizations[match.label] = visualization_name

                inlier_reference = match.reference_points[match.inlier_mask]
                inlier_sensed = match.sensed_points[match.inlier_mask]
                projected = _project(inlier_sensed, match.transform)
                errors = np.linalg.norm(projected - inlier_reference, axis=1)
                sift_metrics[match.label] = calculate_registration_metrics(
                    reference_points=inlier_reference,
                    errors=errors,
                    image_shape=reference_image.shape[:2],
                    source_points=inlier_sensed,
                )

            valid_indices = {
                match.representation_index for match in self.representation_matches
            }
            for representation_index, (path, label) in enumerate(
                zip(self.representation_paths, self.representation_labels)
            ):
                if representation_index in valid_indices:
                    continue
                sensed_image = _read_rgb(Path(path))
                safe_label = "".join(
                    char if char.isalnum() or char in "-_" else "_"
                    for char in label
                ).strip("_") or f"layer_{representation_index}"
                visualization_name = (
                    f"sift_matches_{representation_index:02d}_{safe_label}.png"
                )
                visualization = _draw_pair_visualization(
                    reference_image,
                    sensed_image,
                    np.empty((0, 2), dtype=np.float64),
                    np.empty((0, 2), dtype=np.float64),
                    np.empty((0,), dtype=bool),
                    title=f"SAR-SIFT {label}: no valid transform",
                )
                Image.fromarray(visualization).save(output_dir / visualization_name)
                sift_visualizations[label] = visualization_name
                sift_metrics[label] = {"Nred": 0, "RMSEall": None, "RMSEloo": None, "Pquad": None, "status": "sift_failed"}

        report = {
            "coarse_matcher": "SAR_SIFT_Harris_136D_angular_NNDR_FSC",
            "transform_estimator": "deduplicated_spatial_weighted_affine_RANSAC_IRLS",
            "metric_population": "retained_geometric_inliers",
            "phase_correlation_role": "diagnostic_only",
            "metric_note": "Internal correspondence residuals; no ground truth used",
            "coordinate_model": "All Original/SR restoration layers share one pixel grid.",
            "transform_direction": "Original/SR sensed coordinates -> HR reference coordinates",
            "representation_labels": self.representation_labels,
            "representation_paths": [str(path) for path in self.representation_paths],
            "layer_confidences": {
                label: confidence
                for label, confidence in zip(
                    self.representation_labels, self.layer_confidences
                )
            },
            "valid_transform_count": len(self.representation_matches),
            "consensus_match_indices": self.consensus_indices,
            "final_T0": self.transform.tolist(),
            "candidate_count": int(len(self.sensed_candidate_points)),
            "rejected_low_confidence_count": self.rejected_low_confidence_count,
            "candidate_categories": {
                name: self.candidate_categories.count(name)
                for name in sorted(set(self.candidate_categories))
            },
            "matches": [item.summary() for item in self.representation_matches],
            "sift_visualizations": sift_visualizations,
            "sift_metrics": sift_metrics,
        }
        (output_dir / "fusion_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _read_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("RGB"))


def _project(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = homogeneous @ transform.T
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.abs(projected[:, 2]) > 1e-9
    result[valid] = projected[valid, :2] / projected[valid, 2:3]
    return result


def _draw_pair_visualization(
    reference: np.ndarray,
    sensed: np.ndarray,
    reference_points: np.ndarray,
    sensed_points: np.ndarray,
    inlier_mask: np.ndarray,
    title: str,
    gap_width: int = 30,
) -> np.ndarray:
    """Draw reference/sensed correspondences side by side in RGB format."""
    if reference.ndim == 2:
        reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2RGB)
    if sensed.ndim == 2:
        sensed = cv2.cvtColor(sensed, cv2.COLOR_GRAY2RGB)
    ref_height, ref_width = reference.shape[:2]
    sensed_height, sensed_width = sensed.shape[:2]
    canvas_height = max(ref_height, sensed_height) + 42
    canvas_width = ref_width + gap_width + sensed_width
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    canvas[42 : 42 + ref_height, :ref_width] = reference
    sensed_x_offset = ref_width + gap_width
    canvas[42 : 42 + sensed_height, sensed_x_offset:] = sensed

    cv2.putText(
        canvas,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    for reference_point, sensed_point, is_inlier in zip(
        reference_points, sensed_points, np.asarray(inlier_mask, dtype=bool)
    ):
        reference_xy = tuple(np.rint(reference_point).astype(int) + [0, 42])
        sensed_xy = tuple(
            np.rint(sensed_point).astype(int) + [sensed_x_offset, 42]
        )
        color = (0, 180, 0) if is_inlier else (220, 60, 60)
        cv2.line(canvas, reference_xy, sensed_xy, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, reference_xy, 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, sensed_xy, 4, color, -1, cv2.LINE_AA)
    return canvas


def _resize_for_matching(image: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    if not max_size or max(image.shape[:2]) <= max_size:
        return image, 1.0
    scale = float(max_size) / float(max(image.shape[:2]))
    h, w = image.shape[:2]
    resized = cv2.resize(
        image,
        (max(16, round(w * scale)), max(16, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _coarse_phase_shift(
    sensed: np.ndarray, reference: np.ndarray
) -> tuple[tuple[float, float], float]:
    """Estimate sensed->reference translation only to reject gross SIFT outliers."""
    sensed_gray = cv2.cvtColor(sensed, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    reference_gray = (
        cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    )
    # Log compression reduces the influence of a few very bright SAR scatterers.
    sensed_gray = np.log1p(10.0 * sensed_gray)
    reference_gray = np.log1p(10.0 * reference_gray)
    window = cv2.createHanningWindow(
        (sensed_gray.shape[1], sensed_gray.shape[0]), cv2.CV_32F
    )
    shift, response = cv2.phaseCorrelate(
        sensed_gray * window, reference_gray * window
    )
    return (float(shift[0]), float(shift[1])), float(response)


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.ones(len(values), dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks


class ThreeLayerSIFTFusion:
    """Fuse Original, round1 and round2 representations with SIFT."""

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        if not 0.0 < self.config.sift_ratio_threshold < 1.0:
            raise ValueError("sift_ratio_threshold must lie strictly between 0 and 1")
        for name in ('grid_rows','grid_cols','max_points_per_grid_cell','max_candidate_points',
                     'sift_max_features_per_image'):
            if getattr(self.config,name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.config.min_matches < 4 or self.config.ransac_threshold <= 0:
            raise ValueError('Affine fitting requires min_matches >= 4 and a positive threshold')

    def _sift_correspondences(self, reference, sensed):
        """Use the supplied OTUS SAR-SIFT angular NNDR and FSC consensus."""
        matcher = make_detector(self.config.sift_max_features_per_image,
                                self.config.sift_ratio_threshold)
        reference = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
        sensed = cv2.cvtColor(sensed, cv2.COLOR_RGB2GRAY)
        ref, sen, raw_ref, raw_sen = matcher.match(reference, sensed, return_raw=True)
        print(f"SAR-SIFT raw={len(raw_sen)}, FSC inliers={len(sen)}", flush=True)
        return ref, sen

    def _match_representation(
        self,
        reference: np.ndarray,
        sensed: np.ndarray,
        index: int,
        label: str,
        path: Path,
    ) -> Optional[RepresentationMatch]:
        reference_match, reference_scale = _resize_for_matching(
            reference, self.config.max_match_size
        )
        sensed_match, sensed_scale = _resize_for_matching(
            sensed, self.config.max_match_size
        )
        reference_points, sensed_points = self._sift_correspondences(
            reference_match, sensed_match
        )
        if len(reference_points) < self.config.min_matches:
            return None
        # Preserve the existing phase guard only when both grids are comparable.
        # phaseCorrelate requires equal dimensions; unequal scales invalidate
        # the translation-only displacement comparison even for equal sizes.
        coarse_shift, coarse_response = (0.0, 0.0), 0.0
        if (reference_match.shape == sensed_match.shape
                and np.isclose(reference_scale, sensed_scale)):
            coarse_shift, coarse_response = _coarse_phase_shift(
                sensed_match, reference_match
            )
        reference_points = reference_points * np.array([reference.shape[1] / reference_match.shape[1], reference.shape[0] / reference_match.shape[0]])
        sensed_points = sensed_points * np.array([sensed.shape[1] / sensed_match.shape[1], sensed.shape[0] / sensed_match.shape[0]])
        if len(reference_points) < self.config.min_matches:
            return None
        transform, mask = robust_affine(sensed_points, reference_points,
                                         self.config.ransac_threshold)
        if transform is None or int(mask.sum()) < self.config.min_matches:
            return None
        coarse_shift_original = np.asarray(coarse_shift) / reference_scale
        errors = np.linalg.norm(
            _project(sensed_points, transform) - reference_points, axis=1
        )
        median_error = float(np.median(errors[mask]))
        inlier_ratio = float(mask.mean())
        inlier_reference = reference_points[mask]
        hull_area = (
            float(cv2.contourArea(cv2.convexHull(inlier_reference.astype(np.float32))))
            if len(inlier_reference) >= 3
            else 0.0
        )
        coverage = min(
            1.0,
            hull_area / max(float(reference.shape[0] * reference.shape[1]), 1.0),
        )
        quality = (
            float(mask.sum())
            * (0.5 + inlier_ratio)
            * (0.2 + math.sqrt(coverage))
            / (1.0 + median_error)
        )
        return RepresentationMatch(
            index,
            label,
            str(path),
            transform,
            sensed_points,
            reference_points,
            mask,
            quality,
            median_error,
            coverage,
            inlier_ratio,
            (
                float(coarse_shift_original[0]),
                float(coarse_shift_original[1]),
            ),
            coarse_response,
        )

    def _transform_distance(
        self, first: np.ndarray, second: np.ndarray, width: int, height: int
    ) -> float:
        anchors = np.asarray(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
                [(width - 1) / 2, 0],
                [(width - 1) / 2, height - 1],
                [0, (height - 1) / 2],
                [width - 1, (height - 1) / 2],
                [(width - 1) / 2, (height - 1) / 2],
            ],
            dtype=np.float64,
        )
        return float(
            np.median(
                np.linalg.norm(
                    _project(anchors, first) - _project(anchors, second), axis=1
                )
            )
        )

    def _consensus_component(
        self, matches: list[RepresentationMatch], width: int, height: int
    ) -> list[int]:
        adjacency = [set([index]) for index in range(len(matches))]
        for left in range(len(matches)):
            for right in range(left + 1, len(matches)):
                distance = self._transform_distance(
                    matches[left].transform, matches[right].transform, width, height
                )
                if distance <= self.config.transform_consensus_threshold:
                    adjacency[left].add(right)
                    adjacency[right].add(left)
        # Complete-link consensus prevents A~B~C chains admitting A!~C.
        components = [list(group) for count in range(1, len(matches) + 1)
                      for group in itertools.combinations(range(len(matches)), count)
                      if all(j in adjacency[i] for i, j in itertools.combinations(group, 2))]

        def component_score(component: list[int]) -> tuple:
            original_support = int(
                any(matches[index].representation_index == 0 for index in component)
            )
            total_inliers = sum(
                int(matches[index].inlier_mask.sum()) for index in component
            )
            total_quality = sum(matches[index].quality for index in component)
            return len(component), total_quality, total_inliers

        return max(components, key=component_score)

    def _refit_transform(
        self, matches: list[RepresentationMatch], component: list[int]
    ) -> np.ndarray:
        sensed = np.concatenate(
            [matches[index].sensed_points[matches[index].inlier_mask] for index in component]
        )
        reference = np.concatenate(
            [
                matches[index].reference_points[matches[index].inlier_mask]
                for index in component
            ]
        )
        # Same-grid representations often yield repeated SIFT pairs. Quantized
        # pair deduplication prevents one restoration layer from dominating.
        pair_key = np.rint(np.column_stack([sensed, reference]) * 2.0).astype(np.int64)
        _, unique_indices = np.unique(pair_key, axis=0, return_index=True)
        sensed = sensed[unique_indices]
        reference = reference[unique_indices]
        # Balance crowded cells and retain only a refinement that improves the
        # same pooled correspondence set, rather than comparing different sets.
        cells = np.floor(sensed / 32).astype(int)
        _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        weights = 1. / np.sqrt(counts[inverse])
        fitted, mask = robust_affine(sensed, reference, self.config.ransac_threshold, weights)
        hypotheses = [matches[index].transform for index in component]
        if fitted is not None:
            hypotheses.append(fitted)
        def loss(h):
            e = np.linalg.norm(project(sensed, h) - reference, axis=1)
            return float(np.average(np.minimum(e, self.config.ransac_threshold) ** 2, weights=weights))
        return min(hypotheses, key=loss).copy()

    def _detect_sift_observations(
        self, images: list[np.ndarray]
    ) -> list[tuple[np.ndarray, int, float]]:
        observations = []
        height, width = images[0].shape[:2]
        margin = max(0, int(self.config.sift_edge_margin))
        for representation_index, image in enumerate(images):
            points = detect_points(image, self.config.sift_max_features_per_image)
            # Core returns descending SAR-Harris response; preserve that rank.
            responses = np.linspace(1., 0., len(points))
            for point, response in zip(points, responses):
                if margin <= point[0] < width-margin and margin <= point[1] < height-margin:
                    observations.append((point, representation_index, float(response)))
        return observations

    def _cluster_sift(
        self,
        observations: list[tuple[np.ndarray, int, float]],
        transform: np.ndarray,
        reference_shape: tuple[int, int],
        representation_count: int,
        layer_confidences: list[float],
        anchor_points: np.ndarray | None = None,
    ) -> tuple[list[CandidatePoint], int]:
        observations.sort(key=lambda item: item[2], reverse=True)
        clusters: list[list[tuple[np.ndarray, int, float]]] = []
        centers: list[np.ndarray] = []
        for observation in observations:
            distances = (
                np.linalg.norm(np.asarray(centers) - observation[0], axis=1)
                if centers
                else np.asarray([])
            )
            index = int(np.argmin(distances)) if len(distances) else -1
            if index >= 0 and distances[index] <= self.config.sift_cluster_radius:
                clusters[index].append(observation)
                centers[index] = np.median(
                    np.asarray([item[0] for item in clusters[index]]), axis=0
                )
            else:
                clusters.append([observation])
                centers.append(observation[0].copy())

        candidates: list[CandidatePoint] = []
        anchor_tree = cKDTree(anchor_points) if anchor_points is not None and len(anchor_points) else None
        rejected_low_confidence = 0
        reference_height, reference_width = reference_shape
        margin = int(self.config.reference_patch_margin)
        for cluster in clusters:
            source_indices = tuple(sorted({item[1] for item in cluster}))
            points = np.asarray([item[0] for item in cluster])
            weights = np.asarray([(0.1 + item[2]) * layer_confidences[item[1]] for item in cluster])
            pair_distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
            representative = points[int(np.argmin((pair_distances * weights[None, :]).sum(axis=1)))]
            projected = _project(representative[None], transform)[0]
            if not np.isfinite(projected).all():
                continue
            if not (
                margin <= projected[0] < reference_width - margin
                and margin <= projected[1] < reference_height - margin
            ):
                continue
            responses = np.asarray([item[2] for item in cluster])
            spread = float(
                np.median(
                    np.linalg.norm(
                        np.asarray([item[0] for item in cluster]) - representative,
                        axis=1,
                    )
                )
            )
            support_count = len(source_indices)
            supports_original = 0 in source_indices
            stability = 1.0 / (1.0 + spread)
            source_responses = {
                source_index: max(
                    item[2] for item in cluster if item[1] == source_index
                )
                for source_index in source_indices
            }
            prior_weights = np.asarray(
                [0.1 + source_responses[index] for index in source_indices],
                dtype=np.float64,
            )
            restoration_prior = float(
                np.average(
                    [layer_confidences[index] for index in source_indices],
                    weights=prior_weights,
                )
            )
            # Layer identities have equal priors by default. Strong
            # restored-only detections can survive; proximity to verified pairs
            # rescues weaker responses, but never certifies a correspondence.
            local_support = 0.
            if anchor_tree is not None:
                distance, _ = anchor_tree.query(representative)
                local_support = float(np.exp(-distance / 32.))
            score = (
                0.45 * float(np.max(responses))
                + 0.15 * support_count / max(representation_count, 1)
                + 0.10 * stability
                + 0.15 * restoration_prior
                + 0.15 * local_support
            )
            if score < self.config.min_candidate_confidence:
                rejected_low_confidence += 1
                continue
            if supports_original:
                category = "original"
            elif support_count >= 2:
                category = "multi_restoration"
            else:
                category = f"single_{source_indices[0]}"
            candidates.append(
                CandidatePoint(
                    representative,
                    projected,
                    score,
                    support_count,
                    supports_original,
                    category,
                    source_indices,
                    restoration_prior,
                )
            )
        return candidates, rejected_low_confidence

    @staticmethod
    def _resolve_layer_confidences(
        representation_count: int,
        provided: Optional[Iterable[float]],
    ) -> list[float]:
        if provided is None:
            return [1.0] * representation_count
        values = [float(value) for value in provided]
        if len(values) != representation_count:
            raise ValueError(
                "layer_confidences must include Original and match all representations"
            )
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("layer_confidences values must be within [0, 1]")
        return values

    def _spatially_select(
        self,
        candidates: list[CandidatePoint],
        sensed_shape: tuple[int, int],
    ) -> list[CandidatePoint]:
        height, width = sensed_shape
        cell_counts: dict[tuple[int, int], int] = {}

        def cell(candidate: CandidatePoint) -> tuple[int, int]:
            col = min(
                self.config.grid_cols - 1,
                int(candidate.point[0] * self.config.grid_cols / max(width, 1)),
            )
            row = min(
                self.config.grid_rows - 1,
                int(candidate.point[1] * self.config.grid_rows / max(height, 1)),
            )
            return row, col

        buckets = {}
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            buckets.setdefault(cell(candidate), []).append(candidate)
        selected = []
        # Round-robin equalizes coverage without fixed Original/restoration quotas.
        depth = 0
        while len(selected) < self.config.max_candidate_points:
            available = [items[depth] for items in buckets.values() if len(items) > depth]
            if not available:
                break
            available.sort(key=lambda item: item.score, reverse=True)
            selected.extend(available[:self.config.max_candidate_points - len(selected)])
            depth += 1
            if depth >= self.config.max_points_per_grid_cell:
                break
        # The per-cell quota is soft: do not discard otherwise valid points just
        # because many cells are empty (coasts, ships, partial overlap).
        ids = {id(item) for item in selected}
        remaining = sorted((item for item in candidates if id(item) not in ids),
                           key=lambda item: item.score, reverse=True)
        selected.extend(remaining[:max(0, self.config.max_candidate_points - len(selected))])
        return sorted(selected, key=lambda item: item.score, reverse=True)

    def run(
        self,
        reference_path: Path | str,
        representation_paths: Iterable[Path | str],
        representation_labels: Optional[Iterable[str]] = None,
        layer_confidences: Optional[Iterable[float]] = None,
    ) -> FusionResult:
        reference_path = Path(reference_path)
        paths = [Path(path) for path in representation_paths]
        if not paths:
            raise ValueError("Original image plus at least zero SR images is required")
        labels = (
            list(representation_labels)
            if representation_labels is not None
            else ["Original"] + [path.stem for path in paths[1:]]
        )
        if len(labels) != len(paths):
            raise ValueError("representation_labels must match representation_paths")
        resolved_confidences = self._resolve_layer_confidences(
            len(paths), layer_confidences
        )
        images = [_read_rgb(path) for path in paths]
        sensed_shape = images[0].shape[:2]
        for path, image in zip(paths, images):
            if image.shape[:2] != sensed_shape:
                raise ValueError(
                    "All Original/SR restoration layers must have the same pixel size: "
                    f"{paths[0]}={sensed_shape}, {path}={image.shape[:2]}"
                )
        reference = _read_rgb(reference_path)
        matches = []
        for index, (path, label, image) in enumerate(zip(paths, labels, images)):
            print(f"Matching SAR-SIFT layer: {label}", flush=True)
            item = self._match_representation(
                reference, image, index, label, path
            )
            if item is not None:
                matches.append(item)
        if not matches:
            raise RuntimeError("No Original/SR restoration layer produced a valid SIFT T0")
        component = self._consensus_component(
            matches, sensed_shape[1], sensed_shape[0]
        )
        transform = self._refit_transform(matches, component)

        observations = self._detect_sift_observations(images)
        candidates, rejected_low_confidence = self._cluster_sift(
            observations,
            transform,
            reference.shape[:2],
            len(images),
            resolved_confidences,
            np.concatenate([matches[i].sensed_points[matches[i].inlier_mask] for i in component]),
        )
        candidates = self._spatially_select(candidates, sensed_shape)
        if len(candidates) < 4:
            raise RuntimeError(
                f"Only {len(candidates)} SIFT candidates remain after projection/boundary filtering"
            )
        return FusionResult(
            transform=transform,
            sensed_candidate_points=np.asarray(
                [item.point for item in candidates], dtype=np.float64
            ).reshape(-1, 2),
            projected_reference_points=np.asarray(
                [item.projected_point for item in candidates], dtype=np.float64
            ).reshape(-1, 2),
            candidate_scores=np.asarray(
                [item.score for item in candidates], dtype=np.float64
            ),
            candidate_support_counts=np.asarray(
                [item.support_count for item in candidates], dtype=np.int64
            ),
            candidate_categories=[item.category for item in candidates],
            candidate_restoration_priors=np.asarray(
                [item.restoration_prior for item in candidates], dtype=np.float64
            ),
            layer_confidences=resolved_confidences,
            rejected_low_confidence_count=rejected_low_confidence,
            representation_matches=matches,
            consensus_indices=component,
            representation_paths=paths,
            representation_labels=labels,
        )


