from __future__ import annotations

import numpy as np


RMSELOO_METHOD = "leave_one_out_affine_prediction"


def predictive_loo_rmse(source, target, weights=None) -> float:
    """Refit on N-1 pairs and predict the excluded pair; never reselect points."""
    from sar_registration.geometry import project, weighted_affine

    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    if source.shape != target.shape or not np.isfinite([source, target]).all():
        raise ValueError("One finite source point per reference point is required")
    weights = np.ones(len(source)) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(weights) != len(source) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("One finite nonnegative weight per correspondence is required")
    if len(source) < 4:
        return float("inf")
    errors = []
    for index in range(len(source)):
        keep = np.arange(len(source)) != index
        try:
            transform = weighted_affine(source[keep], target[keep], weights[keep])
            errors.append(np.linalg.norm(project(source[index:index+1], transform)[0] - target[index]))
        except (ValueError, np.linalg.LinAlgError):
            return float("inf")
    return float(np.sqrt(np.mean(np.square(errors))))


def calculate_registration_metrics(
    reference_points: np.ndarray,
    errors: np.ndarray,
    image_shape: tuple[int, int],
    *,
    source_points: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict:
    """Calculate final point-based registration metrics.

    Pquad is the maximum fraction of correspondences in any of the four
    image quadrants, following Mao et al., IEEE TGRS, 2023. Lower is better.
    RMSEloo refits an affine transform on N-1 retained pairs and measures the
    excluded pair's prediction residual. Learned fits use confidence weights;
    SAR-SIFT fits use equal weights. Undefined/degenerate LOO is reported as None.
    """
    nred = int(len(reference_points))
    if nred == 0:
        return {"Nred": 0, "RMSEall": None, "RMSEloo": None, "Pquad": None,
                "RMSEloo_method": RMSELOO_METHOD}

    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    if len(errors) != nred or not np.isfinite(errors).all():
        raise ValueError("One finite residual per correspondence is required")
    reference_points = np.asarray(reference_points, dtype=np.float64).reshape(-1,2)
    rmse_all = float(np.sqrt(np.mean(np.square(errors))))
    if len(reference_points) != nred or len(source_points) != nred:
        raise ValueError("Source/reference point counts must match the residual count")
    loo = predictive_loo_rmse(source_points, reference_points, weights)
    rmse_loo = loo if np.isfinite(loo) else None

    height, width = image_shape[:2]
    cx, cy = width / 2.0, height / 2.0
    x, y = reference_points[:, 0], reference_points[:, 1]
    counts = np.array([
        np.count_nonzero((x < cx) & (y < cy)),
        np.count_nonzero((x >= cx) & (y < cy)),
        np.count_nonzero((x < cx) & (y >= cy)),
        np.count_nonzero((x >= cx) & (y >= cy)),
    ], dtype=np.int64)

    return {
        "Nred": nred,
        "RMSEall": rmse_all,
        "RMSEloo": rmse_loo,
        "RMSEloo_method": RMSELOO_METHOD,
        "Pquad": float(counts.max() / nred),
        "Pquad_quadrant_counts": counts.tolist(),
    }
