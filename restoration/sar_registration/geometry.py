"""Robust, normalized affine fitting shared by fusion and learned matching."""
from __future__ import annotations
import cv2
import numpy as np


def project(points, transform):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    q = np.column_stack((p, np.ones(len(p)))) @ np.asarray(transform).T
    return np.divide(q[:, :2], q[:, 2:3], out=np.full((len(p), 2), np.nan),
                     where=np.abs(q[:, 2:3]) > 1e-10)


def sane(transform):
    if transform is None or not np.isfinite(transform).all():
        return False
    s = np.linalg.svd(transform[:2, :2], compute_uv=False)
    return bool(np.linalg.det(transform[:2, :2]) > 0 and s.min() > .05
                and s.max() < 20 and s.max() / s.min() < 10)


def weighted_affine(source, target, weights):
    center = np.mean(source, axis=0)
    scale = max(float(np.std(source)), 1.)
    x = np.column_stack(((source - center) / scale, np.ones(len(source))))
    w = np.sqrt(np.maximum(weights, 1e-8))[:, None]
    coef, _, rank, _ = np.linalg.lstsq(x * w, target * w, rcond=None)
    if rank < 3:
        raise ValueError('Degenerate/collinear affine correspondences')
    transform = np.eye(3)
    transform[:2, :2] = coef[:2].T / scale
    transform[:2, 2] = coef[2] - transform[:2, :2] @ center
    return transform


def robust_affine(source, target, threshold=5., weights=None):
    """RANSAC initialization + spatially weighted Huber IRLS; returns H, mask."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if len(source) < 4 or source.shape != target.shape or not np.isfinite([source, target]).all():
        return None, np.zeros(len(source), bool)
    affine, mask = cv2.estimateAffine2D(source, target, method=cv2.RANSAC,
        ransacReprojThreshold=threshold, maxIters=20000, confidence=.999, refineIters=20)
    if affine is None or mask is None or mask.sum() < 4:
        return None, np.zeros(len(source), bool)
    h = np.vstack((affine, [0., 0., 1.]))
    base = np.ones(len(source)) if weights is None else np.asarray(weights, float)
    keep = mask.ravel().astype(bool)
    for _ in range(8):
        errors = np.linalg.norm(project(source, h) - target, axis=1)
        sigma = max(.5, 1.4826 * np.median(np.abs(errors[keep] - np.median(errors[keep]))))
        robust = np.minimum(1., 1.345 * sigma / np.maximum(errors, 1e-8))
        try:
            proposal = weighted_affine(source[keep], target[keep], (base * robust)[keep])
        except (ValueError, np.linalg.LinAlgError):
            break
        if not sane(proposal):
            break
        h = proposal
        next_keep = np.linalg.norm(project(source, h) - target, axis=1) <= threshold
        if next_keep.sum() < 4:
            break
        keep = next_keep
    if not sane(h):
        return None, np.zeros(len(source), bool)
    return h, np.linalg.norm(project(source, h) - target, axis=1) <= threshold


def coverage(points, shape):
    if len(points) < 3:
        return 0.
    return float(cv2.contourArea(cv2.convexHull(np.asarray(points, np.float32)))) / max(1, shape[0] * shape[1])
