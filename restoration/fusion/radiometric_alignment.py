from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return image.astype(np.float32) / 255.0


def _robust_signal_gain(source: np.ndarray, target: np.ndarray) -> float:
    """Estimate display-intensity gain from non-background SAR signal pixels."""
    if source.size < 100:
        return 1.0
    ratios = []
    for quantile in (90.0, 95.0, 97.0):
        src_q = float(np.percentile(source, quantile))
        tgt_q = float(np.percentile(target, quantile))
        if src_q > 1e-6:
            ratios.append(tgt_q / src_q)
    if not ratios:
        return 1.0
    return float(np.clip(np.median(ratios), 0.5, 4.0))


def _monotonic_quantile_lut(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build the original zero-preserving monotonic signal mapping."""
    if source.size < 100 or target.size != source.size:
        return None
    quantiles = np.asarray([0, 25, 50, 75, 90, 95, 97, 99, 100], dtype=np.float64)
    source_knots = np.percentile(source, quantiles)
    target_knots = np.percentile(target, quantiles)
    keep = np.r_[True, np.diff(source_knots) > 1e-6]
    source_knots = source_knots[keep]
    target_knots = target_knots[keep]
    if len(source_knots) < 3:
        return None
    source_knots = np.maximum.accumulate(source_knots)
    target_knots = np.maximum.accumulate(np.clip(target_knots, 0.0, 1.0))
    return np.r_[0.0, source_knots], np.r_[0.0, target_knots]


def align_radiometry(restored_path, reference_path, t0_path, output_path, fit_mask=None) -> dict:
    """Original flow: radiometrically align the restored image after NAFNet."""
    restored = _read_gray(restored_path)
    reference = _read_gray(reference_path)
    if restored.shape != reference.shape:
        raise ValueError("辐射校正要求复原图和参考图尺寸相同")
    transform = np.loadtxt(str(t0_path), dtype=np.float64).reshape(2, 3)
    valid = cv2.warpAffine(
        np.ones_like(restored), transform,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
    ) > 0.5
    aligned = cv2.warpAffine(
        restored, transform,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    if fit_mask is not None:
        valid &= np.asarray(fit_mask, bool)
    base_threshold = 2.0 / 255.0
    src_positive = aligned[valid & (aligned > base_threshold)]
    ref_positive = reference[valid & (reference > base_threshold)]
    if len(src_positive) >= 100 and len(ref_positive) >= 100:
        src_threshold = max(base_threshold, float(np.percentile(src_positive, 10.0)))
        ref_threshold = max(base_threshold, float(np.percentile(ref_positive, 10.0)))
        signal = valid & (aligned >= src_threshold) & (reference >= ref_threshold)
        source_fit = aligned[signal].astype(np.float64)
        target_fit = reference[signal].astype(np.float64)
    else:
        src_threshold = base_threshold
        ref_threshold = base_threshold
        source_fit = np.empty((0,), dtype=np.float64)
        target_fit = np.empty((0,), dtype=np.float64)
    if len(source_fit) > 300000:
        rng = np.random.default_rng(2026)
        selected = rng.choice(len(source_fit), size=300000, replace=False)
        source_fit = source_fit[selected]
        target_fit = target_fit[selected]
    lut = _monotonic_quantile_lut(source_fit, target_fit)
    if lut is None:
        gain = _robust_signal_gain(source_fit, target_fit)
        source_knots = np.asarray([0.0, 1.0], dtype=np.float64)
        target_knots = np.asarray([0.0, gain], dtype=np.float64).clip(0.0, 1.0)
        method = "robust_signal_only_multiplicative_gain_fallback"
    else:
        source_knots, target_knots = lut
        gain = float(target_knots[min(2, len(target_knots) - 1)] / max(
            source_knots[min(2, len(source_knots) - 1)], 1e-6
        ))
        method = "robust_signal_only_monotonic_quantile_mapping"
    corrected = np.zeros_like(restored, dtype=np.float32)
    positive = restored > base_threshold
    corrected[positive] = np.interp(restored[positive], source_knots, target_knots).astype(np.float32)
    corrected = np.clip(corrected, 0.0, 1.0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.rint(corrected * 255.0).astype(np.uint8))
    report = {
        "method": method,
        "restored_input": str(restored_path),
        "reference": str(reference_path),
        "t0": str(t0_path),
        "output": str(output_path),
        "gain": gain,
        "bias": 0.0,
        "gain_a": gain,
        "bias_b": 0.0,
        "fit_pixels": int(len(source_fit)),
        "valid_overlap_pixels": int(valid.sum()),
        "source_signal_threshold": src_threshold,
        "reference_signal_threshold": ref_threshold,
        "mapping_source_knots": source_knots.tolist(),
        "mapping_target_knots": target_knots.tolist(),
        "preserves_zero_background": True,
    }
    (output_path.parent / "radiometric_alignment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
