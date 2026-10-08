"""SAR-SIFT implementation aligned with the supplied MATLAB version.

The reference implementation is
``Image-Registration-master/SAR-SIFT-matlab-V1.0``.  This module keeps the
small Python API used by the registration pipeline, but follows the MATLAB
choices that affect the correspondences: SAR-Harris response thresholding,
the 136-D log-polar descriptor, one-way angular NNDR matching, and FSC
affine consensus with a one-pixel error threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SARSIFTConfig:
    sigma: float = 2.0
    layers: int = 8
    ratio: float = 2.0 ** (1.0 / 3.0)
    harris_d: float = 0.04
    harris_threshold: float = 0.8
    orientation_bins: int = 36
    descriptor_angular_bins: int = 8
    descriptor_spatial_bins: int = 8
    max_features: int | None = None
    ratio_test: float = 0.90
    # FSC geometric-consistency threshold in pixels. A larger value is a
    # looser gate and is needed here to recover more SAR candidates.
    fsc_threshold: float = 3.0
    processing_scale: float = 0.5
    fsc_iterations: int = 800


def _matlab_round(values: np.ndarray | float) -> np.ndarray:
    """MATLAB round for the signed values used by SAR-SIFT."""
    array = np.asarray(values, dtype=np.float64)
    return np.sign(array) * np.floor(np.abs(array) + 0.5)


def _gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image.astype(np.float64, copy=False)
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0
    # sar_sift.m adds this offset before build_scale. It prevents the
    # logarithmic ratio filters from producing invalid values in dark areas.
    return np.maximum(image, 0.0) + 0.001


def _matlab_gaussian_kernel(scale: float) -> np.ndarray:
    sigma = np.sqrt(2.0) * scale
    width = int(_matlab_round(3.0 * sigma))
    axis = np.arange(-width, width + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    kernel[(xx * xx + yy * yy) > width * width] = 0.0
    return kernel.astype(np.float32)


def _filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(
        image.astype(np.float32, copy=False),
        cv2.CV_64F,
        kernel.astype(np.float64, copy=False),
        borderType=cv2.BORDER_REPLICATE,
    )


def _sar_scale(image: np.ndarray, scale: float, d: float):
    """Port of SAR-SIFT MATLAB ``build_scale.m`` for one scale."""
    radius = int(_matlab_round(2.0 * scale))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    weight = np.exp(-(np.abs(xx) + np.abs(yy)) / scale)
    center = radius
    w34 = np.zeros_like(weight)
    w12 = np.zeros_like(weight)
    w14 = np.zeros_like(weight)
    w23 = np.zeros_like(weight)
    w34[center + 1 :, :] = weight[center + 1 :, :]
    w12[:center, :] = weight[:center, :]
    w14[:, center + 1 :] = weight[:, center + 1 :]
    w23[:, :center] = weight[:, :center]

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gx = np.log(_filter(image, w14) / _filter(image, w23))
        gy = np.log(_filter(image, w34) / _filter(image, w12))
    gx[~np.isfinite(gx)] = 0.0
    gy[~np.isfinite(gy)] = 0.0

    gradient = np.hypot(gx, gy)
    angle = np.degrees(np.arctan2(gy, gx))
    angle[angle < 0.0] += 360.0

    c11 = scale * scale * gx * gx
    c12 = scale * scale * gx * gy
    c22 = scale * scale * gy * gy
    gaussian = _matlab_gaussian_kernel(scale)
    c11 = _filter(c11, gaussian)
    c12 = _filter(c12, gaussian)
    c22 = _filter(c22, gaussian)
    response = c11 * c22 - c12 * c12 - d * (c11 + c22) ** 2
    return response, gradient, angle


def _orientation_hist(gradient, angle, x, y, scale, bins=36):
    """Port of ``calculate_oritation_hist.m``."""
    height, width = gradient.shape
    radius = int(_matlab_round(min(6.0 * scale, height / 2.0, width / 2.0)))
    x0, x1 = max(1, x - radius), min(width, x + radius)
    y0, y1 = max(1, y - radius), min(height, y + radius)
    gx = gradient[y0 - 1 : y1, x0 - 1 : x1]
    ga = angle[y0 - 1 : y1, x0 - 1 : x1]
    yy, xx = np.mgrid[y0 - 1 - y : y1 - y, x0 - 1 - x : x1 - x]
    inside = xx * xx + yy * yy <= radius * radius
    raw_bin = _matlab_round(ga * bins / 360.0).astype(np.int64)
    raw_bin[raw_bin >= bins] -= bins
    raw_bin[raw_bin < 0] += bins
    histogram = np.bincount(
        raw_bin[inside].ravel(), weights=gx[inside].ravel(), minlength=bins
    ).astype(np.float64)
    smoothed = (
        np.roll(histogram, 2)
        + 4.0 * np.roll(histogram, 1)
        + 6.0 * histogram
        + 4.0 * np.roll(histogram, -1)
        + np.roll(histogram, -2)
    ) / 16.0
    return smoothed


def _descriptor(gradient, angle, x, y, scale, main_angle, d=8, n=8):
    """Port of ``calc_log_polar_descriptor.m`` (hard histogram bins)."""
    height, width = gradient.shape
    radius = int(_matlab_round(min(12.0 * scale, min(height, width) / 2.0)))
    x0, x1 = max(1, x - radius), min(width, x + radius)
    y0, y1 = max(1, y - radius), min(height, y + radius)
    g = gradient[y0 - 1 : y1, x0 - 1 : x1]
    a = angle[y0 - 1 : y1, x0 - 1 : x1]
    yy, xx = np.mgrid[y0 - 1 - y : y1 - y, x0 - 1 - x : x1 - x]

    # MATLAB uses 1..n bins and maps zero to bin n. Keep that convention
    # explicitly; using modulo-zero here changes the descriptor phase.
    vertical = _matlab_round((a - main_angle) * n / 360.0).astype(np.int64)
    vertical[vertical <= 0] += n
    vertical[vertical == 0] = n
    vertical -= 1

    theta = np.deg2rad(-main_angle)
    c, s = np.cos(theta), np.sin(theta)
    c_rot = xx * c - yy * s
    r_rot = xx * s + yy * c
    log_angle = np.degrees(np.arctan2(r_rot, c_rot))
    log_angle[log_angle < 0.0] += 360.0
    log_angle = _matlab_round(log_angle * d / 360.0).astype(np.int64)
    log_angle[log_angle <= 0] += d
    log_angle[log_angle > d] -= d

    with np.errstate(divide="ignore", invalid="ignore"):
        log_amplitude = np.log2(np.hypot(c_rot, r_rot))
    r1 = np.log2(radius * 0.25)
    r2 = np.log2(radius * 0.73)
    amplitude = np.where(log_amplitude <= r1, 1, np.where(log_amplitude <= r2, 2, 3))
    inside = xx * xx + yy * yy <= radius * radius

    descriptor = np.zeros((2 * d + 1) * n, dtype=np.float64)
    rows, cols = np.nonzero(inside)
    for row, col in zip(rows.tolist(), cols.tolist()):
        orient = int(vertical[row, col])
        amp = int(amplitude[row, col])
        if amp == 1:
            index = orient
        else:
            index = ((amp - 2) * d + int(log_angle[row, col]) - 1) * n
            index += orient + n
        descriptor[index] += g[row, col]

    norm = float(np.sqrt(descriptor @ descriptor))
    if norm > 0.0:
        descriptor /= norm
        descriptor[descriptor > 0.2] = 0.2
        norm = float(np.sqrt(descriptor @ descriptor))
        if norm > 0.0:
            descriptor /= norm
    return descriptor.astype(np.float32)


def _fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    if len(source) < 3:
        return None
    design = np.column_stack([source, np.ones(len(source), dtype=np.float64)])
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return np.asarray(
        [
            [coefficients[0, 0], coefficients[1, 0], coefficients[2, 0]],
            [coefficients[0, 1], coefficients[1, 1], coefficients[2, 1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _project(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = homogeneous @ transform.T
    return projected[:, :2] / projected[:, 2:3]


def _fsc(
    source: np.ndarray,
    target: np.ndarray,
    error_threshold: float,
    iterations_limit: int,
) -> tuple[np.ndarray | None, np.ndarray]:
    """FSC affine consensus matching the MATLAB implementation."""
    count = len(source)
    if count < 3:
        return None, np.zeros(count, dtype=bool)
    iterations = min(iterations_limit, count * (count - 1) * (count - 2) // 6)
    rng = np.random.RandomState(0)
    best_count = 0
    best_mask = np.zeros(count, dtype=bool)
    for _ in range(max(0, iterations)):
        while True:
            sample = np.floor(1.0 + (count - 1) * rng.rand(3)).astype(np.int64)
            if len(np.unique(sample)) == 3:
                break
        sample_source = source[sample]
        sample_target = target[sample]
        if np.linalg.matrix_rank(np.column_stack([sample_source, np.ones(3)])) < 3:
            continue
        transform = _fit_affine(sample_source, sample_target)
        if transform is None:
            continue
        errors = np.linalg.norm(_project(source, transform) - target, axis=1)
        mask = errors < error_threshold
        consensus = int(mask.sum())
        if consensus > best_count:
            best_count = consensus
            best_mask = mask

    if best_count < 3:
        return None, best_mask
    transform = _fit_affine(source[best_mask], target[best_mask])
    return transform, best_mask


class SARSIFT:
    def __init__(self, config: SARSIFTConfig | None = None):
        self.config = config or SARSIFTConfig()

    def detect_and_compute(self, image: np.ndarray):
        image = _gray_float(image)
        height, width = image.shape
        responses, gradients, angles = [], [], []
        for layer in range(self.config.layers):
            scale = self.config.sigma * self.config.ratio**layer
            response, gradient, angle = _sar_scale(image, scale, self.config.harris_d)
            responses.append(response)
            gradients.append(gradient)
            angles.append(angle)

        candidates = []
        border = 2
        peak_ratio = 0.8
        for layer, response in enumerate(responses):
            scale = self.config.sigma * self.config.ratio**layer
            for y in range(border, height - border):
                for x in range(border, width - border):
                    value = response[y, x]
                    if value <= self.config.harris_threshold:
                        continue
                    neighborhood = response[y - 1 : y + 2, x - 1 : x + 2]
                    if not np.all(value > np.delete(neighborhood.ravel(), 4)):
                        continue
                    histogram = _orientation_hist(
                        gradients[layer], angles[layer], x + 1, y + 1, scale,
                        self.config.orientation_bins,
                    )
                    threshold = peak_ratio * float(histogram.max(initial=0.0))
                    for bin_index in range(len(histogram)):
                        left = histogram[(bin_index - 1) % len(histogram)]
                        right = histogram[(bin_index + 1) % len(histogram)]
                        if not (histogram[bin_index] > left
                                and histogram[bin_index] > right
                                and histogram[bin_index] > threshold):
                            continue
                        denominator = left + right - 2.0 * histogram[bin_index]
                        offset = (
                            0.5 * (left - right) / denominator
                            if abs(denominator) > 1e-12 else 0.0
                        )
                        peak_bin = bin_index + offset
                        if peak_bin < 0:
                            peak_bin += len(histogram)
                        elif peak_bin >= len(histogram):
                            peak_bin -= len(histogram)
                        candidates.append(
                            (
                                float(value),
                                x + 1,
                                y + 1,
                                scale,
                                (360.0 / len(histogram)) * peak_bin,
                                layer + 1,
                            )
                        )

        candidates.sort(key=lambda item: item[0], reverse=True)
        if self.config.max_features is not None:
            candidates = candidates[: self.config.max_features]
        points, descriptors = [], []
        for _, x, y, scale, main_angle, layer in candidates:
            points.append((float(x - 1), float(y - 1)))
            descriptors.append(
                _descriptor(
                    gradients[layer - 1],
                    angles[layer - 1],
                    x,
                    y,
                    scale,
                    main_angle,
                    self.config.descriptor_spatial_bins,
                    self.config.descriptor_angular_bins,
                )
            )
        if not points:
            return np.empty((0, 2), np.float32), np.empty((0, 136), np.float32)
        return np.asarray(points, dtype=np.float32), np.asarray(descriptors, dtype=np.float32)

    def match(
        self,
        reference: np.ndarray,
        sensed: np.ndarray,
        processing_scale: float | None = None,
        return_raw: bool = False,
    ):
        """Return reference/sensed FSC inliers in original-image coordinates."""
        scale = (
            self.config.processing_scale
            if processing_scale is None else processing_scale
        )
        if not 0.0 < scale <= 1.0:
            raise ValueError("processing_scale must lie in (0, 1]")
        if scale != 1.0:
            ref_h, ref_w = reference.shape[:2]
            sen_h, sen_w = sensed.shape[:2]
            reference_work = cv2.resize(
                reference, (max(1, round(ref_w * scale)), max(1, round(ref_h * scale))),
                interpolation=cv2.INTER_LINEAR,
            )
            sensed_work = cv2.resize(
                sensed, (max(1, round(sen_w * scale)), max(1, round(sen_h * scale))),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            reference_work, sensed_work = reference, sensed

        ref_points, ref_desc = self.detect_and_compute(reference_work)
        sen_points, sen_desc = self.detect_and_compute(sensed_work)
        if len(ref_desc) < 2 or len(sen_desc) < 2:
            empty = np.empty((0, 2), dtype=np.float64)
            return (empty, empty, empty, empty) if return_raw else (empty, empty)

        # MATLAB match.m uses one-way NNDR from sensed descriptors to reference
        # descriptors and ranks acos(dot products), not Euclidean distances.
        dot_products = np.clip(sen_desc @ ref_desc.T, -1.0, 1.0)
        distances = np.arccos(dot_products)
        raw_sensed, raw_reference = [], []
        for index in range(len(sen_desc)):
            order = np.argsort(distances[index], kind="stable")
            if len(order) >= 2 and distances[index, order[0]] < self.config.ratio_test * distances[index, order[1]]:
                raw_sensed.append(sen_points[index])
                raw_reference.append(ref_points[order[0]])
        raw_sensed = np.asarray(raw_sensed, dtype=np.float64).reshape(-1, 2)
        raw_reference = np.asarray(raw_reference, dtype=np.float64).reshape(-1, 2)

        if len(raw_sensed):
            paired = np.column_stack([raw_sensed, raw_reference])
            _, unique = np.unique(paired, axis=0, return_index=True)
            unique = np.sort(unique)
            raw_sensed = raw_sensed[unique]
            raw_reference = raw_reference[unique]

        transform, mask = _fsc(
            raw_sensed,
            raw_reference,
            self.config.fsc_threshold,
            self.config.fsc_iterations,
        )
        if transform is None:
            inlier_sensed = np.empty((0, 2), dtype=np.float64)
            inlier_reference = np.empty((0, 2), dtype=np.float64)
        else:
            inlier_sensed = raw_sensed[mask]
            inlier_reference = raw_reference[mask]
            # MATLAB removes duplicate source/target points after FSC and then
            # refits the affine model. The points themselves are the result
            # consumed by the caller and by the matching-line visualization.
            if len(inlier_sensed):
                _, keep_s = np.unique(inlier_sensed, axis=0, return_index=True)
                keep_s = np.sort(keep_s)
                inlier_sensed = inlier_sensed[keep_s]
                inlier_reference = inlier_reference[keep_s]
                _, keep_r = np.unique(inlier_reference, axis=0, return_index=True)
                keep_r = np.sort(keep_r)
                inlier_sensed = inlier_sensed[keep_r]
                inlier_reference = inlier_reference[keep_r]

        if scale != 1.0:
            inlier_sensed /= scale
            inlier_reference /= scale
            raw_sensed /= scale
            raw_reference /= scale
        if return_raw:
            return inlier_reference, inlier_sensed, raw_reference, raw_sensed
        return inlier_reference, inlier_sensed
