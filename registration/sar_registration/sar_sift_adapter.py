"""Adapter for the user's OTUS SAR-SIFT core (kept byte-for-byte unchanged).

Memoize detection across the three pairwise matches and candidate extraction.
Matching remains the source implementation's angular NNDR + FSC, scale=0.5.
"""
from __future__ import annotations
import hashlib
import cv2
import numpy as np
from sar_registration.sar_sift_core import SARSIFT, SARSIFTConfig

_CACHE = {}


class CachedSARSIFT(SARSIFT):
    def detect_and_compute(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        key = (gray.shape, gray.dtype.str, repr(self.config), hashlib.sha256(gray.tobytes()).digest())
        if key not in _CACHE:
            points, descriptors = super().detect_and_compute(gray)
            # Keep bounded memory across repeated pipeline invocations.
            if len(_CACHE) >= 8:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = points, descriptors
        return _CACHE[key]


def make_detector(maximum=6000, ratio=.90):
    return CachedSARSIFT(SARSIFTConfig(max_features=maximum, ratio_test=ratio,
                                     processing_scale=.5, fsc_threshold=3.0))


def detect_points(image, maximum=6000):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    h,w = gray.shape[:2]
    work = cv2.resize(gray,(max(1,round(w*.5)),max(1,round(h*.5))),interpolation=cv2.INTER_LINEAR)
    points,_ = make_detector(maximum).detect_and_compute(work)
    return points.astype(np.float64)/.5
