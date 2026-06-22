from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class SharpnessEvaluator(ABC):
    """Interface for face-crop sharpness evaluation."""

    @abstractmethod
    def score(self, gray: np.ndarray) -> tuple[float, float, float]:
        """
        Compute a normalised sharpness score for a greyscale image patch.

        Returns
        -------
        sharpness_score : float in [0, 1]  — 1 = perfectly sharp, 0 = completely blurry
        metric_a        : float            — first raw metric value
        metric_b        : float            — second raw metric value
        """


class LaplacianTenengradEvaluator(SharpnessEvaluator):
    """
    Sharpness evaluator combining Laplacian variance and Tenengrad.

    Each raw metric is normalised with  1 / (1 + x / scale), where *scale* is
    calibrated so that a value at the scale equals 0.5 — placing it on the
    medium sensitivity boundary.  The two components are then blended 60 / 40
    and inverted so that 1 = sharp and 0 = blurry.
    """

    _LAP_SCALE: float = 100.0    # Laplacian variance reference
    _TEN_SCALE: float = 5_000.0  # Tenengrad reference

    def score(self, gray: np.ndarray) -> tuple[float, float, float]:
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        gx      = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy      = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        ten     = float(np.mean(gx ** 2 + gy ** 2))

        # Map each raw metric to a blur component in (0, 1) using a monotone-
        # decreasing curve; sharper images yield blur components closer to 0.
        lap_component = 1.0 / (1.0 + lap_var / self._LAP_SCALE)
        ten_component = 1.0 / (1.0 + ten    / self._TEN_SCALE)

        # Weighted combination then invert so that 1 = sharp, 0 = blurry.
        blur = float(np.clip(0.6 * lap_component + 0.4 * ten_component, 0.0, 1.0))
        return 1.0 - blur, lap_var, ten


class GeometricMeanEvaluator(SharpnessEvaluator):
    """
    Sharpness evaluator using the geometric mean of Laplacian variance and Tenengrad.

    Unlike the weighted-average approach, the geometric mean only scores high
    when *both* metrics agree — penalising mismatches where one metric is
    inflated by noise, compression artefacts, eyelashes, or over-sharpening
    while the other stays low.

    score = sqrt((1 - lap_component) * (1 - ten_component))
    """

    _LAP_SCALE: float = 100.0
    _TEN_SCALE: float = 5_000.0

    def score(self, gray: np.ndarray) -> tuple[float, float, float]:
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        gx      = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy      = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        ten     = float(np.mean(gx ** 2 + gy ** 2))

        lap_sharp = 1.0 - 1.0 / (1.0 + lap_var / self._LAP_SCALE)
        ten_sharp = 1.0 - 1.0 / (1.0 + ten    / self._TEN_SCALE)

        score = float(np.sqrt(np.clip(lap_sharp * ten_sharp, 0.0, 1.0)))
        return score, lap_var, ten


sharpness_evaluator: SharpnessEvaluator = GeometricMeanEvaluator()
