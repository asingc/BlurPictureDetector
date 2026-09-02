from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from algo.config import app_config


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


class WeightedGeometricMeanEvaluator(SharpnessEvaluator):
    """
    Sharpness evaluator using a weighted geometric mean of Tenengrad and
    Laplacian variance, leaning heavily (80/20) on Tenengrad.

    Validated (2026-07-27) against real ground truth (`Anno_Blur` in album
    info.json, 823 images / 2499 face crops across 2 albums) via
    `_setup_tmp/sharpness_eval/`: Tenengrad alone is the strongest standalone
    focus-blur signal found (AUC 0.686 vs plain GeometricMeanEvaluator's
    0.676), and this 80/20 blend keeps a small Laplacian "agreement" term so
    a lone Tenengrad outlier still gets penalised somewhat (AUC 0.683). Beats
    the previous default at "medium" sensitivity (acc 0.779 vs 0.774) — see
    SENSITIVITY_PRESETS in culling_app.py, whose "high" threshold was
    recalibrated alongside this swap to preserve the same recall the old
    evaluator achieved at its "high" setting.

    score = (lap_sharp ** (1 - w)) * (ten_sharp ** w), w = 0.8
    """

    _LAP_SCALE: float = 100.0
    _TEN_SCALE: float = 5_000.0
    _TEN_WEIGHT: float = 0.8

    def score(self, gray: np.ndarray) -> tuple[float, float, float]:
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        gx      = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy      = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        ten     = float(np.mean(gx ** 2 + gy ** 2))

        lap_sharp = 1.0 - 1.0 / (1.0 + lap_var / self._LAP_SCALE)
        ten_sharp = 1.0 - 1.0 / (1.0 + ten    / self._TEN_SCALE)

        w = self._TEN_WEIGHT
        score = float(np.clip(max(lap_sharp, 0.0) ** (1 - w) * max(ten_sharp, 0.0) ** w, 0.0, 1.0))
        return score, lap_var, ten


def normalize_patch_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Rescale *gray* so its standard deviation matches a fixed reference,
    returning a float64 array (values may fall outside 0-255; every metric in
    this module works on floats, so no clipping is applied).

    Laplacian variance and Tenengrad both measure *absolute* gradient energy,
    which scales with the square of the patch's signal amplitude.  Two faces
    photographed at identical focus therefore score very differently when one
    reflects less light than the other — dark skin tones, backlit subjects and
    underexposed frames all get systematically penalised against the fixed
    ``_LAP_SCALE`` / ``_TEN_SCALE`` references.

    Dividing by the patch's own contrast cancels that amplitude factor, so the
    metrics become *relative* sharpness measures ("how much gradient energy per
    unit of contrast") rather than absolute ones.  Standard deviation is used
    as the contrast estimate because a face patch's spread is dominated by
    large-scale structure (hair vs. skin, eye sockets, shadow terminator) that
    survives defocus, whereas the gradient metrics being normalised are driven
    by the fine detail that defocus destroys — so the ratio keeps its blur
    sensitivity.

    Returns the input unchanged (as float64) when normalization is disabled or
    the patch is degenerate.
    """
    g = gray.astype(np.float64)
    if not app_config.sharpness_contrast_normalize:
        return g
    sd = float(g.std())
    if sd <= 1e-6:
        return g
    gain = float(np.clip(
        app_config.sharpness_contrast_reference / sd,
        app_config.sharpness_contrast_min_gain,
        app_config.sharpness_contrast_max_gain,
    ))
    return g * gain


class ContrastNormalizedEvaluator(WeightedGeometricMeanEvaluator):
    """
    Production evaluator: :class:`WeightedGeometricMeanEvaluator` applied to a
    contrast-normalised patch (see :func:`normalize_patch_contrast`).

    Motivation (2026-08-31): faces with dark skin tones were being reported as
    blurry when they were plainly in focus.  Measured on the cached real-photo
    ground truth in ``_setup_tmp/sharpness_eval`` (823 photos / 2499 face crops
    across two albums), the un-normalised score correlates +0.31 with face-crop
    brightness, and among photos the photographer marked *sharp* the darkest
    half was flagged blurry far more often than the brightest half
    (19.9 % vs 8.7 % at the "low" preset).

    Normalising each crop's contrast removes that dependence (correlation
    +0.31 -> +0.07) *and* improves raw discrimination — ROC AUC 0.672 -> 0.713
    overall, improving on both albums individually (0.603 -> 0.712 and
    0.716 -> 0.711), with accuracy/precision/F1 up at every sensitivity preset.

    The score scale shifts slightly, so the "low" and "high" sensitivity
    presets were recalibrated to preserve the previous recall (0.35 -> 0.40 and
    0.68 -> 0.62); "medium" stays at 0.50.  See ``SENSITIVITY_PRESETS`` in
    culling_app.py and ``SENSITIVITY_THRESHOLDS`` in 1_prep_review.py.
    """

    def score(self, gray: np.ndarray) -> tuple[float, float, float]:
        return super().score(normalize_patch_contrast(gray))


# ---------------------------------------------------------------------------
# Noise-robust building blocks
# ---------------------------------------------------------------------------

# 3x3 kernel orthogonal to constant / linear (gradient) / quadratic image
# content (Immerkaer, "Fast Noise Variance Estimation", CVIU 1996).  Convolving
# a clean image with it yields ~0 almost everywhere except at high-order
# (noise-like) content, so its response isolates additive sensor noise from
# genuine low-order image structure (edges, gradients).
_NOISE_KERNEL = np.array([[1.0, -2.0, 1.0],
                           [-2.0,  4.0, -2.0],
                           [1.0, -2.0, 1.0]])
_NOISE_KERNEL_SUMSQ = float(np.sum(_NOISE_KERNEL ** 2))  # = 36

# Sum of squared coefficients of the two operators used elsewhere in this
# module -- used to convert an estimated noise sigma into the variance that
# additive i.i.d. noise alone would contribute to each raw metric, so that
# contribution can be subtracted back out.
_LAPLACIAN_KERNEL_SUMSQ = 4.0 ** 2 + 4 * 1.0 ** 2       # cv2.Laplacian 3x3 default: [[0,1,0],[1,-4,1],[0,1,0]]  == 20
_SOBEL_KERNEL_SUMSQ     = 4 * 1.0 ** 2 + 4 * 2.0 ** 2    # cv2.Sobel ksize=3: [[-1,0,1],[-2,0,2],[-1,0,1]]        == 20


def _estimate_noise_sigma(gray: np.ndarray) -> float:
    """
    Fast estimate of the additive (sensor/ISO) noise standard deviation in a
    greyscale patch, using Immerkaer's method.  Returns 0.0 for patches too
    small to estimate reliably.
    """
    h, w = gray.shape[:2]
    if h < 5 or w < 5:
        return 0.0
    conv = cv2.filter2D(gray.astype(np.float64), -1, _NOISE_KERNEL, borderType=cv2.BORDER_REPLICATE)
    # Drop the outer ring: BORDER_REPLICATE biases the response right at the edge.
    inner = conv[1:-1, 1:-1]
    if inner.size == 0:
        return 0.0
    sigma = np.sqrt(np.pi / 2.0) * float(np.mean(np.abs(inner))) / np.sqrt(_NOISE_KERNEL_SUMSQ)
    return max(0.0, sigma)


def _denoise_corrected_metrics(gray: np.ndarray) -> tuple[float, float, float]:
    """
    Compute Laplacian-variance and Tenengrad the same way as the other
    evaluators, then analytically subtract the contribution that additive
    sensor/ISO noise alone would be expected to make to each metric.

    Both raw metrics are, at their core, the output variance of a small
    high-pass kernel.  For i.i.d. noise with standard deviation *sigma*, that
    kernel contributes an expected variance of ``sigma**2 * sum(kernel**2)``
    regardless of the underlying (noise-free) image content.  Subtracting
    this expected contribution keeps genuine edge energy while discounting
    grain, so a noisy-but-blurry crop no longer reads as sharp.

    Returns (sigma, corrected_lap_var, corrected_ten).
    """
    sigma = _estimate_noise_sigma(gray)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx      = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy      = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    ten     = float(np.mean(gx ** 2 + gy ** 2))

    noise_var = sigma ** 2
    lap_var_corrected = max(0.0, lap_var - _LAPLACIAN_KERNEL_SUMSQ * noise_var)
    ten_corrected      = max(0.0, ten     - 2 * _SOBEL_KERNEL_SUMSQ * noise_var)  # gx and gy each contribute
    return sigma, lap_var_corrected, ten_corrected


def _blur_effect_metric(gray: np.ndarray, kernel_size: int = 9) -> float:
    """
    No-reference perceptual blur metric (Crete et al., "The Blur Effect:
    Perception and Estimation with a New No-Reference Perceptual Blur
    Metric", 2007).

    Idea: re-blur the image with a known low-pass filter and see how much
    local pixel-to-pixel variation survives.  An edge that is already blurry
    loses little extra variation when blurred again; a genuinely sharp edge
    loses a lot.  Because the metric is a *ratio* of variation-lost to
    variation-present at each pixel (evaluated only where the original image
    actually has some local variation), it is largely insensitive to the
    absolute contrast/noise level of the crop -- unlike raw gradient-energy
    metrics, sensor noise inflates numerator and denominator together instead
    of only inflating the "looks sharp" signal.

    Returns a sharpness score in [0, 1] where 1 = sharp, 0 = blurry.
    """
    img = gray.astype(np.float64)
    k = max(3, kernel_size | 1)  # ensure odd

    blur_h = cv2.blur(img, (1, k))  # blurred vertically -> probes horizontal detail loss
    blur_v = cv2.blur(img, (k, 1))  # blurred horizontally -> probes vertical detail loss

    d_f_h = np.abs(img[:, 1:] - img[:, :-1])
    d_f_v = np.abs(img[1:, :] - img[:-1, :])
    d_b_h = np.abs(blur_h[:, 1:] - blur_h[:, :-1])
    d_b_v = np.abs(blur_v[1:, :] - blur_v[:-1, :])

    v_h = np.clip(d_f_h - d_b_h, 0.0, None)
    v_v = np.clip(d_f_v - d_b_v, 0.0, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        s_h = np.where(d_f_h > 1e-6, v_h / d_f_h, 0.0)
        s_v = np.where(d_f_v > 1e-6, v_v / d_f_v, 0.0)

    # Only average over pixels with meaningful original variation -- flat
    # regions (background, saturated skin) carry no blur information and
    # would otherwise dilute the score toward 0 regardless of true sharpness.
    informative_h = d_f_h > 1e-6
    informative_v = d_f_v > 1e-6
    n_informative = int(informative_h.sum() + informative_v.sum())
    if n_informative == 0:
        return 0.0

    blur_extent = (s_h[informative_h].sum() + s_v[informative_v].sum()) / n_informative
    return float(np.clip(1.0 - blur_extent, 0.0, 1.0))


class NoiseRobustEvaluator(SharpnessEvaluator):
    """
    Sharpness evaluator designed to resist the two failure modes reported in
    the field: (a) high-ISO grain inflating gradient/Laplacian energy enough
    to read a blurry crop as sharp, and (b) genuinely sharp crops scoring low
    because a single noise-prone metric dipped.

    Combines three independent signals via geometric mean (all three must
    roughly agree for a high score):
      1. Noise-corrected Laplacian variance  -- fine-detail energy, with the
         expected sensor-noise contribution analytically subtracted out.
      2. Noise-corrected Tenengrad           -- gradient energy, same
         noise correction applied.
      3. Blur-effect ratio (Crete et al.)    -- a contrast/noise-insensitive
         perceptual measure of how much detail a further blur would destroy.

    metric_a / metric_b returned are the noise-corrected Laplacian variance
    and Tenengrad (for continuity with existing logging/CSV columns).
    """

    _LAP_SCALE: float = 100.0
    _TEN_SCALE: float = 5_000.0

    def score(self, gray: np.ndarray) -> tuple[float, float, float]:
        _sigma, lap_var, ten = _denoise_corrected_metrics(gray)

        lap_sharp = 1.0 - 1.0 / (1.0 + lap_var / self._LAP_SCALE)
        ten_sharp = 1.0 - 1.0 / (1.0 + ten    / self._TEN_SCALE)
        blur_sharp = _blur_effect_metric(gray)

        score = float(np.clip(
            (max(lap_sharp, 0.0) * max(ten_sharp, 0.0) * max(blur_sharp, 0.0)) ** (1.0 / 3.0),
            0.0, 1.0,
        ))
        return score, lap_var, ten


sharpness_evaluator: SharpnessEvaluator = ContrastNormalizedEvaluator()
