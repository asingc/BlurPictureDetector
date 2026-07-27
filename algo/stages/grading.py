from __future__ import annotations

import logging

import cv2
import numpy as np

from algo.config import AppConfig
from algo.frame import Frame
from algo.models import Body, ColorLab
from algo.scorers import (
    BodyArrayScorer,
    BodyHeadKPVisibilityScorer,
    FaceSharpnessScorer,
    FaceSizeScorer,
    MatchedFaceScorer,
)
from algo.stage import ProcessStage

log = logging.getLogger("BlurPictureDetector")


# ---------------------------------------------------------------------------
# Cloth colour predictor  (moved from 1_prep_review.py)
# ---------------------------------------------------------------------------

class ClothColorPredictor:
    """Predicts the dominant jersey/cloth color for a body.

    Strategy
    --------
    1. Crop the torso region using COCO keypoints 5/6 (shoulders) and 11/12
       (hips) when at least two are confident.  Falls back to the middle band
       of the body bbox (skip top 25 % head, bottom 20 % legs).
    2. Resize the crop to a small 24 × 24 sample grid.
    3. Convert to CIE L*a*b* and assign each pixel to the nearest reference
       color by Euclidean distance.  Skin-tone pixels are skipped.
    4. The colour with the most votes is returned.  ``mean_lab`` (in the
       returned detail dict) is the per-channel *median* L*a*b* of only the
       pixels that voted for the winning color — restricting to the winning
       color's pixels avoids diluting the value with shadowed folds, jersey
       numbers/logos or background bleed at the crop edges, and the median
       (vs. a plain mean) limits the influence of any remaining outlier
       pixels (specular highlights, stray contamination).
    """

    _TORSO_KP_INDICES: tuple[int, ...] = (5, 6, 11, 12)
    _TORSO_KP_CONF:    float            = 0.30

    _COLORS: list[ColorLab] = [
        ColorLab("Red",        "Crimson", ( 40.0,  65.0,  40.0)),
        ColorLab("Orange",     "Vivid",   ( 65.0,  35.0,  55.0)),
        ColorLab("Yellow",     "Gold",    ( 85.0,  -5.0,  75.0)),
        ColorLab("Green",      "Emerald", ( 45.0, -40.0,  25.0)),
        ColorLab("Light Blue", "Sky",     ( 70.0,  -8.0, -30.0)),
        ColorLab("Blue",       "Royal",   ( 35.0,   5.0, -55.0)),
        ColorLab("Blue",       "Navy",    ( 15.0,   5.0, -25.0)),
        ColorLab("Blue",       "Deep Blue", ( 12.0,  10.0, -42.0)),
        ColorLab("Purple",     "Violet",  ( 30.0,  30.0, -35.0)),
        ColorLab("Pink",       "Magenta", ( 55.0,  60.0, -20.0)),
        ColorLab("Pink",       "Deep Magenta", ( 28.0,  48.0, -12.0)),
        ColorLab("White",      "Bright",  ( 95.0,   0.0,   0.0)),
        ColorLab("Gray",       "75%",     ( 75.0,   0.0,   0.0)),
        ColorLab("Gray",       "Medium",  ( 50.0,   0.0,   0.0)),
        ColorLab("Gray",       "25%",     ( 25.0,   0.0,   0.0)),
        ColorLab("Black",      "Deep",    (  8.0,   0.0,   0.0)),
    ]

    _SKIN_LAB:         tuple[float, float, float] = (65.0, 18.0, 22.0)
    _SKIN_DIST_THRESH: float                      = 35.0

    def predict(self, body: Body, normalized_image: np.ndarray) -> tuple[str, dict]:
        torso = self._torso_crop(body, normalized_image)
        if torso is None or torso.size == 0:
            return "N/A", {}
        sample = cv2.resize(torso, (24, 24), interpolation=cv2.INTER_AREA)
        lab    = cv2.cvtColor(sample.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape(-1, 3)

        colors = self._COLORS
        refs   = np.array([c.lab for c in colors], dtype=np.float32)
        skin   = np.array(self._SKIN_LAB, dtype=np.float32)

        diffs       = pixels[:, None, :] - refs[None, :, :]
        nearest_idx = (diffs ** 2).sum(axis=2).argmin(axis=1)

        skin_dists = np.sqrt(((pixels - skin) ** 2).sum(axis=1))
        is_skin    = skin_dists < self._SKIN_DIST_THRESH

        votes: dict[int, int] = {}
        pixels_by_color: dict[int, list[np.ndarray]] = {}
        for i, (idx, skip) in enumerate(zip(nearest_idx.tolist(), is_skin.tolist())):
            if skip:
                continue
            votes[idx] = votes.get(idx, 0) + 1
            pixels_by_color.setdefault(idx, []).append(pixels[i])

        if not votes:
            return "Unknown", {"votes": {}, "mean_lab": None}
        winner_idx = max(votes, key=votes.__getitem__)
        winner     = colors[winner_idx]
        # Use only the pixels that actually voted for the winning color — mixing
        # in pixels nearest to *other* reference colors (shadowed folds, jersey
        # numbers, background bleed at the crop edges) would drag the average
        # away from the true jersey color instead of just reflecting brightness.
        # A per-channel median (not mean) further limits the influence of any
        # remaining small contaminated patch (specular highlight, stray
        # background pixels) within the winning-color pixel set.
        winner_pixels = pixels_by_color[winner_idx]
        mean_lab = [round(float(v), 1) for v in np.median(winner_pixels, axis=0)]
        votes_by_label = {colors[k].label: v for k, v in votes.items()}
        return winner.label, {"votes": votes_by_label, "mean_lab": mean_lab}

    def _torso_crop(self, body: Body, image: np.ndarray) -> np.ndarray | None:
        h_img, w_img = image.shape[:2]
        kps = body.keypoints
        pts = [
            (kps[i].point.x * w_img, kps[i].point.y * h_img)
            for i in self._TORSO_KP_INDICES
            if i < len(kps) and kps[i].confidence >= self._TORSO_KP_CONF
        ]
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1 = max(0, int(np.floor(min(xs))))
            y1 = max(0, int(np.floor(min(ys))))
            x2 = min(w_img, int(np.ceil(max(xs))))
            y2 = min(h_img, int(np.ceil(max(ys))))
        else:
            b  = body.bbox
            bh = b.y2 - b.y1
            x1 = int(np.floor(b.x1 * w_img))
            x2 = int(np.ceil(b.x2 * w_img))
            y1 = int(np.floor((b.y1 + (bh * 0.25)) * h_img))
            y2 = int(np.ceil((b.y2 - (bh * 0.20)) * h_img))

        x1 = max(0, min(x1, w_img))
        x2 = max(0, min(x2, w_img))
        y1 = max(0, min(y1, h_img))
        y2 = max(0, min(y2, h_img))
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]


cloth_color_predictor = ClothColorPredictor()


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class GradingStage(ProcessStage):
    """Run all body scorers against every Frame and predict cloth colours.

    Scorers are applied in order; a body that fails early is short-circuited.
    Cloth colour is predicted only for bodies that pass all scorers.
    All Frame/Body mutations are in-place (shallow — no deep copies).
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def process(self, frames: list[Frame], config: AppConfig) -> list[Frame]:
        scorer = BodyArrayScorer([
            MatchedFaceScorer(),
            FaceSizeScorer(),
            BodyHeadKPVisibilityScorer(),
            FaceSharpnessScorer(self.threshold),
        ])

        total = len(frames)
        log.info("[GradingStage] grading %d frame(s)", total)
        progress_step = max(1, total // 20)
        width = len(str(total))
        passed_bodies = 0
        total_bodies = 0

        for idx, frame in enumerate(frames, 1):
            if frame.normalized_image is None:
                continue

            try:
                frame.bodies = scorer.process(frame.normalized_image, frame.bodies)

                for body in frame.bodies:
                    total_bodies += 1
                    if not body.passed:
                        continue
                    passed_bodies += 1
                    body.cloth_color, body.cloth_color_detail = cloth_color_predictor.predict(
                        body, frame.normalized_image
                    )
                    bx1, by1, bx2, by2 = body.bbox.as_px_ints(
                        frame.normalized_image.shape[1], frame.normalized_image.shape[0]
                    )
                    log.debug("[GradingStage] %s — body bbox=(%d,%d,%d,%d) score=%.4f cloth=%s",
                              frame.path.name, bx1, by1, bx2, by2,
                              body.sharpness_score, body.cloth_color)
            except Exception as exc:  # noqa: BLE001 — never let one frame abort the batch
                log.exception("[GradingStage] %s — error while grading: %s",
                              frame.path.name, exc)
                for body in frame.bodies:
                    body.passed = False
                    body.rejection_reason = f"exception while grading: {exc}"

            if idx % progress_step == 0 or idx == total:
                log.info("[GradingStage] [%*d/%d] frame(s) graded", width, idx, total)

        log.info("[GradingStage] done: %d/%d body(ies) passed across %d frame(s)",
                 passed_bodies, total_bodies, total)
        return frames
