from __future__ import annotations

import logging

import cv2
import numpy as np

from algo.config import AppConfig
from algo.frame import Frame
from algo.models import AutoAdjustment, Body
from algo.stage import ProcessStage

log = logging.getLogger("BlurPictureDetector")


def _mean_brightness(image: np.ndarray | None) -> float:
    """Return mean gray-level brightness of *image*, normalised to [0, 1]."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) / 255.0


def _main_body(frame: Frame) -> Body | None:
    """Pick the frame's main subject: the highest-sharpness passed body,
    falling back to the highest-sharpness body overall when none passed."""
    if not frame.bodies:
        return None
    passing = [b for b in frame.bodies if b.passed]
    pool = passing or frame.bodies
    return max(pool, key=lambda b: b.sharpness_score)


def compute_auto_adjustment(frame: Frame, config: AppConfig) -> AutoAdjustment:
    """Derive a simple, discretised exposure (brightness) correction for *frame*.

    Blends whole-image brightness with the main subject's face crop (50/50)
    so the correction favours neither an over/under-exposed background nor
    the subject alone, then rounds the result to simple steps (e.g. EV +0.5)
    so the prescription stays easy to reason about and re-apply later purely
    from the stored number.
    """
    image = frame.normalized_image
    if image is None or image.size == 0:
        return AutoAdjustment()

    overall_brightness = _mean_brightness(image)

    body = _main_body(frame)
    face_crop = body.crop if body is not None else None
    if face_crop is not None and face_crop.size > 0:
        face_brightness = _mean_brightness(face_crop)
    else:
        # No usable subject crop — fall back to whole-image measurement so
        # the 50/50 blend degrades to a plain whole-image correction.
        face_brightness = overall_brightness

    blended_brightness = 0.5 * overall_brightness + 0.5 * face_brightness

    # --- Exposure (EV) -----------------------------------------------------
    if blended_brightness > 1e-6:
        ev_raw = float(np.log2(config.auto_adjust_target_brightness / blended_brightness))
    else:
        ev_raw = 0.0
    ev = round(ev_raw / config.auto_adjust_ev_step) * config.auto_adjust_ev_step
    ev = max(-config.auto_adjust_max_ev, min(config.auto_adjust_max_ev, ev))

    return AutoAdjustment(ev=round(float(ev), 3))


class AutoAdjustStage(ProcessStage):
    """Compute a simple auto exposure (brightness) correction per frame.

    The prescription is stored on ``frame.auto_adjustment`` so downstream
    stages (preview annotation) and 2_apply_changes.py (final output) can
    apply the exact same correction without recomputing it.
    """

    def process(self, frames: list[Frame], config: AppConfig) -> list[Frame]:
        for frame in frames:
            frame.auto_adjustment = compute_auto_adjustment(frame, config)
            log.debug(
                "[AutoAdjustStage] %s — EV=%+.2f",
                frame.path.name, frame.auto_adjustment.ev,
            )
        log.info("[AutoAdjustStage] computed auto adjustment for %d frame(s)", len(frames))
        return frames
