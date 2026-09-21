from __future__ import annotations

import logging

import numpy as np

from algo.config import AppConfig
from algo.frame import Frame
from algo.models import AutoAdjustment, Body
from algo.stage import ProcessStage

log = logging.getLogger("BlurPictureDetector")


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

    Both brightness measurements are taken during analysis (see
    populate_frame_cache) so this stage never needs the pixels.
    """
    overall_brightness = frame.overall_brightness
    if overall_brightness <= 0.0:
        return AutoAdjustment()

    body = _main_body(frame)
    if body is not None and body.crop_brightness > 0.0:
        face_brightness = body.crop_brightness
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
    stages (preview annotation, final export) can apply the exact same
    correction without recomputing it.
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
