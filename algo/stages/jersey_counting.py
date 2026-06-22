from __future__ import annotations

import logging

from algo.config import AppConfig
from algo.frame import Frame
from algo.stage import ProcessStage
from algo.utils import _matches_allowed_jersey_color

log = logging.getLogger("BlurPictureDetector")


def _poll_jersey_color(frames: list[Frame]) -> str:
    """Tally cloth colors across all passing bodies and return the most common.

    Colors labeled 'Unknown' or 'N/A' are excluded.
    Returns 'Unknown' when no body has a usable color.
    """
    counts: dict[str, int] = {}
    for frame in frames:
        for body in frame.bodies:
            if body.passed and body.cloth_color not in ("N/A", "Unknown"):
                counts[body.cloth_color] = counts.get(body.cloth_color, 0) + 1
    if not counts:
        return "Unknown"
    summary = "  ".join(f"{c}={n}" for c, n in sorted(counts.items(), key=lambda x: -x[1]))
    log.info("[JerseyCountingStage] colour distribution: %s", summary)
    return max(counts, key=counts.__getitem__)


class JerseyCountingStage(ProcessStage):
    """Determine the dominant jersey colour and invalidate non-matching bodies.

    1. Tally cloth colours across all currently-passing bodies to find
       "our" team colour.
    2. Mark any body whose colour does not match the allow-list or the
       polled team colour as ``passed=False``.

    When *jersey_colors* is empty, the allow-list check is skipped and only
    the polled team colour is used for filtering.
    """

    def __init__(self, jersey_colors: frozenset[str]) -> None:
        self.jersey_colors = jersey_colors

    def process(self, frames: list[Frame], config: AppConfig) -> list[Frame]:
        our_color = _poll_jersey_color(frames)
        log.info("[JerseyCountingStage] polled team colour: %s", our_color)

        for frame in frames:
            for body in frame.bodies:
                if not body.passed:
                    continue

                # Allow-list check (CLI --jerseycolor).
                if self.jersey_colors and not _matches_allowed_jersey_color(
                    body.cloth_color, self.jersey_colors
                ):
                    body.passed = False
                    log.debug("[JerseyCountingStage] %s — body cloth=%s not in allow-list → fail",
                              frame.path.name, body.cloth_color)
                    continue

                # Team colour check (polled from all frames).
                if our_color != "Unknown" and body.cloth_color != our_color:
                    body.passed = False
                    log.debug("[JerseyCountingStage] %s — body cloth=%s != team colour %s → fail",
                              frame.path.name, body.cloth_color, our_color)

        return frames
