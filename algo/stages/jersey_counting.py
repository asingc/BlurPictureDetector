from __future__ import annotations

import logging

from algo.config import AppConfig
from algo.frame import Frame
from algo.models import ColorLab
from algo.stage import ProcessStage
from algo.utils import _color_from_label, _colors_match, _matches_allowed_jersey_color

log = logging.getLogger("BlurPictureDetector")


def _poll_jersey_color(frames: list[Frame]) -> ColorLab | None:
    """Tally cloth colors across all passing bodies and return the most common as a ColorLab.

    Colors labeled 'Unknown' or 'N/A' are excluded.
    Returns None when no body has a usable color.
    """
    counts: dict[str, int] = {}
    for frame in frames:
        for body in frame.bodies:
            if body.passed and body.cloth_color not in ("N/A", "Unknown"):
                counts[body.cloth_color] = counts.get(body.cloth_color, 0) + 1
    if not counts:
        return None
    summary = "  ".join(f"{c}={n}" for c, n in sorted(counts.items(), key=lambda x: -x[1]))
    log.info("[JerseyCountingStage] colour distribution: %s", summary)
    winner_label = max(counts, key=counts.__getitem__)
    return _color_from_label(winner_label)


class JerseyCountingStage(ProcessStage):
    """Determine the dominant jersey colour and invalidate non-matching bodies.

    1. Tally cloth colours across all currently-passing bodies to find
       "our" team colour.
    2. Mark any body whose colour does not match the allow-list or the
       polled team colour as ``passed=False``.

    *forced_colors* are always accepted regardless of the polled team colour
    (e.g. goalies wearing a distinct colour).  *allowed_colors* are the
    regular team colours subject to both the allow-list and the polled-colour
    check.  When both sets are empty, only the polled team colour is used.

    When *no_team* is True all jersey-colour filtering is skipped and every
    body passes this stage unconditionally.
    """

    def __init__(
        self,
        forced_colors:  frozenset[str],
        allowed_colors: frozenset[str],
        no_team: bool = False,
    ) -> None:
        if forced_colors is None:
            raise TypeError("forced_colors must be a frozenset, not None")
        if allowed_colors is None:
            raise TypeError("allowed_colors must be a frozenset, not None")
        self.forced_colors  = forced_colors
        self.allowed_colors = allowed_colors
        self.no_team        = no_team
        self.our_color: ColorLab | None = None

    def process(self, frames: list[Frame], config: AppConfig) -> list[Frame]:
        if self.no_team:
            return frames

        our_color = _poll_jersey_color(frames)
        if our_color is None:
            log.warning("[JerseyCountingStage] no usable cloth colour found — skipping team-colour filter")
            return frames
        self.our_color = our_color
        log.info("[JerseyCountingStage] polled team colour: %s", our_color.label)

        for frame in frames:
            for body in frame.bodies:
                if not body.passed:
                    continue

                bcolor = _color_from_label(body.cloth_color)

                # Forced colors (e.g. goalies) always pass — skip further checks.
                if _matches_allowed_jersey_color(bcolor, self.forced_colors):
                    continue

                # Allow-list check (CLI --jerseycolor regular entries).
                if not _matches_allowed_jersey_color(bcolor, self.allowed_colors):
                    body.passed = False
                    body.rejection_reason = f"jersey color '{body.cloth_color}' not in allow-list"
                    log.debug("[JerseyCountingStage] %s — body cloth=%s not in allow-list → fail",
                              frame.path.name, body.cloth_color)
                    continue

                # Team colour check (polled from all frames).
                if not _colors_match(bcolor, our_color):
                    body.passed = False
                    body.rejection_reason = f"jersey color '{body.cloth_color}' != team colour '{our_color.label}'"
                    log.debug("[JerseyCountingStage] %s — body cloth=%s != team colour %s → fail",
                              frame.path.name, body.cloth_color, our_color.label)

        return frames

