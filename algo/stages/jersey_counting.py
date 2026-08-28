from __future__ import annotations

import logging
import statistics

from algo.config import AppConfig
from algo.frame import Frame
from algo.models import Body, ColorLab
from algo.stage import ProcessStage
from algo.stages.grading import ClothColorPredictor
from algo.utils import _color_from_label, _colors_match, _matches_allowed_jersey_color

log = logging.getLogger("BlurPictureDetector")


# Reference L*a*b* values for every predicted Hue:Shade label, used as a
# fallback when a body has no measured mean LAB available.
_REF_LAB_BY_LABEL: dict[str, tuple[float, float, float]] = {
    c.label: c.lab for c in ClothColorPredictor._COLORS
}


def _lightness_class(lab: tuple[float, float, float], l_min: float, chroma_max: float) -> str:
    """Classify a CIE L*a*b* colour as 'Light' or 'Dark'.

    'Light' (white / pale) requires both high lightness (L* >= *l_min*) and low
    chroma (sqrt(a*^2 + b*^2) <= *chroma_max*).  The chroma gate keeps brightly-
    lit *coloured* jerseys in the 'Dark' bucket so they are not mistaken for
    white.  Everything else is 'Dark'.
    """
    L, a, b = lab
    chroma = (a * a + b * b) ** 0.5
    if L >= l_min and chroma <= chroma_max:
        return "Light"
    return "Dark"


def _body_lightness_bucket(body: Body, config: AppConfig) -> str | None:
    """Return the 'Light'/'Dark' bucket for *body*, or None when unavailable.

    Prefers the body's measured mean LAB (reflects the actual lighting on this
    jersey); falls back to the reference LAB of the predicted colour label.
    """
    if body.cloth_color in ("N/A", "Unknown"):
        return None
    mean = (body.cloth_color_detail or {}).get("mean_lab")
    if mean and len(mean) == 3:
        lab = (float(mean[0]), float(mean[1]), float(mean[2]))
    else:
        ref = _REF_LAB_BY_LABEL.get(body.cloth_color)
        if ref is None:
            return None
        lab = ref
    return _lightness_class(lab, config.jersey_light_l_min, config.jersey_light_chroma_max)


def _poll_lightness_bucket(frames: list[Frame], config: AppConfig) -> str | None:
    """Tally Light/Dark buckets across all passing bodies and return the most common.

    Returns None when no passing body has a usable colour.
    """
    counts: dict[str, int] = {}
    for frame in frames:
        for body in frame.bodies:
            if body.passed:
                bucket = _body_lightness_bucket(body, config)
                if bucket is not None:
                    counts[bucket] = counts.get(bucket, 0) + 1
    if not counts:
        return None
    summary = "  ".join(f"{b}={n}" for b, n in sorted(counts.items(), key=lambda x: -x[1]))
    log.info("[JerseyCountingStage] lightness distribution: %s", summary)
    return max(counts, key=counts.__getitem__)


def _body_lab(body: Body) -> tuple[float, float, float] | None:
    """Return the body's representative L*a*b*.

    Prefers the measured mean LAB (reflects the actual lighting on this jersey);
    falls back to the reference LAB of the predicted colour label.  Returns None
    when no usable colour is available.
    """
    if body.cloth_color in ("N/A", "Unknown"):
        return None
    mean = (body.cloth_color_detail or {}).get("mean_lab")
    if mean and len(mean) == 3:
        return (float(mean[0]), float(mean[1]), float(mean[2]))
    return _REF_LAB_BY_LABEL.get(body.cloth_color)


def _weighted_lab_distance(
    lab_a: tuple[float, float, float],
    lab_b: tuple[float, float, float],
    l_weight: float,
    c_weight: float = 1.0,
    h_weight: float = 1.0,
) -> float:
    """LCh-decomposed colour distance between two L*a*b* colours.

    Decomposes the raw a*/b* difference into a Chroma (saturation magnitude)
    component and a Hue (colour angle) component — the same split
    CIE94/CIEDE2000 use for perceptual colour differences — instead of a flat
    Euclidean distance over a*/b*.  This matters because lighting/shadow moves
    a jersey's Lightness *and* Chroma a lot (shadows both darken and desaturate)
    while barely touching its Hue (a yellow jersey in shadow is still "yellow",
    just darker/less vivid).  By down-weighting L and C while keeping H at
    (near) full weight, the same real-world jersey colour matches across a wide
    brightness/shadow range without becoming forgiving of an actual colour
    (hue) change, e.g. yellow vs. green.

    ``ΔH² = Δa² + Δb² - ΔC²`` is the exact algebraic decomposition of the a*/b*
    Euclidean distance into its chroma and hue parts (no trig needed); clamped
    to >= 0 to guard against floating-point noise.
    """
    dL = lab_a[0] - lab_b[0]
    da = lab_a[1] - lab_b[1]
    db = lab_a[2] - lab_b[2]
    c_a = (lab_a[1] ** 2 + lab_a[2] ** 2) ** 0.5
    c_b = (lab_b[1] ** 2 + lab_b[2] ** 2) ** 0.5
    dC = c_a - c_b
    dH_sq = max(0.0, da * da + db * db - dC * dC)
    return (l_weight * dL * dL + c_weight * dC * dC + h_weight * dH_sq) ** 0.5


def _allowed_reference_labs(allowed_colors: frozenset[str]) -> list[tuple[float, float, float]]:
    """Resolve allow-list entries to the reference L*a*b* values they name.

    Entries may be a hue ("Blue"), a shade ("Navy") or a "Hue:Shade" label and
    are matched against the canonical reference colours exactly as
    :func:`_matches_allowed_jersey_color` does (shade or hue equality).  The
    returned LABs are the anchors a body's measured colour is compared against
    under brightness-forgiving distance matching.
    """
    labs: list[tuple[float, float, float]] = []
    for c in ClothColorPredictor._COLORS:
        hue_l = c.hue.strip().lower()
        shade_l = c.shade.strip().lower()
        for allowed in allowed_colors:
            a = allowed.strip().lower()
            if not a:
                continue
            if ":" in a:
                a_hue, _, a_shade = a.partition(":")
                if a_shade == shade_l or a_hue == hue_l:
                    labs.append(c.lab)
                    break
            elif a == shade_l or a == hue_l:
                labs.append(c.lab)
                break
    return labs


def _matches_lab_refs(
    body: Body, ref_labs: list[tuple[float, float, float]], config: AppConfig
) -> bool:
    """True when the body's measured L*a*b* is within distance of any reference.

    Uses the same brightness-forgiving weighted distance as team matching, so a
    forced/allow-list colour is recognised across light and shadow.
    """
    if not ref_labs:
        return False
    blab = _body_lab(body)
    if blab is None:
        return False
    return any(
        _weighted_lab_distance(
            blab, ref, config.jersey_lab_l_weight, config.jersey_lab_c_weight, config.jersey_lab_h_weight
        ) <= config.jersey_lab_max_dist
        for ref in ref_labs
    )


def _poll_team_target_lab(
    frames: list[Frame], our_label: str
) -> tuple[float, float, float] | None:
    """Compute the team's target L*a*b* anchor for distance matching.

    Averages the measured mean LAB of every passing body that shares the polled
    dominant label *our_label* (so the opponent's colour cannot skew it); falls
    back to that label's canonical reference LAB when no measurements exist.
    """
    samples: list[tuple[float, float, float]] = []
    for frame in frames:
        for body in frame.bodies:
            if body.passed and body.cloth_color == our_label:
                mean = (body.cloth_color_detail or {}).get("mean_lab")
                if mean and len(mean) == 3:
                    samples.append((float(mean[0]), float(mean[1]), float(mean[2])))
    if samples:
        # Per-channel median across bodies (not a plain mean) so a single
        # outlier body — e.g. one still sitting mostly in deep shadow — can't
        # skew the whole team's anchor colour.
        return (
            statistics.median(s[0] for s in samples),
            statistics.median(s[1] for s in samples),
            statistics.median(s[2] for s in samples),
        )
    return _REF_LAB_BY_LABEL.get(our_label)


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


def classify_body_jersey(
    body: Body,
    forced_colors: frozenset[str],
    allowed_colors: frozenset[str],
    forced_labs: list[tuple[float, float, float]],
    allowed_labs: list[tuple[float, float, float]],
    team_target_lab: tuple[float, float, float] | None,
    team_bucket: str | None,
    our_color: ColorLab,
    config: AppConfig,
    *,
    log_prefix: str = "",
) -> None:
    """Apply the forced-colour / allow-list / polled-team-colour decision to a
    single already-colour-predicted *body*, mutating ``body.passed`` /
    ``body.rejection_reason`` in place.

    Split out of :meth:`JerseyCountingStage.process` so the exact same
    per-body decision can be replayed later against just-recovered bodies
    (see algo/regrade.py) without duplicating the logic.
    """
    bcolor = _color_from_label(body.cloth_color)

    # Forced colors (e.g. goalies) always pass — skip further checks.
    # In LAB mode, also accept a brightness-forgiving distance match.
    if (_matches_allowed_jersey_color(bcolor, forced_colors)
            or _matches_lab_refs(body, forced_labs, config)):
        log.debug("[JerseyCountingStage] %s — body cloth=%s in forced-colors → pass",
                  log_prefix, body.cloth_color)
        return

    # Allow-list check (CLI --jerseycolor regular entries).
    # In LAB mode, also accept a brightness-forgiving distance match.
    if not (_matches_allowed_jersey_color(bcolor, allowed_colors)
            or _matches_lab_refs(body, allowed_labs, config)):
        body.passed = False
        body.rejection_reason = f"jersey color '{body.cloth_color}' not in allow-list"
        log.debug("[JerseyCountingStage] %s — body cloth=%s not in allow-list → fail",
                  log_prefix, body.cloth_color)
        return

    # Team colour check (polled from all frames).
    if team_target_lab is not None:
        blab = _body_lab(body)
        dist = (
            _weighted_lab_distance(
                blab, team_target_lab,
                config.jersey_lab_l_weight, config.jersey_lab_c_weight, config.jersey_lab_h_weight,
            )
            if blab is not None else None
        )
        if dist is None or dist > config.jersey_lab_max_dist:
            body.passed = False
            body.rejection_reason = (
                f"jersey color '{body.cloth_color}' LAB dist {dist:.1f} > {config.jersey_lab_max_dist:.1f}"
                if dist is not None
                else f"jersey color '{body.cloth_color}' has no usable LAB"
            )
            log.debug("[JerseyCountingStage] %s — body cloth=%s LAB dist=%s > %.1f → fail",
                      log_prefix, body.cloth_color,
                      f"{dist:.1f}" if dist is not None else "N/A",
                      config.jersey_lab_max_dist)
    elif team_bucket is not None:
        body_bucket = _body_lightness_bucket(body, config)
        if body_bucket != team_bucket:
            body.passed = False
            body.rejection_reason = (
                f"jersey lightness '{body_bucket or 'Unknown'}' != team bucket '{team_bucket}'"
            )
            log.debug("[JerseyCountingStage] %s — body cloth=%s bucket=%s != team bucket %s → fail",
                      log_prefix, body.cloth_color, body_bucket, team_bucket)
    elif not _colors_match(bcolor, our_color):
        body.passed = False
        body.rejection_reason = f"jersey color '{body.cloth_color}' != team colour '{our_color.label}'"
        log.debug("[JerseyCountingStage] %s — body cloth=%s != team colour %s → fail",
                  log_prefix, body.cloth_color, our_color.label)


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

    The polled team-colour check supports three strategies, chosen via
    :class:`AppConfig`:

    * ``jersey_lab_match`` (default) — match each body by a weighted L*a*b*
      distance to the team's target colour, with the L* (brightness) axis
      down-weighted so the same jersey matches across light and shadow.
    * ``jersey_binary_lightness`` — collapse jerseys into "Light" vs "Dark"
      lightness buckets.
    * otherwise — exact Hue:Shade matching.

    The forced- and allow-list checks are unaffected by the chosen strategy.
    """

    def __init__(
        self,
        forced_colors:  frozenset[str],
        allowed_colors: frozenset[str],
        no_team: bool = False,
        team_color_override: str | None = None,
    ) -> None:
        if forced_colors is None:
            raise TypeError("forced_colors must be a frozenset, not None")
        if allowed_colors is None:
            raise TypeError("allowed_colors must be a frozenset, not None")
        self.forced_colors  = forced_colors
        self.allowed_colors = allowed_colors
        self.no_team        = no_team
        # "Hue:Shade" label pinning the team colour instead of polling it.
        self.team_color_override = (team_color_override or "").strip() or None
        self.our_color: ColorLab | None = None
        self.our_bucket: str | None = None

    def process(self, frames: list[Frame], config: AppConfig) -> list[Frame]:
        if self.no_team:
            return frames

        if self.team_color_override:
            our_color = _color_from_label(self.team_color_override)
            log.info("[JerseyCountingStage] team colour pinned: %s", our_color.label)
        else:
            our_color = _poll_jersey_color(frames)
            if our_color is None:
                log.warning("[JerseyCountingStage] no usable cloth colour found — skipping team-colour filter")
                return frames
            log.info("[JerseyCountingStage] polled team colour: %s", our_color.label)
        self.our_color = our_color

        team_bucket: str | None = None
        team_target_lab: tuple[float, float, float] | None = None
        forced_labs: list[tuple[float, float, float]] = []
        allowed_labs: list[tuple[float, float, float]] = []
        if config.jersey_lab_match:
            team_target_lab = _poll_team_target_lab(frames, our_color.label)
            forced_labs = _allowed_reference_labs(self.forced_colors)
            allowed_labs = _allowed_reference_labs(self.allowed_colors)
            log.info("[JerseyCountingStage] LAB-distance mode — team target L*a*b*: %s "
                     "(l_weight=%.2f, c_weight=%.2f, h_weight=%.2f, max_dist=%.1f)",
                     tuple(round(v, 1) for v in team_target_lab) if team_target_lab else "N/A",
                     config.jersey_lab_l_weight, config.jersey_lab_c_weight,
                     config.jersey_lab_h_weight, config.jersey_lab_max_dist)
        elif config.jersey_binary_lightness:
            # A pinned colour must anchor the bucket too, otherwise this
            # strategy would still judge against the polled majority.
            if self.team_color_override:
                ref = _REF_LAB_BY_LABEL.get(our_color.label)
                team_bucket = _lightness_class(
                    ref, config.jersey_light_l_min, config.jersey_light_chroma_max
                ) if ref else None
            else:
                team_bucket = _poll_lightness_bucket(frames, config)
            self.our_bucket = team_bucket
            log.info("[JerseyCountingStage] binary lightness mode — team bucket: %s (%s)",
                     team_bucket or "N/A", "pinned" if self.team_color_override else "polled")

        if self.forced_colors:
            log.info("[JerseyCountingStage] forced colors: %s", ", ".join(sorted(self.forced_colors)))

        if self.allowed_colors:
            log.info("[JerseyCountingStage] allowed colors: %s", ", ".join(sorted(self.allowed_colors)))

        for frame in frames:
            for body in frame.bodies:
                if not body.passed:
                    continue
                classify_body_jersey(
                    body, self.forced_colors, self.allowed_colors,
                    forced_labs, allowed_labs, team_target_lab, team_bucket, our_color,
                    config, log_prefix=frame.path.name,
                )

        return frames

