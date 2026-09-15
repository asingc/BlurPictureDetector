"""Shallow regrade: re-derive an already-imported album's Blur/Sharp verdicts
at a NEW sensitivity threshold from the per-body sharpness scores already
stored on the `Album` (see algo/album.py) — without re-running person/pose
or face detection.

Used by the webui's Apply/Summary page ("Quick Regrade"). The "Deep Regrade"
button instead re-runs the real detection pipeline via
``1_prep_review.py --regrade-only``; both end up applying the same
jersey/team-colour rules, this one just replays them against persisted data.

Why this isn't a pure "score > threshold" flip
-----------------------------------------------
A body's verdict at import time comes from FOUR short-circuiting grading
gates (see algo/scorers.py's BodyArrayScorer): matched-face -> face-size ->
keypoint-visibility -> sharpness; then a separate JerseyCountingStage
filters on team colour. Cloth colour is only ever predicted for a body that
cleared all four gates, so a body rejected on sharpness has no colour
recorded at all.

Changing the threshold therefore reshuffles which bodies are even eligible
for the jersey check, which in turn moves the polled team colour. So a
regrade runs in three passes, mirroring the real pipeline:

1. Measure cloth colour (from the original pixels) for any newly-eligible
   body that never had one.
2. Re-poll the team's dominant jersey colour / target L*a*b* / lightness
   bucket across the whole eligible population.
3. Re-apply :func:`algo.stages.jersey_counting.classify_body_jersey` to every
   eligible body against that fresh reference, then recompute each photo's
   overall status, its info.json bucket, and its annotated preview.

A body that failed one of the first three gates is never revisited — those
gates don't depend on the threshold, and a looser blur setting cannot
conjure a face that was never detected. Albums imported before
``rejection_reason`` was persisted can't prove which gate a body failed, so
their currently-failing bodies are conservatively left alone (run a deep
regrade once to populate it).
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

from algo.config import app_config
from algo.frame import Frame
from algo.models import ColorLab
from algo.results import baseline_stars
from algo.stages.annotation import _annotate_frame
from algo.stages.grading import cloth_color_predictor
from algo.stages.jersey_counting import (
    _REF_LAB_BY_LABEL,
    _allowed_reference_labs,
    _lightness_class,
    classify_body_jersey,
)
from algo.album import Album, AlbumImage, PersonRecord
from algo.utils import _color_from_label, atomic_save_and_backup

log = logging.getLogger("BlurPictureDetector")

_REVIEW_INFO_KEY = {"blurry": "Anno_Blur", "sharp": "Anno_Sharp"}


@dataclass
class RegradeSummary:
    threshold: float
    images_considered: int = 0
    recovered: int = 0            # blurry -> sharp
    demoted: int = 0              # sharp -> blurry
    jersey_rechecked: int = 0     # bodies whose cloth colour was (re-)predicted from pixels
    jersey_recheck_unreadable: int = 0  # ...of which the source photo couldn't be re-read
    team_color: str | None = None      # jersey colour used for this regrade
    team_color_pinned: bool = False    # True when it came from a manual override, not a poll
    stars_rebaselined: int = 0         # flipped photos whose star rating was reset
    previews_regenerated: int = 0      # verdict flipped -> preview/thumbnail redrawn
    previews_regen_failed: int = 0     # ...of which the source photo couldn't be re-read


def _parse_jersey_colors(jerseycolor_arg: str | None) -> tuple[frozenset[str], frozenset[str]]:
    """Mirrors 1_prep_review.py main()'s forced/regular colour parsing."""
    raw = jerseycolor_arg or ""
    forced = frozenset(
        c.strip().lstrip("+").strip().title()
        for c in raw.split(";")
        if c.strip().startswith("+") and c.strip().lstrip("+").strip()
    )
    regular = frozenset(
        c.strip().title()
        for c in raw.split(";")
        if c.strip() and not c.strip().startswith("+")
    )
    return forced, regular


def _regenerate_preview(image: AlbumImage, album: Album) -> bool:
    """Redraw <album>/previews/<key>.jpg (+ thumbnail) from the photo's
    just-updated per-body verdicts, so the pass/fail badges and rejection-
    reason labels baked into the preview stay in sync with the new overall
    status. Returns False (leaving the stale preview in place) when the
    source photo can no longer be re-read."""
    file_path = image.source_file
    decoded = _read_source_image(file_path)
    if decoded is None:
        return False

    frame = Frame(
        path=Path(file_path),
        bodies=[record.to_body() for record in image.bodies],
        image=decoded,
        auto_adjustment=image.auto_adjustment,
        output_key=image.preview_stem,
    )
    _annotate_frame(frame, album.path, app_config)
    return True


def _read_source_image(file_path: str):
    """Decode one album source photo, or None if unreadable.

    Deliberately not cached: an album can hold thousands of photos and
    holding every decoded frame would exhaust memory. Each pass that needs
    pixels re-decodes only the images it actually touches.
    """
    if not file_path:
        return None
    # Imported lazily -- pulls in ultralytics/YOLO, only worth paying for
    # when a regrade actually needs to re-decode a source photo.
    from algo.stages.image_analysis import _read_image
    return _read_image(Path(file_path))


def _stored_lab(record: PersonRecord) -> tuple[float, float, float] | None:
    """The body's representative L*a*b*, preferring its measured median over
    the reference LAB of its predicted label — mirrors
    algo/stages/jersey_counting.py::_body_lab, but reading persisted data."""
    color = record.cloth_color
    if color in ("N/A", "Unknown"):
        return None
    mean = record.cloth_color_detail.get("mean_lab")
    if mean and len(mean) == 3:
        return (float(mean[0]), float(mean[1]), float(mean[2]))
    return _REF_LAB_BY_LABEL.get(color)


def _poll_jersey(candidates: list[PersonRecord], config, pinned_label: str | None = None) -> tuple[
    str | None, tuple[float, float, float] | None, str | None
]:
    """Determine the team's jersey colour across every body eligible for the
    jersey check, returning ``(label, team target LAB, lightness bucket)``.

    Replays algo/stages/jersey_counting.py's ``_poll_jersey_color`` /
    ``_poll_team_target_lab`` / ``_poll_lightness_bucket`` against persisted
    cloth colours instead of live Frame objects. *candidates* is the same
    population the stage polls from: bodies that cleared grading, before the
    jersey filter narrows them down.

    When *pinned_label* is given it replaces the polled dominant colour, but
    the LAB anchor is still measured from the bodies actually wearing it so
    shadow/brightness tolerance keeps working.
    """
    label_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    labs_by_label: dict[str, list[tuple[float, float, float]]] = {}

    for record in candidates:
        color = record.cloth_color
        if color in ("N/A", "Unknown"):
            continue
        label_counts[color] = label_counts.get(color, 0) + 1
        lab = _stored_lab(record)
        if lab is None:
            continue
        labs_by_label.setdefault(color, []).append(lab)
        bucket = _lightness_class(lab, config.jersey_light_l_min, config.jersey_light_chroma_max)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    if not label_counts and pinned_label is None:
        return None, None, None

    summary = "  ".join(f"{c}={n}" for c, n in sorted(label_counts.items(), key=lambda x: -x[1]))
    log.info("[regrade] jersey colour distribution: %s", summary or "(none)")
    if pinned_label:
        our_label = pinned_label
        log.info("[regrade] team colour pinned to %s (%d matching body/bodies)",
                 our_label, label_counts.get(our_label, 0))
    else:
        our_label = max(label_counts, key=label_counts.__getitem__)

    samples = labs_by_label.get(our_label) or []
    if samples:
        team_target_lab = (
            statistics.median(s[0] for s in samples),
            statistics.median(s[1] for s in samples),
            statistics.median(s[2] for s in samples),
        )
    else:
        team_target_lab = _REF_LAB_BY_LABEL.get(our_label)

    # A pinned colour must anchor the Light/Dark bucket too, otherwise the
    # fallback strategy would still be judging against the polled majority.
    if pinned_label and team_target_lab is not None:
        team_bucket = _lightness_class(
            team_target_lab, config.jersey_light_l_min, config.jersey_light_chroma_max
        )
    else:
        team_bucket = max(bucket_counts, key=bucket_counts.__getitem__) if bucket_counts else None
    return our_label, team_target_lab, team_bucket


def regrade_sensitivity(
    album_path: Path,
    new_threshold: float,
    team_color_override: str | None = None,
) -> RegradeSummary:
    """Re-derive Blur/Sharp verdicts at *new_threshold*.

    *team_color_override* pins the team's jersey colour to a "Hue:Shade"
    label instead of polling it from the photos; None restores auto-polling.
    """
    info_path = album_path / "info.json"

    with open(info_path, encoding="utf-8") as fh:
        info = json.load(fh)
    album = Album(album_path)
    payload = album.payload

    run_settings: dict = payload.get("run_settings") or {}
    no_team = bool(run_settings.get("noteam"))
    summary = RegradeSummary(threshold=new_threshold)

    # An explicit argument wins; otherwise fall back to whatever pin the
    # album already carries, so a plain sensitivity regrade doesn't silently
    # revert a colour the user chose earlier.
    pinned = team_color_override if team_color_override is not None else run_settings.get("team_color_override")
    pinned = (pinned or "").strip() or None
    summary.team_color_pinned = pinned is not None

    forced_colors, regular_colors = _parse_jersey_colors(run_settings.get("jerseycolor"))
    forced_labs = _allowed_reference_labs(forced_colors)
    allowed_labs = _allowed_reference_labs(regular_colors)

    # Entries worth re-grading at all, paired with per-body flags captured
    # BEFORE anything is mutated: whether the body cleared the
    # threshold-independent grading gates, and whether it clears the new
    # threshold. Eligible bodies are exactly the population
    # JerseyCountingStage evaluates.
    gradable: list[tuple[AlbumImage, list[PersonRecord], list[bool], list[bool]]] = []
    candidates: list[PersonRecord] = []
    for image in album.image_list:
        if image.status not in ("blurry", "sharp"):
            continue
        bodies = image.bodies
        if not bodies:
            continue
        summary.images_considered += 1
        cleared = [b.cleared_grading_gates for b in bodies]
        eligible = [
            c and b.sharpness_score > new_threshold
            for b, c in zip(bodies, cleared)
        ]
        gradable.append((image, bodies, cleared, eligible))
        candidates.extend(b for b, e in zip(bodies, eligible) if e)

    # ---- Pass 1: make sure every candidate has a measured cloth colour ----
    # Bodies rejected for sharpness at import time never reached the colour
    # predictor (GradingStage skips failed bodies), so theirs must be
    # measured from pixels now. Bodies that got as far as the jersey check
    # already carry one and are reused as-is.
    if not no_team:
        for image, bodies, _cleared, eligible in gradable:
            needs_color = [
                b for b, e in zip(bodies, eligible)
                if e and b.cloth_color in ("N/A", None)
            ]
            if not needs_color:
                continue
            file_path = image.source_file
            decoded = _read_source_image(file_path)
            if decoded is None:
                summary.jersey_recheck_unreadable += len(needs_color)
                log.warning("[regrade] cannot re-read source photo for jersey colour: %s", file_path)
                continue
            for record in needs_color:
                record.cloth_color, record.cloth_color_detail = cloth_color_predictor.predict(
                    record.to_body(), decoded
                )
                summary.jersey_rechecked += 1

    # ---- Pass 2: re-poll the team's jersey colour from those candidates ----
    our_label: str | None = None
    team_target_lab: tuple[float, float, float] | None = None
    team_bucket: str | None = None
    our_color: ColorLab | None = None
    if not no_team:
        our_label, team_target_lab, team_bucket = _poll_jersey(candidates, app_config, pinned)
        if our_label is None:
            log.warning("[regrade] no usable cloth colour found — skipping team-colour filter")
        else:
            our_color = _color_from_label(our_label)
            log.info("[regrade] team colour: %s (%s)", our_label, "pinned" if pinned else "polled")
    summary.team_color = our_label
    apply_jersey_filter = our_color is not None

    # ---- Pass 3: final verdicts, info.json bookkeeping, previews ----
    # key -> the info.json Anno_Blur/Anno_Sharp item, so a status flip can
    # move it between the two lists without disturbing its "src"/"srcPath".
    info_items_by_key: dict[str, dict] = {}
    for status, info_key in _REVIEW_INFO_KEY.items():
        for item in info.get(info_key, []):
            src = item.get("src")
            if src:
                info_items_by_key[src] = item

    for image, bodies, cleared, eligible in gradable:
        old_status = image.status

        for record, was_cleared, is_eligible in zip(bodies, cleared, eligible):
            score = record.sharpness_score
            if not is_eligible:
                record.is_blurry = True
                if was_cleared:
                    # Only the threshold pushed it out; say so explicitly.
                    record.rejection_reason = (
                        f"sharpness score {score:.4f} <= threshold {new_threshold:.2f}"
                    )
                continue

            if not apply_jersey_filter:
                record.is_blurry = False
                record.rejection_reason = ""
                continue

            body = record.to_body()
            classify_body_jersey(
                body, forced_colors, regular_colors, forced_labs, allowed_labs,
                team_target_lab, team_bucket, our_color, app_config,
                log_prefix=Path(image.source_file).name,
            )
            record.passed = body.passed
            record.rejection_reason = body.rejection_reason

        new_overall_blurry = all(b.is_blurry for b in bodies)
        new_status = "blurry" if new_overall_blurry else "sharp"

        passing_scores = [b.sharpness_score for b in bodies if not b.is_blurry]
        best_score = max(passing_scores) if passing_scores else max(
            (b.sharpness_score for b in bodies), default=0.0
        )
        image.set_sharpness_score(best_score)
        image.overall_blurry = new_overall_blurry
        image.status = new_status

        if new_status != old_status:
            if new_status == "sharp":
                summary.recovered += 1
            else:
                summary.demoted += 1
            item = info_items_by_key.get(image.key)
            if item is not None:
                old_list = info.get(_REVIEW_INFO_KEY[old_status], [])
                if item in old_list:
                    old_list.remove(item)
                info.setdefault(_REVIEW_INFO_KEY[new_status], []).append(item)

            # A photo that changed side of the keep line carries a star
            # rating that no longer reflects it (LLM culling rated it under
            # the old verdict). Reset to the baseline unless the user rated
            # it by hand -- see culling_app.py's "stars_manual" marker.
            if image.apply_auto_rating(
                baseline_stars(image.sharpness_score, new_status, new_threshold)
            ):
                summary.stars_rebaselined += 1

            if _regenerate_preview(image, album):
                summary.previews_regenerated += 1
            else:
                summary.previews_regen_failed += 1
                log.warning("[regrade] verdict changed but preview could not be regenerated (source unreadable): %s",
                            image.source_file)

    # The team colour (polled or pinned) can move, so persist it alongside
    # the verdicts it just produced.
    if summary.team_color is not None:
        payload["our_jersey_color"] = summary.team_color
        info["OurJerseyColor"] = summary.team_color

    # Lock the new threshold in as this album's sensitivity going forward,
    # so a later "Import more images" merge (which reuses run_settings)
    # grades newly-added photos the same way instead of silently reverting
    # to whatever sensitivity was used on the very first import. The colour
    # pin rides along for the same reason ("" clears it back to auto).
    payload.setdefault("run_settings", {})["sensitivity"] = str(new_threshold)
    payload["run_settings"]["team_color_override"] = pinned or ""

    album.save()
    atomic_save_and_backup(json.dumps(info, indent=4), info_path)
    log.info(
        "[regrade] threshold=%.2f team_colour=%s(%s) — recovered=%d demoted=%d stars_rebaselined=%d "
        "cloth_colours_measured=%d (unreadable=%d) previews_regenerated=%d (failed=%d) of %d image(s)",
        new_threshold, summary.team_color or "n/a", "pinned" if pinned else "polled",
        summary.recovered, summary.demoted, summary.stars_rebaselined,
        summary.jersey_rechecked, summary.jersey_recheck_unreadable,
        summary.previews_regenerated, summary.previews_regen_failed, summary.images_considered,
    )
    return summary
