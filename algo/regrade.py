"""Shallow regrade: re-derive an already-imported album's Blur/Sharp verdicts
at a NEW sensitivity threshold from the per-body sharpness scores already
stored in album.json — without re-running person/pose or face detection.

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

import numpy as np

from algo.config import app_config
from algo.frame import Frame
from algo.models import AutoAdjustment, Body, Box, ColorLab, Face, Point, PredictedKeyPoint
from algo.results import baseline_stars
from algo.stages.annotation import _annotate_frame
from algo.stages.grading import cloth_color_predictor
from algo.stages.jersey_counting import (
    _REF_LAB_BY_LABEL,
    _allowed_reference_labs,
    _lightness_class,
    classify_body_jersey,
)
from algo.utils import _color_from_label, atomic_save_and_backup

log = logging.getLogger("BlurPictureDetector")

# A body reconstructed from stored JSON has no real face crop (sharpness was
# already scored at import time) -- cloth_color_predictor.predict() never
# touches body.crop, so a tiny placeholder is enough.
_DUMMY_CROP = np.zeros((1, 1, 3), dtype=np.uint8)

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


def _body_from_entry(body_dict: dict) -> Body:
    bbox = Box(**body_dict["body_bbox"])
    keypoints = [
        PredictedKeyPoint(Point(kp["x"], kp["y"]), kp["conf"], kp.get("passed", True))
        for kp in body_dict.get("body_keypoints", [])
    ]
    return Body(crop=_DUMMY_CROP, bbox=bbox, faces=[], keypoints=keypoints)


def _cleared_grading_gates(body_dict: dict) -> bool:
    """True when this body passed every gate BEFORE the sharpness threshold —
    i.e. it has a matched face, of adequate size, with enough visible head
    keypoints (see algo/scorers.py's short-circuiting BodyArrayScorer).

    Those three gates don't depend on the threshold, so such a body is a
    legitimate candidate to re-evaluate at any new one. A body rejected for
    sharpness OR for jersey colour necessarily got past them; a body rejected
    for anything else did not. Albums imported before ``rejection_reason`` was
    persisted have no reason string, so a currently-failing body there can't
    be shown to qualify and is conservatively left alone.
    """
    if not body_dict.get("is_blurry", True):
        return True
    reason = body_dict.get("rejection_reason") or ""
    return reason.startswith("sharpness score") or reason.startswith("jersey ")


def _full_body_from_dict(b: dict) -> Body:
    """Reconstruct a Body with every field algo/stages/annotation.py's drawer
    reads (unlike :func:`_body_from_entry`, which only carries the bbox/
    keypoints needed for a cloth-colour re-prediction)."""
    keypoints = [
        PredictedKeyPoint(Point(kp["x"], kp["y"]), kp["conf"], kp.get("passed", True))
        for kp in b.get("body_keypoints", [])
    ]
    best_face: Face | None = None
    face_kps = b.get("face_kps")
    if face_kps:
        best_face = Face(
            bbox=Box(**face_kps["bbox"]),
            confidence=float(face_kps.get("confidence", 0.0)),
            landmarks=[
                PredictedKeyPoint(Point(lm["x"], lm["y"]), lm["conf"], lm.get("passed", True))
                for lm in face_kps.get("landmarks", [])
            ],
            passed=bool(face_kps.get("passed", True)),
        )
    return Body(
        crop=_DUMMY_CROP,
        bbox=Box(**b["body_bbox"]),
        faces=[best_face] if best_face else [],
        keypoints=keypoints,
        passed=not b.get("is_blurry", True),
        rejection_reason=b.get("rejection_reason") or "",
        sharpness_score=float(b.get("sharpness_score", 0.0)),
        best_face=best_face,
        best_narrow_box=Box(**b["narrow_face_bbox"]) if b.get("narrow_face_bbox") else None,
        lap_var=float(b.get("lap_var", 0.0)),
        ten=float(b.get("ten", 0.0)),
        cloth_color=b.get("cloth_color", "N/A"),
        cloth_color_detail=b.get("cloth_color_detail") or {},
    )


def _regenerate_preview(entry: dict, bodies: list[dict], album_path: Path) -> bool:
    """Redraw <album>/previews/<key>.jpg (+ thumbnail) from the entry's
    just-updated per-body verdicts, so the pass/fail badges and rejection-
    reason labels baked into the preview stay in sync with the new overall
    status. Returns False (leaving the stale preview in place) when the
    source photo can no longer be re-read."""
    file_path = entry.get("file", "")
    image = _read_source_image(file_path)
    if image is None:
        return False

    preview_path = entry.get("preview_path")
    output_key = Path(preview_path).stem if preview_path else (entry.get("key") or Path(file_path).stem)
    auto_adj = entry.get("auto_adjustment")

    frame = Frame(
        path=Path(file_path),
        bodies=[_full_body_from_dict(b) for b in bodies],
        image=image,
        auto_adjustment=AutoAdjustment(ev=float(auto_adj["ev"])) if auto_adj else None,
        output_key=output_key,
    )
    _annotate_frame(frame, album_path, app_config)
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


def _stored_lab(body_dict: dict) -> tuple[float, float, float] | None:
    """The body's representative L*a*b*, preferring its measured median over
    the reference LAB of its predicted label — mirrors
    algo/stages/jersey_counting.py::_body_lab, but reading persisted JSON."""
    color = body_dict.get("cloth_color", "N/A")
    if color in ("N/A", "Unknown"):
        return None
    mean = (body_dict.get("cloth_color_detail") or {}).get("mean_lab")
    if mean and len(mean) == 3:
        return (float(mean[0]), float(mean[1]), float(mean[2]))
    return _REF_LAB_BY_LABEL.get(color)


def _poll_jersey(candidates: list[dict], config, pinned_label: str | None = None) -> tuple[
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

    for body_dict in candidates:
        color = body_dict.get("cloth_color", "N/A")
        if color in ("N/A", "Unknown"):
            continue
        label_counts[color] = label_counts.get(color, 0) + 1
        lab = _stored_lab(body_dict)
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
    results_path = album_path / "album.json"

    with open(info_path, encoding="utf-8") as fh:
        info = json.load(fh)
    with open(results_path, encoding="utf-8") as fh:
        payload = json.load(fh)

    results: list[dict] = payload.get("results", [])
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
    gradable: list[tuple[dict, list[bool], list[bool]]] = []
    candidates: list[dict] = []
    for entry in results:
        if entry.get("status") not in ("blurry", "sharp"):
            continue
        ann = entry.get("annotation_data")
        bodies = ann.get("evaluated", []) if ann else []
        if not bodies:
            continue
        summary.images_considered += 1
        cleared = [_cleared_grading_gates(b) for b in bodies]
        eligible = [
            c and float(b.get("sharpness_score", 0.0)) > new_threshold
            for b, c in zip(bodies, cleared)
        ]
        gradable.append((entry, cleared, eligible))
        candidates.extend(b for b, e in zip(bodies, eligible) if e)

    # ---- Pass 1: make sure every candidate has a measured cloth colour ----
    # Bodies rejected for sharpness at import time never reached the colour
    # predictor (GradingStage skips failed bodies), so theirs must be
    # measured from pixels now. Bodies that got as far as the jersey check
    # already carry one and are reused as-is.
    if not no_team:
        for entry, _cleared, eligible in gradable:
            bodies = entry["annotation_data"]["evaluated"]
            needs_color = [
                b for b, e in zip(bodies, eligible)
                if e and b.get("cloth_color", "N/A") in ("N/A", None)
            ]
            if not needs_color:
                continue
            file_path = entry.get("file", "")
            image = _read_source_image(file_path)
            if image is None:
                summary.jersey_recheck_unreadable += len(needs_color)
                log.warning("[regrade] cannot re-read source photo for jersey colour: %s", file_path)
                continue
            for body_dict in needs_color:
                body = _body_from_entry(body_dict)
                body.cloth_color, body.cloth_color_detail = cloth_color_predictor.predict(body, image)
                body_dict["cloth_color"] = body.cloth_color
                body_dict["cloth_color_detail"] = body.cloth_color_detail
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

    for entry, cleared, eligible in gradable:
        old_status = entry["status"]
        ann = entry["annotation_data"]
        bodies = ann["evaluated"]

        for body_dict, was_cleared, is_eligible in zip(bodies, cleared, eligible):
            score = float(body_dict.get("sharpness_score", 0.0))
            if not is_eligible:
                body_dict["is_blurry"] = True
                if was_cleared:
                    # Only the threshold pushed it out; say so explicitly.
                    body_dict["rejection_reason"] = (
                        f"sharpness score {score:.4f} <= threshold {new_threshold:.2f}"
                    )
                continue

            if not apply_jersey_filter:
                body_dict["is_blurry"] = False
                body_dict["rejection_reason"] = ""
                continue

            body = _body_from_entry(body_dict)
            body.cloth_color = body_dict.get("cloth_color", "N/A")
            body.cloth_color_detail = body_dict.get("cloth_color_detail") or {}
            classify_body_jersey(
                body, forced_colors, regular_colors, forced_labs, allowed_labs,
                team_target_lab, team_bucket, our_color, app_config,
                log_prefix=Path(entry.get("file", "")).name,
            )
            body_dict["is_blurry"] = not body.passed
            body_dict["rejection_reason"] = body.rejection_reason

        new_overall_blurry = all(b.get("is_blurry", True) for b in bodies)
        new_status = "blurry" if new_overall_blurry else "sharp"

        passing_scores = [
            float(b.get("sharpness_score", 0.0)) for b in bodies if not b.get("is_blurry", True)
        ]
        best_score = max(passing_scores) if passing_scores else max(
            (float(b.get("sharpness_score", 0.0)) for b in bodies), default=0.0
        )
        entry["sharpness_score"] = round(best_score, 4)
        entry["sharpness_grade"] = round(best_score * 100, 1)
        ann["overall_blurry"] = new_overall_blurry
        entry["status"] = new_status

        if new_status != old_status:
            if new_status == "sharp":
                summary.recovered += 1
            else:
                summary.demoted += 1
            key = entry.get("key") or Path(entry.get("file", "")).name
            item = info_items_by_key.get(key)
            if item is not None:
                old_list = info.get(_REVIEW_INFO_KEY[old_status], [])
                if item in old_list:
                    old_list.remove(item)
                info.setdefault(_REVIEW_INFO_KEY[new_status], []).append(item)

            # A photo that changed side of the keep line carries a star
            # rating that no longer reflects it (LLM culling rated it under
            # the old verdict). Reset to the baseline unless the user rated
            # it by hand -- see culling_app.py's "stars_manual" marker.
            if not entry.get("stars_manual"):
                entry["stars"] = baseline_stars(entry, new_status, new_threshold)
                entry["keep"] = entry["stars"] >= 3
                summary.stars_rebaselined += 1

            if _regenerate_preview(entry, bodies, album_path):
                summary.previews_regenerated += 1
            else:
                summary.previews_regen_failed += 1
                log.warning("[regrade] verdict changed but preview could not be regenerated (source unreadable): %s",
                            entry.get("file"))

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

    atomic_save_and_backup(json.dumps(payload, indent=2), results_path)
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
