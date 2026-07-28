from __future__ import annotations

import base64
import json
import logging
import math
from pathlib import Path

import cv2

from algo.config import AppConfig
from algo.frame import Frame
from algo.llm.culling_provider import BurstFrameInput, BurstRankingResult, CullingProvider
from algo.stage import ProcessStage
from algo.stages.image_analysis import _read_image
from algo.utils import atomic_save_and_backup, cap_long_edge, image_capture_timestamp

log = logging.getLogger("BlurPictureDetector")

# Consecutive sharp photos within this many seconds of each other are treated
# as the same "burst" — mirrors culling_app.py's review-page grouping
# (BURST_GAP_SECONDS), but computed here over sharp frames only: a burst is a
# run of *already-qualified* shots, so one sharp frame surrounded by 100
# blurry ones is never treated as part of a "sequence".
DEFAULT_BURST_GAP_SECONDS = 1.0
# Bursts smaller than this are left untouched (nothing to rank between).
DEFAULT_MIN_GROUP_SIZE = 2
# Long-edge size (px) of the down-sized copy sent to the LLM.
DEFAULT_IMAGE_MAX_LONG_EDGE = 999
# Sharp frames that don't belong to a qualifying burst (see
# DEFAULT_MIN_GROUP_SIZE) are still sent to the LLM, but graded
# independently rather than ranked/captioned as a group — they're batched
# together purely to amortize each request's fixed overhead (system prompt,
# instructions) across multiple images, not because they're related.
DEFAULT_SINGLETON_BATCH_SIZE = 6
# Fraction of those standalone images marked as keepers, highest llm_grade
# first. Every standalone image keeps its llm_grade regardless of whether it
# lands in the kept fraction, so the score stays visible for manual review.
DEFAULT_SINGLETON_KEEP_FRACTION = 0.6

# Star-rating thresholds (fraction of all LLM-graded sharp frames, by
# llm_grade, highest first) — see _assign_star_ratings.
STAR_4_TOP_FRACTION = 0.4
STAR_5_TOP_FRACTION = 0.1


def _sorted_sharp_entries(payload: dict) -> list[dict]:
    """Sharp entries from *payload*, sorted chronologically. Each entry gets
    a transient "_timestamp" key added (caller is responsible for popping it
    before writing the payload back, if it intends to)."""
    entries = payload.get("results", [])
    sharp_entries = [e for e in entries if e.get("status") == "sharp"]
    for entry in sharp_entries:
        entry["_timestamp"] = image_capture_timestamp(Path(entry["file"]))
    sharp_entries.sort(key=lambda e: (e["_timestamp"], e["file"]))
    return sharp_entries


def _group_bursts(sharp_entries: list[dict], burst_gap_seconds: float) -> list[list[dict]]:
    """Group chronologically-sorted *sharp_entries* (as returned by
    :func:`_sorted_sharp_entries`) into bursts, splitting wherever the gap to
    the previous frame exceeds *burst_gap_seconds*."""
    bursts: list[list[dict]] = []
    for entry in sharp_entries:
        if bursts and entry["_timestamp"] - bursts[-1][-1]["_timestamp"] <= burst_gap_seconds:
            bursts[-1].append(entry)
        else:
            bursts.append([entry])
    return bursts


def load_qualifying_bursts(
    results_path: Path,
    burst_gap_seconds: float = DEFAULT_BURST_GAP_SECONDS,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> list[list[dict]]:
    """Read *results_path* (read-only — nothing is written back) and return
    the bursts of consecutive sharp frames that qualify for LLM ranking
    (>= *min_group_size* frames). Shared by :meth:`LLMCullingStage.process`
    and pre-flight cost estimates (e.g. RunLLMCulling.py) so the two can
    never drift out of sync with each other's burst definition."""
    with open(results_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    sharp_entries = _sorted_sharp_entries(payload)
    bursts = _group_bursts(sharp_entries, burst_gap_seconds)
    return [b for b in bursts if len(b) >= min_group_size]


class LLMCullingStage(ProcessStage):
    """Rank bursts of continuous, already-sharp frames with an LLM and mark
    the best shot in each burst as a keeper.

    Like :class:`~algo.stages.face_reco.FaceRecoStage`, this stage reads and
    rewrites ``album.json`` directly rather than operating purely on the
    in-memory ``Frame`` list — it is meant to run as a post-processing step
    (after face recognition) once ``album.json`` already exists on disk.

    Only entries with ``status == "sharp"`` participate. A burst is a run of
    sharp images whose capture timestamps are each within
    ``burst_gap_seconds`` of the previous one (sorted chronologically).
    Bursts with fewer than ``min_group_size`` frames don't qualify for
    ranking — those frames are instead graded individually (never ranked or
    captioned against each other) by :meth:`_grade_standalone_entries`,
    batched only for request efficiency: the top ``singleton_keep_fraction``
    of them (by ``llm_grade``) are marked keepers, the rest dropped, and
    every one of them keeps its ``llm_grade`` regardless for manual review.

    All qualifying bursts are handed to ``provider.rank_bursts`` in one call
    so a provider that supports it (e.g. :class:`~algo.llm.culling_provider.
    OpenAIProvider`) can fan the LLM calls out concurrently instead of
    waiting on each burst's round-trip one at a time. For each burst the
    provider picks the top ``min(3, len(burst))`` shots. Rank 1 gets
    ``keep=True``; every other frame in the burst — ranked #2/#3 or not
    ranked at all — gets ``keep=False``. Ranked (but not #1) frames
    additionally get a ``burst_ranking`` entry (rank + reason) recorded so
    the review UI can still show why they were close contenders, even
    though they're dropped. Separately, EVERY frame in the burst (not just
    the ranked top picks) gets an ``llm_grade`` (0.0-1.0 quality score) when
    the provider returns one, and every frame in the burst gets the same
    ``burst_caption`` (a short, punchy caption for the burst as a whole)
    when the provider returns one.

    A token-usage/cost summary (``provider.get_cost_summary()``) is logged
    and written to ``album.json`` as ``llm_cost_summary`` once processing
    finishes.
    """

    def __init__(
        self,
        output_dir: Path,
        provider: CullingProvider,
        threshold: float,
        burst_gap_seconds: float = DEFAULT_BURST_GAP_SECONDS,
        min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
        image_max_long_edge: int = DEFAULT_IMAGE_MAX_LONG_EDGE,
        singleton_batch_size: int = DEFAULT_SINGLETON_BATCH_SIZE,
        singleton_keep_fraction: float = DEFAULT_SINGLETON_KEEP_FRACTION,
    ) -> None:
        self.output_dir = output_dir
        self.provider = provider
        # The same blur-sensitivity threshold used earlier (GradingStage) to
        # split sharp/blurry — reused here to split the *discard* side of the
        # star scale (1 vs 2 stars) between "clearly bad" and "close to the
        # keep line". See _assign_star_ratings.
        self.threshold = threshold
        self.burst_gap_seconds = burst_gap_seconds
        self.min_group_size = min_group_size
        self.image_max_long_edge = image_max_long_edge
        self.singleton_batch_size = singleton_batch_size
        self.singleton_keep_fraction = singleton_keep_fraction

    def process(self, frames: list[Frame], config: AppConfig) -> list[Frame]:
        results_path = self.output_dir / "album.json"
        if not results_path.exists():
            log.warning("[LLMCullingStage] album.json not found at %s — skipping", results_path)
            return frames

        with open(results_path, encoding="utf-8") as fh:
            payload = json.load(fh)

        sharp_entries = _sorted_sharp_entries(payload)
        bursts = _group_bursts(sharp_entries, self.burst_gap_seconds)

        qualifying = [b for b in bursts if len(b) >= self.min_group_size]
        log.info(
            "[LLMCullingStage] %d sharp frame(s) -> %d burst(s), %d qualify (>= %d frames)",
            len(sharp_entries), len(bursts), len(qualifying), self.min_group_size,
        )

        # Build every qualifying burst's provider input up front so they can
        # all be handed to the provider in one batch call — this is what lets
        # a concurrency-capable provider (OpenAIProvider) fan the LLM calls
        # out in parallel instead of waiting on each burst serially.
        prepared: list[tuple[str, list[dict]]] = []
        burst_inputs_batch: list[list[BurstFrameInput]] = []
        if qualifying:
            log.info("[LLMCullingStage] preparing %d qualifying burst(s) for ranking …", len(qualifying))
        for idx, burst in enumerate(qualifying):
            group_id = f"burst-{idx:04d}"
            burst_inputs = [
                inp for inp in (self._build_frame_input(e) for e in burst) if inp is not None
            ]
            if len(burst_inputs) < self.min_group_size:
                log.debug(
                    "[LLMCullingStage] %s: only %d/%d frame(s) loaded — skipping",
                    group_id, len(burst_inputs), len(burst),
                )
                continue
            prepared.append((group_id, burst))
            burst_inputs_batch.append(burst_inputs)
            log.debug("[LLMCullingStage] prepared %d/%d burst(s)", idx + 1, len(qualifying))

        if prepared:
            log.info("[LLMCullingStage] sending %d burst(s) to LLM provider for ranking …", len(prepared))
            try:
                all_rankings = self.provider.rank_bursts(burst_inputs_batch)
            except Exception as exc:  # noqa: BLE001 — a batch failure must not abort the whole run
                log.error("[LLMCullingStage] provider.rank_bursts failed: %s", exc, exc_info=True)
                all_rankings = [BurstRankingResult(rankings=[], grades={}, caption="") for _ in prepared]
            for (group_id, burst), result in zip(prepared, all_rankings):
                self._apply_rankings(burst, group_id, result)
            log.info("[LLMCullingStage] burst ranking complete: %d burst(s) processed", len(prepared))

        standalone_entries = [e for b in bursts if len(b) < self.min_group_size for e in b]
        if standalone_entries:
            log.info("[LLMCullingStage] grading %d standalone image(s) …", len(standalone_entries))
            self._grade_standalone_entries(standalone_entries)

        all_entries = payload.get("results", [])
        blurry_entries = [e for e in all_entries if e.get("status") == "blurry"]
        skipped_entries = [e for e in all_entries if e.get("status") in ("skipped", "error")]
        self._assign_star_ratings(sharp_entries, blurry_entries, skipped_entries)

        for entry in sharp_entries:
            entry.pop("_timestamp", None)

        cost_summary = self.provider.get_cost_summary()
        payload["llm_cost_summary"] = {
            "session_count": cost_summary.session_count,
            "total_input_tokens": cost_summary.total_input_tokens,
            "total_output_tokens": cost_summary.total_output_tokens,
            "total_cost_usd": round(cost_summary.total_cost_usd, 6),
        }
        log.info(
            "[LLMCullingStage] LLM cost summary: %d session(s), %d input token(s), "
            "%d output token(s), $%.4f total",
            cost_summary.session_count, cost_summary.total_input_tokens,
            cost_summary.total_output_tokens, cost_summary.total_cost_usd,
        )

        atomic_save_and_backup(json.dumps(payload, indent=2), results_path)
        log.info("[LLMCullingStage] album.json updated: %s", results_path)

        return frames

    def _assign_star_ratings(
        self,
        sharp_entries: list[dict],
        blurry_entries: list[dict],
        skipped_entries: list[dict],
    ) -> None:
        """Assign a 1-5 star rating to every analysed entry (sharp, blurry,
        or skipped) so the whole album can be ranked on one scale:

        - 1 star:  no face detected (``skipped``/``error``), or a blurry
          frame whose sharpness_score is below ``min(threshold, 0.4)`` —
          clearly discard.
        - 2 stars: a blurry frame whose sharpness_score is between
          ``min(threshold, 0.4)`` and ``threshold`` — still below the keep
          line, but close to it.
        - 3 stars: baseline for every ``sharp`` frame (it already qualified
          as a keeper).
        - 4 stars: top ``STAR_4_TOP_FRACTION`` of sharp frames by
          ``llm_grade`` percentile (measured across every sharp frame that
          received a grade, whether ranked as part of a burst or graded
          standalone).
        - 5 stars: top ``STAR_5_TOP_FRACTION`` by that same percentile.

        Burst ranking then applies a *floor* on top of the 3-5 tiers: a
        burst's #1 pick is always at least 5 stars and its #2/#3 picks are
        always at least 4 stars, regardless of where their llm_grade falls
        in the global percentile (it can never be lowered by this step —
        only raised).
        """
        low_cutoff = min(self.threshold, 0.4)
        for entry in skipped_entries:
            entry["stars"] = 1
        for entry in blurry_entries:
            score = entry.get("sharpness_score")
            entry["stars"] = 1 if score is None or score < low_cutoff else 2

        for entry in sharp_entries:
            entry["stars"] = 3

        graded = [e for e in sharp_entries if e.get("llm_grade") is not None]
        graded.sort(key=lambda e: e["llm_grade"], reverse=True)
        top_4_count = math.ceil(len(graded) * STAR_4_TOP_FRACTION)
        top_5_count = math.ceil(len(graded) * STAR_5_TOP_FRACTION)
        for entry in graded[:top_4_count]:
            entry["stars"] = 4
        for entry in graded[:top_5_count]:
            entry["stars"] = 5

        for entry in sharp_entries:
            rank = (entry.get("burst_ranking") or {}).get("rank")
            if rank == 1:
                entry["stars"] = max(entry["stars"], 5)
            elif rank in (2, 3):
                entry["stars"] = max(entry["stars"], 4)

        log.info(
            "[LLMCullingStage] star ratings: %d1\u2605, %d2\u2605, %d3\u2605, %d4\u2605, %d5\u2605 (%d sharp/%d blurry/%d skipped)",
            sum(1 for e in skipped_entries + blurry_entries if e["stars"] == 1),
            sum(1 for e in blurry_entries if e["stars"] == 2),
            sum(1 for e in sharp_entries if e["stars"] == 3),
            sum(1 for e in sharp_entries if e["stars"] == 4),
            sum(1 for e in sharp_entries if e["stars"] == 5),
            len(sharp_entries), len(blurry_entries), len(skipped_entries),
        )

    def _apply_rankings(self, burst: list[dict], group_id: str, result: BurstRankingResult) -> None:
        rank_by_file = {r.file: r for r in result.rankings}
        if not rank_by_file:
            log.warning("[LLMCullingStage] %s: provider returned no usable ranking — leaving as-is", group_id)
            return

        by_name = {Path(e["file"]).name: e for e in burst}
        for filename, entry in by_name.items():
            ranked = rank_by_file.get(filename)
            if ranked is not None:
                entry["keep"] = ranked.rank == 1
                entry["burst_ranking"] = {
                    "rank": ranked.rank,
                    "reason": ranked.reason,
                    "group_id": group_id,
                }
            else:
                entry["keep"] = False
            grade = result.grades.get(filename)
            if grade is not None:
                entry["llm_grade"] = grade
            if result.caption:
                entry["burst_caption"] = result.caption

        top = next((r.file for r in result.rankings if r.rank == 1), "n/a")
        log.info("[LLMCullingStage] %s: %d frame(s) ranked, top pick=%s", group_id, len(result.rankings), top)

    def _grade_standalone_entries(self, entries: list[dict]) -> None:
        """Grade sharp frames that don't belong to a qualifying burst
        (``entries``) individually via ``provider.grade_image_batches``.
        Entries are chunked into ``singleton_batch_size``-sized groups purely
        to amortize each request's fixed overhead across multiple images —
        the LLM still grades each one independently, never comparing them.

        Every entry that gets a grade back has its ``llm_grade`` recorded.
        The top ``singleton_keep_fraction`` of *graded* entries (by grade,
        highest first) are marked ``keep=True``, the rest ``keep=False`` —
        every graded entry keeps its ``llm_grade`` regardless, so the score
        stays visible for manual review even when dropped. Entries whose
        batch fails (no grade returned) are left untouched.
        """
        batches: list[list[BurstFrameInput]] = []
        by_name_per_batch: list[dict[str, dict]] = []
        for i in range(0, len(entries), self.singleton_batch_size):
            chunk = entries[i:i + self.singleton_batch_size]
            inputs = [inp for inp in (self._build_frame_input(e) for e in chunk) if inp is not None]
            if not inputs:
                continue
            batches.append(inputs)
            by_name_per_batch.append({Path(e["file"]).name: e for e in chunk})
            log.debug(
                "[LLMCullingStage] standalone: prepared batch %d (%d image(s))",
                len(batches), len(inputs),
            )

        if not batches:
            return

        log.info(
            "[LLMCullingStage] sending %d standalone batch(es) (%d image(s)) to LLM provider for grading …",
            len(batches), len(entries),
        )
        try:
            grade_results = self.provider.grade_image_batches(batches)
        except Exception as exc:  # noqa: BLE001 — a batch failure must not abort the whole run
            log.error("[LLMCullingStage] provider.grade_image_batches failed: %s", exc, exc_info=True)
            grade_results = [{} for _ in batches]

        graded_entries: list[dict] = []
        for by_name, grades in zip(by_name_per_batch, grade_results):
            for filename, grade in grades.items():
                entry = by_name.get(filename)
                if entry is None:
                    continue
                entry["llm_grade"] = grade
                graded_entries.append(entry)

        if not graded_entries:
            log.warning(
                "[LLMCullingStage] standalone grading returned no usable grade for any of %d entr(y/ies)",
                len(entries),
            )
            return

        graded_entries.sort(key=lambda e: e["llm_grade"], reverse=True)
        keep_count = round(len(graded_entries) * self.singleton_keep_fraction)
        for idx, entry in enumerate(graded_entries):
            entry["keep"] = idx < keep_count

        log.info(
            "[LLMCullingStage] standalone: %d/%d image(s) graded, top %d (%.0f%%) marked keep",
            len(graded_entries), len(entries), keep_count, self.singleton_keep_fraction * 100,
        )

    def _build_frame_input(self, entry: dict) -> BurstFrameInput | None:
        file_path = Path(entry["file"])
        image = _read_image(file_path)
        if image is None:
            log.warning("[LLMCullingStage] could not read %s — excluding from burst", file_path)
            return None

        resized = cap_long_edge(image, self.image_max_long_edge)
        ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            log.warning("[LLMCullingStage] could not encode %s — excluding from burst", file_path)
            return None
        image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        ann = entry.get("annotation_data") or {}
        evaluated = ann.get("evaluated", [])
        best_body = max(
            (b for b in evaluated if not b.get("is_blurry", True)),
            key=lambda b: b.get("sharpness_score", 0.0),
            default=None,
        )
        face_bbox = None
        if best_body is not None:
            box = best_body.get("narrow_face_bbox") or best_body.get("face_bbox")
            if box is not None:
                face_bbox = (box["x1"], box["y1"], box["x2"], box["y2"])

        return BurstFrameInput(
            file=file_path.name,
            image_b64=image_b64,
            face_bbox=face_bbox,
            sharpness_score=float(entry.get("sharpness_score", 0.0)),
        )
