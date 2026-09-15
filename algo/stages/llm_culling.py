from __future__ import annotations

import base64
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cv2

from algo.album import Album, AlbumImage
from algo.config import AppConfig
from algo.frame import Frame
from algo.llm.culling_provider import BurstFrameInput, BurstRankingResult, CullingProvider
from algo.stage import ProcessStage
from algo.stages.image_analysis import _read_image
from algo.utils import cap_long_edge, image_capture_timestamp

log = logging.getLogger("BlurPictureDetector")

# Consecutive sharp photos within this many seconds of each other are treated
# as the same "burst" — mirrors culling_app.py's review-page grouping
# (BURST_GAP_SECONDS), but computed here over sharp frames only: a burst is a
# run of *already-qualified* shots, so one sharp frame surrounded by 100
# blurry ones is never treated as part of a "sequence".
DEFAULT_BURST_GAP_SECONDS = 1.0
# A burst only qualifies as a "sequence" (see _is_qualifying_sequence) once
# its duration exceeds this many seconds OR it has more than
# DEFAULT_MIN_SEQUENCE_FRAMES frames — anything smaller is graded like a
# standalone image instead (a burst always has >= 2 frames by construction,
# so both bounds already imply that).
DEFAULT_MIN_SEQUENCE_SECONDS = 0.5
DEFAULT_MIN_SEQUENCE_FRAMES = 8
# Long-edge size (px) of the down-sized copy sent to the LLM.
DEFAULT_IMAGE_MAX_LONG_EDGE = 999
# Sharp frames that don't belong to a qualifying sequence (see
# _is_qualifying_sequence) are still sent to the LLM, but graded
# independently rather than ranked/captioned as a group — they're batched
# together purely to amortize each request's fixed overhead (system prompt,
# instructions) across multiple images, not because they're related.
DEFAULT_SINGLETON_BATCH_SIZE = 6
# Fraction of those standalone images marked as keepers, highest llm_grade
# first. Every standalone image keeps its llm_grade regardless of whether it
# lands in the kept fraction, so the score stays visible for manual review.
DEFAULT_SINGLETON_KEEP_FRACTION = 0.6

# Star-rating thresholds for standalone (non-sequence) sharp frames —
# fraction of all LLM-graded standalone frames, by llm_grade, highest
# first — see _assign_star_ratings.
STAR_4_TOP_FRACTION = 0.4
STAR_5_TOP_FRACTION = 0.1

# Star-rating tiers for frames inside a qualifying sequence (see
# _assign_sequence_stars): one 4-star pick per this many seconds of burst
# duration, then the top SEQUENCE_STAR_3_TOP_FRACTION of what's left get 3
# stars, everything else gets 2 stars. No 5-star tier applies within a
# sequence.
SEQUENCE_STAR_4_QUOTA_SECONDS = 0.5
SEQUENCE_STAR_3_TOP_FRACTION = 0.30


@dataclass(frozen=True)
class TimedImage:
    """An album photo paired with its capture time — burst grouping's unit
    of work.

    The timestamp is derived (EXIF, else file times) rather than stored, so
    it lives here instead of on the photo's album entry, where it used to be
    injected as a transient ``_timestamp`` key that every writer then had to
    remember to strip before saving.
    """
    image: AlbumImage
    timestamp: float

    @property
    def key(self) -> str:
        return self.image.key


def _sorted_sharp_images(album: Album) -> list[TimedImage]:
    """Every sharp photo in *album*, chronologically."""
    timed = [
        TimedImage(image, image_capture_timestamp(Path(image.source_file)))
        for image in album.image_list if image.is_sharp
    ]
    timed.sort(key=lambda t: (t.timestamp, t.image.source_file))
    return timed


def _group_bursts(sharp_images: list[TimedImage], burst_gap_seconds: float) -> list[list[TimedImage]]:
    """Group chronologically-sorted *sharp_images* (as returned by
    :func:`_sorted_sharp_images`) into bursts, splitting wherever the gap to
    the previous frame exceeds *burst_gap_seconds*."""
    bursts: list[list[TimedImage]] = []
    for timed in sharp_images:
        if bursts and timed.timestamp - bursts[-1][-1].timestamp <= burst_gap_seconds:
            bursts[-1].append(timed)
        else:
            bursts.append([timed])
    return bursts


def _is_qualifying_sequence(
    burst: list[TimedImage],
    min_seconds: float = DEFAULT_MIN_SEQUENCE_SECONDS,
    min_frames: int = DEFAULT_MIN_SEQUENCE_FRAMES,
) -> bool:
    """Whether *burst* (a group from :func:`_group_bursts`) is long/large
    enough to be treated as a "sequence" (ranked as a group, and star-rated
    via :meth:`LLMCullingStage._assign_sequence_stars`) rather than as a set
    of standalone images. Requires the burst's capture-time span to exceed
    *min_seconds*, OR its frame count to exceed *min_frames*."""
    if len(burst) < 2:
        return False
    duration = burst[-1].timestamp - burst[0].timestamp
    return duration > min_seconds or len(burst) > min_frames


def load_qualifying_bursts(
    album_dir: Path,
    burst_gap_seconds: float = DEFAULT_BURST_GAP_SECONDS,
    min_sequence_seconds: float = DEFAULT_MIN_SEQUENCE_SECONDS,
    min_sequence_frames: int = DEFAULT_MIN_SEQUENCE_FRAMES,
) -> list[list[TimedImage]]:
    """Read *album_dir*'s `Album` (read-only — nothing is written back) and
    return the bursts of consecutive sharp frames that qualify as sequences
    (see :func:`_is_qualifying_sequence`) for LLM group-ranking. Shared by
    :meth:`LLMCullingStage.process` and pre-flight cost estimates (e.g.
    RerunLLMCulling.py) so the two can never drift out of sync with each
    other's burst definition."""
    sharp_images = _sorted_sharp_images(Album(album_dir))
    bursts = _group_bursts(sharp_images, burst_gap_seconds)
    return [b for b in bursts if _is_qualifying_sequence(b, min_sequence_seconds, min_sequence_frames)]


class LLMCullingStage(ProcessStage):
    """Rank bursts of continuous, already-sharp frames with an LLM and mark
    the best shot in each burst as a keeper.

    Like :class:`~algo.stages.face_reco.FaceRecoStage`, this stage reads and
    rewrites the `Album` (see algo/album.py) directly rather than operating
    purely on the in-memory ``Frame`` list — it is meant to run as a
    post-processing step (after face recognition) once the Album already
    exists on disk.

    Only entries with ``status == "sharp"`` participate. A burst is a run of
    sharp images whose capture timestamps are each within
    ``burst_gap_seconds`` of the previous one (sorted chronologically). A
    burst only qualifies as a "sequence" (see :func:`_is_qualifying_sequence`)
    once its capture-time span exceeds ``min_sequence_seconds`` OR it has
    more than ``min_sequence_frames`` frames — smaller bursts don't qualify
    for group ranking; those frames are instead graded individually (never
    ranked or captioned against each other) by
    :meth:`_grade_standalone_entries`, batched only for request efficiency:
    the top ``singleton_keep_fraction`` of them (by ``llm_grade``) are marked
    keepers, the rest dropped, and every one of them keeps its ``llm_grade``
    regardless for manual review.

    All qualifying sequences are handed to ``provider.rank_bursts`` in one
    call so a provider that supports it (e.g. :class:`~algo.llm.culling_provider.
    OpenAIProvider`) can fan the LLM calls out concurrently instead of
    waiting on each burst's round-trip one at a time. For each sequence the
    provider picks the top ``min(3, len(burst))`` shots. Rank 1 gets
    ``keep=True``; every other frame in the burst — ranked #2/#3 or not
    ranked at all — gets ``keep=False``. Ranked (but not #1) frames
    additionally get a ``burst_ranking`` entry (rank + reason) recorded so
    the review UI can still show why they were close contenders, even
    though they're dropped. Separately, EVERY frame in the burst (not just
    the ranked top picks) gets an ``llm_grade`` (0.0-1.0 quality score) when
    the provider returns one, and every frame in the burst gets the same
    ``burst_caption`` (a short, punchy caption for the burst as a whole)
    when the provider returns one. Star ratings within a qualifying sequence
    are then assigned by :meth:`_assign_sequence_stars` instead of the
    global-percentile rule used for standalone frames — see
    :meth:`_assign_star_ratings`.

    A token-usage/cost summary (``provider.get_cost_summary()``) is logged
    and written to the `Album` as ``llm_cost_summary`` once processing
    finishes.
    """

    def __init__(
        self,
        output_dir: Path,
        provider: CullingProvider,
        threshold: float,
        burst_gap_seconds: float = DEFAULT_BURST_GAP_SECONDS,
        min_sequence_seconds: float = DEFAULT_MIN_SEQUENCE_SECONDS,
        min_sequence_frames: int = DEFAULT_MIN_SEQUENCE_FRAMES,
        image_max_long_edge: int = DEFAULT_IMAGE_MAX_LONG_EDGE,
        singleton_batch_size: int = DEFAULT_SINGLETON_BATCH_SIZE,
        singleton_keep_fraction: float = DEFAULT_SINGLETON_KEEP_FRACTION,
        new_keys: frozenset[str] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.provider = provider
        # The same blur-sensitivity threshold used earlier (GradingStage) to
        # split sharp/blurry — reused here to split the *discard* side of the
        # star scale (1 vs 2 stars) between "clearly bad" and "close to the
        # keep line". See _assign_star_ratings.
        self.threshold = threshold
        self.burst_gap_seconds = burst_gap_seconds
        self.min_sequence_seconds = min_sequence_seconds
        self.min_sequence_frames = min_sequence_frames
        self.image_max_long_edge = image_max_long_edge
        self.singleton_batch_size = singleton_batch_size
        self.singleton_keep_fraction = singleton_keep_fraction
        # Bookkeeping keys (see algo/utils.py::make_unique_import_key) of the
        # frames processed THIS run. When provided, only bursts/standalone
        # entries containing at least one of these are sent to the LLM --
        # "import more images" would otherwise re-rank (and re-bill) every
        # burst in the whole album on every single import. None (the
        # default) disables this filter so a fresh, non-merge run still
        # ranks every qualifying sequence as before.
        self.new_keys = new_keys

    def process(self, frames: list[Frame], config: AppConfig) -> list[Frame]:
        album = Album(self.output_dir)
        if not album.exists:
            log.warning("[LLMCullingStage] no Album found at %s — skipping", album.path)
            return frames

        sharp_images = _sorted_sharp_images(album)
        bursts = _group_bursts(sharp_images, self.burst_gap_seconds)

        sequences = [
            b for b in bursts
            if _is_qualifying_sequence(b, self.min_sequence_seconds, self.min_sequence_frames)
        ]
        log.info(
            "[LLMCullingStage] %d sharp frame(s) -> %d burst(s), %d qualify as sequences "
            "(duration > %.1fs or > %d frames)",
            len(sharp_images), len(bursts), len(sequences),
            self.min_sequence_seconds, self.min_sequence_frames,
        )

        to_rank = sequences
        if self.new_keys is not None:
            before = len(to_rank)
            to_rank = [b for b in to_rank if any(t.key in self.new_keys for t in b)]
            log.info(
                "[LLMCullingStage] import-more: %d/%d qualifying sequence(s) touch a newly-imported "
                "photo -- only those will be (re-)ranked",
                len(to_rank), before,
            )

        # Build every qualifying sequence's provider input up front so they
        # can all be handed to the provider in one batch call — this is what
        # lets a concurrency-capable provider (OpenAIProvider) fan the LLM
        # calls out in parallel instead of waiting on each burst serially.
        prepared: list[tuple[str, list[TimedImage]]] = []
        burst_inputs_batch: list[list[BurstFrameInput]] = []
        if to_rank:
            log.info("[LLMCullingStage] preparing %d qualifying sequence(s) for ranking …", len(to_rank))
        for idx, burst in enumerate(to_rank):
            group_id = f"burst-{idx:04d}"
            burst_inputs = [
                inp for inp in (self._build_frame_input(t.image) for t in burst) if inp is not None
            ]
            if len(burst_inputs) < 2:
                log.debug(
                    "[LLMCullingStage] %s: only %d/%d frame(s) loaded — skipping",
                    group_id, len(burst_inputs), len(burst),
                )
                continue
            prepared.append((group_id, burst))
            burst_inputs_batch.append(burst_inputs)
            log.debug("[LLMCullingStage] prepared %d/%d sequence(s)", idx + 1, len(to_rank))

        if prepared:
            log.info("[LLMCullingStage] sending %d sequence(s) to LLM provider for ranking …", len(prepared))
            try:
                all_rankings = self.provider.rank_bursts(burst_inputs_batch)
            except Exception as exc:  # noqa: BLE001 — a batch failure must not abort the whole run
                log.error("[LLMCullingStage] provider.rank_bursts failed: %s", exc, exc_info=True)
                all_rankings = [BurstRankingResult(rankings=[], grades={}, caption="") for _ in prepared]
            for (group_id, burst), result in zip(prepared, all_rankings):
                self._apply_rankings(burst, group_id, result)
            log.info("[LLMCullingStage] sequence ranking complete: %d sequence(s) processed", len(prepared))

        standalone_images = [
            t for b in bursts
            if not _is_qualifying_sequence(b, self.min_sequence_seconds, self.min_sequence_frames)
            for t in b
        ]
        if self.new_keys is not None:
            before = len(standalone_images)
            standalone_images = [t for t in standalone_images if t.key in self.new_keys]
            if before:
                log.info(
                    "[LLMCullingStage] import-more: %d/%d standalone image(s) are newly-imported -- "
                    "only those will be graded",
                    len(standalone_images), before,
                )
        if standalone_images:
            log.info("[LLMCullingStage] grading %d standalone image(s) …", len(standalone_images))
            self._grade_standalone_images([t.image for t in standalone_images])

        all_images = album.image_list
        blurry_images = [i for i in all_images if i.is_blurry]
        skipped_images = [i for i in all_images if i.status in ("skipped", "error")]
        self._assign_star_ratings(sharp_images, blurry_images, skipped_images, sequences)

        cost_summary = self.provider.get_cost_summary()
        album.llm_cost_summary = {
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

        album.save()
        log.info("[LLMCullingStage] Album updated: %s", album.path)

        return frames

    def _assign_star_ratings(
        self,
        sharp_images: list[TimedImage],
        blurry_images: list[AlbumImage],
        skipped_images: list[AlbumImage],
        sequences: list[list[TimedImage]],
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

        Sharp frames belonging to a qualifying *sequence* (see
        :func:`_is_qualifying_sequence`) then get their 2/3/4-star tier
        overridden by :meth:`_assign_sequence_stars` instead — sequences
        never compete against the rest of the album on one global
        percentile, and have no 5-star tier.

        Every OTHER sharp frame (standalone images, and frames in a burst
        too short/small to qualify as a sequence) keeps the original
        global-percentile rule:

        - 4 stars: top ``STAR_4_TOP_FRACTION`` of those frames by
          ``llm_grade`` percentile.
        - 5 stars: top ``STAR_5_TOP_FRACTION`` by that same percentile.
        """
        low_cutoff = min(self.threshold, 0.4)
        for image in skipped_images:
            image.stars = 1
        for image in blurry_images:
            score = image.sharpness_score
            image.stars = 1 if score is None or score < low_cutoff else 2

        for timed in sharp_images:
            timed.image.stars = 3

        for burst in sequences:
            self._assign_sequence_stars(burst)

        sequence_keys = {t.key for burst in sequences for t in burst}
        non_sequence_sharp = [t.image for t in sharp_images if t.key not in sequence_keys]

        graded = [i for i in non_sequence_sharp if i.llm_grade is not None]
        graded.sort(key=lambda i: i.llm_grade, reverse=True)
        top_4_count = math.ceil(len(graded) * STAR_4_TOP_FRACTION)
        top_5_count = math.ceil(len(graded) * STAR_5_TOP_FRACTION)
        for image in graded[:top_4_count]:
            image.stars = 4
        for image in graded[:top_5_count]:
            image.stars = 5

        sharp = [t.image for t in sharp_images]
        log.info(
            "[LLMCullingStage] star ratings: %d1\u2605, %d2\u2605, %d3\u2605, %d4\u2605, %d5\u2605 (%d sharp/%d blurry/%d skipped)",
            sum(1 for i in skipped_images + blurry_images if i.stars == 1),
            sum(1 for i in blurry_images if i.stars == 2),
            sum(1 for i in sharp if i.stars == 3),
            sum(1 for i in sharp if i.stars == 4),
            sum(1 for i in sharp if i.stars == 5),
            len(sharp), len(blurry_images), len(skipped_images),
        )

    def _assign_sequence_stars(self, burst: list[TimedImage]) -> None:
        """Star tiers for one qualifying burst sequence, scored independently
        per-sequence (never pooled against other sequences or the rest of
        the album):

        - 4 stars: the top ``floor(duration / SEQUENCE_STAR_4_QUOTA_SECONDS)``
          frames by ``llm_grade`` (clamped to at least 1, and at most every
          graded frame in the burst) — e.g. a 1.5s sequence picks its top 3.
        - 3 stars: the top ``SEQUENCE_STAR_3_TOP_FRACTION`` of whatever's
          left (by ``llm_grade``).
        - 2 stars: everything else in the sequence.

        There is no 5-star tier here. If the burst has no graded frames at
        all (e.g. the LLM call failed), it's left at the baseline 3 stars
        every sharp frame already got, rather than guessing.
        """
        graded = [t for t in burst if t.image.llm_grade is not None]
        if not graded:
            return
        graded.sort(key=lambda t: t.image.llm_grade, reverse=True)

        duration = burst[-1].timestamp - burst[0].timestamp
        four_star_count = min(
            max(math.floor(duration / SEQUENCE_STAR_4_QUOTA_SECONDS), 1),
            len(graded),
        )
        four_star_keys = {t.key for t in graded[:four_star_count]}

        remaining = graded[four_star_count:]
        three_star_count = math.ceil(len(remaining) * SEQUENCE_STAR_3_TOP_FRACTION)
        three_star_keys = {t.key for t in remaining[:three_star_count]}

        for timed in burst:
            if timed.key in four_star_keys:
                timed.image.stars = 4
            elif timed.key in three_star_keys:
                timed.image.stars = 3
            else:
                timed.image.stars = 2

    def _apply_rankings(self, burst: list[TimedImage], group_id: str, result: BurstRankingResult) -> None:
        rank_by_file = {r.file: r for r in result.rankings}
        if not rank_by_file:
            log.warning("[LLMCullingStage] %s: provider returned no usable ranking — leaving as-is", group_id)
            return

        for timed in burst:
            image = timed.image
            ranked = rank_by_file.get(timed.key)
            if ranked is not None:
                image.keep = ranked.rank == 1
                image.burst_ranking = {
                    "rank": ranked.rank,
                    "reason": ranked.reason,
                    "group_id": group_id,
                }
            else:
                image.keep = False
            grade = result.grades.get(timed.key)
            if grade is not None:
                image.llm_grade = grade
            if result.caption:
                image.burst_caption = result.caption

        top = next((r.file for r in result.rankings if r.rank == 1), "n/a")
        log.info("[LLMCullingStage] %s: %d frame(s) ranked, top pick=%s", group_id, len(result.rankings), top)

    def _grade_standalone_images(self, images: list[AlbumImage]) -> None:
        """Grade sharp frames that don't belong to a qualifying burst
        (``images``) individually via ``provider.grade_image_batches``.
        Images are chunked into ``singleton_batch_size``-sized groups purely
        to amortize each request's fixed overhead across multiple images —
        the LLM still grades each one independently, never comparing them.

        Every image that gets a grade back has its ``llm_grade`` recorded.
        The top ``singleton_keep_fraction`` of *graded* images (by grade,
        highest first) are marked ``keep=True``, the rest ``keep=False`` —
        every graded image keeps its ``llm_grade`` regardless, so the score
        stays visible for manual review even when dropped. Images whose
        batch fails (no grade returned) are left untouched.
        """
        batches: list[list[BurstFrameInput]] = []
        by_name_per_batch: list[dict[str, AlbumImage]] = []
        for i in range(0, len(images), self.singleton_batch_size):
            chunk = images[i:i + self.singleton_batch_size]
            inputs = [inp for inp in (self._build_frame_input(im) for im in chunk) if inp is not None]
            if not inputs:
                continue
            batches.append(inputs)
            by_name_per_batch.append({im.key: im for im in chunk})
            log.debug(
                "[LLMCullingStage] standalone: prepared batch %d (%d image(s))",
                len(batches), len(inputs),
            )

        if not batches:
            return

        log.info(
            "[LLMCullingStage] sending %d standalone batch(es) (%d image(s)) to LLM provider for grading …",
            len(batches), len(images),
        )
        try:
            grade_results = self.provider.grade_image_batches(batches)
        except Exception as exc:  # noqa: BLE001 — a batch failure must not abort the whole run
            log.error("[LLMCullingStage] provider.grade_image_batches failed: %s", exc, exc_info=True)
            grade_results = [{} for _ in batches]

        graded_images: list[AlbumImage] = []
        for by_name, grades in zip(by_name_per_batch, grade_results):
            for filename, grade in grades.items():
                image = by_name.get(filename)
                if image is None:
                    continue
                image.llm_grade = grade
                graded_images.append(image)

        if not graded_images:
            log.warning(
                "[LLMCullingStage] standalone grading returned no usable grade for any of %d image(s)",
                len(images),
            )
            return

        graded_images.sort(key=lambda i: i.llm_grade, reverse=True)
        keep_count = round(len(graded_images) * self.singleton_keep_fraction)
        for idx, image in enumerate(graded_images):
            image.keep = idx < keep_count

        log.info(
            "[LLMCullingStage] standalone: %d/%d image(s) graded, top %d (%.0f%%) marked keep",
            len(graded_images), len(images), keep_count, self.singleton_keep_fraction * 100,
        )

    def _build_frame_input(self, album_image: AlbumImage) -> BurstFrameInput | None:
        file_path = album_image.original_path
        image = _read_image(file_path) if file_path else None
        if image is None:
            log.warning("[LLMCullingStage] could not read %s — excluding from burst", file_path)
            return None

        resized = cap_long_edge(image, self.image_max_long_edge)
        ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            log.warning("[LLMCullingStage] could not encode %s — excluding from burst", file_path)
            return None
        image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        passing = [b for b in album_image.bodies if b.passed]
        best_body = max(passing, key=lambda b: b.sharpness_score, default=None)
        face_bbox = None
        if best_body is not None:
            box = best_body.narrow_face_bbox or best_body.face_bbox
            if box is not None:
                face_bbox = (box.x1, box.y1, box.x2, box.y2)

        return BurstFrameInput(
            file=album_image.key,
            image_b64=image_b64,
            face_bbox=face_bbox,
            sharpness_score=album_image.sharpness_score or 0.0,
        )
