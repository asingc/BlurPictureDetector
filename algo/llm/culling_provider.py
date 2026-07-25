"""LLM burst-culling provider abstraction.

:class:`CullingProvider` is the interface every backend implements. Naming
convention: concrete backends live in this same module as
``<Backend>Provider`` (e.g. :class:`OpenAIProvider`) so call sites just need
``from algo.llm.culling_provider import CullingProvider, OpenAIProvider``
instead of a longer ``OpenAILLMCullingProvider``-style name.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from algo.llm import prompts

log = logging.getLogger("BlurPictureDetector")


@dataclass
class BurstFrameInput:
    """One candidate frame within a burst, ready to send to a provider."""
    file:            str                                        # original filename (unique key within the burst)
    image_b64:       str                                        # base64-encoded JPEG, downsized for LLM consumption
    face_bbox:       tuple[float, float, float, float] | None    # normalised (x1, y1, x2, y2), 0..1
    sharpness_score: float


@dataclass
class RankedFrame:
    """One of the provider's top picks within a burst."""
    file:   str
    rank:   int   # 1 (best) .. 3
    reason: str


@dataclass
class BurstRankingResult:
    """Everything a provider returns for one burst: the top-N ranking (as
    before) plus a 0.0-1.0 quality grade for EVERY frame shown (not just the
    top picks) and a short caption describing the burst as a whole. All
    three fields default to "nothing usable" (empty list/dict/string) so a
    total parse/API failure can be represented without raising."""
    rankings: list[RankedFrame]
    grades:   dict[str, float]   # file -> grade 0.0 (worst) .. 1.0 (best)
    caption:  str                # short, punchy caption for the whole burst


@dataclass
class SessionCost:
    """Token usage + actual dollar cost of a single LLM call ("session")."""
    input_tokens:  int
    output_tokens: int
    cost_usd:      float


@dataclass
class CostSummary:
    """Aggregate cost across every session a provider has made so far."""
    session_count:       int
    total_input_tokens:  int
    total_output_tokens: int
    total_cost_usd:      float


class CullingProvider(ABC):
    """Abstract interface for an LLM backend that ranks the top shots within
    a burst of continuous, already-sharp frames.

    Implementations translate the shared prompt (``algo/llm/prompts.py``)
    into their own API's request shape and parse the reply back into
    :class:`RankedFrame` objects. Cost tracking (:class:`SessionCost` /
    :class:`CostSummary`) is handled once here, in the base class, via
    :meth:`_record_cost` / :meth:`get_cost_summary` — shared by every backend
    so callers don't need provider-specific code to get a cost report.
    """

    def __init__(self) -> None:
        self._session_costs: list[SessionCost] = []
        self._cost_lock = threading.Lock()

    @abstractmethod
    def rank_burst(self, frames: list[BurstFrameInput]) -> BurstRankingResult:
        """Return a :class:`BurstRankingResult` for this burst: the top
        ``min(3, len(frames))`` ranked picks, a 0.0-1.0 grade for every frame
        shown, and a short caption for the burst as a whole.

        Implementations should return an all-empty ``BurstRankingResult()``
        (rather than raise) when the backend call fails or the response
        can't be parsed, so a single burst failure never aborts the whole
        processing run.
        """
        raise NotImplementedError

    def rank_bursts(self, bursts: list[list[BurstFrameInput]]) -> list[BurstRankingResult]:
        """Rank multiple independent bursts, returned in the same order as
        *bursts*. The default implementation processes them one at a time;
        providers whose backend supports concurrent requests (e.g.
        :class:`OpenAIProvider`) should override this to fan the calls out
        instead of paying for each burst's round-trip serially.
        """
        return [self.rank_burst(burst) for burst in bursts]

    @abstractmethod
    def grade_images(self, frames: list[BurstFrameInput]) -> dict[str, float]:
        """Grade a batch of unrelated STANDALONE images (frames that don't
        belong to a qualifying burst) independently on a 0.0-1.0 quality
        scale (1.0 = best). Unlike :meth:`rank_burst`, these images are NOT
        the same moment/sequence -- callers batch several together purely to
        amortize the request's fixed overhead, so implementations must grade
        each image on its own merits, never comparing them against each
        other, and must not produce a ranking or shared caption.

        Implementations should return ``{}`` (rather than raise) when the
        backend call fails or the response can't be parsed, so a single
        batch's failure never aborts the whole run.
        """
        raise NotImplementedError

    def grade_image_batches(self, batches: list[list[BurstFrameInput]]) -> list[dict[str, float]]:
        """Grade multiple independent batches of standalone images, returned
        in the same order as *batches*. The default implementation processes
        them one at a time; providers whose backend supports concurrent
        requests (e.g. :class:`OpenAIProvider`) should override this.
        """
        return [self.grade_images(batch) for batch in batches]

    def _record_cost(self, cost: SessionCost) -> None:
        """Thread-safe: subclasses may call this from multiple worker
        threads when ``rank_bursts`` fans requests out concurrently."""
        with self._cost_lock:
            self._session_costs.append(cost)

    def get_cost_summary(self) -> CostSummary:
        """Aggregate cost across every session recorded so far."""
        with self._cost_lock:
            costs = list(self._session_costs)
        return CostSummary(
            session_count=len(costs),
            total_input_tokens=sum(c.input_tokens for c in costs),
            total_output_tokens=sum(c.output_tokens for c in costs),
            total_cost_usd=sum(c.cost_usd for c in costs),
        )


def _parse_llm_response(raw: str | None, frames: list[BurstFrameInput], top_n: int) -> BurstRankingResult:
    """Shared, provider-agnostic parsing of the
    ``{"caption": ..., "rankings": [...], "grades": [...]}`` JSON contract
    described in ``prompts.build_user_instructions``. Silently drops any
    ranking/grade entry that doesn't reference a real file in *frames*; an
    out-of-range rank or duplicate rank drops that ranking entry; an
    out-of-range grade is clamped into [0.0, 1.0] rather than dropped.
    Never raises — a totally unparseable response yields an all-empty
    :class:`BurstRankingResult`."""
    valid_files = {f.file for f in frames}
    try:
        payload = json.loads(raw or "")
    except (json.JSONDecodeError, TypeError) as exc:
        log.warning("[CullingProvider] failed to parse LLM response as JSON: %s — raw=%r", exc, raw)
        return BurstRankingResult(rankings=[], grades={}, caption="")

    rankings: list[RankedFrame] = []
    seen_ranks: set[int] = set()
    for item in payload.get("rankings", []):
        try:
            file = str(item["file"])
            rank = int(item["rank"])
            reason = str(item.get("reason", "")).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if file not in valid_files or not (1 <= rank <= top_n) or rank in seen_ranks:
            continue
        seen_ranks.add(rank)
        rankings.append(RankedFrame(file=file, rank=rank, reason=reason))
    rankings.sort(key=lambda r: r.rank)

    grades: dict[str, float] = {}
    for item in payload.get("grades", []):
        try:
            file = str(item["file"])
            grade = float(item["grade"])
        except (KeyError, TypeError, ValueError):
            continue
        if file not in valid_files:
            continue
        grades[file] = min(max(grade, 0.0), 1.0)

    caption = str(payload.get("caption", "")).strip()

    return BurstRankingResult(rankings=rankings, grades=grades, caption=caption)


def _parse_grade_response(raw: str | None, frames: list[BurstFrameInput]) -> dict[str, float]:
    """Shared, provider-agnostic parsing of the ``{"grades": [...]}`` JSON
    contract used for standalone (non-burst) image grading -- see
    ``prompts.build_single_image_grade_instructions``. Silently drops any
    grade entry that doesn't reference a real file in *frames*; an
    out-of-range grade is clamped into [0.0, 1.0] rather than dropped. Never
    raises -- a totally unparseable response yields an empty dict."""
    valid_files = {f.file for f in frames}
    try:
        payload = json.loads(raw or "")
    except (json.JSONDecodeError, TypeError) as exc:
        log.warning("[CullingProvider] failed to parse LLM grade response as JSON: %s — raw=%r", exc, raw)
        return {}

    grades: dict[str, float] = {}
    for item in payload.get("grades", []):
        try:
            file = str(item["file"])
            grade = float(item["grade"])
        except (KeyError, TypeError, ValueError):
            continue
        if file not in valid_files:
            continue
        grades[file] = min(max(grade, 0.0), 1.0)
    return grades


# USD price per 1,000,000 tokens: (input, output). Used only to *estimate*
# actual_cost alongside the real token counts returned by the API — verify
# against https://openai.com/api/pricing/ when adding new models.
_MODEL_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":  (0.15, 0.60),
    "gpt-4o":       (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1":      (2.00, 8.00),
    "o4-mini":      (1.10, 4.40),
}
_DEFAULT_PRICING_PER_MILLION = _MODEL_PRICING_PER_MILLION["gpt-4o-mini"]

# Default OpenAI model used when a caller doesn't specify one — shared by
# OpenAIProvider's constructor default and RunLLMCulling.py's CLI default so
# the two can never drift apart.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def known_model_pricing() -> dict[str, tuple[float, float]]:
    """Public, read-only view of the static (model -> (price_in, price_out)
    USD-per-1M-tokens) pricing table. OpenAI has no pricing API, so this
    hand-maintained table is the only source — callers (e.g. RunLLMCulling.py)
    should treat it as informational and verify against
    https://openai.com/api/pricing/ if it looks stale."""
    return dict(_MODEL_PRICING_PER_MILLION)


def _lookup_pricing(model: str) -> tuple[float, float]:
    """(price_per_million_input, price_per_million_output) for *model*, with
    a startswith fallback for dated model suffixes (e.g. "gpt-4o-mini-2024-
    07-18") and a conservative default (with a warning) for unknown models."""
    if model in _MODEL_PRICING_PER_MILLION:
        return _MODEL_PRICING_PER_MILLION[model]
    for prefix, pricing in _MODEL_PRICING_PER_MILLION.items():
        if model.startswith(prefix):
            return pricing
    log.warning(
        "[OpenAIProvider] no known pricing for model %r — using gpt-4o-mini "
        "pricing as an estimate for cost reporting", model,
    )
    return _DEFAULT_PRICING_PER_MILLION


# --- Rough pre-flight cost estimate (NOT authoritative billing) -----------
#
# OpenAI does not expose a pricing or token-counting API, so there is no way
# to "query" an exact cost ahead of time — these helpers only approximate it,
# for a ballpark figure shown to the user *before* a real run (see
# RunLLMCulling.py). The real, authoritative cost is always the token usage
# reported back on each response (see SessionCost / CostSummary / OpenAIProvider
# ._record_usage above).
#
# IMPORTANT: different model families use *different* image-tokenization
# formulas (verified against https://developers.openai.com/api/docs/guides/
# images-vision — "Calculating costs"), and the per-image token counts differ
# by over 30x between them. Using the wrong formula/constants for a model
# silently produces wildly wrong cost estimates (confirmed via a real test
# run: gpt-4o-mini actually costs ~33x more per image than the gpt-4o
# formula predicts).
#
# 1. Tile-based models (GPT-4o, GPT-4.1, GPT-4o-mini, computer-use-preview,
#    o1/o1-pro/o3): scale to fit within 2048x2048, then scale so the short
#    side is 768px, then tile in 512x512 squares —
#    tokens = base_tokens + tokens_per_tile * tiles.
_VISION_TILE_PX = 512
_TILE_BASED_TOKEN_PARAMS: dict[str, tuple[int, int]] = {
    # model prefix -> (base_tokens, tokens_per_tile)
    "gpt-4o-mini":           (2833, 5667),
    "gpt-4o":                (85, 170),
    "gpt-4.1":               (85, 170),
    "gpt-4.5":               (85, 170),
    "o1":                    (75, 150),
    "o3":                    (75, 150),
    "computer-use-preview":  (65, 129),
}
# 2. Patch-based models (gpt-4.1-mini/-nano, o4-mini, gpt-5-mini/-nano
#    family): cover the image in 32x32px patches, shrink proportionally if
#    over the model's patch budget, then multiply by a per-model factor.
_PATCH_PX = 32
_PATCH_BUDGET = 1536
_PATCH_BASED_TOKEN_MULTIPLIERS: dict[str, float] = {
    # model prefix -> multiplier
    "gpt-4.1-mini": 1.62,
    "gpt-4.1-nano": 2.46,
    "o4-mini":      1.72,
}
# Typical DSLR aspect ratio (3:2), used when the caller has no real image to
# measure (e.g. estimating cost before any images have even been listed).
_ASSUMED_ASPECT_RATIO = 3 / 2
# Rough, fixed allowances for the non-image parts of a burst-ranking call —
# small relative to image tokens, not worth modelling more precisely.
_ESTIMATED_PROMPT_TEXT_TOKENS = 250
_ESTIMATED_OUTPUT_TOKENS_PER_BURST = 150


def _lookup_prefixed(table: dict, model: str):
    """Exact match first, then longest-matching-prefix (for dated model
    snapshots like "gpt-4o-mini-2024-07-18"). Returns None if nothing matches."""
    if model in table:
        return table[model]
    best_prefix, best_value = None, None
    for prefix, value in table.items():
        if model.startswith(prefix) and (best_prefix is None or len(prefix) > len(best_prefix)):
            best_prefix, best_value = prefix, value
    return best_value


def _tiles_for_dims(width_px: float, height_px: float) -> tuple[int, int]:
    """Apply OpenAI's "high detail" scaling rule (fit within 2048x2048, then
    scale so the short side is 768px) and return the (tiles_w, tiles_h) count
    of 512x512 tiles needed to cover the result."""
    if max(width_px, height_px) > 2048:
        scale = 2048 / max(width_px, height_px)
        width_px, height_px = width_px * scale, height_px * scale
    short_px = min(width_px, height_px)
    if short_px > 768:
        scale = 768 / short_px
        width_px, height_px = width_px * scale, height_px * scale
    return math.ceil(width_px / _VISION_TILE_PX), math.ceil(height_px / _VISION_TILE_PX)


def _patch_based_tokens(width_px: float, height_px: float, multiplier: float) -> int:
    """OpenAI's patch-based image tokenization (gpt-4.1-mini/-nano, o4-mini):
    cover the image in 32x32px patches, shrink proportionally if over the
    1536-patch budget, then multiply by the model's per-tile multiplier."""
    original_patches = math.ceil(width_px / _PATCH_PX) * math.ceil(height_px / _PATCH_PX)
    if original_patches <= _PATCH_BUDGET:
        resized_patches = original_patches
    else:
        shrink = math.sqrt((_PATCH_PX ** 2 * _PATCH_BUDGET) / (width_px * height_px))
        shrunk_w, shrunk_h = width_px * shrink, height_px * shrink
        adjusted_shrink = shrink * min(
            math.floor(shrunk_w / _PATCH_PX) / (shrunk_w / _PATCH_PX),
            math.floor(shrunk_h / _PATCH_PX) / (shrunk_h / _PATCH_PX),
        )
        resized_w, resized_h = width_px * adjusted_shrink, height_px * adjusted_shrink
        resized_patches = math.ceil(resized_w / _PATCH_PX) * math.ceil(resized_h / _PATCH_PX)
    return math.ceil(resized_patches * multiplier)


def estimate_tokens_for_dims(width_px: float, height_px: float, model: str = DEFAULT_OPENAI_MODEL) -> int:
    """Vision token count for an image of the given exact pixel dimensions,
    using the correct formula/constants for *model* (tile-based or
    patch-based — see module note above). Use this over
    :func:`estimate_image_tokens` whenever real dimensions are available
    (e.g. :func:`estimate_tokens_for_b64_image`) — it's exact, not an
    aspect-ratio guess."""
    multiplier = _lookup_prefixed(_PATCH_BASED_TOKEN_MULTIPLIERS, model)
    if multiplier is not None:
        return _patch_based_tokens(width_px, height_px, multiplier)
    params = _lookup_prefixed(_TILE_BASED_TOKEN_PARAMS, model)
    if params is None:
        log.warning(
            "[OpenAIProvider] no known vision-token formula for model %r — "
            "using gpt-4o-mini's tiling constants as an estimate", model,
        )
        params = _TILE_BASED_TOKEN_PARAMS["gpt-4o-mini"]
    base_tokens, tokens_per_tile = params
    tiles_w, tiles_h = _tiles_for_dims(width_px, height_px)
    return base_tokens + tokens_per_tile * tiles_w * tiles_h


def estimate_tokens_for_b64_image(image_b64: str, model: str = DEFAULT_OPENAI_MODEL) -> int:
    """Vision token count for a base64-encoded image, decoding its real
    dimensions first (via Pillow's header-only parse, cheap — does not
    decode pixel data) for an exact estimate. Falls back to
    :func:`estimate_image_tokens`'s assumed-aspect-ratio guess if the image
    can't be parsed for any reason (never raises)."""
    try:
        from PIL import Image
        with Image.open(io.BytesIO(base64.b64decode(image_b64))) as img:
            width_px, height_px = img.size
        return estimate_tokens_for_dims(width_px, height_px, model)
    except Exception as exc:  # noqa: BLE001 — this must never abort a real ranking call
        log.warning("Could not decode image dimensions for a precise token estimate: %s", exc)
        return estimate_image_tokens(2048, model)  # conservative fallback


def estimate_image_tokens(long_edge_px: int, model: str = DEFAULT_OPENAI_MODEL, aspect_ratio: float = _ASSUMED_ASPECT_RATIO) -> int:
    """Rough estimate of vision tokens for one image whose long edge is
    *long_edge_px*, assuming *aspect_ratio* (used when no real image is
    available yet — e.g. a pre-flight estimate before any files are read).
    Prefer :func:`estimate_tokens_for_dims` / :func:`estimate_tokens_for_b64_image`
    when real dimensions are known."""
    short_edge_px = long_edge_px / aspect_ratio
    return estimate_tokens_for_dims(long_edge_px, short_edge_px, model)


def estimate_burst_cost(num_images: int, model: str, image_long_edge_px: int) -> tuple[int, int, float]:
    """Rough (input_tokens, output_tokens, cost_usd) pre-flight estimate for
    ranking one burst of *num_images* frames with *model*. Not authoritative
    — see the module-level note above."""
    input_tokens = estimate_image_tokens(image_long_edge_px, model) * num_images + _ESTIMATED_PROMPT_TEXT_TOKENS
    output_tokens = _ESTIMATED_OUTPUT_TOKENS_PER_BURST
    price_in, price_out = _lookup_pricing(model)
    cost_usd = (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out
    return input_tokens, output_tokens, cost_usd


# Conservative default tokens-per-minute budget for OpenAIProvider's rate
# limiter — matches the TPM cap OpenAI reports for gpt-4o-mini on a fresh/
# lower-usage-tier account (verified against a real 429 response: "Limit
# 200000 ... tokens per min (TPM)"). Override via OpenAIProvider's
# tokens_per_minute constructor arg (or RunLLMCulling.py's --tpm-limit) if
# your account has a higher tier. OpenAI does not expose account rate limits
# via any API, so this can't be auto-detected.
DEFAULT_TPM_LIMIT = 200_000

# Fraction of tokens_per_minute the rate limiter actually targets, leaving
# headroom below the real budget for token-estimation error and concurrent-
# request timing jitter (see TokenBucketRateLimiter). 0.9 = throttle at 90%
# of the configured TPM limit.
DEFAULT_TPM_SAFETY_FACTOR = 0.9


class TokenBucketRateLimiter:
    """Thread-safe token-bucket approximating an OpenAI tokens-per-minute
    (TPM) budget. The bucket starts full (the account's whole per-minute
    allowance is assumed available immediately) and continuously refills at
    ``tokens_per_minute / 60`` tokens/sec, mirroring OpenAI's rolling-window
    rate limiting (not a fixed calendar-minute reset).

    :meth:`acquire` blocks the calling thread until enough budget is
    available, so concurrent worker threads (see OpenAIProvider.rank_bursts's
    ThreadPoolExecutor) naturally trickle their requests out instead of all
    firing at once and instantly 429'ing.
    """

    def __init__(self, tokens_per_minute: int, safety_factor: float = DEFAULT_TPM_SAFETY_FACTOR) -> None:
        # safety_factor leaves headroom below the account's real TPM budget —
        # our per-request token estimates are approximate (not the exact
        # value OpenAI will bill), and concurrent threads can still overlap
        # slightly within the same instant, so targeting 100% of the real
        # limit leaves no margin for either. Throttling to e.g. 90% trades a
        # bit of throughput for far fewer 429s/retries in practice.
        safety_factor = min(max(safety_factor, 0.01), 1.0)
        self.capacity = max(1, int(tokens_per_minute * safety_factor))
        self._rate_per_sec = self.capacity / 60.0
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: int) -> None:
        """Block until *tokens* worth of budget is available, then consume
        it. A single request larger than the whole bucket capacity is capped
        to the capacity so it can still eventually proceed (after a full
        refill) rather than waiting forever."""
        tokens = min(max(0, tokens), self.capacity)
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self.capacity, self._tokens + elapsed * self._rate_per_sec)
                self._last_refill = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait_seconds = (tokens - self._tokens) / self._rate_per_sec
            time.sleep(min(wait_seconds, 5.0))  # re-check periodically rather than one huge sleep


class OpenAIProvider(CullingProvider):
    """OpenAI (chat completions + vision) implementation of CullingProvider.

    ``rank_bursts`` fans independent bursts out across a thread pool instead
    of processing them one at a time — each burst is a fully independent LLM
    call, and the OpenAI SDK releases the GIL while waiting on the network,
    so it's safe (and much faster for a large shoot) to have up to
    ``max_concurrency`` sessions in flight at once. To avoid instantly
    blowing through the account's real rate limit when many bursts fire at
    once, a :class:`TokenBucketRateLimiter` (sized by ``tokens_per_minute``)
    throttles how fast requests actually go out — see :meth:`rank_burst`.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_OPENAI_MODEL,
        timeout: float = 180.0,
        max_concurrency: int = 50,
        tokens_per_minute: int = DEFAULT_TPM_LIMIT,
        tpm_safety_factor: float = DEFAULT_TPM_SAFETY_FACTOR,
    ) -> None:
        super().__init__()
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAIProvider — pip install openai"
            ) from exc
        self._client = openai.OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.max_concurrency = max_concurrency
        self._pricing_per_million = _lookup_pricing(model)
        self._rate_limiter = (
            TokenBucketRateLimiter(tokens_per_minute, tpm_safety_factor) if tokens_per_minute > 0 else None
        )

    def rank_bursts(self, bursts: list[list[BurstFrameInput]]) -> list[BurstRankingResult]:
        if not bursts:
            return []
        max_workers = max(1, min(self.max_concurrency, len(bursts)))
        results: list[BurstRankingResult] = [BurstRankingResult(rankings=[], grades={}, caption="") for _ in bursts]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self.rank_burst, burst): i for i, burst in enumerate(bursts)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # noqa: BLE001 — one burst's failure must not sink the whole batch
                    log.warning("[OpenAIProvider] burst #%d failed: %s", idx, exc)
        return results

    def rank_burst(self, frames: list[BurstFrameInput]) -> BurstRankingResult:
        if not frames:
            return BurstRankingResult(rankings=[], grades={}, caption="")
        top_n = min(3, len(frames))
        messages = [
            {"role": "system", "content": prompts.CULLING_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_content(frames, top_n)},
        ]
        if self._rate_limiter is not None:
            estimated_tokens = sum(estimate_tokens_for_b64_image(f.image_b64, self.model) for f in frames) \
                + _ESTIMATED_PROMPT_TEXT_TOKENS
            self._rate_limiter.acquire(estimated_tokens)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = response.choices[0].message.content
            self._record_usage(response)
        except Exception as exc:  # noqa: BLE001 — a single burst's LLM failure must not abort the run
            log.warning("[OpenAIProvider] LLM call failed: %s", exc)
            return BurstRankingResult(rankings=[], grades={}, caption="")
        return _parse_llm_response(raw, frames, top_n)

    def grade_image_batches(self, batches: list[list[BurstFrameInput]]) -> list[dict[str, float]]:
        if not batches:
            return []
        max_workers = max(1, min(self.max_concurrency, len(batches)))
        results: list[dict[str, float]] = [{} for _ in batches]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self.grade_images, batch): i for i, batch in enumerate(batches)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # noqa: BLE001 — one batch's failure must not sink the whole run
                    log.warning("[OpenAIProvider] standalone grade batch #%d failed: %s", idx, exc)
        return results

    def grade_images(self, frames: list[BurstFrameInput]) -> dict[str, float]:
        if not frames:
            return {}
        messages = [
            {"role": "system", "content": prompts.SINGLE_IMAGE_GRADE_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_grade_user_content(frames)},
        ]
        if self._rate_limiter is not None:
            estimated_tokens = sum(estimate_tokens_for_b64_image(f.image_b64, self.model) for f in frames) \
                + _ESTIMATED_PROMPT_TEXT_TOKENS
            self._rate_limiter.acquire(estimated_tokens)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = response.choices[0].message.content
            self._record_usage(response)
        except Exception as exc:  # noqa: BLE001 — a single batch's LLM failure must not abort the run
            log.warning("[OpenAIProvider] standalone grade LLM call failed: %s", exc)
            return {}
        return _parse_grade_response(raw, frames)

    def _record_usage(self, response) -> None:
        """Extract token usage from *response* and record its cost. Thread-
        safe — called from worker threads when invoked via ``rank_bursts``."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        price_in, price_out = self._pricing_per_million
        cost_usd = (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out
        self._record_cost(SessionCost(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd))

    @staticmethod
    def _build_user_content(frames: list[BurstFrameInput], top_n: int) -> list[dict]:
        content: list[dict] = [
            {"type": "text", "text": prompts.build_user_instructions(frames, top_n)},
        ]
        for i, frame in enumerate(frames):
            content.append({"type": "text", "text": f"Image {i + 1} ({frame.file}):"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame.image_b64}"},
            })
        return content

    @staticmethod
    def _build_grade_user_content(frames: list[BurstFrameInput]) -> list[dict]:
        content: list[dict] = [
            {"type": "text", "text": prompts.build_single_image_grade_instructions(frames)},
        ]
        for i, frame in enumerate(frames):
            content.append({"type": "text", "text": f"Image {i + 1} ({frame.file}):"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame.image_b64}"},
            })
        return content
