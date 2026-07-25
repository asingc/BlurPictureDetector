#!/usr/bin/env python3
"""
RunLLMCulling.py
Re-run LLM-assisted burst culling against an existing albums/<run> directory
without re-doing image analysis / face recognition.

Usage:
    python RunLLMCulling.py <output_dir> [--openaikey KEY] [--model MODEL]
        [--burst-gap-seconds N] [--min-group-size N] [--image-max-long-edge N]

    python RunLLMCulling.py
        (no arguments: prints usage plus known OpenAI model pricing and a
        rough per-burst cost estimate, then exits without processing anything)

<output_dir> must already contain a album.json (produced by
1_prep_review.py). Every qualifying burst of consecutive "sharp" frames
(grouped by capture-timestamp gap) is re-ranked from scratch by an
OpenAI-backed CullingProvider (see algo/llm/culling_provider.py) via the
existing algo/stages/llm_culling.py::LLMCullingStage. album.json is
overwritten in place: "keep" and "burst_ranking" on every frame in a
qualifying burst are fully replaced (not merged) with the new verdict, and
llm_cost_summary is updated with this run's token usage/cost.

Note on cost estimates: OpenAI does not expose a pricing or token-counting
API, so nothing here is queried "live" from OpenAI except the list of
currently available model IDs (client.models.list(), a free metadata call —
it does not consume any input/output tokens). Actual per-model USD pricing
comes from a hand-maintained table (algo/llm/culling_provider.py
known_model_pricing()) and per-burst cost is only a rough estimate based on
OpenAI's published vision-token tiling formula — treat it as a ballpark, not
a bill.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from algo.config import app_config
from algo.llm.culling_provider import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TPM_LIMIT,
    DEFAULT_TPM_SAFETY_FACTOR,
    OpenAIProvider,
    estimate_burst_cost,
    known_model_pricing,
)
from algo.stages.llm_culling import (
    DEFAULT_BURST_GAP_SECONDS,
    DEFAULT_IMAGE_MAX_LONG_EDGE,
    DEFAULT_MIN_GROUP_SIZE,
    LLMCullingStage,
    load_qualifying_bursts,
)

log = logging.getLogger("BlurPictureDetector")

# Burst sizes shown in the no-args pricing preview (no album.json to
# measure real bursts from yet).
_EXAMPLE_BURST_SIZES = (2, 3, 5, 10)


def _setup_console_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _list_available_models(api_key: str) -> list[str] | None:
    """Real, currently-available OpenAI model IDs via client.models.list() —
    a free metadata call (no input/output tokens, not billed like a
    completion). Returns None if the call fails (e.g. bad key, no network)."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        return sorted(m.id for m in client.models.list().data)
    except Exception as exc:  # noqa: BLE001 — this is a best-effort info lookup
        log.warning("Could not list available OpenAI models: %s", exc)
        return None


def _print_pricing_info(image_max_long_edge: int) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    available_models = _list_available_models(api_key) if api_key else None

    print()
    print("OpenAI pricing note:")
    print("  OpenAI does not expose a pricing or token-counting API, so exact costs")
    print("  can't be queried ahead of time. Prices below are a hand-maintained table")
    print("  (see algo/llm/culling_provider.py) -- verify against https://openai.com/api/pricing/")
    print("  if they look stale. Listing available models IS free (no tokens billed).")
    print()

    if api_key:
        if available_models is not None:
            print(f"Models currently available to your account ({len(available_models)}):")
            for m in available_models:
                print(f"  {m}")
        else:
            print("Could not fetch your account's available models (see warning above).")
    else:
        print("Set OPENAI_API_KEY (or pass --openaikey) to also see your account's available models.")
    print()

    print("Known model pricing (USD per 1,000,000 tokens) and a ROUGH per-burst cost")
    print(f"estimate (images downsized to {image_max_long_edge}px long edge, assumed 3:2 aspect ratio):")
    print(f"  {'model':<16}{'input $/1M':>12}{'output $/1M':>14}   " +
          "  ".join(f"{n}-img burst" for n in _EXAMPLE_BURST_SIZES))
    for model, (price_in, price_out) in sorted(known_model_pricing().items()):
        costs = []
        for n in _EXAMPLE_BURST_SIZES:
            _, _, cost = estimate_burst_cost(n, model, image_max_long_edge)
            costs.append(f"${cost:.5f}".rjust(len(f"{n}-img burst") + 2))
        print(f"  {model:<16}{price_in:>12.2f}{price_out:>14.2f}   " + "  ".join(costs))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run LLM-assisted burst culling against an existing albums/<run> directory.",
    )
    parser.add_argument("output_dir", nargs="?", default=None,
                         help="Existing albums/<run> directory containing album.json.")
    parser.add_argument(
        "--openaikey",
        default=None,
        metavar="KEY",
        help="OpenAI API key (falls back to the OPENAI_API_KEY environment variable).",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help=(
            f"OpenAI model name to use for ranking (default: {DEFAULT_OPENAI_MODEL}). "
            "If set, you are responsible for providing a valid model name."
        ),
    )
    parser.add_argument(
        "--burst-gap-seconds",
        type=float,
        default=DEFAULT_BURST_GAP_SECONDS,
        help=f"Max seconds between consecutive sharp frames to be treated as the same burst (default: {DEFAULT_BURST_GAP_SECONDS}).",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=DEFAULT_MIN_GROUP_SIZE,
        help=f"Minimum burst size to be re-ranked; smaller bursts are left untouched (default: {DEFAULT_MIN_GROUP_SIZE}).",
    )
    parser.add_argument(
        "--image-max-long-edge",
        type=int,
        default=DEFAULT_IMAGE_MAX_LONG_EDGE,
        help=f"Long-edge size (px) of the downsized images sent to the LLM (default: {DEFAULT_IMAGE_MAX_LONG_EDGE}).",
    )
    parser.add_argument(
        "--tpm-limit",
        type=int,
        default=DEFAULT_TPM_LIMIT,
        metavar="N",
        help=(
            f"Your OpenAI account's tokens-per-minute budget for --model, used to "
            f"throttle concurrent requests so a large batch doesn't instantly trip "
            f"rate limits (default: {DEFAULT_TPM_LIMIT}, a conservative lower-tier "
            "estimate -- raise it if your account has a higher tier; OpenAI does "
            "not expose this via any API). Pass 0 to disable throttling."
        ),
    )
    parser.add_argument(
        "--tpm-safety-factor",
        type=float,
        default=DEFAULT_TPM_SAFETY_FACTOR,
        metavar="F",
        help=(
            f"Fraction of --tpm-limit the rate limiter actually targets "
            f"(default: {DEFAULT_TPM_SAFETY_FACTOR}, i.e. throttle at "
            f"{int(DEFAULT_TPM_SAFETY_FACTOR * 100)}% of the budget). Leaves headroom "
            "below the real limit for token-estimation error and concurrent-request "
            "timing jitter -- lower this if you still see 429s, raise it (up to 1.0) "
            "for more throughput once you're confident in the estimate."
        ),
    )
    args = parser.parse_args()

    _setup_console_logging()

    if args.output_dir is None:
        # No positional argument at all: show usage + pricing info instead of
        # erroring on a missing required argument, then exit cleanly.
        parser.print_help()
        _print_pricing_info(args.image_max_long_edge)
        sys.exit(0)

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_dir():
        log.error("Output directory does not exist: %s", output_dir)
        sys.exit(1)
    results_path = output_dir / "album.json"
    if not results_path.exists():
        log.error("No album.json found in %s -- run 1_prep_review.py first.", output_dir)
        sys.exit(1)

    api_key = args.openaikey or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("No OpenAI API key provided (use --openaikey or set OPENAI_API_KEY).")
        sys.exit(1)

    model = args.model or DEFAULT_OPENAI_MODEL
    available_models = _list_available_models(api_key)
    if available_models is not None and model not in available_models:
        log.warning(
            "Model %r was not found in your account's available models list -- "
            "proceeding anyway since you asked for it explicitly.", model,
        )

    bursts = load_qualifying_bursts(results_path, args.burst_gap_seconds, args.min_group_size)
    if bursts:
        total_input = total_output = 0
        total_cost = 0.0
        log.info("Estimated cost for %d qualifying burst(s) (rough, not authoritative):", len(bursts))
        for idx, burst in enumerate(bursts):
            input_tokens, output_tokens, cost = estimate_burst_cost(len(burst), model, args.image_max_long_edge)
            total_input += input_tokens
            total_output += output_tokens
            total_cost += cost
            log.info("  burst-%04d: %d frame(s) -> ~%d input / ~%d output tokens, ~$%.5f",
                      idx, len(burst), input_tokens, output_tokens, cost)
        log.info("  TOTAL: ~%d input / ~%d output tokens, ~$%.5f (model=%s)",
                 total_input, total_output, total_cost, model)
    else:
        log.info("No qualifying bursts found (>= %d frames) -- nothing to rank.", args.min_group_size)

    provider = OpenAIProvider(
        api_key=api_key,
        model=model,
        tokens_per_minute=args.tpm_limit,
        tpm_safety_factor=args.tpm_safety_factor,
    )

    stage = LLMCullingStage(
        output_dir,
        provider=provider,
        burst_gap_seconds=args.burst_gap_seconds,
        min_group_size=args.min_group_size,
        image_max_long_edge=args.image_max_long_edge,
    )
    stage.process([], app_config)

    log.info("Done. Review updated album.json at %s", output_dir / "album.json")


if __name__ == "__main__":
    main()

