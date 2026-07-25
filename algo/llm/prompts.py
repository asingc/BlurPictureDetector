"""Shared prompt text for LLM-assisted burst culling.

Kept independent of any specific provider implementation (see
``culling_provider.py``) so every backend — OpenAI today, others (Anthropic,
local models, ...) later — sends the model the exact same instructions.
Only the transport (how images/text get packaged into a request, and how the
reply gets parsed) is provider-specific.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algo.llm.culling_provider import BurstFrameInput

CULLING_SYSTEM_PROMPT = (
    "You are an expert sports photo editor helping cull a burst of nearly "
    "identical action photos taken back-to-back of the same subject. "
    "You will be shown several images from one continuous sequence, each "
    "labeled with a sharpness score (0.0-1.0, higher = sharper) and the "
    "normalised bounding box of the main subject's face (x1, y1, x2, y2, "
    "each a fraction 0-1 of the image width/height). Every image shown has "
    "already passed a sharpness pre-filter, so IGNORE the sharpness score "
    "entirely — it must NOT factor into your ranking or grading at all. "
    "Judge the images purely on composition/framing and the decisive "
    "moment: which frame best conveys genuine emotion, peak action, and a "
    "compelling, well-composed shot (e.g. clean framing, strong body "
    "language, eyes open, good posture/form, ball in frame, uncluttered "
    "background). In addition to picking your top choice(s), grade EVERY "
    "image shown (not just your top picks) on the same 0.0-1.0 scale, and "
    "write one short, punchy caption for the whole burst — catchy, "
    "attention-grabbing, occasionally cheeky or rage-bait-y in tone (like a "
    "highlight-reel social post), while staying tasteful and appropriate. "
    "Respond with JSON only — no prose, no markdown fences."
)


def build_frame_label(index: int, frame: "BurstFrameInput") -> str:
    """One line of context describing a single frame within the burst."""
    if frame.face_bbox is not None:
        x1, y1, x2, y2 = frame.face_bbox
        bbox_txt = f"[{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]"
    else:
        bbox_txt = "unknown"
    return (
        f"Image {index + 1} — file: {frame.file}, "
        f"sharpness_score: {frame.sharpness_score:.3f}, "
        f"face_bbox: {bbox_txt}"
    )


SINGLE_IMAGE_GRADE_SYSTEM_PROMPT = (
    "You are an expert sports photo editor grading individual action photos "
    "on their own merits. Each image shown is UNRELATED to the others -- they "
    "are not the same moment or burst, so do NOT compare or rank them against "
    "each other. Each is labeled with a sharpness score (0.0-1.0, higher = "
    "sharper) and the normalised bounding box of the main subject's face "
    "(x1, y1, x2, y2, each a fraction 0-1 of the image width/height). Every "
    "image shown has already passed a sharpness pre-filter, so IGNORE the "
    "sharpness score entirely -- it must NOT factor into your grading at "
    "all. Grade each image purely on composition/framing and the decisive "
    "moment: how well it conveys genuine emotion, peak action, and a "
    "compelling, well-composed shot (e.g. clean framing, strong body "
    "language, eyes open, good posture/form, ball in frame, uncluttered "
    "background). Respond with JSON only -- no prose, no markdown fences."
)


def build_single_image_grade_instructions(frames: list["BurstFrameInput"]) -> str:
    """User-turn instructions for grading a batch of unrelated, standalone
    images independently (no ranking, no shared caption since they aren't
    from the same burst/moment) -- provider-agnostic; providers attach the
    actual images alongside this text in whatever shape their API wants."""
    labels = "\n".join(build_frame_label(i, f) for i, f in enumerate(frames))
    files = ", ".join(f.file for f in frames)
    return (
        f"Below are {len(frames)} unrelated standalone images (NOT from the same "
        "burst/moment -- do not compare or rank them against each other), shown "
        "in this order:\n"
        f"{labels}\n\n"
        "Grade EVERY one of the images above from 0.0 (worst) to 1.0 (best) "
        "purely on composition and the decisive moment (emotion, peak action) "
        "-- disregard the sharpness_score field completely, it is not a "
        "grading criterion. Respond with a single JSON object in exactly "
        "this shape:\n"
        '{"grades": [{"file": "<filename>", "grade": 0.0}, ...]}\n'
        f"\"grades\" must include exactly one entry for EVERY one of the {len(frames)} "
        f"images. Each \"file\" value must be exactly one of: {files}."
    )


def build_user_instructions(frames: list["BurstFrameInput"], top_n: int) -> str:
    """The shared user-turn instructions, provider-agnostic. Providers attach
    the actual images alongside this text in whatever shape their API wants."""
    labels = "\n".join(build_frame_label(i, f) for i, f in enumerate(frames))
    files = ", ".join(f.file for f in frames)
    return (
        f"This sequence has {len(frames)} images from the same continuous burst, "
        "shown below in chronological order:\n"
        f"{labels}\n\n"
        f"Choose the best {top_n} shot(s), ranked best (1) to worst, based only on "
        "composition and the decisive moment (emotion, peak action) — disregard "
        "the sharpness_score field completely, it is not a ranking criterion. "
        "Also grade EVERY one of the images above (all "
        f"{len(frames)}, including your top picks) from 0.0 (worst) to 1.0 "
        "(best) using the same composition/decisive-moment criteria. Also "
        "write one short, catchy caption (a single sentence, punchy and "
        "sometimes cheeky/rage-bait-y, tasteful and appropriate) that "
        "captures this whole burst's moment. "
        "Respond with a single JSON object in exactly this shape:\n"
        '{"caption": "<short punchy caption for the whole burst>", '
        '"rankings": [{"file": "<filename>", "rank": 1, "reason": "<short reason>"}, ...], '
        '"grades": [{"file": "<filename>", "grade": 0.0}, ...]}\n'
        f"\"rankings\" must include exactly {top_n} entries with ranks 1..{top_n}, each "
        "with a short (one sentence) reason. \"grades\" must include exactly one entry "
        f"for EVERY one of the {len(frames)} images (not just the top {top_n}). Each "
        f"\"file\" value must be exactly one of: {files}."
    )

