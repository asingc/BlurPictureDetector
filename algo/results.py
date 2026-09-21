"""Serialization of processed :class:`~algo.frame.Frame` objects into the
`Album`'s ``results`` entries (see algo/album.py).

Lives here rather than in ``1_prep_review.py`` so every caller that needs to
build those entries — the normal import path, the incremental "import more
images" merge, and the deep-regrade re-analysis (see ``--regrade-only``) —
shares one definition of the on-disk entry schema instead of each growing
its own drifting copy.
"""

from __future__ import annotations

from algo.album import PersonRecord
from algo.frame import Frame
from algo.models import Box, Face, NumpyEncoder, PredictedKeyPoint

__all__ = ["NumpyEncoder", "baseline_stars", "build_result_entries",
           "serial_box", "serial_face", "serial_keypoint"]


def serial_box(b: Box) -> dict:
    return b.to_wire()


def serial_keypoint(kp: PredictedKeyPoint) -> dict:
    return kp.to_wire()


def serial_face(f: Face) -> dict:
    return f.to_wire()


def baseline_stars(sharpness_score: float | None, status: str, threshold: float) -> int:
    """Star rating for a photo whose verdict just changed.

    Mirrors the floor of algo/stages/llm_culling.py::_assign_star_ratings:
    3 for sharp, and for blurry 2 if it's close to the keep line or 1 if
    clearly bad. The 4/5 tiers there come from llm_grade percentiles across
    the whole album, which only that stage can recompute -- a regrade drops
    a re-verdicted photo back to the baseline rather than guessing.
    """
    if status == "sharp":
        return 3
    if sharpness_score is None:
        return 1
    return 1 if float(sharpness_score) < min(threshold, 0.4) else 2


def build_result_entries(frames: list[Frame]) -> list[dict]:
    """Serialize *frames* into `Album` ``results`` entries (scores,
    bboxes, keypoints)."""
    serializable = []
    for frame in frames:
        auto_adj = frame.auto_adjustment
        auto_adj_entry = {"ev": auto_adj.ev} if auto_adj is not None else None
        key = frame.output_key or frame.path.name
        if not frame.bodies:
            entry: dict = {
                "file": str(frame.path),
                "key": key,
                "status": "skipped",
                "auto_adjustment": auto_adj_entry,
                "preview_path": f"previews/{frame.key_stem}.jpg",
            }
        else:
            overall_blurry = not frame.is_sharp()
            passing = [b for b in frame.bodies if b.passed]
            best = max(passing or frame.bodies, key=lambda b: b.sharpness_score)
            entry = {
                "file":               str(frame.path),
                "key":                key,
                "status":             "blurry" if overall_blurry else "sharp",
                "sharpness_score":    round(best.sharpness_score, 4),
                "sharpness_grade":    round(best.sharpness_score * 100, 1),
                "laplacian_variance": round(best.lap_var, 2),
                "tenengrad_score":    round(best.ten, 2),
                "auto_adjustment":    auto_adj_entry,
                "preview_path":       f"previews/{frame.key_stem}.jpg",
                "annotation_data": {
                    "processing_shape": [frame.img_h, frame.img_w],
                    "overall_blurry":   overall_blurry,
                    "evaluated": [PersonRecord.from_body(body).wire for body in frame.bodies],
                },
            }
        serializable.append(entry)
    return serializable
