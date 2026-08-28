"""Serialization of processed :class:`~algo.frame.Frame` objects into
album.json ``results`` entries.

Lives here rather than in ``1_prep_review.py`` so every caller that needs to
write album.json — the normal import path, the incremental "import more
images" merge, and the deep-regrade re-analysis (see ``--regrade-only``) —
shares one definition of the on-disk entry schema instead of each growing
its own drifting copy.
"""

from __future__ import annotations

import json

import numpy as np

from algo.frame import Frame
from algo.models import Box, Face, PredictedKeyPoint


def serial_box(b: Box) -> dict:
    return {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2}


def serial_keypoint(kp: PredictedKeyPoint) -> dict:
    return {"x": kp.point.x, "y": kp.point.y, "conf": kp.confidence, "passed": kp.passed}


def serial_face(f: Face) -> dict:
    return {
        "bbox":       serial_box(f.bbox),
        "confidence": f.confidence,
        "landmarks":  [serial_keypoint(lm) for lm in f.landmarks],
        "passed":     f.passed,
    }


class NumpyEncoder(json.JSONEncoder):
    """Encode numpy scalar types as their Python equivalents."""

    def default(self, obj: object) -> object:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def baseline_stars(entry: dict, status: str, threshold: float) -> int:
    """Star rating for a photo whose verdict just changed.

    Mirrors the floor of algo/stages/llm_culling.py::_assign_star_ratings:
    3 for sharp, and for blurry 2 if it's close to the keep line or 1 if
    clearly bad. The 4/5 tiers there come from llm_grade percentiles across
    the whole album, which only that stage can recompute -- a regrade drops
    a re-verdicted photo back to the baseline rather than guessing.
    """
    if status == "sharp":
        return 3
    score = entry.get("sharpness_score")
    return 1 if score is None or float(score) < min(threshold, 0.4) else 2


def build_result_entries(frames: list[Frame]) -> list[dict]:
    """Serialize *frames* into album.json ``results`` entries (scores,
    bboxes, keypoints)."""
    serializable = []
    for frame in frames:
        norm_img = frame.normalized_image
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
                    "processing_shape": list(norm_img.shape[:2]) if norm_img is not None else [0, 0],
                    "overall_blurry":   overall_blurry,
                    "evaluated": [
                        {
                            "body_bbox":          serial_box(body.bbox),
                            "body_keypoints":     [serial_keypoint(kp) for kp in body.keypoints],
                            "face_bbox":          serial_box(body.best_face.bbox) if body.best_face else None,
                            "narrow_face_bbox":   serial_box(body.best_narrow_box) if body.best_narrow_box else None,
                            "face_kps":           serial_face(body.best_face) if body.best_face else None,
                            "sharpness_score":    body.sharpness_score,
                            "lap_var":            body.lap_var,
                            "ten":                body.ten,
                            "is_blurry":          not body.passed,
                            "rejection_reason":   body.rejection_reason,
                            "cloth_color":        body.cloth_color,
                            "cloth_color_detail": body.cloth_color_detail,
                        }
                        for body in frame.bodies
                    ],
                },
            }
        serializable.append(entry)
    return serializable
