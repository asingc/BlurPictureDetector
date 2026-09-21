"""The import serializer must keep writing byte-identical album.json.

`build_result_entries` now builds each body through `PersonRecord.from_body`
instead of its own dict literal. The expected values below are transcribed
from the pre-refactor implementation, key order included -- album.json is
written with ``indent=2`` and read by humans, so a reordering is a real
(if cosmetic) regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from algo.album import PersonRecord
from algo.frame import Frame
from algo.models import AutoAdjustment, Body, Box, Face, Point, PredictedKeyPoint
from algo.results import baseline_stars, build_result_entries


def _body(passed: bool = True) -> Body:
    face = Face(
        bbox=Box(0.12, 0.21, 0.18, 0.27),
        confidence=0.77,
        landmarks=[PredictedKeyPoint(Point(0.14, 0.23), 0.8, True)],
        passed=True,
    )
    return Body(
        crop=np.zeros((4, 4, 3), dtype=np.uint8),
        bbox=Box(0.1, 0.2, 0.3, 0.4),
        faces=[face],
        keypoints=[PredictedKeyPoint(Point(0.15, 0.25), 0.9, True)],
        passed=passed,
        rejection_reason="" if passed else "sharpness score 0.1 <= threshold 0.35",
        sharpness_score=0.62,
        best_face=face,
        best_narrow_box=Box(0.13, 0.22, 0.17, 0.26),
        lap_var=123.45,
        ten=67.89,
        cloth_color="Blue:Royal",
        cloth_color_detail={"votes": {"Blue:Royal": 400}, "mean_lab": [40.0, 8.0, -30.0]},
    )


def test_entry_shape_and_key_order_are_unchanged():
    frame = Frame(
        path=Path("C:/photos/IMG_0001.jpg"),
        bodies=[_body()],
        img_w=1920,
        img_h=1080,
        auto_adjustment=AutoAdjustment(ev=0.5),
        output_key="IMG_0001.jpg",
    )

    entry = build_result_entries([frame])[0]

    assert list(entry) == [
        "file", "key", "status", "sharpness_score", "sharpness_grade",
        "laplacian_variance", "tenengrad_score", "auto_adjustment",
        "preview_path", "annotation_data",
    ]
    assert list(entry["annotation_data"]) == ["processing_shape", "overall_blurry", "evaluated"]
    assert list(entry["annotation_data"]["evaluated"][0]) == [
        "body_bbox", "body_keypoints", "face_bbox", "narrow_face_bbox", "face_kps",
        "sharpness_score", "lap_var", "ten", "is_blurry", "rejection_reason",
        "cloth_color", "cloth_color_detail",
    ]

    assert entry["status"] == "sharp"
    assert entry["sharpness_score"] == 0.62
    assert entry["sharpness_grade"] == 62.0
    assert entry["laplacian_variance"] == 123.45
    assert entry["tenengrad_score"] == 67.89
    assert entry["auto_adjustment"] == {"ev": 0.5}
    assert entry["preview_path"] == "previews/IMG_0001.jpg"
    assert entry["annotation_data"]["processing_shape"] == [1080, 1920]
    assert entry["annotation_data"]["overall_blurry"] is False

    body = entry["annotation_data"]["evaluated"][0]
    assert body["body_bbox"] == {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}
    assert body["body_keypoints"] == [{"x": 0.15, "y": 0.25, "conf": 0.9, "passed": True}]
    assert body["face_bbox"] == {"x1": 0.12, "y1": 0.21, "x2": 0.18, "y2": 0.27}
    assert body["narrow_face_bbox"] == {"x1": 0.13, "y1": 0.22, "x2": 0.17, "y2": 0.26}
    assert body["face_kps"] == {
        "bbox": {"x1": 0.12, "y1": 0.21, "x2": 0.18, "y2": 0.27},
        "confidence": 0.77,
        "landmarks": [{"x": 0.14, "y": 0.23, "conf": 0.8, "passed": True}],
        "passed": True,
    }
    assert body["is_blurry"] is False
    assert body["rejection_reason"] == ""
    assert body["cloth_color"] == "Blue:Royal"
    assert body["cloth_color_detail"]["mean_lab"] == [40.0, 8.0, -30.0]


def test_a_frame_with_no_bodies_is_skipped_with_no_annotation_data():
    frame = Frame(path=Path("C:/photos/IMG_0002.jpg"), bodies=[], output_key="IMG_0002.jpg")

    entry = build_result_entries([frame])[0]

    assert list(entry) == ["file", "key", "status", "auto_adjustment", "preview_path"]
    assert entry["status"] == "skipped"
    assert "annotation_data" not in entry


def test_all_bodies_failing_marks_the_photo_blurry():
    frame = Frame(
        path=Path("C:/photos/IMG_0003.jpg"),
        bodies=[_body(passed=False)],
        output_key="IMG_0003.jpg",
    )

    entry = build_result_entries([frame])[0]

    assert entry["status"] == "blurry"
    assert entry["annotation_data"]["overall_blurry"] is True
    assert entry["annotation_data"]["evaluated"][0]["is_blurry"] is True


def test_person_record_survives_a_body_roundtrip():
    """`to_body` feeds the annotation drawer and the cloth-colour predictor;
    anything it drops is something those two silently stop seeing."""
    record = PersonRecord.from_body(_body())
    restored = record.to_body()

    assert restored.bbox == Box(0.1, 0.2, 0.3, 0.4)
    assert restored.keypoints == [PredictedKeyPoint(Point(0.15, 0.25), 0.9, True)]
    assert restored.best_face.confidence == 0.77
    assert restored.best_face.landmarks[0].point.x == 0.14
    assert restored.best_narrow_box == Box(0.13, 0.22, 0.17, 0.26)
    assert restored.passed is True
    assert restored.rejection_reason == ""
    assert restored.sharpness_score == 0.62
    assert restored.lap_var == 123.45
    assert restored.ten == 67.89
    assert restored.cloth_color == "Blue:Royal"
    assert restored.cloth_color_detail["votes"] == {"Blue:Royal": 400}
    assert PersonRecord.from_body(restored).wire == record.wire


def test_person_record_wire_is_json_serializable_and_passed_through():
    """algo/facereco.py embeds `wire` verbatim into face.json, so it has to
    stay the album's own dict, not a rebuilt copy."""
    source = {"body_bbox": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}, "custom": 7}
    record = PersonRecord(source)

    assert record.wire is source
    json.dumps(record.wire)


def test_baseline_stars_matches_the_documented_rule():
    assert baseline_stars(0.01, "sharp", 0.35) == 3
    assert baseline_stars(0.5, "blurry", 0.35) == 2
    assert baseline_stars(0.1, "blurry", 0.35) == 1
    assert baseline_stars(None, "blurry", 0.35) == 1
    # The floor is capped at 0.4 however strict the album's setting is.
    assert baseline_stars(0.45, "blurry", 0.9) == 2
    assert baseline_stars(0.3, "blurry", 0.9) == 1
