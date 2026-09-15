"""Shared fixtures for the Album/AlbumImage/PersonRecord tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def make_body_dict(**overrides) -> dict:
    """One ``annotation_data.evaluated[]`` entry in the shape the import
    serializer writes, key order included."""
    body = {
        "body_bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4},
        "body_keypoints": [
            {"x": 0.15, "y": 0.25, "conf": 0.9, "passed": True},
            {"x": 0.16, "y": 0.26, "conf": 0.4, "passed": False},
        ],
        "face_bbox": {"x1": 0.12, "y1": 0.21, "x2": 0.18, "y2": 0.27},
        "narrow_face_bbox": {"x1": 0.13, "y1": 0.22, "x2": 0.17, "y2": 0.26},
        "face_kps": {
            "bbox": {"x1": 0.12, "y1": 0.21, "x2": 0.18, "y2": 0.27},
            "confidence": 0.77,
            "landmarks": [{"x": 0.14, "y": 0.23, "conf": 0.8, "passed": True}],
            "passed": True,
        },
        "sharpness_score": 0.62,
        "lap_var": 123.45,
        "ten": 67.89,
        "is_blurry": False,
        "rejection_reason": "",
        "cloth_color": "Blue:Royal",
        "cloth_color_detail": {"votes": {"Blue:Royal": 400}, "mean_lab": [40.0, 8.0, -30.0]},
    }
    body.update(overrides)
    return body


def make_entry(key: str = "IMG_0001.jpg", **overrides) -> dict:
    """One ``results[]`` entry in the shape the import serializer writes."""
    entry = {
        "file": f"C:/photos/{key}",
        "key": key,
        "status": "sharp",
        "sharpness_score": 0.62,
        "sharpness_grade": 62.0,
        "laplacian_variance": 123.45,
        "tenengrad_score": 67.89,
        "auto_adjustment": {"ev": 0.5},
        "preview_path": f"previews/{Path(key).stem}.jpg",
        "annotation_data": {
            "processing_shape": [1080, 1920],
            "overall_blurry": False,
            "evaluated": [make_body_dict()],
        },
    }
    entry.update(overrides)
    return entry


def write_album(album_dir: Path, entries: list[dict], **payload_overrides) -> Path:
    payload = {
        "team_id": "team-1",
        "our_jersey_color": "Blue:Royal",
        "import_status": "complete",
        "run_settings": {"sensitivity": "0.35"},
        "results": entries,
    }
    payload.update(payload_overrides)
    album_dir.mkdir(parents=True, exist_ok=True)
    album_json = album_dir / "album.json"
    album_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return album_json


@pytest.fixture
def album_dir(tmp_path: Path) -> Path:
    return tmp_path / "20260914-000000-test album"
