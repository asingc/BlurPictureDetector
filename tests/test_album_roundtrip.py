"""The regression guard for the typed Album layer.

The refactor's central bet is that `AlbumImage`/`PersonRecord` are a typed
VIEW over the entry dict, not a replacement for it. If that ever stops being
true, a load/save cycle starts quietly deleting fields this code doesn't
model -- and albums on disk carry fields from older schema versions. These
tests fail the moment that happens.
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import make_body_dict, make_entry, write_album

from algo.album import Album, PersonRecord


def test_load_save_roundtrip_is_byte_identical(album_dir: Path):
    album_json = write_album(album_dir, [make_entry(), make_entry("IMG_0002.jpg")])
    before = album_json.read_text(encoding="utf-8")

    album = Album(album_dir)
    assert len(album.image_list) == 2
    album.save()

    assert album_json.read_text(encoding="utf-8") == before


def test_unknown_keys_survive_a_mutating_save(album_dir: Path):
    """Fields no property models -- legacy, transient, or from a future
    version -- must not be collateral damage of an unrelated edit."""
    entry = make_entry()
    entry["legacy_reason"] = "No person detected"
    entry["persons_detail"] = [{"anything": 1}]
    entry["annotation_data"]["some_future_field"] = {"nested": True}
    entry["annotation_data"]["evaluated"][0]["qualified_for_sharpness"] = True
    entry["annotation_data"]["evaluated"][0]["unheard_of"] = [1, 2, 3]
    album_json = write_album(album_dir, [entry])

    album = Album(album_dir)
    image = album.image("IMG_0001.jpg")
    image.stars = 5
    album.save()

    saved = json.loads(album_json.read_text(encoding="utf-8"))["results"][0]
    assert saved["legacy_reason"] == "No person detected"
    assert saved["persons_detail"] == [{"anything": 1}]
    assert saved["annotation_data"]["some_future_field"] == {"nested": True}
    body = saved["annotation_data"]["evaluated"][0]
    assert body["qualified_for_sharpness"] is True
    assert body["unheard_of"] == [1, 2, 3]
    assert saved["stars"] == 5


def test_mutations_reach_the_payload_through_the_typed_layer(album_dir: Path):
    album_json = write_album(album_dir, [make_entry()])
    album = Album(album_dir)
    image = album.image("IMG_0001.jpg")

    image.status = "blurry"
    image.stars = 2
    image.stars_manual = True
    image.keep = False
    image.llm_grade = 0.42
    image.burst_ranking = {"rank": 1, "reason": "best", "group_id": 3}
    image.burst_caption = "The tackle"
    image.overall_blurry = True
    image.set_sharpness_score(0.1234567)
    image.bodies[0].cloth_color = "Red:Crimson"
    album.save()

    saved = json.loads(album_json.read_text(encoding="utf-8"))["results"][0]
    assert saved["status"] == "blurry"
    assert saved["stars"] == 2
    assert saved["stars_manual"] is True
    assert saved["keep"] is False
    assert saved["llm_grade"] == 0.42
    assert saved["burst_ranking"]["rank"] == 1
    assert saved["burst_caption"] == "The tackle"
    assert saved["annotation_data"]["overall_blurry"] is True
    assert saved["sharpness_score"] == 0.1235
    assert saved["sharpness_grade"] == 12.3
    assert saved["annotation_data"]["evaluated"][0]["cloth_color"] == "Red:Crimson"


def test_bodies_are_live_views_not_copies(album_dir: Path):
    write_album(album_dir, [make_entry()])
    album = Album(album_dir)
    image = album.image("IMG_0001.jpg")

    image.bodies[0].rejection_reason = "jersey colour"
    assert image.bodies[0].rejection_reason == "jersey colour"
    assert image.entry["annotation_data"]["evaluated"][0]["rejection_reason"] == "jersey colour"


def test_set_bodies_replaces_the_whole_set(album_dir: Path):
    write_album(album_dir, [make_entry()])
    album = Album(album_dir)
    image = album.image("IMG_0001.jpg")

    replacement = PersonRecord(make_body_dict(cloth_color="Red:Crimson"))
    image.set_bodies([replacement])

    assert len(image.bodies) == 1
    assert image.bodies[0].cloth_color == "Red:Crimson"


def test_typed_reads_match_the_wire_values(album_dir: Path):
    write_album(album_dir, [make_entry()])
    image = Album(album_dir).image("IMG_0001.jpg")

    assert image.key == "IMG_0001.jpg"
    assert image.status == "sharp"
    assert image.is_sharp and not image.is_blurry
    assert image.sharpness_score == 0.62
    assert image.sharpness_grade == 62.0
    assert image.laplacian_variance == 123.45
    assert image.tenengrad_score == 67.89
    assert image.auto_adjustment.ev == 0.5
    assert image.processing_shape == (1080, 1920)
    assert image.has_annotation
    assert image.overall_blurry is False
    assert image.preview_stem == "IMG_0001"
    assert image.original_path == Path("C:/photos/IMG_0001.jpg")

    body = image.bodies[0]
    assert body.body_bbox.x1 == 0.1
    assert len(body.body_keypoints) == 2
    assert body.body_keypoints[1].passed is False
    assert body.face.confidence == 0.77
    assert body.face_confidence == 0.77
    assert body.narrow_face_bbox.x2 == 0.17
    assert body.sharpness_score == 0.62
    assert body.passed is True and body.is_blurry is False
    assert body.cloth_color == "Blue:Royal"
    assert body.cloth_color_detail["mean_lab"] == [40.0, 8.0, -30.0]


def test_missing_optional_fields_read_as_none_not_zero(album_dir: Path):
    """A photo the pipeline skipped has no scores at all. Reporting 0.0
    would make it look like a measured-as-terrible photo."""
    write_album(album_dir, [make_entry(
        "IMG_0003.jpg", status="skipped", annotation_data=None,
        sharpness_score=None, sharpness_grade=None,
        laplacian_variance=None, tenengrad_score=None,
        auto_adjustment=None,
    )])
    image = Album(album_dir).image("IMG_0003.jpg")

    assert image.sharpness_score is None
    assert image.stars is None
    assert image.llm_grade is None
    assert image.auto_adjustment is None
    assert image.bodies == []
    assert image.has_annotation is False
    assert image.best_body() is None


def test_album_source_paths_is_the_current_contents(album_dir: Path):
    write_album(album_dir, [make_entry(), make_entry("IMG_0002.jpg")])
    paths = Album(album_dir).source_paths

    assert Path("C:/photos/IMG_0001.jpg").resolve() in paths
    assert Path("C:/photos/IMG_0002.jpg").resolve() in paths
    assert len(paths) == 2
