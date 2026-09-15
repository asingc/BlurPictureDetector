"""Behaviour that the typed-layer migration had to preserve exactly.

Each test here pins down a rule that was previously expressed as raw dict
manipulation in a consumer, and is now expressed through `AlbumImage` /
`PersonRecord`. They exist to prove the translation didn't change meaning.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from conftest import make_body_dict, make_entry, write_album

from algo import album as album_module
from algo.album import Album
from algo.imagefiles import IMAGE_EXTENSIONS, collect_images, images_to_import


# --------------------------------------------------------------------------- #
# keep / stars
# --------------------------------------------------------------------------- #

def test_keep_is_independent_of_stars(album_dir: Path):
    """LLM culling sets keep from burst rank, not from the star rating --
    deriving one from the other would silently change its output."""
    write_album(album_dir, [make_entry()])
    image = Album(album_dir).image("IMG_0001.jpg")

    image.stars = 5
    image.keep = False
    assert image.keep is False

    image.stars = 1
    image.keep = True
    assert image.keep is True


def test_auto_rating_applies_the_stars_rule(album_dir: Path):
    write_album(album_dir, [make_entry()])
    image = Album(album_dir).image("IMG_0001.jpg")

    assert image.apply_auto_rating(3) is True
    assert image.stars == 3 and image.keep is True

    assert image.apply_auto_rating(2) is True
    assert image.stars == 2 and image.keep is False


def test_auto_rating_never_overwrites_a_hand_rating(album_dir: Path):
    write_album(album_dir, [make_entry(stars=5, stars_manual=True, keep=True)])
    image = Album(album_dir).image("IMG_0001.jpg")

    assert image.apply_auto_rating(1) is False
    assert image.stars == 5
    assert image.keep is True


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #

def test_status_is_an_unconstrained_passthrough(album_dir: Path):
    """Status keeps today's wire values and today's (re-)writability --
    regrade legitimately re-verdicts an already-graded photo."""
    album_json = write_album(album_dir, [make_entry()])
    album = Album(album_dir)
    image = album.image("IMG_0001.jpg")

    assert image.status == "sharp"
    image.status = "blurry"
    assert image.is_blurry and not image.is_sharp
    image.status = "sharp"
    album.save()

    assert json.loads(album_json.read_text(encoding="utf-8"))["results"][0]["status"] == "sharp"


def test_missing_status_reads_as_empty_not_an_error(album_dir: Path):
    entry = make_entry()
    del entry["status"]
    write_album(album_dir, [entry])

    assert Album(album_dir).image("IMG_0001.jpg").status == ""


# --------------------------------------------------------------------------- #
# best_body -- the rule the import serializer uses for the photo's score
# --------------------------------------------------------------------------- #

def test_best_body_prefers_a_passing_person(album_dir: Path):
    entry = make_entry()
    entry["annotation_data"]["evaluated"] = [
        make_body_dict(sharpness_score=0.9, is_blurry=True),
        make_body_dict(sharpness_score=0.5, is_blurry=False),
    ]
    write_album(album_dir, [entry])

    assert Album(album_dir).image("IMG_0001.jpg").best_body().sharpness_score == 0.5


def test_best_body_falls_back_to_the_sharpest_when_none_passed(album_dir: Path):
    """A fully-rejected photo still reports a real measured score rather
    than zero -- the fallback a naive "best passing body" rule would lose."""
    entry = make_entry()
    entry["annotation_data"]["evaluated"] = [
        make_body_dict(sharpness_score=0.3, is_blurry=True),
        make_body_dict(sharpness_score=0.7, is_blurry=True),
    ]
    write_album(album_dir, [entry])

    assert Album(album_dir).image("IMG_0001.jpg").best_body().sharpness_score == 0.7


# --------------------------------------------------------------------------- #
# cleared_grading_gates -- regrade eligibility
# --------------------------------------------------------------------------- #

def test_cleared_grading_gates_matches_the_documented_rule(album_dir: Path):
    entry = make_entry()
    entry["annotation_data"]["evaluated"] = [
        make_body_dict(is_blurry=False),
        make_body_dict(is_blurry=True, rejection_reason="sharpness score 0.1 <= threshold 0.35"),
        make_body_dict(is_blurry=True, rejection_reason="jersey colour mismatch"),
        make_body_dict(is_blurry=True, rejection_reason="no matched face"),
        make_body_dict(is_blurry=True, rejection_reason=""),
    ]
    write_album(album_dir, [entry])

    cleared = [b.cleared_grading_gates for b in Album(album_dir).image("IMG_0001.jpg").bodies]
    # Passing, sharpness-rejected and jersey-rejected bodies are revisitable;
    # a body that failed an earlier gate -- or a pre-rejection_reason album
    # that can't prove which gate it failed -- is left alone.
    assert cleared == [True, True, True, False, False]


# --------------------------------------------------------------------------- #
# face tagging
# --------------------------------------------------------------------------- #

def test_player_tagging_lives_on_the_person_not_the_photo(album_dir: Path):
    entry = make_entry()
    entry["annotation_data"]["evaluated"] = [make_body_dict(), make_body_dict()]
    album_json = write_album(album_dir, [entry])
    album = Album(album_dir)
    image = album.image("IMG_0001.jpg")

    assert image.bodies[0].player_name == ""
    image.bodies[0].player_name = "Sam"
    image.bodies[0].player_number = "7"
    album.save()

    saved = json.loads(album_json.read_text(encoding="utf-8"))["results"][0]
    assert saved["annotation_data"]["evaluated"][0]["player_name"] == "Sam"
    assert saved["annotation_data"]["evaluated"][0]["player_number"] == "7"
    assert "player_name" not in saved["annotation_data"]["evaluated"][1]
    assert "player_name" not in saved


# --------------------------------------------------------------------------- #
# album.json backups
# --------------------------------------------------------------------------- #

def test_backups_are_written_by_default(album_dir: Path):
    write_album(album_dir, [make_entry()])
    album = Album(album_dir)
    album.image("IMG_0001.jpg").stars = 4
    album.save()

    backups = list((album_dir / "backup").glob("album_*.gz"))
    assert len(backups) == 1
    with gzip.open(backups[0], "rt", encoding="utf-8") as fh:
        assert json.load(fh)["results"][0].get("stars") is None


def test_backups_can_be_turned_off_without_losing_atomicity(album_dir: Path, monkeypatch):
    album_json = write_album(album_dir, [make_entry()])
    monkeypatch.setattr(album_module, "album_json_backups_enabled", False)

    album = Album(album_dir)
    album.image("IMG_0001.jpg").stars = 4
    album.save()

    assert not (album_dir / "backup").exists()
    assert json.loads(album_json.read_text(encoding="utf-8"))["results"][0]["stars"] == 4
    # No temp file left stranded beside the album.
    assert not list(album_dir.glob("*.tmp"))


# --------------------------------------------------------------------------- #
# enumeration
# --------------------------------------------------------------------------- #

def test_collect_images_is_sorted_and_extension_filtered(tmp_path: Path):
    for name in ("b.jpg", "a.JPG", "c.cr3", "notes.txt", "clip.mp4"):
        (tmp_path / name).write_bytes(b"x")

    found = [p.name for p in collect_images(tmp_path)]

    assert found == ["a.JPG", "b.jpg", "c.cr3"]
    assert ".cr3" in IMAGE_EXTENSIONS


def test_images_to_import_skips_what_the_album_already_has(tmp_path: Path):
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(b"x")

    remaining = images_to_import(tmp_path, exclude={tmp_path / "b.jpg"})

    assert [p.name for p in remaining] == ["a.jpg", "c.jpg"]


def test_album_create_assigns_keys_up_front(tmp_path: Path):
    """Keys are needed to name previews, so they're assigned at creation --
    before a single pixel has been decoded."""
    src_a, src_b = tmp_path / "a", tmp_path / "b"
    for directory in (src_a, src_b):
        directory.mkdir()
        (directory / "IMG_0001.jpg").write_bytes(b"x")

    album = Album.create(
        tmp_path / "album", team_id="t1",
        paths=[src_a / "IMG_0001.jpg", src_b / "IMG_0001.jpg"],
    )

    keys = [image.key for image in album.image_list]
    # Two source directories can share a filename; the second is disambiguated.
    assert len(set(keys)) == 2
    assert keys[0] == "IMG_0001.jpg"
    assert (tmp_path / "album" / "album.json").is_file()
    assert album.team_id == "t1"
    assert album.import_status == "in_progress"


def test_album_create_with_no_paths_is_an_empty_album(tmp_path: Path):
    album = Album.create(tmp_path / "album", team_id="t1")

    assert album.image_list == []
    assert album.results == []
