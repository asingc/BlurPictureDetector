"""info.json's typed layer, and the album.json/info.json consistency it exists to keep.

The bug this class was built for: `Album.import_from` disambiguates a
bookkeeping key that collides with one already in the album, but info.json
indexes the same photos by that same key. Left un-renamed, two photos end up
sharing one `src`, and every consumer that builds a *set* of kept keys
(apply_export, the culling page) silently collapses them into one -- so the
renamed photo never exports.
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import make_entry, write_album

from algo.album import Album
from algo.album_info import CATEGORIES, AlbumInfo, category_for_status


def write_info(album_dir: Path, **overrides) -> Path:
    payload = {
        "SrcDir": "C:/photos",
        "SrcDirs": ["C:/photos"],
        "SrcType": "Directory",
        "Timestamp": "20260914-010203",
        "LastImportTimestamp": "20260914-010203",
        "OurJerseyColor": "Blue:Royal",
        "Anno_Blur": [],
        "Anno_Sharp": [{"src": "IMG_0001.jpg", "srcPath": "C:/photos/IMG_0001.jpg"}],
        "Anno_Skipped": [],
    }
    payload.update(overrides)
    album_dir.mkdir(parents=True, exist_ok=True)
    path = album_dir / "info.json"
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #

def test_typed_reads_match_the_wire_values(album_dir: Path):
    write_info(album_dir)
    info = AlbumInfo(album_dir)

    assert info.exists
    assert info.src_dir == "C:/photos"
    assert info.src_dirs == ["C:/photos"]
    assert info.src_type == "Directory"
    assert info.timestamp == "20260914-010203"
    assert info.last_import_timestamp == "20260914-010203"
    assert info.our_jersey_color == "Blue:Royal"
    assert info.count("sharp") == 1 and info.count("blur") == 0
    assert info.total == 1

    entry = info.entries("sharp")[0]
    assert entry.key == "IMG_0001.jpg"
    assert entry.source_path == Path("C:/photos/IMG_0001.jpg")


def test_a_missing_info_json_reads_as_empty(album_dir: Path):
    album_dir.mkdir(parents=True, exist_ok=True)
    info = AlbumInfo(album_dir)

    assert info.exists is False
    assert info.src_dir == "" and info.src_dirs == []
    assert info.total == 0
    assert info.all_entries == []
    assert info.category_of("anything") is None


def test_category_for_status_maps_the_two_vocabularies():
    assert category_for_status("blurry") == "blur"
    assert category_for_status("sharp") == "sharp"
    assert category_for_status("skipped") == "skipped"
    # A status with no bucket must not silently land in one.
    assert category_for_status("error") is None
    assert category_for_status("") is None


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #

def test_unknown_keys_survive_a_save(album_dir: Path):
    path = write_info(album_dir, SomeFutureField={"kept": True})
    info = AlbumInfo(album_dir)
    info.our_jersey_color = "Red:Crimson"
    info.save()

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["SomeFutureField"] == {"kept": True}
    assert saved["OurJerseyColor"] == "Red:Crimson"


def test_move_relocates_a_photo_and_keeps_its_source_path(album_dir: Path):
    write_info(album_dir)
    info = AlbumInfo(album_dir)

    assert info.move("IMG_0001.jpg", "blur") is True
    assert info.category_of("IMG_0001.jpg") == "blur"
    assert info.count("sharp") == 0
    assert info.entries("blur")[0].source_path == Path("C:/photos/IMG_0001.jpg")


def test_move_reports_an_unknown_key_instead_of_inventing_one(album_dir: Path):
    write_info(album_dir)
    info = AlbumInfo(album_dir)

    assert info.move("NOT_THERE.jpg", "blur") is False
    assert info.total == 1


def test_move_to_the_same_category_is_a_no_op(album_dir: Path):
    write_info(album_dir)
    info = AlbumInfo(album_dir)

    assert info.move("IMG_0001.jpg", "sharp") is True
    assert info.count("sharp") == 1


def test_record_import_seeds_once_and_appends_thereafter(album_dir: Path, tmp_path: Path):
    second = tmp_path / "more photos"
    second.mkdir()
    write_info(album_dir)
    info = AlbumInfo(album_dir)

    info.record_import(second, "20260915-111111")

    # SrcDir/Timestamp describe the FIRST import and must not move.
    assert info.src_dir == "C:/photos"
    assert info.timestamp == "20260914-010203"
    assert info.last_import_timestamp == "20260915-111111"
    assert str(second.resolve()) in info.src_dirs
    assert len(info.src_dirs) == 2

    # Re-importing the same directory doesn't duplicate it.
    info.record_import(second, "20260915-222222")
    assert len(info.src_dirs) == 2


def test_replace_entries_rebuilds_every_bucket(album_dir: Path):
    write_info(album_dir)
    info = AlbumInfo(album_dir)

    info.replace_entries({
        "blur": [("A.jpg", "C:/x/A.jpg")],
        "sharp": [],
        "skipped": [("B.jpg", "C:/x/B.jpg")],
    })

    assert info.count("sharp") == 0
    assert [e.key for e in info.entries("blur")] == ["A.jpg"]
    assert [e.key for e in info.entries("skipped")] == ["B.jpg"]


# --------------------------------------------------------------------------- #
# the import-more key-collision bug
# --------------------------------------------------------------------------- #

def test_rename_key_moves_only_the_entry_for_that_source_file(album_dir: Path, tmp_path: Path):
    """Two entries share the old key at rename time -- the photo already in
    the album and the one arriving. Only the arriving one may move."""
    existing = tmp_path / "a" / "IMG_0001.jpg"
    arriving = tmp_path / "b" / "IMG_0001.jpg"
    for path in (existing, arriving):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    write_info(album_dir, Anno_Sharp=[
        {"src": "IMG_0001.jpg", "srcPath": str(existing)},
        {"src": "IMG_0001.jpg", "srcPath": str(arriving)},
    ])
    info = AlbumInfo(album_dir)

    assert info.rename_key("IMG_0001.jpg", "IMG_0001__2.jpg", arriving) is True

    by_path = {str(e.source_path): e.key for e in info.entries("sharp")}
    assert by_path[str(existing)] == "IMG_0001.jpg"
    assert by_path[str(arriving)] == "IMG_0001__2.jpg"


def test_rename_key_reports_a_source_it_does_not_have(album_dir: Path, tmp_path: Path):
    write_info(album_dir)
    info = AlbumInfo(album_dir)

    assert info.rename_key("IMG_0001.jpg", "X.jpg", tmp_path / "elsewhere.jpg") is False
    assert info.entries("sharp")[0].key == "IMG_0001.jpg"


def test_import_more_keeps_info_json_in_step_with_the_renamed_key(tmp_path: Path):
    """The regression this whole class exists for: a colliding key renamed
    in album.json must be renamed in info.json too, or the photo vanishes
    from every consumer that keys off info.json."""
    src_a, src_b = tmp_path / "a", tmp_path / "b"
    for directory in (src_a, src_b):
        directory.mkdir()
        (directory / "IMG_0001.jpg").write_bytes(b"x")

    target = tmp_path / "album"
    write_album(target, [make_entry("IMG_0001.jpg", file=str(src_a / "IMG_0001.jpg"))])
    write_info(target, Anno_Sharp=[{"src": "IMG_0001.jpg", "srcPath": str(src_a / "IMG_0001.jpg")}])

    staging = tmp_path / "staging"
    write_album(staging, [make_entry("IMG_0001.jpg", file=str(src_b / "IMG_0001.jpg"))])
    # The incoming photo is appended to the target's info.json under its
    # pre-merge key, exactly as 1_prep_review.py writes it before merging.
    info = AlbumInfo(target)
    info.add("sharp", "IMG_0001.jpg", src_b / "IMG_0001.jpg")
    info.save()

    summary = Album(target).import_from(Album(staging))

    assert summary.added == 1
    assert summary.renamed == 1
    assert summary.info_entries_renamed == 1

    album_keys = sorted(image.key for image in Album(target).image_list)
    info_keys = sorted(entry.key for entry in AlbumInfo(target).all_entries)
    assert album_keys == ["IMG_0001.jpg", "IMG_0001__2.jpg"]
    # The two must agree, or a set-based consumer drops one of the photos.
    assert info_keys == album_keys

    # ...and each key must still point at the photo it actually describes.
    info_by_key = {e.key: e.source_path for e in AlbumInfo(target).all_entries}
    assert info_by_key["IMG_0001.jpg"] == src_a / "IMG_0001.jpg"
    assert info_by_key["IMG_0001__2.jpg"] == src_b / "IMG_0001.jpg"


def test_import_without_a_collision_leaves_info_json_alone(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "IMG_0009.jpg").write_bytes(b"x")

    target = tmp_path / "album"
    write_album(target, [make_entry("IMG_0001.jpg")])
    write_info(target)

    staging = tmp_path / "staging"
    write_album(staging, [make_entry("IMG_0009.jpg", file=str(src / "IMG_0009.jpg"))])

    summary = Album(target).import_from(Album(staging))

    assert summary.renamed == 0
    assert summary.info_entries_renamed == 0
    assert [e.key for e in AlbumInfo(target).entries("sharp")] == ["IMG_0001.jpg"]


def test_every_category_has_a_bucket():
    info = AlbumInfo(Path("nonexistent"))
    for category in CATEGORIES:
        assert info.count(category) == 0
