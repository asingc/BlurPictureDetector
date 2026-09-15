"""Cached, in-memory view of an album's ``album.json``.

``album.json`` is the album's system of record (every photo's verdict,
sharpness scores, per-body annotation data, star ratings, LLM culling
results, edited-image pointers, run settings). It routinely reaches tens of
megabytes — a 1600-photo album measures ~33 MB on disk and ~50 MB parsed —
so re-reading it per HTTP request made even a 5 KB thumbnail cost ~0.4 s to
serve in culling_app.py.

`Album` wraps the payload so every consumer shares ONE parse and ONE set of
derived indexes, and `album_for()` is the single place caching happens.
Freshness is stat-based (see `Album.refresh`): any external writer -- a
1_prep_review.py subprocess, algo/regrade.py, another process entirely --
invalidates the cache implicitly on its next write, so no writer needs to
know this cache exists.
"""

from __future__ import annotations

import json
import shutil
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from algo.models import (
    AutoAdjustment,
    Body,
    Box,
    Face,
    NumpyEncoder,
    PredictedKeyPoint,
)
from algo.utils import (
    THUMBNAILS_SUBDIR,
    atomic_save_and_backup,
    ensure_cover_thumbnail,
    make_unique_import_key,
    unique_path,
)

ALBUM_JSON_NAME = "album.json"
INFO_JSON_NAME = "info.json"

EDITEDIMAGES_SUBDIR = "editedimages"

# Whether `Album.save()` keeps a gzipped copy of the previous album.json.
#
# Deliberately a single global switch rather than a per-call parameter: it is
# an ambient policy ("is this process doing bulk rewrites?"), not a property
# of any one save, and threading it through every call site would put the
# decision in places that have no basis for making it. Do not "fix" this into
# a parameter. Callers that rewrite an album repeatedly in one run set it to
# False once, explicitly, where that reasoning is visible.
album_json_backups_enabled: bool = True
# How many albums may hold a fully parsed payload at once. Bounded because a
# parsed payload is ~50 MB for a large album and the album-listing page walks
# every album directory; the working set in every other flow is a single
# album, so a small cap costs at most one re-parse after visiting that page.
CACHE_MAX_ALBUMS = 4


def entry_key(entry: dict) -> str:
    """The album-wide bookkeeping key for a raw ``results[]`` entry dict (see
    algo/utils.py::make_unique_import_key), falling back to the plain
    basename for albums written before ``key`` existed.

    A module-level function (not just `AlbumImage.key`) because several
    batch scripts (algo/regrade.py, algo/stages/llm_culling.py) mutate
    entries in place without wrapping every one in an `AlbumImage`.
    """
    return entry.get("key") or Path(entry.get("file", "")).name


class PersonRecord:
    """One detected person inside a photo — the persisted, pixel-free record.

    Replaces the four ad-hoc shapes this data used to take: the raw
    ``annotation_data.evaluated[i]`` dict, algo/regrade.py's two partial
    ``Body`` reconstructions, and algo/facereco_provider.py's ``BodyRecord``.

    Backed by the wire dict rather than copying out of it (see `wire`), so a
    record read from an album and mutated is the album's own data. The live
    pixel-bearing counterpart is algo/models.py::Body, which this class
    deliberately does not replace: that one carries a face crop and belongs
    to detection, not to persistence. `to_body()` bridges between them.
    """

    def __init__(self, data: dict) -> None:
        self._d = data

    def __repr__(self) -> str:
        return f"<PersonRecord bbox={self._d.get('body_bbox')} sharpness={self.sharpness_score:.3f}>"

    @property
    def wire(self) -> dict:
        """The exact ``evaluated[]`` dict this record wraps.

        Public because algo/facereco.py embeds it verbatim into
        ``.FaceReco/*/face.json`` as ``"Body"``, and face databases already
        on disk contain those copies — anything that reconstructs the dict
        instead of passing this one through would break them. Not an
        invitation to read it by key; use the properties.
        """
        return self._d

    # -- geometry -------------------------------------------------------- #
    @property
    def body_bbox(self) -> Optional[Box]:
        return Box.from_wire(self._d.get("body_bbox"))

    @property
    def body_keypoints(self) -> list[PredictedKeyPoint]:
        """The 17 COCO body keypoints, empty for albums that never stored them."""
        return [PredictedKeyPoint.from_wire(kp) for kp in self._d.get("body_keypoints", [])]

    @property
    def face_bbox(self) -> Optional[Box]:
        return Box.from_wire(self._d.get("face_bbox"))

    @property
    def narrow_face_bbox(self) -> Optional[Box]:
        """Tight box around the face landmarks, used for sharpness scoring."""
        return Box.from_wire(self._d.get("narrow_face_bbox"))

    @property
    def face(self) -> Optional[Face]:
        """The matched face with its landmarks, or None when no face matched."""
        return Face.from_wire(self._d.get("face_kps"))

    @property
    def face_confidence(self) -> Optional[float]:
        face_kps = self._d.get("face_kps") or {}
        return face_kps.get("confidence")

    # -- scores ---------------------------------------------------------- #
    @property
    def sharpness_score(self) -> float:
        return float(self._d.get("sharpness_score", 0.0))

    @property
    def lap_var(self) -> float:
        return float(self._d.get("lap_var", 0.0))

    @property
    def ten(self) -> float:
        return float(self._d.get("ten", 0.0))

    # -- verdict --------------------------------------------------------- #
    @property
    def is_blurry(self) -> bool:
        return bool(self._d.get("is_blurry", True))

    @is_blurry.setter
    def is_blurry(self, value: bool) -> None:
        self._d["is_blurry"] = bool(value)

    @property
    def passed(self) -> bool:
        """The same fact as `is_blurry`, inverted — algo/models.py::Body
        calls it ``passed`` and the wire format calls it ``is_blurry``."""
        return not self.is_blurry

    @passed.setter
    def passed(self, value: bool) -> None:
        self.is_blurry = not value

    @property
    def rejection_reason(self) -> str:
        return self._d.get("rejection_reason") or ""

    @rejection_reason.setter
    def rejection_reason(self, value: str) -> None:
        self._d["rejection_reason"] = value

    @property
    def qualified_for_sharpness(self) -> bool:
        """Whether this body cleared every gate before sharpness scoring.

        Only ever written by the legacy ``1_prep_review.py::analyse_image``
        path, not by the serializer the live pipeline uses, so it is absent
        (and therefore False) in every album the current code writes. Kept
        because algo/facereco.py reads it; see `cleared_grading_gates` for
        the check that actually works today.
        """
        return bool(self._d.get("qualified_for_sharpness", False))

    @property
    def cleared_grading_gates(self) -> bool:
        """True when this body passed every gate BEFORE the sharpness
        threshold — i.e. it has a matched face, of adequate size, with
        enough visible head keypoints (see algo/scorers.py's
        short-circuiting BodyArrayScorer).

        Those three gates don't depend on the threshold, so such a body is a
        legitimate candidate to re-evaluate at any new one. A body rejected
        for sharpness OR for jersey colour necessarily got past them; a body
        rejected for anything else did not. Albums imported before
        ``rejection_reason`` was persisted have no reason string, so a
        currently-failing body there can't be shown to qualify and is
        conservatively reported as not cleared.
        """
        if self.passed:
            return True
        reason = self.rejection_reason
        return reason.startswith("sharpness score") or reason.startswith("jersey ")

    # -- appearance ------------------------------------------------------ #
    @property
    def cloth_color(self) -> str:
        return self._d.get("cloth_color", "N/A")

    @cloth_color.setter
    def cloth_color(self, value: str) -> None:
        self._d["cloth_color"] = value

    @property
    def cloth_color_detail(self) -> dict:
        return self._d.get("cloth_color_detail") or {}

    @cloth_color_detail.setter
    def cloth_color_detail(self, value: dict) -> None:
        self._d["cloth_color_detail"] = value

    # -- roster tagging -------------------------------------------------- #
    @property
    def player_name(self) -> str:
        """Roster name attached by face tagging, "" if this person is
        untagged. Lives here rather than on `AlbumImage` because a photo
        can contain several players."""
        return self._d.get("player_name") or ""

    @player_name.setter
    def player_name(self, value: str) -> None:
        self._d["player_name"] = value

    @property
    def player_number(self) -> str:
        return self._d.get("player_number") or ""

    @player_number.setter
    def player_number(self, value: str) -> None:
        self._d["player_number"] = value

    # -- pixels ---------------------------------------------------------- #
    def crop_from(self, image: np.ndarray, box: Optional[Box] = None) -> Optional[np.ndarray]:
        """Cut this person out of *image* (the already-decoded source photo).

        Defaults to the body box; pass `face_bbox`/`narrow_face_bbox` for a
        face crop. Returns None when the box is missing or lands outside the
        image. Bbox maths lives here so no consumer re-derives it.
        """
        box = box if box is not None else self.body_bbox
        if box is None or image is None or image.size == 0:
            return None
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box.as_px_ints(w, h)
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]

    # -- bridge to the live detection model ------------------------------ #
    def to_body(self, crop: Optional[np.ndarray] = None) -> Body:
        """This record as an algo/models.py::Body, for the code that still
        speaks that type (the annotation drawer, the cloth-colour
        predictor). *crop* is the face crop; neither of those consumers
        reads it, so it defaults to a 1x1 placeholder rather than forcing
        callers to decode pixels they don't need."""
        face = self.face
        body_bbox = self.body_bbox
        return Body(
            crop=crop if crop is not None else _PLACEHOLDER_CROP,
            bbox=body_bbox if body_bbox is not None else Box(0.0, 0.0, 0.0, 0.0),
            faces=[face] if face else [],
            keypoints=self.body_keypoints,
            passed=self.passed,
            rejection_reason=self.rejection_reason,
            sharpness_score=self.sharpness_score,
            best_face=face,
            best_narrow_box=self.narrow_face_bbox,
            lap_var=self.lap_var,
            ten=self.ten,
            cloth_color=self.cloth_color,
            cloth_color_detail=self.cloth_color_detail,
        )

    @classmethod
    def from_body(cls, body: Body) -> "PersonRecord":
        """Serialize a freshly detected `Body` into a new record.

        The one definition of the ``evaluated[]`` wire shape — key order
        included, since album.json is written with ``indent=2`` and diffed
        by humans.
        """
        return cls({
            "body_bbox":          body.bbox.to_wire(),
            "body_keypoints":     [kp.to_wire() for kp in body.keypoints],
            "face_bbox":          body.best_face.bbox.to_wire() if body.best_face else None,
            "narrow_face_bbox":   body.best_narrow_box.to_wire() if body.best_narrow_box else None,
            "face_kps":           body.best_face.to_wire() if body.best_face else None,
            "sharpness_score":    body.sharpness_score,
            "lap_var":            body.lap_var,
            "ten":                body.ten,
            "is_blurry":          not body.passed,
            "rejection_reason":   body.rejection_reason,
            "cloth_color":        body.cloth_color,
            "cloth_color_detail": body.cloth_color_detail,
        })


# `PersonRecord.to_body` hands this to consumers that need a Body but never
# look at its pixels (the annotation drawer, the cloth-colour predictor).
_PLACEHOLDER_CROP = np.zeros((1, 1, 3), dtype=np.uint8)


class AlbumImage:
    """One photo in an album: every field album.json records about it, plus
    where its files live on disk.

    A TYPED VIEW OVER THE ENTRY DICT, not a replacement for it. `_d` is the
    live payload dict, so a mutation here is persisted by the next
    `Album.save()`, and any key this class doesn't model survives a
    load/save cycle untouched. That matters: albums on disk carry fields
    from older schema versions, and rebuilding the dict from typed fields
    would silently delete them.

    An album keeps up to four files per photo — the imported original, an
    accepted AI edit, the annotated preview, and the preview's thumbnail —
    and callers kept re-deriving those paths from raw dict fields, each with
    its own idea of which one to prefer. `image_path` is the answer to "show
    me this photo": the accepted edit when there is one, the original
    otherwise.

    LIFETIME: valid until the next `Album.refresh()` sees a changed file.
    Re-fetch with `album.image(key)` after anything that could have let
    another process write album.json.
    """

    def __init__(self, album: "Album", entry: dict) -> None:
        self.album = album
        self._d = entry

    def __repr__(self) -> str:
        return f"<AlbumImage {self.key!r}>"

    @property
    def entry(self) -> dict:
        """The live ``results[]`` dict behind this photo.

        For algo/album.py's own merge/import code and for callers that must
        hand the whole entry to something outside this module. Reading it by
        key path is what this class exists to replace — use the properties.
        """
        return self._d

    # -- identity ------------------------------------------------------- #
    @property
    def key(self) -> str:
        """The album-wide bookkeeping key (see algo/utils.py::
        make_unique_import_key), falling back to the plain basename for
        albums written before ``key`` existed."""
        return entry_key(self._d)

    @property
    def source_file(self) -> str:
        """Where this photo was imported from, as stored. Prefer
        `original_path` unless you specifically need the raw string."""
        return self._d.get("file") or ""

    # -- verdict -------------------------------------------------------- #
    @property
    def status(self) -> str:
        """This photo's verdict: "sharp" / "blurry" / "skipped" / "error",
        or "" before grading has run.

        Deliberately a plain string rather than an enum or a state machine.
        The lifecycle is genuinely unsettled — "skipped" conflates "no
        person detected" with "excluded", "error" is written by a legacy
        path, and regrade re-verdicts already-graded photos — and pinning it
        down is its own task. Routing every reader and writer through this
        property is what will make that task cheap.
        """
        return self._d.get("status", "")

    @status.setter
    def status(self, value: str) -> None:
        self._d["status"] = value

    @property
    def is_sharp(self) -> bool:
        return self.status == "sharp"

    @property
    def is_blurry(self) -> bool:
        return self.status == "blurry"

    @property
    def overall_blurry(self) -> bool:
        """The annotation stage's own verdict, used to label the preview.
        Kept distinct from `status` because it is what was drawn onto the
        preview image, which a later regrade may not have redrawn."""
        return bool(self.annotation.get("overall_blurry", False))

    @overall_blurry.setter
    def overall_blurry(self, value: bool) -> None:
        self._ensure_annotation()["overall_blurry"] = bool(value)

    # -- scores --------------------------------------------------------- #
    @property
    def sharpness_score(self) -> Optional[float]:
        score = self._d.get("sharpness_score")
        return float(score) if score is not None else None

    @property
    def sharpness_grade(self) -> Optional[float]:
        """`sharpness_score` as a 0-100 percentage, for display."""
        grade = self._d.get("sharpness_grade")
        return float(grade) if grade is not None else None

    @property
    def laplacian_variance(self) -> Optional[float]:
        value = self._d.get("laplacian_variance")
        return float(value) if value is not None else None

    @property
    def tenengrad_score(self) -> Optional[float]:
        value = self._d.get("tenengrad_score")
        return float(value) if value is not None else None

    def set_sharpness_score(self, score: float) -> None:
        """Record this photo's overall sharpness. Sets `sharpness_grade`
        too, because it is just *score* rescaled for display and letting the
        two be written separately is how they drift apart."""
        self._d["sharpness_score"] = round(score, 4)
        self._d["sharpness_grade"] = round(score * 100, 1)

    def best_body(self) -> Optional[PersonRecord]:
        """The person whose sharpness this photo is judged by: the sharpest
        one that passed, or — when nothing passed — the sharpest overall, so
        a fully-rejected photo still reports a real score. Mirrors the rule
        the import serializer uses."""
        bodies = self.bodies
        if not bodies:
            return None
        passing = [b for b in bodies if b.passed]
        return max(passing or bodies, key=lambda b: b.sharpness_score)

    # -- rating / culling ----------------------------------------------- #
    @property
    def stars(self) -> Optional[int]:
        stars = self._d.get("stars")
        return int(stars) if stars is not None else None

    @stars.setter
    def stars(self, value: Optional[int]) -> None:
        self._d["stars"] = value

    @property
    def stars_manual(self) -> bool:
        """True once the user has rated this photo by hand, which stops
        automatic re-rating from overwriting their judgement."""
        return bool(self._d.get("stars_manual", False))

    @stars_manual.setter
    def stars_manual(self, value: bool) -> None:
        self._d["stars_manual"] = bool(value)

    @property
    def keep(self) -> bool:
        return bool(self._d.get("keep", False))

    @keep.setter
    def keep(self, value: bool) -> None:
        """Whether this photo survives culling.

        NOT derived from `stars`: LLM culling sets it from burst rank (only
        the best frame in a burst is kept, whatever its rating). The
        stars-based rule is one of several writers — see `apply_auto_rating`.
        """
        self._d["keep"] = bool(value)

    def apply_auto_rating(self, stars: int) -> bool:
        """Re-rate this photo automatically (stars, and keep = stars >= 3).

        No-op, returning False, once the user has rated it by hand.
        """
        if self.stars_manual:
            return False
        self._d["stars"] = stars
        self._d["keep"] = stars >= 3
        return True

    @property
    def llm_grade(self) -> Optional[float]:
        """The LLM's 0.0-1.0 quality score, None if culling never ran."""
        grade = self._d.get("llm_grade")
        return float(grade) if grade is not None else None

    @llm_grade.setter
    def llm_grade(self, value: Optional[float]) -> None:
        self._d["llm_grade"] = value

    @property
    def burst_ranking(self) -> Optional[dict]:
        """``{"rank", "reason", "group_id"}`` for a photo the LLM ranked in
        the top 3 of its burst, None otherwise."""
        return self._d.get("burst_ranking")

    @burst_ranking.setter
    def burst_ranking(self, value: Optional[dict]) -> None:
        self._d["burst_ranking"] = value

    @property
    def burst_caption(self) -> str:
        return self._d.get("burst_caption") or ""

    @burst_caption.setter
    def burst_caption(self, value: str) -> None:
        self._d["burst_caption"] = value

    # -- detected people ------------------------------------------------- #
    @property
    def annotation(self) -> dict:
        """Read-only view of ``annotation_data`` (``{}`` when absent)."""
        return self._d.get("annotation_data") or {}

    def _ensure_annotation(self) -> dict:
        return self._d.setdefault("annotation_data", {})

    @property
    def has_annotation(self) -> bool:
        """False for a photo the analysis stage never produced detections
        for — a skipped or errored one."""
        return self._d.get("annotation_data") is not None

    @property
    def bodies(self) -> list[PersonRecord]:
        """Everyone detected in this photo.

        Each record wraps the album's own dict, so mutating one and saving
        persists it. Replacing the whole set is `set_bodies` — a detection
        pass always produces a complete set, never an incremental addition.
        """
        return [PersonRecord(b) for b in self.annotation.get("evaluated", [])]

    def set_bodies(self, records: Sequence[PersonRecord]) -> None:
        self._ensure_annotation()["evaluated"] = [r.wire for r in records]

    @property
    def processing_shape(self) -> tuple[int, int]:
        """``(height, width)`` of the downscaled image the detections were
        computed against — the frame the normalised bboxes refer to."""
        shape = self.annotation.get("processing_shape") or [0, 0]
        return (int(shape[0]), int(shape[1]))

    # -- exposure -------------------------------------------------------- #
    @property
    def auto_adjustment(self) -> Optional[AutoAdjustment]:
        """The auto-exposure correction computed at import, or None."""
        adjustment = self._d.get("auto_adjustment")
        if not adjustment or adjustment.get("ev") is None:
            return None
        return AutoAdjustment(ev=float(adjustment["ev"]))

    @auto_adjustment.setter
    def auto_adjustment(self, value: Optional[AutoAdjustment]) -> None:
        self._d["auto_adjustment"] = {"ev": value.ev} if value is not None else None

    # -- files ---------------------------------------------------------- #
    @property
    def original_path(self) -> Optional[Path]:
        """The imported source photo, wherever it was imported from. May be
        a RAW file, and may no longer exist (the album only ever references
        originals, it never copies them)."""
        file_path = self._d.get("file")
        return Path(file_path) if file_path else None

    @property
    def edited_path(self) -> Optional[Path]:
        """The accepted AI edit, or None if this photo has none. Only
        returned when the file is actually present, so a manually deleted
        edit degrades to the original instead of a broken path."""
        edited = self._d.get("edited_image")
        if not edited:
            return None
        path = self.album.path / edited
        return path if path.is_file() else None

    @property
    def image_path(self) -> Optional[Path]:
        """What this photo currently looks like — the accepted edit if there
        is one, else the original. Use this anywhere the user is shown "the
        photo" rather than specifically its unedited state."""
        return self.edited_path or self.original_path

    @property
    def preview_path(self) -> Optional[Path]:
        """The annotated (boxes/badges drawn on) preview written at import,
        or None for an entry that never got one."""
        preview = self._d.get("preview_path")
        return self.album.path / preview if preview else None

    @property
    def thumbnail_path(self) -> Optional[Path]:
        """The preview's square cover-crop thumbnail. Written alongside the
        preview at import, and regenerated on demand by `ensure_thumbnail()`
        for albums imported before thumbnails existed."""
        preview = self.preview_path
        return self.album.path / THUMBNAILS_SUBDIR / preview.name if preview else None

    def ensure_thumbnail(self) -> Optional[Path]:
        """The thumbnail to serve for this photo, generating it from the
        preview first if it is missing or stale. Falls back to the preview
        itself if generation isn't possible."""
        preview, thumbnail = self.preview_path, self.thumbnail_path
        if preview is None or thumbnail is None or not preview.is_file():
            return None
        return ensure_cover_thumbnail(preview, thumbnail)

    @property
    def preview_stem(self) -> str:
        """Filename stem shared by this photo's preview and thumbnail —
        falling back to the key's stem, which is what the annotation stage
        would name them (see algo/frame.py::Frame.key_stem)."""
        preview = self._d.get("preview_path")
        return Path(preview).stem if preview else Path(self.key).stem

    def set_edited_image(self, path: Path) -> None:
        """Record *path* (inside the album directory) as the accepted edit.
        Stored relative to the album so the folder stays movable."""
        self._d["edited_image"] = f"{EDITEDIMAGES_SUBDIR}/{Path(path).name}"



@dataclass
class ImportSummary:
    """Result of one `Album.import_from` call, for logging."""
    added: int = 0
    renamed: int = 0
    facereco_clusters_added: int = 0
    facereco_clusters_merged: int = 0


class Album:
    """One album directory's ``album.json``, parsed on demand.

    Every accessor calls `refresh()` first, so a stale payload is never
    handed out even though the object itself is cached. Mutating callers
    should edit the dicts reached through `results`/`entry()` and then call
    `save()`, which rewrites the file atomically and keeps this object's
    freshness stamp in step with what it just wrote.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.album_json = self.path / ALBUM_JSON_NAME
        self._lock = threading.RLock()
        self._stamp: Optional[tuple] = None
        self._payload: Optional[dict] = None
        self._entries: Optional[dict] = None
        self._images: Optional[dict] = None
        self._source_index: Optional[dict] = None

    # -- construction ----------------------------------------------------- #
    @classmethod
    def create(
        cls,
        path: Union[str, Path],
        *,
        team_id: str,
        paths: Optional[Sequence[Path]] = None,
        our_jersey_color: Optional[str] = None,
        run_settings: Optional[dict] = None,
        import_status: str = "in_progress",
    ) -> "Album":
        """Factory for a brand-new album (a permanent one, or a temp staging
        album for "import more images" -- see 1_prep_review.py): creates
        *path* if needed and writes an initial album.json immediately, so
        the basic properties (team_id/run_settings/...) are set and readable
        from the moment the caller gets the object back, before any analysis
        pipeline stage has run.

        *paths* is the complete set of source photos this album is being
        created from, already enumerated and filtered by the caller (see
        algo/imagefiles.py::images_to_import). One entry is created per
        path, with its bookkeeping key assigned up front -- so preview
        filenames are known before a single pixel is decoded, and nothing
        downstream has to discover images for itself.

        Use plain ``Album(path)`` (or `album_for`) to load an ALREADY
        existing album instead -- this always (re)creates one.
        """
        new_album = cls(path)
        new_album.path.mkdir(parents=True, exist_ok=True)
        used_keys: dict[str, Path] = {}
        results = []
        for source in paths or ():
            source = Path(source)
            results.append({
                "file": str(source),
                "key": make_unique_import_key(source.name, used_keys, source.resolve()),
            })
        new_album._payload = {
            "team_id": team_id,
            "our_jersey_color": our_jersey_color,
            "import_status": import_status,
            "run_settings": run_settings or {},
            "results": results,
        }
        new_album.save()
        return new_album

    # -- freshness ------------------------------------------------------ #
    def _file_stamp(self) -> Optional[tuple]:
        try:
            st = self.album_json.stat()
        except OSError:
            return None
        return (st.st_mtime, st.st_size)

    def refresh(self) -> None:
        """Drop everything derived from album.json if the file changed."""
        with self._lock:
            stamp = self._file_stamp()
            if stamp != self._stamp:
                self.invalidate()

    def invalidate(self) -> None:
        with self._lock:
            self._stamp = None
            self._payload = None
            self._entries = None
            self._images = None
            self._source_index = None

    # -- payload -------------------------------------------------------- #
    @property
    def payload(self) -> dict:
        """The whole parsed album.json (``{}`` if missing or unreadable)."""
        with self._lock:
            self.refresh()
            if self._payload is None:
                stamp = self._file_stamp()
                try:
                    with open(self.album_json, encoding="utf-8") as fh:
                        self._payload = json.load(fh)
                except (json.JSONDecodeError, OSError):
                    self._payload = {}
                    stamp = None  # don't cache a failed read
                self._stamp = stamp
            return self._payload

    @property
    def results(self) -> list:
        return self.payload.get("results", []) or []

    @property
    def run_settings(self) -> dict:
        return self.payload.get("run_settings") or {}

    @property
    def team_id(self) -> str:
        return self.payload.get("team_id") or ""

    @property
    def our_jersey_color(self) -> str:
        return self.payload.get("our_jersey_color") or ""

    @property
    def import_status(self) -> str:
        return self.payload.get("import_status", "complete")

    @property
    def llm_cost_summary(self) -> dict:
        """What the last LLM culling run cost (token counts + USD), ``{}``
        if culling never ran."""
        return self.payload.get("llm_cost_summary") or {}

    @llm_cost_summary.setter
    def llm_cost_summary(self, value: dict) -> None:
        with self._lock:
            self.payload["llm_cost_summary"] = value

    @property
    def exists(self) -> bool:
        return self.album_json.is_file()

    @property
    def is_complete(self) -> bool:
        """True once 1_prep_review.py has finished writing this album.

        album.json and info.json are always written together (barring a
        completely empty input folder), regardless of --skip-facereco or
        whether any blurry images were found. An interrupted "import more
        images" run leaves ``import_status`` set to "in_progress", so a
        crashed/partial import is never mistaken for a finished album.
        """
        if not (self.exists and (self.path / INFO_JSON_NAME).is_file()):
            return False
        return self.import_status != "in_progress"

    # -- derived indexes ------------------------------------------------ #
    @property
    def entries(self) -> dict:
        """``{key: result entry}`` for every photo in the album.

        Keyed by the disambiguated bookkeeping key, falling back to the
        plain basename for albums written before ``key`` existed (see
        algo/utils.py::make_unique_import_key). The entries are the live
        payload dicts, so mutating one and calling `save()` persists it.
        """
        with self._lock:
            payload = self.payload
            if self._entries is None:
                index: dict = {}
                for entry in payload.get("results", []) or []:
                    file_path = entry.get("file")
                    if not file_path and not entry.get("key"):
                        continue
                    index[entry.get("key") or Path(file_path).name] = entry
                self._entries = index
            return self._entries

    def entry(self, key: str) -> Optional[dict]:
        return self.entries.get(key)

    @property
    def images(self) -> dict:
        """``{key: AlbumImage}`` — the same photos as `entries`, wrapped so
        callers get file-path resolution instead of raw dict fields."""
        with self._lock:
            entries = self.entries
            if self._images is None:
                self._images = {key: AlbumImage(self, entry) for key, entry in entries.items()}
            return self._images

    def image(self, key: str) -> Optional["AlbumImage"]:
        return self.images.get(key)

    @property
    def image_list(self) -> list["AlbumImage"]:
        """Every photo, in album.json order. Iterate this rather than
        `results` so per-photo data is reached through `AlbumImage`."""
        return list(self.images.values())

    @property
    def source_paths(self) -> frozenset[Path]:
        """Every source photo this album already holds, resolved.

        The exclusion set for a follow-up import (see
        algo/imagefiles.py::images_to_import): the album's CURRENT contents,
        computed fresh each time. There is deliberately no record of past
        imports to reconstruct it from.
        """
        return frozenset(
            Path(file_path).resolve()
            for file_path in (entry.get("file") for entry in self.results)
            if file_path
        )

    @property
    def source_index(self) -> dict:
        """``{key: absolute source path}`` — where each photo was imported
        from. Retained separately from `entries` because it is the only
        thing the per-image endpoints need, and it stays small."""
        with self._lock:
            entries = self.entries
            if self._source_index is None:
                self._source_index = {
                    key: entry["file"] for key, entry in entries.items() if entry.get("file")
                }
            return self._source_index

    # -- persistence ---------------------------------------------------- #
    def save(self) -> None:
        """Atomically rewrite album.json from the in-memory payload.

        Backs up the previous contents (see algo/utils.py) so a crash
        mid-write can never corrupt the album.
        """
        with self._lock:
            if self._payload is None:
                return
            try:
                atomic_save_and_backup(
                    json.dumps(self._payload, indent=2, cls=NumpyEncoder),
                    self.album_json,
                    backup=album_json_backups_enabled,
                )
            except Exception:
                # The file on disk and this object may now disagree; force
                # the next reader to go back to disk.
                self.invalidate()
                raise
            self._stamp = self._file_stamp()

    def write_results(
        self,
        results: list[dict],
        *,
        our_jersey_color: Optional[str] = None,
        team_id: Optional[str] = None,
        import_status: str = "complete",
        run_settings: Optional[dict] = None,
    ) -> None:
        """Replace this album's entire payload with a freshly-built one and
        save it -- the only place the top-level payload is assembled from
        scratch (the normal import path, "import more images", and a deep
        regrade; see 1_prep_review.py). *results* must already include
        whatever prior entries the caller wants carried forward -- this
        does not merge with what's currently on disk.
        """
        with self._lock:
            self._payload = {
                "team_id": team_id,
                "our_jersey_color": our_jersey_color,
                "import_status": import_status,
                "run_settings": run_settings or {},
                "results": results,
            }
            self._entries = None
            self._images = None
            self._source_index = None
            self.save()

    def mark_import_complete(self) -> None:
        """Flip ``import_status`` to "complete". Called once the full
        pipeline (analysis + FaceReco + LLM culling) has finished, so a
        crash in between leaves the album correctly marked "in_progress"
        instead of falsely looking finished."""
        with self._lock:
            payload = self.payload
            payload["import_status"] = "complete"
            self.save()

    def update_settings(self, *, our_jersey_color: Optional[str] = None, run_settings: Optional[dict] = None) -> None:
        """Patch metadata only known once processing finished (e.g. the
        jersey colour polled from this run's photos), without touching
        `results`."""
        with self._lock:
            payload = self.payload
            if our_jersey_color is not None:
                payload["our_jersey_color"] = our_jersey_color
            if run_settings is not None:
                payload["run_settings"] = run_settings
            self.save()

    # -- merging another album in ---------------------------------------- #
    def import_from(self, other: "Album") -> ImportSummary:
        """Merge *other* (typically a temp staging album produced by one
        "import more images" run -- see 1_prep_review.py) into this album:
        appends its `results` entries (renaming the bookkeeping ``key`` --
        and the preview/thumbnail/edited-image files derived from it --
        only on collision with an entry already in THIS album), and folds
        its ``.FaceReco`` clusters into this album's own. *other* is left
        untouched on disk; the caller deletes its directory afterward.
        """
        with self._lock:
            used_keys: dict[str, Path] = {
                key: Path(entry["file"]).resolve()
                for key, entry in self.entries.items() if entry.get("file")
            }
            key_renames: dict[str, str] = {}
            merged_entries: list[dict] = []
            for entry in other.results:
                new_entry = dict(entry)
                old_key = entry_key(entry)
                file_path = new_entry.get("file")
                new_key = (
                    make_unique_import_key(Path(file_path).name, used_keys, Path(file_path).resolve())
                    if file_path else old_key
                )
                if new_key != old_key:
                    new_entry["key"] = new_key
                    key_renames[old_key] = new_key
                self._copy_entry_files(other, new_entry, old_key, new_key)
                merged_entries.append(new_entry)

            facereco_merged, facereco_added = self._merge_facereco(other, key_renames)

            payload = self.payload
            payload.setdefault("results", []).extend(merged_entries)
            self._entries = None
            self._images = None
            self._source_index = None
            self.save()

        return ImportSummary(
            added=len(merged_entries), renamed=len(key_renames),
            facereco_clusters_added=facereco_added, facereco_clusters_merged=facereco_merged,
        )

    def _copy_entry_files(self, other: "Album", entry: dict, old_key: str, new_key: str) -> None:
        """Copy *entry*'s preview/thumbnail/edited-image files from *other*
        into this album, renaming them to match *new_key* when it differs
        from *old_key*, and updating *entry*'s path fields in place."""
        renamed = new_key != old_key
        preview_rel = entry.get("preview_path")
        if preview_rel:
            src = other.path / preview_rel
            if src.is_file():
                dest_name = f"{Path(new_key).stem}{Path(preview_rel).suffix}" if renamed else Path(preview_rel).name
                previews_dir = self.path / "previews"
                previews_dir.mkdir(parents=True, exist_ok=True)
                dest = unique_path(previews_dir, dest_name)
                shutil.copy2(src, dest)
                entry["preview_path"] = f"previews/{dest.name}"
                # The thumbnail is looked up BY the preview's own filename
                # (see AlbumImage.thumbnail_path) so it must share dest.name,
                # not go through its own unique_path resolution.
                thumb_src = other.path / THUMBNAILS_SUBDIR / Path(preview_rel).name
                if thumb_src.is_file():
                    thumb_dir = self.path / THUMBNAILS_SUBDIR
                    thumb_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(thumb_src, thumb_dir / dest.name)

        edited_rel = entry.get("edited_image")
        if edited_rel:
            src = other.path / edited_rel
            if src.is_file():
                dest_dir = self.path / EDITEDIMAGES_SUBDIR
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = unique_path(dest_dir, Path(edited_rel).name)
                shutil.copy2(src, dest)
                entry["edited_image"] = f"{EDITEDIMAGES_SUBDIR}/{dest.name}"

    # -- FaceReco merging -------------------------------------------------- #
    def _merge_facereco(self, other: "Album", key_renames: dict[str, str]) -> tuple[int, int]:
        """Fold *other*'s ``.FaceReco`` clusters into this album's own.
        Numeric (unmatched) clusters are renumbered past this album's
        highest existing id; named (matched) clusters are merged into an
        existing same-named cluster if present, else copied over as-is.
        Returns ``(clusters_merged, clusters_added)``."""
        other_facereco = other.path / ".FaceReco"
        if not other_facereco.is_dir():
            return (0, 0)
        self_facereco = self.path / ".FaceReco"
        self_facereco.mkdir(parents=True, exist_ok=True)

        existing_numeric_ids = [
            int(d.name) for d in self_facereco.iterdir()
            if d.is_dir() and d.name.isdigit()
        ]
        next_numeric = (max(existing_numeric_ids) + 1) if existing_numeric_ids else 0

        merged = added = 0
        for cluster_dir in sorted(other_facereco.iterdir()):
            # Dot-prefixed dirs (.AllFaces, .debug) are diagnostic-only, not
            # consumed by any reader -- left behind rather than merged.
            if not cluster_dir.is_dir() or cluster_dir.name.startswith("."):
                continue
            if cluster_dir.name.isdigit():
                dest_name = f"{next_numeric:04d}"
                next_numeric += 1
                shutil.copytree(cluster_dir, self_facereco / dest_name)
                self._rewrite_facereco_origfilenames(self_facereco / dest_name / "face.json", key_renames)
                added += 1
                continue
            target_cluster_dir = self_facereco / cluster_dir.name
            if target_cluster_dir.is_dir():
                self._merge_facereco_cluster(cluster_dir, target_cluster_dir, key_renames)
                merged += 1
            else:
                shutil.copytree(cluster_dir, target_cluster_dir)
                self._rewrite_facereco_origfilenames(target_cluster_dir / "face.json", key_renames)
                added += 1
        return (merged, added)

    @staticmethod
    def _rewrite_facereco_origfilenames(face_json_path: Path, key_renames: dict[str, str]) -> None:
        if not key_renames or not face_json_path.is_file():
            return
        with open(face_json_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        changed = False
        for face in payload.get("faces", []):
            renamed = key_renames.get(face.get("origFilename"))
            if renamed:
                face["origFilename"] = renamed
                changed = True
        if changed:
            with open(face_json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)

    @staticmethod
    def _merge_facereco_cluster(src_dir: Path, dest_dir: Path, key_renames: dict[str, str]) -> None:
        """Combine a same-named cluster from another album into *dest_dir*
        (crop files copied over -- renamed on filename collision -- and
        *src_dir*'s face.json entries appended to dest's own)."""
        crop_renames: dict[str, str] = {}
        src_face_dir = src_dir / "Face"
        if src_face_dir.is_dir():
            dest_face_dir = dest_dir / "Face"
            dest_face_dir.mkdir(parents=True, exist_ok=True)
            for crop in sorted(src_face_dir.iterdir()):
                if not crop.is_file():
                    continue
                dest = unique_path(dest_face_dir, crop.name)
                shutil.copy2(crop, dest)
                crop_renames[crop.name] = dest.name

        src_annotated_dir = src_dir / "Face.annotated"
        if src_annotated_dir.is_dir():
            dest_annotated_dir = dest_dir / "Face.annotated"
            for annotated in sorted(src_annotated_dir.iterdir()):
                if not annotated.is_file():
                    continue
                dest_annotated_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(annotated, dest_annotated_dir / crop_renames.get(annotated.name, annotated.name))

        src_face_json = src_dir / "face.json"
        if not src_face_json.is_file():
            return
        with open(src_face_json, encoding="utf-8") as fh:
            src_payload = json.load(fh)
        dest_face_json = dest_dir / "face.json"
        dest_payload: dict = {}
        if dest_face_json.is_file():
            with open(dest_face_json, encoding="utf-8") as fh:
                dest_payload = json.load(fh)
        dest_faces = dest_payload.setdefault("faces", [])
        dest_payload.setdefault("provider", src_payload.get("provider", ""))
        dest_payload.setdefault("aligned", src_payload.get("aligned", False))
        for face in src_payload.get("faces", []):
            face = dict(face)
            renamed = key_renames.get(face.get("origFilename"))
            if renamed:
                face["origFilename"] = renamed
            if face.get("cropFileName") in crop_renames:
                face["cropFileName"] = crop_renames[face["cropFileName"]]
            dest_faces.append(face)
        with open(dest_face_json, "w", encoding="utf-8") as fh:
            json.dump(dest_payload, fh, indent=2)


# --------------------------------------------------------------------------- #
# The single cache. Keyed by resolved album directory, LRU-bounded, and safe
# to call from FastAPI's request threadpool.
# --------------------------------------------------------------------------- #
_cache: "OrderedDict[Path, Album]" = OrderedDict()
_cache_lock = threading.Lock()


def album_for(album_dir: Union[str, Path]) -> Album:
    """The shared `Album` for *album_dir*, parsing album.json at most once
    per on-disk revision."""
    key = Path(album_dir).resolve()
    with _cache_lock:
        album = _cache.get(key)
        if album is None:
            album = Album(key)
            _cache[key] = album
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX_ALBUMS:
            _cache.popitem(last=False)
    album.refresh()
    return album


def invalidate_all() -> None:
    """Force every cached album back to disk on next access. Only needed
    when something bypasses both `Album.save()` and the file's mtime/size
    (nothing does today) — kept as an escape hatch for callers that have
    just run an external tool and want to be certain."""
    with _cache_lock:
        albums = list(_cache.values())
    for album in albums:
        album.invalidate()
