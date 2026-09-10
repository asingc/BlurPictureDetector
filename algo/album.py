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
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

from algo.utils import THUMBNAILS_SUBDIR, atomic_save_and_backup, ensure_cover_thumbnail

ALBUM_JSON_NAME = "album.json"
INFO_JSON_NAME = "info.json"

EDITEDIMAGES_SUBDIR = "editedimages"
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


class AlbumImage:
    """One photo in an album: its ``results[]`` entry plus where its files
    live on disk.

    An album keeps up to four files per photo — the imported original, an
    accepted AI edit, the annotated preview, and the preview's thumbnail —
    and callers kept re-deriving those paths from raw dict fields, each with
    its own idea of which one to prefer. `image_path` is the answer to "show
    me this photo": the accepted edit when there is one, the original
    otherwise.

    `entry` is the live payload dict, so mutating it and calling
    `Album.save()` persists the change.
    """

    def __init__(self, album: "Album", entry: dict) -> None:
        self.album = album
        self.entry = entry

    def __repr__(self) -> str:
        return f"<AlbumImage {self.key!r}>"

    # -- identity ------------------------------------------------------- #
    @property
    def key(self) -> str:
        """The album-wide bookkeeping key (see algo/utils.py::
        make_unique_import_key), falling back to the plain basename for
        albums written before ``key`` existed."""
        return entry_key(self.entry)

    @property
    def status(self) -> str:
        return self.entry.get("status", "")

    @property
    def stars(self) -> Optional[int]:
        stars = self.entry.get("stars")
        return int(stars) if stars is not None else None

    # -- files ---------------------------------------------------------- #
    @property
    def original_path(self) -> Optional[Path]:
        """The imported source photo, wherever it was imported from. May be
        a RAW file, and may no longer exist (the album only ever references
        originals, it never copies them)."""
        file_path = self.entry.get("file")
        return Path(file_path) if file_path else None

    @property
    def edited_path(self) -> Optional[Path]:
        """The accepted AI edit, or None if this photo has none. Only
        returned when the file is actually present, so a manually deleted
        edit degrades to the original instead of a broken path."""
        edited = self.entry.get("edited_image")
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
        preview = self.entry.get("preview_path")
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
        preview = self.entry.get("preview_path")
        return Path(preview).stem if preview else Path(self.key).stem

    def set_edited_image(self, path: Path) -> None:
        """Record *path* (inside the album directory) as the accepted edit.
        Stored relative to the album so the folder stays movable."""
        self.entry["edited_image"] = f"{EDITEDIMAGES_SUBDIR}/{Path(path).name}"


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
                atomic_save_and_backup(json.dumps(self._payload, indent=2), self.album_json)
            except Exception:
                # The file on disk and this object may now disagree; force
                # the next reader to go back to disk.
                self.invalidate()
                raise
            self._stamp = self._file_stamp()


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
