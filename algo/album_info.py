"""An album directory's ``info.json`` — the run summary beside album.json.

Where `Album` (see algo/album.py) is the system of record for every photo's
data, ``info.json`` records how the album was PRODUCED: which source
directories were imported and when, the team colour that was polled, and
which bucket each photo landed in (blur / sharp / skipped).

The two files index the same photos by the same bookkeeping key (see
algo/utils.py::make_unique_import_key), so they have to be kept in step.
Before this class existed each consumer opened the file itself and reached
in by key path -- which is how `Album.import_from` came to rename a key in
album.json while leaving info.json pointing at the old one, quietly
dropping that photo from exports. `rename_keys` is the fix, and living
here is what keeps it from drifting again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from algo.utils import atomic_save_and_backup

INFO_JSON_NAME = "info.json"

# The buckets a reviewed photo can land in, and the payload key each maps to.
CATEGORIES = ("blur", "sharp", "skipped")
_ANNO_KEY = {"blur": "Anno_Blur", "sharp": "Anno_Sharp", "skipped": "Anno_Skipped"}

# album.json's per-photo ``status`` -> the info.json bucket it belongs in.
# Two vocabularies for one fact, kept mapped in one place rather than
# re-derived by every caller that has a status and needs a bucket.
_CATEGORY_FOR_STATUS = {"blurry": "blur", "sharp": "sharp", "skipped": "skipped"}


def category_for_status(status: str) -> Optional[str]:
    """The info.json bucket for an `AlbumImage.status`, or None for a status
    that has no bucket (e.g. "error", or an ungraded photo)."""
    return _CATEGORY_FOR_STATUS.get(status)


class InfoEntry:
    """One photo's line in an ``Anno_*`` list.

    Like `AlbumImage`, a typed view over the live dict rather than a copy,
    so an edit here is persisted by the next `AlbumInfo.save()` and fields
    this class doesn't model survive untouched.
    """

    def __init__(self, data: dict) -> None:
        self._d = data

    def __repr__(self) -> str:
        return f"<InfoEntry {self.key!r}>"

    @property
    def wire(self) -> dict:
        return self._d

    @property
    def key(self) -> str:
        """The album-wide bookkeeping key this entry refers to — the same
        value as `AlbumImage.key` (older albums store a plain filename)."""
        return self._d.get("src") or ""

    @key.setter
    def key(self, value: str) -> None:
        self._d["src"] = value

    @property
    def source_path(self) -> Optional[Path]:
        """Where this photo was imported from. Recorded per entry because
        one album can be built from several source directories, so the
        album-wide `AlbumInfo.src_dir` cannot resolve it on its own."""
        path = self._d.get("srcPath")
        return Path(path) if path else None


class AlbumInfo:
    """One album directory's ``info.json``, parsed on demand.

    Deliberately NOT cached the way `Album` is: info.json is small (a few
    hundred KB against album.json's tens of megabytes), so a per-caller
    parse costs nothing and the object can't go stale behind a writer's
    back.
    """

    def __init__(self, album_path: Union[str, Path]) -> None:
        self.path = Path(album_path)
        self.info_json = self.path / INFO_JSON_NAME
        self._payload: Optional[dict] = None

    def __repr__(self) -> str:
        return f"<AlbumInfo {self.path.name!r}>"

    # -- loading -------------------------------------------------------- #
    @property
    def exists(self) -> bool:
        return self.info_json.is_file()

    @property
    def payload(self) -> dict:
        """The whole parsed info.json (``{}`` if missing or unreadable)."""
        if self._payload is None:
            try:
                with open(self.info_json, encoding="utf-8") as fh:
                    self._payload = json.load(fh)
            except (json.JSONDecodeError, OSError):
                self._payload = {}
        return self._payload

    # -- provenance ----------------------------------------------------- #
    @property
    def src_dir(self) -> str:
        """The FIRST source directory ever imported into this album.

        Kept for consumers that predate multi-directory imports; prefer each
        entry's own `InfoEntry.source_path`, which is correct even when an
        album was built from several folders.
        """
        return self.payload.get("SrcDir") or ""

    @property
    def src_dirs(self) -> list[str]:
        """Every source directory imported into this album, in import order."""
        return list(self.payload.get("SrcDirs") or [])

    @property
    def src_type(self) -> str:
        """"File" or "Directory" — what the most recent import pointed at."""
        return self.payload.get("SrcType") or ""

    @property
    def timestamp(self) -> str:
        """When the album was first created (``yyyymmdd-hhmmss``)."""
        return self.payload.get("Timestamp") or ""

    @property
    def last_import_timestamp(self) -> str:
        return self.payload.get("LastImportTimestamp") or ""

    @property
    def our_jersey_color(self) -> Optional[str]:
        return self.payload.get("OurJerseyColor")

    @our_jersey_color.setter
    def our_jersey_color(self, value: Optional[str]) -> None:
        self.payload["OurJerseyColor"] = value

    def record_import(self, source_path: Union[str, Path], timestamp: str) -> None:
        """Note that *source_path* was imported at *timestamp*.

        Appends to `src_dirs` (ignoring a repeat of a directory already
        imported) and seeds `src_dir`/`timestamp` the first time only, so
        they keep meaning "the first import" across later merges.
        """
        source_path = Path(source_path)
        resolved = str(source_path.resolve())
        payload = self.payload
        src_dirs = list(payload.get("SrcDirs") or [])
        if resolved not in src_dirs:
            src_dirs.append(resolved)
        payload["SrcDirs"] = src_dirs
        payload["SrcDir"] = payload.get("SrcDir") or resolved
        payload["SrcType"] = "File" if source_path.is_file() else "Directory"
        payload["Timestamp"] = payload.get("Timestamp") or timestamp
        payload["LastImportTimestamp"] = timestamp

    # -- entries -------------------------------------------------------- #
    def entries(self, category: str) -> list[InfoEntry]:
        """Every photo in one bucket, in album order."""
        return [InfoEntry(item) for item in self.payload.get(_ANNO_KEY[category], [])]

    @property
    def all_entries(self) -> list[InfoEntry]:
        return [entry for category in CATEGORIES for entry in self.entries(category)]

    def count(self, category: str) -> int:
        return len(self.payload.get(_ANNO_KEY[category], []))

    @property
    def total(self) -> int:
        return sum(self.count(category) for category in CATEGORIES)

    def category_of(self, key: str) -> Optional[str]:
        """Which bucket *key* is currently in, or None if it isn't listed."""
        for category in CATEGORIES:
            if any(entry.key == key for entry in self.entries(category)):
                return category
        return None

    def add(self, category: str, key: str, source_path: Union[str, Path]) -> InfoEntry:
        """Append a newly-imported photo to a bucket."""
        item = {"src": key, "srcPath": str(source_path)}
        self.payload.setdefault(_ANNO_KEY[category], []).append(item)
        return InfoEntry(item)

    def move(self, key: str, category: str) -> bool:
        """Move *key* into *category*, preserving its recorded source path.

        Returns False when the key isn't listed at all (an album written
        before it existed, say) so a regrade can count what it actually
        moved instead of assuming.
        """
        for current in CATEGORIES:
            items = self.payload.get(_ANNO_KEY[current], [])
            for item in items:
                if item.get("src") != key:
                    continue
                if current == category:
                    return True
                items.remove(item)
                self.payload.setdefault(_ANNO_KEY[category], []).append(item)
                return True
        return False

    def rename_key(self, old_key: str, new_key: str, source_path: Union[str, Path]) -> bool:
        """Point the entry for *source_path* at *new_key*. Returns False if
        no such entry is listed.

        Matching on the source path as well as the old key is the whole
        point: at the moment `Album.import_from` renames a colliding key,
        TWO entries carry that key -- the photo already in the album and the
        one arriving from the staging album. Only the arriving one moves,
        and its source path is what tells them apart.

        Keeping album.json and info.json in step matters because consumers
        build a *set* of keys from info.json (see apply_export.py); two
        entries sharing one key collapse into a single member and the
        renamed photo is silently dropped from exports.
        """
        try:
            wanted = Path(source_path).resolve()
        except OSError:
            wanted = Path(source_path)
        for category in CATEGORIES:
            for item in self.payload.get(_ANNO_KEY[category], []):
                if item.get("src") != old_key:
                    continue
                recorded = item.get("srcPath")
                if not recorded:
                    continue
                try:
                    same = Path(recorded).resolve() == wanted
                except OSError:
                    same = Path(recorded) == wanted
                if same:
                    item["src"] = new_key
                    return True
        return False

    def replace_entries(self, entries_by_category: dict) -> None:
        """Replace all three buckets at once — for a deep regrade, which
        re-derives every photo's verdict from scratch rather than moving
        individual photos between buckets.

        *entries_by_category* maps a category to an iterable of
        ``(key, source_path)`` pairs.
        """
        for category in CATEGORIES:
            self.payload[_ANNO_KEY[category]] = [
                {"src": key, "srcPath": str(source_path)}
                for key, source_path in entries_by_category.get(category, ())
            ]

    # -- persistence ---------------------------------------------------- #
    # The historical field order, kept so a hand-read info.json looks the
    # same as it always has and diffs stay small. Anything not listed here
    # (a field from another version, say) is preserved and written after.
    _FIELD_ORDER = (
        "SrcDir", "SrcDirs", "SrcType", "Timestamp", "LastImportTimestamp",
        "OurJerseyColor", "Anno_Blur", "Anno_Sharp", "Anno_Skipped",
    )

    def save(self) -> None:
        """Atomically rewrite info.json, backing up the previous contents.

        All three ``Anno_*`` buckets are always emitted, even when empty --
        consumers have read them unguarded since before this class existed.
        """
        if self._payload is None:
            return
        for category in CATEGORIES:
            self._payload.setdefault(_ANNO_KEY[category], [])
        ordered = {k: self._payload[k] for k in self._FIELD_ORDER if k in self._payload}
        ordered.update({k: v for k, v in self._payload.items() if k not in ordered})
        atomic_save_and_backup(json.dumps(ordered, indent=4), self.info_json)
