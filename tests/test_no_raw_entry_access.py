"""Nobody reads album.json's per-photo data by key path any more.

The whole point of `AlbumImage`/`PersonRecord` is that the entry dict's
shape is known in exactly one place. This test is what stops that from
quietly eroding: a new `entry["annotation_data"]["evaluated"]` anywhere
outside the typed layer fails the build instead of being noticed a year
later when the shape changes.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Key paths that only algo/album.py may know about. Bracket/`.get()` forms
# only -- bare mentions in log messages and docstrings are fine.
#
# Deliberately limited to keys unique to album.json's per-photo entries.
# ``body_bbox`` and friends are excluded because .FaceReco's face.json and
# its manual-overrides file use the same names for their own records, so
# matching on them would flag code that has nothing to do with album.json.
FORBIDDEN = re.compile(
    r"""\[\s*["'](annotation_data|evaluated|stars_manual|burst_ranking"""
    r"""|burst_caption|llm_grade|sharpness_grade)["']\s*\]"""
    r"""|\.get\(\s*["'](annotation_data|evaluated|stars_manual|burst_ranking"""
    r"""|burst_caption|llm_grade)["']"""
)

# Files allowed to speak the wire format, each for a stated reason.
ALLOWED = {
    # The typed layer itself.
    "algo/album.py",
    # Dead legacy code (1_prep_review.py::analyse_image and friends, reached
    # only by the removed process() path) plus _merge_preserved_fields, which
    # belongs to deep regrade -- both explicitly out of scope for this
    # refactor. Remove from this list when they are.
    "1_prep_review.py",
    # The tests construct and assert on the wire format on purpose.
    "tests/conftest.py",
    "tests/test_album_roundtrip.py",
    "tests/test_consumer_semantics.py",
    "tests/test_results_serializer.py",
    "tests/test_no_raw_entry_access.py",
    "tests/e2e_check.py",
}

SKIP_DIRS = {".git", "__pycache__", "albums", "output", "gfpgan", "_setup_tmp", "test", "webui"}


def _python_files() -> list[Path]:
    return [
        p for p in REPO_ROOT.rglob("*.py")
        if not any(part in SKIP_DIRS for part in p.relative_to(REPO_ROOT).parts)
    ]


def test_no_raw_album_entry_access_outside_the_typed_layer():
    offenders: list[str] = []
    for path in _python_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWED:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Raw album.json key-path access found outside algo/album.py.\n"
        "Use AlbumImage/PersonRecord properties instead:\n  " + "\n  ".join(offenders)
    )


def test_the_allow_list_has_no_stale_entries():
    """An allow-listed file that no longer needs the exemption should be
    removed from the list, not left to hide a future regression."""
    stale = [
        rel for rel in ALLOWED
        if rel.startswith(("algo/", "1_prep"))
        and (REPO_ROOT / rel).is_file()
        and not any(FORBIDDEN.search(line) for line in (REPO_ROOT / rel).read_text(encoding="utf-8").splitlines())
    ]
    assert not stale, f"These files no longer need an exemption: {stale}"
