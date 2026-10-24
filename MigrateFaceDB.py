"""One-time migration of the flat system face DB to the per-team layout.

Old layout:   .FaceReco/<Player>/face.json ...
New layout:   .FaceReco/<team id>/<Player>/face.json ...

Each flat entry is copied into the folder of EVERY team whose roster contains
that person's name (case-insensitive), so a player who is on two teams ends up
as two independent entries. Entries whose name is on no roster (e.g. a coach)
are left untouched in the flat root and are not used by any team. Once an entry
has been copied everywhere it belongs, the flat original is removed (use
--keep-originals to retain it). calibration.json is copied into each team
folder that received entries.

A timestamped backup (.FaceReco.backup-<stamp>) is made first unless
--no-backup is given. Use --dry-run to only print what would happen.

Usage:
    python MigrateFaceDB.py [--root .FaceReco] [--team-json team.json]
                            [--dry-run] [--no-backup] [--keep-originals]

Team ids in team.json must already be UUIDs; start culling_app.py once if the
file still has pre-UUID ids (it upgrades them on load).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger("migrate_facedb")

REPO_ROOT = Path(__file__).resolve().parent
CALIBRATION_FILE = "calibration.json"


@dataclass
class MigrationReport:
    copied: list[tuple[str, str]] = field(default_factory=list)       # (entry folder, team id)
    skipped_existing: list[tuple[str, str]] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    backup: Path | None = None


def _is_uuid(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value.lower()
    except ValueError:
        return False


def _load_teams(team_json: Path) -> list[dict]:
    with open(team_json, encoding="utf-8") as fh:
        payload = json.load(fh)
    teams = payload.get("Teams") if isinstance(payload, dict) else None
    if not isinstance(teams, list) or not teams:
        raise ValueError(f"No teams found in {team_json}")
    bad = [t.get("name") or t.get("id") for t in teams if not _is_uuid(t.get("id"))]
    if bad:
        raise ValueError(
            "These teams still have a pre-UUID id: %s. Start culling_app.py once "
            "so team.json is upgraded, then re-run." % ", ".join(map(str, bad))
        )
    return teams


def _roster_names(team: dict) -> set[str]:
    return {
        (p.get("name") or "").strip().casefold()
        for p in team.get("players") or []
        if isinstance(p, dict) and (p.get("name") or "").strip()
    }


def _flat_entries(root: Path, team_ids: set[str]) -> list[Path]:
    """Person folders sitting directly under *root* (i.e. the legacy layout)."""
    return [
        child for child in sorted(root.iterdir())
        if child.is_dir()
        and child.name not in team_ids
        and not child.name.startswith((".", "Annotated."))
        and (child / "face.json").is_file()
    ]


def _entry_name(entry: Path) -> str:
    try:
        with open(entry / "face.json", encoding="utf-8-sig") as fh:
            name = json.load(fh).get("name")
    except (OSError, json.JSONDecodeError):
        name = None
    return (name or entry.name).strip()


def migrate(
    root: Path,
    teams: list[dict],
    *,
    dry_run: bool = False,
    backup: bool = True,
    keep_originals: bool = False,
) -> MigrationReport:
    report = MigrationReport()
    if not root.is_dir():
        raise FileNotFoundError(f"Face DB root not found: {root}")

    rosters = {t["id"]: _roster_names(t) for t in teams}
    entries = _flat_entries(root, set(rosters))
    if not entries:
        return report

    if backup and not dry_run:
        report.backup = root.with_name(f"{root.name}.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        shutil.copytree(root, report.backup)

    teams_touched: set[str] = set()
    for entry in entries:
        key = _entry_name(entry).casefold()
        owners = [tid for tid, names in rosters.items() if key in names]
        if not owners:
            report.unmatched.append(entry.name)
            continue
        for tid in owners:
            dest = root / tid / entry.name
            if dest.exists():
                report.skipped_existing.append((entry.name, tid))
                continue
            report.copied.append((entry.name, tid))
            teams_touched.add(tid)
            if not dry_run:
                shutil.copytree(entry, dest)
        if not keep_originals:
            report.removed.append(entry.name)
            if not dry_run:
                shutil.rmtree(entry)

    calibration = root / CALIBRATION_FILE
    if calibration.is_file() and not dry_run:
        for tid in teams_touched:
            if not (root / tid / CALIBRATION_FILE).exists():
                shutil.copy2(calibration, root / tid / CALIBRATION_FILE)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / ".FaceReco", help="Face DB root (default: ./.FaceReco)")
    parser.add_argument("--team-json", type=Path, default=REPO_ROOT / "team.json")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be done.")
    parser.add_argument("--no-backup", action="store_true", help="Skip the .FaceReco.backup-<stamp> copy.")
    parser.add_argument("--keep-originals", action="store_true", help="Do not delete flat entries after copying.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        report = migrate(
            args.root.resolve(), _load_teams(args.team_json),
            dry_run=args.dry_run, backup=not args.no_backup, keep_originals=args.keep_originals,
        )
    except (OSError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    prefix = "[dry-run] would " if args.dry_run else ""
    for name, tid in report.copied:
        log.info("%scopy %s -> %s/", prefix, name, tid)
    for name, tid in report.skipped_existing:
        log.info("skip %s (already in %s/)", name, tid)
    for name in report.unmatched:
        log.info("left in place (on no roster): %s", name)
    log.info("%s%d copied, %d already present, %d left unmatched, %d flat original(s) %s.",
             "[dry-run] " if args.dry_run else "", len(report.copied), len(report.skipped_existing),
             len(report.unmatched), len(report.removed), "to remove" if args.dry_run else "removed")
    if report.backup:
        log.info("Backup: %s", report.backup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
