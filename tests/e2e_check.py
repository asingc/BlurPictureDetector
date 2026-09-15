"""End-to-end verification of the typed-layer refactor against a real album.

Run after 1_prep_review.py has produced C:\\Temp\\bpd_e2e\\album. Exercises
the paths unit tests can't: a genuine multi-source import with a key
collision, a shallow regrade (algo/regrade.py), and the album.json
load/save round-trip against real pipeline output.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ALBUM = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Temp\bpd_e2e\album")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algo.album import Album  # noqa: E402
from algo.regrade import regrade_sensitivity  # noqa: E402

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {label}{f'  -- {detail}' if detail else ''}")
    if not condition:
        failures.append(label)


# --- the import produced what we expect ----------------------------------- #
album = Album(ALBUM)
images = album.image_list
check("album loaded", len(images) == 4, f"{len(images)} image(s)")
check("keys are unique", len({i.key for i in images}) == len(images))
check("collision was disambiguated",
      any(i.key != Path(i.source_file).name for i in images),
      ", ".join(i.key for i in images))
check("every photo has a verdict", all(i.status in ("sharp", "blurry", "skipped") for i in images))
check("bodies are typed", all(
    b.body_bbox is not None for i in images for b in i.bodies
))
check("sharpness is populated", all(
    i.sharpness_score is not None for i in images if i.bodies
))

# --- load/save against REAL pipeline output is byte-identical -------------- #
before = (ALBUM / "album.json").read_text(encoding="utf-8")
album.save()
check("round-trip is byte-identical", (ALBUM / "album.json").read_text(encoding="utf-8") == before)

# --- typed writes land where they should ---------------------------------- #
target = images[0]
target.stars = 4
target.stars_manual = True    # the user rated this one by hand
target.keep = False           # independent of stars, on purpose
target.bodies[0].player_name = "Test Player"
album.save()

reloaded = Album(ALBUM).image(target.key)
check("stars persisted", reloaded.stars == 4)
check("keep stayed independent of stars", reloaded.keep is False)
check("player name persisted on the person", reloaded.bodies[0].player_name == "Test Player")

# --- shallow regrade ------------------------------------------------------- #
snapshot = json.loads((ALBUM / "album.json").read_text(encoding="utf-8"))
statuses_before = {r["key"]: r["status"] for r in snapshot["results"]}

summary = regrade_sensitivity(ALBUM, 0.95)
print(f"      regrade: considered={summary.images_considered} recovered={summary.recovered} "
      f"demoted={summary.demoted} rebaselined={summary.stars_rebaselined} "
      f"previews={summary.previews_regenerated}/{summary.previews_regen_failed}")

after = Album(ALBUM)
statuses_after = {i.key: i.status for i in after.image_list}
check("regrade considered the graded photos", summary.images_considered > 0)
check("a strict threshold demotes at least one photo", summary.demoted > 0,
      f"{statuses_before} -> {statuses_after}")
check("regrade regenerated previews", summary.previews_regen_failed == 0,
      f"{summary.previews_regenerated} regenerated")
check("regrade persisted the new threshold", after.run_settings.get("sensitivity") == "0.95")
check("regrade left the hand-rated photo alone",
      after.image(target.key).stars == 4 and after.image(target.key).keep is False,
      "stars_manual must veto re-baselining")
check("regrade re-baselined everyone else",
      all(i.stars in (1, 2) for i in after.image_list if i.key != target.key),
      {i.key: i.stars for i in after.image_list})

# --- unknown fields survive the whole journey ------------------------------ #
album2 = Album(ALBUM)
entry = album2.image_list[0].entry
entry["some_legacy_field"] = {"kept": True}
album2.save()
check("unknown fields survive save",
      json.loads((ALBUM / "album.json").read_text(encoding="utf-8"))["results"][0]
      .get("some_legacy_field") == {"kept": True})

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("All end-to-end checks passed.")
