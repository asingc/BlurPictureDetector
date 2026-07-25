#!/usr/bin/env python3
"""RebuildFaceDB.py — Rebuild embeddings AND calibrate a .FaceReco/ face database.

One command, two steps, so there is only ever one script to run after curating
a face DB:

  STEP 1 — REBUILD.  For every cluster directory found under <facedb_dir>:
    1. Read face.json to get the provider name, person name, and player number.
    2. Run YOLOv8-face (two-pass, same strategy as 1_prep_review.py) on each
       crop in Face/ to detect the face bbox and refined landmarks, then call
       the provider to extract an embedding. Crops whose cropFileName already
       has an embedding in the previous face.json (same alignment setting)
       reuse it as-is instead of re-running detection — only genuinely NEW
       crops pay the embedding cost.
    3. Do the same for every crop in Negative/.
    4. Rewrite face.json with only what matters:
         { name, playernum, provider, cluster, faces: [...], negative_faces: [...] }
       Each entry contains just cropFileName + embedding (base64 float32).
       Stale Body objects and original file paths are dropped.

  STEP 2 — RETIRE REDUNDANT FACES.  After embedding every positive crop in a
  cluster's Face/ directory, group them into visually-cohesive "look"
  prototypes and keep each prototype's medoid unconditionally (so no
  distinct look is ever lost), plus any other crop that isn't a
  near-duplicate (cosine similarity >= --redundancy-threshold) of something
  already kept. Crops that don't survive are MOVED (never deleted) to
  Face/Retired/ and dropped from face.json, so they're automatically
  excluded from future rebuilds too. Negative/ crops are left untouched.

  STEP 3 — CALIBRATE (runs automatically afterwards, pass --skip-calibration
  to opt out).  Loads the just-rebuilt DB and runs leave-one-out cross-
  validation: every photo is temporarily held out, scored against its own
  person's remaining photos (genuine similarity) and against every other
  person (impostor similarity) -- exactly what a real query faces at match
  time.  Reports a recommended face_db_match_threshold / face_db_match_margin
  and a CONFUSIONS list naming any specific photo that scores closer to a
  DIFFERENT person than to its own -- the fastest way to spot a mislabeled
  crop or a genuine look-alike pair before it causes a wrong match.

Usage:
    python RebuildFaceDB.py <facedb_dir> [--skip-calibration] [--redundancy-threshold 0.93]
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from algo.face_crop_embed import annotate_face_crop, embed_face_crop, load_face_model
from algo.facereco import (
    FaceDb,
    FaceDbEntry,
    FaceRecoConfig,
    normalize_embedding,
    build_prototypes,
    select_useful_subset,
    _decode_embedding,
)

log = logging.getLogger("RebuildFaceDB")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    log.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(handler)


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def _build_provider(provider_name: str):
    if provider_name == "facenet":
        try:
            from algo.facenet_provider import FaceNetFaceRecoProvider
            return FaceNetFaceRecoProvider()
        except ImportError as exc:
            raise SystemExit(f"facenet-pytorch is not installed: {exc}") from exc
    if provider_name == "dlib":
        try:
            from algo.dlib_provider import DlibFaceRecoProvider
            return DlibFaceRecoProvider()
        except ImportError as exc:
            raise SystemExit(f"face-recognition/dlib is not installed: {exc}") from exc
    raise SystemExit(
        f"Unknown provider '{provider_name}' in face.json. Expected 'facenet' or 'dlib'."
    )


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _serialize_embedding(emb: np.ndarray) -> dict:
    emb = emb.astype(np.float32)
    return {
        "dtype": "float32",
        "shape": [int(emb.shape[0])],
        "encoding": "base64",
        "value": base64.b64encode(emb.tobytes()).decode("ascii"),
    }


def _embed_directory(
    image_dir: Path,
    annotated_dir: Path,
    provider,
    face_model,
    label: str,
    align: bool,
    existing_by_crop: dict[str, dict] | None = None,
) -> list[dict]:
    """Compute embeddings for every image in *image_dir*.

    *existing_by_crop* maps cropFileName -> the previous face.json entry for
    that crop (see ``_rebuild_cluster``). When a crop's name is found there
    with a usable embedding, that embedding is reused as-is instead of
    re-running detection+embedding on it — a rebuild then only pays the
    (relatively expensive) landmark+embedding cost for genuinely NEW crops,
    not every crop that was already embedded in a previous run.

    Returns a list of { cropFileName, dtype, shape, encoding, value } dicts.
    """
    if not image_dir.is_dir():
        log.debug("[%s] directory missing: %s", label, image_dir)
        return []

    images = sorted(
        f for f in image_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS
    )
    log.info("[%s] %d image(s) in %s", label, len(images), image_dir.name)

    existing_by_crop = existing_by_crop or {}
    records: list[dict] = []
    ok = skipped = reused = 0
    for img_path in images:
        cached = existing_by_crop.get(img_path.name)
        if cached is not None and cached.get("value"):
            records.append(cached)
            reused += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            log.warning("[%s] cannot read %s — skipped", label, img_path.name)
            skipped += 1
            continue

        player, debug = embed_face_crop(
            provider,
            face_model,
            image,
            align=align,
            collect_debug=True,
        )
        emb = player.internal.get("embedding")
        if emb is None:
            log.warning("[%s] no embedding returned for %s — skipped", label, img_path.name)
            skipped += 1
            continue

        annotated_dir.mkdir(parents=True, exist_ok=True)
        annotated = annotate_face_crop(
            image,
            debug.get("face_bbox"),
            debug.get("narrow_face_bbox"),
            debug.get("landmarks_px", []),
            player.confidence,
        )
        cv2.imwrite(
            str(annotated_dir / img_path.name),
            annotated,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )

        emb_arr = np.asarray(emb, dtype=np.float32)
        records.append({"cropFileName": img_path.name, **_serialize_embedding(emb_arr)})
        ok += 1
        log.debug("[%s] %s  dim=%d  norm=%.4f", label, img_path.name,
                  emb_arr.shape[0], float(np.linalg.norm(emb_arr)))

    log.info("[%s] done: new=%d  reused=%d  skipped=%d", label, ok, reused, skipped)
    return records


# ---------------------------------------------------------------------------
# Per-cluster rebuild
# ---------------------------------------------------------------------------

def _retire_redundant_faces(
    face_dir: Path,
    faces: list[dict],
    redundancy_threshold: float,
    prototype_threshold: float,
) -> list[dict]:
    """Move positive face crops that don't meaningfully expand this person's
    visual coverage into Face/Retired/, and drop them from the embeddings
    returned (so they're excluded from the rewritten face.json). Retired
    crops are kept on disk -- never deleted -- and automatically excluded
    from future rebuilds since Retired/ isn't scanned by _embed_directory().
    """
    if len(faces) <= 1:
        return faces

    decoded = [(f, _decode_embedding(f)) for f in faces]
    usable = [(f, e) for f, e in decoded if e is not None]
    if len(usable) <= 1:
        return faces

    kept_idx = set(select_useful_subset(
        [e for _, e in usable],
        redundancy_threshold=redundancy_threshold,
        prototype_threshold=prototype_threshold,
    ))

    kept_faces: list[dict] = []
    retired_names: list[str] = []
    retired_dir = face_dir / "Retired"
    for i, (f, _e) in enumerate(usable):
        if i in kept_idx:
            kept_faces.append(f)
            continue
        crop_name = f.get("cropFileName", "")
        src = face_dir / crop_name
        if src.is_file():
            retired_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(retired_dir / crop_name))
        retired_names.append(crop_name)

    if retired_names:
        log.info("  retired %d redundant face(s) -> %s: %s",
                 len(retired_names), retired_dir, ", ".join(retired_names))

    return kept_faces


def _rebuild_cluster(
    cluster_dir: Path,
    face_model,
    annotated_root: Path,
    align: bool,
    redundancy_threshold: float,
    prototype_threshold: float,
) -> None:
    face_json_path = cluster_dir / "face.json"
    if not face_json_path.exists():
        log.warning("No face.json in %s — skipping", cluster_dir.name)
        return

    with open(face_json_path, encoding="utf-8-sig") as fh:
        meta = json.load(fh)

    provider_name: str = meta.get("provider", "facenet")
    name: str = meta.get("name", "")
    playernum = meta.get("playernum", None)
    player_dir_name = name or cluster_dir.name

    log.info("--- name=%r  provider=%s ---",  name, provider_name)

    provider = _build_provider(provider_name)

    # Reuse embeddings from the previous face.json for crops that are still
    # present (keyed by cropFileName) -- but only when the previous run used
    # the same alignment setting, since alignment changes the embedding.
    reuse_cache = bool(meta.get("aligned")) == bool(align)
    existing_faces_by_crop = (
        {f["cropFileName"]: f for f in meta.get("faces", []) if f.get("cropFileName")}
        if reuse_cache else {}
    )
    existing_negatives_by_crop = (
        {f["cropFileName"]: f for f in meta.get("negative_faces", []) if f.get("cropFileName")}
        if reuse_cache else {}
    )

    player_annotated_dir = annotated_root / player_dir_name
    faces = _embed_directory(
        cluster_dir / "Face",
        player_annotated_dir / "Face",
        provider,
        face_model,
        "positive",
        align=align,
        existing_by_crop=existing_faces_by_crop,
    )
    faces_before = len(faces)
    faces = _retire_redundant_faces(
        cluster_dir / "Face", faces, redundancy_threshold, prototype_threshold,
    )
    negatives = _embed_directory(
        cluster_dir / "Negative",
        player_annotated_dir / "Negative",
        provider,
        face_model,
        "negative",
        align=align,
        existing_by_crop=existing_negatives_by_crop,
    )

    payload = {
        "name": name,
        "playernum": playernum,
        "provider": provider_name,
        "aligned": align,
        "faces": faces,
        "negative_faces": negatives,
    }

    with open(face_json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    log.info(
        "Wrote face.json for cluster %s: %d positive (%d retired), %d negative",
        cluster_dir.name, len(faces), faces_before - len(faces), len(negatives),
    )


# ---------------------------------------------------------------------------
# Calibration (leave-one-out cross-validation over the rebuilt DB)
# ---------------------------------------------------------------------------

def _leave_one_out(
    entries: list[FaceDbEntry], prototype_threshold: float,
) -> tuple[
    list[tuple[float, str]],
    list[tuple[float, str, str]],
    list[tuple[str, int, float, str, float]],
]:
    """Score every DB photo against the DB as if it were a new query.

    Returns ``(genuine, impostor, confusions)``:
      genuine    — (similarity, person_name) for each held-out photo scored
                   against its OWN person's remaining prototypes.
      impostor   — (similarity, person_name, other_person_name) for the SAME
                   held-out photo scored against the best-matching different
                   person (its real competitor at match time).
      confusions — the subset where the impostor score >= the genuine score,
                   i.e. a real query identical to this photo would have been
                   matched to the WRONG person (or rejected outright).
    """
    genuine: list[tuple[float, str]] = []
    impostor: list[tuple[float, str, str]] = []
    confusions: list[tuple[str, int, float, str, float]] = []

    for i, entry in enumerate(entries):
        others = entries[:i] + entries[i + 1:]
        for idx in range(len(entry.embeddings)):
            rest = entry.embeddings[:idx] + entry.embeddings[idx + 1:]
            if not rest:
                # Sole photo of this person -- nothing to self-validate against.
                continue
            held_out = normalize_embedding(entry.embeddings[idx].astype(np.float32))
            rest_protos = build_prototypes(rest, prototype_threshold)
            genuine_sim = max(float(np.dot(held_out, p.centroid)) for p in rest_protos)

            best_other_sim = -1.0
            best_other_name = ""
            for other in others:
                if not other.prototypes:
                    continue
                sim = max(float(np.dot(held_out, p.centroid)) for p in other.prototypes)
                if sim > best_other_sim:
                    best_other_sim = sim
                    best_other_name = other.name

            genuine.append((genuine_sim, entry.name))
            impostor.append((best_other_sim, entry.name, best_other_name))
            if best_other_sim >= genuine_sim:
                confusions.append((entry.name, idx, genuine_sim, best_other_name, best_other_sim))

    return genuine, impostor, confusions


def _rates_at_threshold(
    genuine: list[tuple[float, str]], impostor: list[tuple[float, str, str]], threshold: float,
) -> tuple[float, float]:
    """Return (false_accept_rate, false_reject_rate) at *threshold*."""
    frr = sum(1 for sim, _ in genuine if sim < threshold) / len(genuine) if genuine else 0.0
    far = sum(1 for sim, _, _ in impostor if sim >= threshold) / len(impostor) if impostor else 0.0
    return far, frr


def _find_eer(
    genuine: list[tuple[float, str]], impostor: list[tuple[float, str, str]],
) -> tuple[float, float]:
    """Sweep thresholds and return (threshold, error_rate) nearest equal-error-rate."""
    candidates = sorted({round(s, 4) for s, _ in genuine} | {round(s, 4) for s, _, _ in impostor})
    best_threshold, best_gap, best_rate = 0.5, float("inf"), 0.0
    for t in candidates:
        far, frr = _rates_at_threshold(genuine, impostor, t)
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap, best_threshold, best_rate = gap, t, (far + frr) / 2
    return best_threshold, best_rate


def _calibrate_entries(entries: list[FaceDbEntry], prototype_threshold: float) -> dict:
    """Run leave-one-out calibration over *entries* and return recommendations.

    Returns a JSON-serializable dict with recommended thresholds and summary
    stats. The same data is logged for humans and persisted by
    :func:`_run_calibration` so future recognition runs can consume it
    automatically.
    """
    total_embeddings = sum(len(e.embeddings) for e in entries)
    log.info("People: %d   Embeddings: %d   Prototype threshold: %.3f",
              len(entries), total_embeddings, prototype_threshold)

    genuine, impostor, confusions = _leave_one_out(entries, prototype_threshold)
    skipped = total_embeddings - len(genuine)
    log.info("Leave-one-out samples: %d usable (%d skipped — sole photo of their person)",
              len(genuine), skipped)
    if not genuine:
        log.warning("No leave-one-out samples available (every person has only 1 photo) — cannot calibrate.")
        return {
            "people": len(entries),
            "embeddings": total_embeddings,
            "usable_samples": len(genuine),
            "skipped_samples": skipped,
            "recommended_match_threshold": None,
            "recommended_match_margin": None,
            "prototype_threshold": prototype_threshold,
            "confusions_count": 0,
            "notes": "insufficient leave-one-out samples",
        }

    genuine_sims = np.array([s for s, _ in genuine])
    impostor_sims = np.array([s for s, _, _ in impostor])
    log.info("")
    log.info("Genuine  similarity:  mean=%.4f  median=%.4f  min=%.4f  p5=%.4f",
              genuine_sims.mean(), float(np.median(genuine_sims)), genuine_sims.min(),
              float(np.percentile(genuine_sims, 5)))
    log.info("Impostor similarity:  mean=%.4f  median=%.4f  max=%.4f  p95=%.4f",
              impostor_sims.mean(), float(np.median(impostor_sims)), impostor_sims.max(),
              float(np.percentile(impostor_sims, 95)))

    eer_threshold, eer_rate = _find_eer(genuine, impostor)
    far, frr = _rates_at_threshold(genuine, impostor, eer_threshold)
    log.info("")
    log.info("Recommended --face-db-match-threshold ~= %.3f  (equal-error-rate ~= %.1f%%)",
              eer_threshold, eer_rate * 100)
    log.info("  at this threshold: false-accept rate=%.1f%%  false-reject rate=%.1f%%", far * 100, frr * 100)

    margins = np.array([g[0] - i[0] for g, i in zip(genuine, impostor)])
    p5_margin = float(np.percentile(margins, 5))
    recommended_margin = max(0.0, round(p5_margin, 3)) if p5_margin > 0 else 0.0
    log.info("")
    log.info("Genuine-vs-runner-up margin:  mean=%.4f  median=%.4f  p5=%.4f",
              margins.mean(), float(np.median(margins)), p5_margin)
    log.info("Recommended --face-db-match-margin ~= %.3f", recommended_margin)
    if p5_margin < 0:
        log.warning(
            "  WARNING: at least 5%% of genuine photos were CLOSER to a different person "
            "than to their own remaining photos -- see CONFUSIONS below before trusting "
            "any margin."
        )

    log.info("")
    log.info("=" * 78)
    if confusions:
        log.info("CONFUSIONS — %d photo(s) where a DIFFERENT person scored >= the true person",
                  len(confusions))
        log.info("These are the most likely root cause of 'faces mixed up between people':")
        log.info("either the two people genuinely look alike, or one of the photos below is")
        log.info("mislabeled / sitting in the wrong person's Face/ folder.")
        log.info("=" * 78)
        for name, idx, genuine_sim, other_name, other_sim in sorted(confusions, key=lambda c: c[2] - c[4]):
            log.info(
                "  %-20s embedding #%-3d  self_sim=%.4f   vs  %-20s other_sim=%.4f",
                name, idx, genuine_sim, other_name, other_sim,
            )
    else:
        log.info("No confusions found — every held-out photo was closer to its own person "
                  "than to anyone else.")
    log.info("=" * 78)

    log.info("")
    log.info("=" * 78)
    log.info("PAIRWISE PROTOTYPE OVERLAP — closest prototype-to-prototype similarity per pair")
    log.info("High values here (near the recommended threshold) mean two people are hard to")
    log.info("tell apart even with clean data; review crops for those pairs.")
    log.info("=" * 78)
    pair_scores: list[tuple[float, str, str]] = []
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            if not a.prototypes or not b.prototypes:
                continue
            best = max(
                float(np.dot(pa.centroid, pb.centroid))
                for pa in a.prototypes for pb in b.prototypes
            )
            pair_scores.append((best, a.name, b.name))
    pair_scores.sort(reverse=True)
    for sim, a_name, b_name in pair_scores[:15]:
        log.info("  %-20s <-> %-20s   %.4f", a_name, b_name, sim)

    log.info("")
    log.info("Suggested FaceRecoConfig for provider used above:")
    log.info("  face_db_match_threshold     = %.3f", eer_threshold)
    log.info("  face_db_match_margin        = %.3f", recommended_margin)
    log.info("  face_db_prototype_threshold = %.3f  (as used for this run)", prototype_threshold)

    return {
        "people": len(entries),
        "embeddings": total_embeddings,
        "usable_samples": len(genuine),
        "skipped_samples": skipped,
        "recommended_match_threshold": float(eer_threshold),
        "recommended_match_margin": float(recommended_margin),
        "prototype_threshold": float(prototype_threshold),
        "equal_error_rate": float(eer_rate),
        "far_at_recommended": float(far),
        "frr_at_recommended": float(frr),
        "genuine_mean": float(genuine_sims.mean()),
        "impostor_mean": float(impostor_sims.mean()),
        "confusions_count": len(confusions),
    }


def _run_calibration(facedb_dir: Path, prototype_threshold: float) -> dict:
    """Reload the just-rebuilt DB and run leave-one-out calibration, grouped by provider.

    Grouped by provider because embeddings from different providers do not
    live in the same space -- mixing them would produce meaningless genuine/
    impostor distributions.
    """
    face_db = FaceDb.load(facedb_dir, prototype_similarity_threshold=prototype_threshold)
    if not face_db.entries:
        log.warning("[calibrate] no embeddings found in %s — skipping calibration", facedb_dir)
        return {"providers": {}}

    by_provider: dict[str, list[FaceDbEntry]] = {}
    for entry in face_db.entries:
        by_provider.setdefault(entry.provider, []).append(entry)

    provider_reports: dict[str, dict] = {}
    for provider_name, entries in by_provider.items():
        log.info("")
        log.info("#" * 78)
        log.info("CALIBRATION — provider=%s", provider_name)
        log.info("#" * 78)
        if len(entries) < 2:
            log.warning(
                "[calibrate] need >=2 people with embeddings to calibrate provider '%s' "
                "(found %d) — skipped", provider_name, len(entries),
            )
            provider_reports[provider_name] = {
                "people": len(entries),
                "embeddings": sum(len(e.embeddings) for e in entries),
                "recommended_match_threshold": None,
                "recommended_match_margin": None,
                "prototype_threshold": float(prototype_threshold),
                "notes": "need >= 2 people",
            }
            continue
        provider_reports[provider_name] = _calibrate_entries(entries, prototype_threshold)

    return {"providers": provider_reports}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Rebuild embeddings in a .FaceReco/ face database from the crop images, "
                    "then calibrate matching thresholds against it."
    )
    parser.add_argument(
        "facedb_dir",
        metavar="<facedb_dir>",
        help="Path to the .FaceReco/ directory (contains one subdirectory per person).",
    )
    parser.add_argument(
        "--prototype-threshold",
        type=float,
        default=0.62,
        help=(
            "Cosine similarity used to split each person's embeddings into "
            "visual sub-clusters for calibration (default: 0.62). Use the same "
            "value you plan to run FaceRecoConfig.face_db_prototype_threshold with."
        ),
    )
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=0.93,
        help=(
            "Cosine similarity above which a positive face crop is considered a "
            "near-duplicate of one already kept, and gets retired to Face/Retired/ "
            "(default: 0.93). Every visual 'look' prototype always keeps at least "
            "one crop regardless of this threshold."
        ),
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Only rebuild embeddings; skip the leave-one-out calibration report that runs afterwards by default.",
    )
    parser.add_argument(
        "--engine",
        choices=("mediapipe", "yolo"),
        default="mediapipe",
        help=(
            "Detector engine used for the per-crop face-landmark refinement "
            "pass (default: mediapipe). Must match the engine used at "
            "prediction time (1_prep_review.py --engine) so embeddings live "
            "in the same space."
        ),
    )
    args = parser.parse_args()

    facedb_dir = Path(args.facedb_dir).resolve()
    if not facedb_dir.is_dir():
        log.error("Not a directory: %s", facedb_dir)
        sys.exit(1)

    cluster_dirs = sorted(d for d in facedb_dir.iterdir() if d.is_dir())
    if not cluster_dirs:
        log.warning("No cluster subdirectories found in %s", facedb_dir)
        sys.exit(0)

    log.info("Face DB : %s", facedb_dir)
    log.info("Clusters: %d", len(cluster_dirs))
    align_faces = FaceRecoConfig().align_faces
    log.info("Face alignment: %s", "ENABLED" if align_faces else "disabled")

    face_model = load_face_model(force_cpu=False, engine=args.engine)
    annotated_root = facedb_dir / datetime.now().strftime("Annotated.%y%m%d-%H%M%S")

    ok = errors = 0
    for cluster_dir in cluster_dirs:
        try:
            _rebuild_cluster(
                cluster_dir, face_model, annotated_root, align=align_faces,
                redundancy_threshold=args.redundancy_threshold,
                prototype_threshold=args.prototype_threshold,
            )
            ok += 1
        except Exception as exc:
            log.error(
                "Failed to rebuild cluster %s: %s", cluster_dir.name, exc, exc_info=True
            )
            errors += 1

    log.info("Done: %d cluster(s) rebuilt, %d error(s)", ok, errors)
    log.info("Annotated rebuild crops: %s", annotated_root)

    if args.skip_calibration:
        log.info("--skip-calibration set: not running the calibration report.")
    else:
        log.info("")
        try:
            report = _run_calibration(facedb_dir, args.prototype_threshold)
            report_payload = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "source": "RebuildFaceDB.py",
                "prototype_threshold": float(args.prototype_threshold),
                "providers": report.get("providers", {}),
            }
            report_path = facedb_dir / "calibration.json"
            with open(report_path, "w", encoding="utf-8") as fh:
                json.dump(report_payload, fh, indent=2)
            log.info("Calibration report written: %s", report_path)
        except Exception as exc:
            log.error("Calibration failed: %s", exc, exc_info=True)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()

