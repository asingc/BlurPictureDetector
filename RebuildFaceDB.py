#!/usr/bin/env python3
"""RebuildFaceDB.py — Rebuild embeddings in a .FaceReco/ face database.

For every cluster directory found under <facedb_dir>:
  1. Read face.json to get the provider name, person name, and player number.
  2. Run YOLOv8-face (two-pass, same strategy as 1_prep_review.py) on each
     crop in Face/ to detect the face bbox and refined landmarks, then call
     the provider to extract an embedding.
  3. Do the same for every crop in Negative/.
  4. Rewrite face.json with only what matters:
       { name, playernum, provider, cluster, faces: [...], negative_faces: [...] }
     Each entry contains just cropFileName + embedding (base64 float32).
     Stale Body objects and original file paths are dropped.

Usage:
    python RebuildFaceDB.py <facedb_dir>
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from algo.face_crop_embed import embed_face_crop, load_face_model

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
    provider,
    face_model,
    label: str,
    align: bool = False,
) -> list[dict]:
    """Compute embeddings for every image in *image_dir*.

    Returns a list of { cropFileName, embedding } dicts.
    """
    if not image_dir.is_dir():
        log.debug("[%s] directory missing: %s", label, image_dir)
        return []

    images = sorted(
        f for f in image_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTENSIONS
    )
    log.info("[%s] %d image(s) in %s", label, len(images), image_dir.name)

    records: list[dict] = []
    ok = skipped = 0
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            log.warning("[%s] cannot read %s — skipped", label, img_path.name)
            skipped += 1
            continue

        player = embed_face_crop(provider, face_model, image, align=align)
        emb = player.internal.get("embedding")
        if emb is None:
            log.warning("[%s] no embedding returned for %s — skipped", label, img_path.name)
            skipped += 1
            continue

        emb_arr = np.asarray(emb, dtype=np.float32)
        records.append(_serialize_embedding(emb_arr))
        ok += 1
        log.debug("[%s] %s  dim=%d  norm=%.4f", label, img_path.name,
                  emb_arr.shape[0], float(np.linalg.norm(emb_arr)))

    log.info("[%s] done: ok=%d  skipped=%d", label, ok, skipped)
    return records


# ---------------------------------------------------------------------------
# Per-cluster rebuild
# ---------------------------------------------------------------------------

def _rebuild_cluster(cluster_dir: Path, face_model, align: bool = False) -> None:
    face_json_path = cluster_dir / "face.json"
    if not face_json_path.exists():
        log.warning("No face.json in %s — skipping", cluster_dir.name)
        return

    with open(face_json_path, encoding="utf-8-sig") as fh:
        meta = json.load(fh)

    provider_name: str = meta.get("provider", "facenet")
    name: str = meta.get("name", "")
    playernum = meta.get("playernum", None)

    log.info("--- name=%r  provider=%s ---",  name, provider_name)

    provider = _build_provider(provider_name)

    faces = _embed_directory(cluster_dir / "Face", provider, face_model, "positive", align=align)
    negatives = _embed_directory(cluster_dir / "Negative", provider, face_model, "negative", align=align)

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
        "Wrote face.json for cluster %s: %d positive, %d negative",
        cluster_dir.name, len(faces), len(negatives),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Rebuild embeddings in a .FaceReco/ face database from the crop images."
    )
    parser.add_argument(
        "facedb_dir",
        metavar="<facedb_dir>",
        help="Path to the .FaceReco/ directory (contains one subdirectory per person).",
    )
    parser.add_argument(
        "--align-faces",
        action="store_true",
        help=(
            "Similarity-align each crop to a canonical 5-point template before "
            "embedding.  Must match the setting used at prediction time "
            "(4_face_reco.py / 1_prep_review.py --align-faces)."
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
    log.info("Face alignment: %s", "ENABLED" if args.align_faces else "disabled")

    face_model = load_face_model()

    ok = errors = 0
    for cluster_dir in cluster_dirs:
        try:
            _rebuild_cluster(cluster_dir, face_model, align=args.align_faces)
            ok += 1
        except Exception as exc:
            log.error(
                "Failed to rebuild cluster %s: %s", cluster_dir.name, exc, exc_info=True
            )
            errors += 1

    log.info("Done: %d cluster(s) rebuilt, %d error(s)", ok, errors)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
