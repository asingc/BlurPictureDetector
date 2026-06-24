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
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from algo.facereco_provider import BodyRecord
from algo.facereco_provider import Box as FRBox
from algo.models import Box as ModelBox
from algo.models import Face, PredictedKeyPoint, Point
from algo.utils import _narrow_face_box

log = logging.getLogger("RebuildFaceDB")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FACE_MODEL_URL = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
)
_FACE_MODEL_PATH = Path(__file__).parent / "yolov8n-face.pt"

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
# Model bootstrap
# ---------------------------------------------------------------------------

def _ensure_face_model() -> Path:
    if _FACE_MODEL_PATH.exists():
        log.debug("Face model present: %s", _FACE_MODEL_PATH)
        return _FACE_MODEL_PATH
    log.info("Downloading yolov8n-face.pt from %s ...", _FACE_MODEL_URL)
    urllib.request.urlretrieve(_FACE_MODEL_URL, _FACE_MODEL_PATH)
    log.info("Download complete: %s", _FACE_MODEL_PATH)
    return _FACE_MODEL_PATH


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
# Two-pass face detection (mirrors image_analysis.py extract_faces)
# ---------------------------------------------------------------------------

def _detect_face_in_crop(
    image: np.ndarray,
    face_model: YOLO,
) -> tuple[FRBox, FRBox | None]:
    """Detect the face bbox and narrow landmark bbox within *image*.

    Uses the same two-pass strategy as image_analysis.py:
      Pass 1 — full image to find the bounding box.
      Pass 2 — padded crop of that box for refined landmark positions.

    Returns (face_bbox, narrow_face_bbox) as normalised FRBox values.
    Falls back to Box(0,0,1,1) with no narrow box when no face is found.
    """
    h, w = image.shape[:2]

    # --- Pass 1: detect on full image ---
    results = face_model.predict(image, verbose=False)
    if not results or len(results[0].boxes) == 0:
        log.debug("[detect] no face found in crop — using full-image fallback")
        return FRBox(0.0, 0.0, 1.0, 1.0), None

    boxes = results[0].boxes.xyxy.cpu().numpy()   # (N, 4) pixel coords
    confs = results[0].boxes.conf.cpu().numpy()   # (N,)

    # Pick the largest detection (crop images should have exactly one face).
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = int(np.argmax(areas))
    bx1, by1, bx2, by2 = boxes[best_idx]

    # Normalise to [0, 1].
    face_box_norm = ModelBox(
        max(0.0, bx1 / w), max(0.0, by1 / h),
        min(1.0, bx2 / w), min(1.0, by2 / h),
    )

    # --- Pass 2: re-run on a padded sub-crop for refined landmarks ---
    px1, py1 = max(0, int(bx1)), max(0, int(by1))
    px2, py2 = min(w, int(bx2)), min(h, int(by2))
    pad = max(10, int(max(px2 - px1, py2 - py1) * 0.2))
    cx1 = max(0, px1 - pad)
    cy1 = max(0, py1 - pad)
    cx2 = min(w, px2 + pad)
    cy2 = min(h, py2 + pad)
    face_crop = image[cy1:cy2, cx1:cx2]

    landmarks: list[PredictedKeyPoint] = []
    if face_crop.size > 0:
        crop_results = face_model.predict(face_crop, verbose=False)
        if (
            crop_results
            and len(crop_results[0].boxes) > 0
            and crop_results[0].keypoints is not None
        ):
            crop_boxes = crop_results[0].boxes.xyxy.cpu().numpy()
            crop_h, crop_w = face_crop.shape[:2]
            # Pick detection closest to the crop centre.
            cx_centres = (crop_boxes[:, 0] + crop_boxes[:, 2]) / 2
            cy_centres = (crop_boxes[:, 1] + crop_boxes[:, 3]) / 2
            dists = np.sqrt(
                (cx_centres - crop_w / 2) ** 2 + (cy_centres - crop_h / 2) ** 2
            )
            best_crop = int(np.argmin(dists))
            kps = crop_results[0].keypoints.data.cpu().numpy()[best_crop]  # (5, 3)
            # Translate crop-local coordinates back to full-image space.
            landmarks = [
                PredictedKeyPoint(
                    Point.from_px(int(kps[i, 0]) + cx1, int(kps[i, 1]) + cy1, w, h),
                    float(kps[i, 2]),
                )
                for i in range(len(kps))
            ]
            log.debug("[detect] pass-2: %d landmark(s)", len(landmarks))

    # Build a temporary Face to reuse the existing _narrow_face_box utility.
    model_face = Face(
        bbox=face_box_norm,
        confidence=float(confs[best_idx]),
        landmarks=landmarks,
    )
    narrow = _narrow_face_box(model_face, pad=0)

    fr_face = FRBox(face_box_norm.x1, face_box_norm.y1, face_box_norm.x2, face_box_norm.y2)
    fr_narrow = FRBox(narrow.x1, narrow.y1, narrow.x2, narrow.y2) if narrow else None
    return fr_face, fr_narrow


def _make_body(fr_face: FRBox, fr_narrow: FRBox | None) -> BodyRecord:
    return BodyRecord(
        orig_filename="",
        body_index=0,
        body_bbox=FRBox(0.0, 0.0, 1.0, 1.0),
        face_bbox=fr_face,
        narrow_face_bbox=fr_narrow,
        cloth_color="N/A",
        qualified_for_sharpness=True,
        is_blurry=False,
        confidence=None,
        raw_body={},
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
    face_model: YOLO,
    label: str,
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

        fr_face, fr_narrow = _detect_face_in_crop(image, face_model)
        body = _make_body(fr_face, fr_narrow)
        player = provider.predict_player(image, body)
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

def _rebuild_cluster(cluster_dir: Path, face_model: YOLO) -> None:
    face_json_path = cluster_dir / "face.json"
    if not face_json_path.exists():
        log.warning("No face.json in %s — skipping", cluster_dir.name)
        return

    with open(face_json_path, encoding="utf-8-sig") as fh:
        meta = json.load(fh)

    provider_name: str = meta.get("provider", "facenet")
    name: str = meta.get("name", "")
    playernum = meta.get("playernum", None)
    cluster: str = meta.get("cluster", cluster_dir.name)

    log.info("--- Cluster %s  name=%r  provider=%s ---", cluster_dir.name, name, provider_name)

    provider = _build_provider(provider_name)

    faces = _embed_directory(cluster_dir / "Face", provider, face_model, "positive")
    negatives = _embed_directory(cluster_dir / "Negative", provider, face_model, "negative")

    payload = {
        "name": name,
        "playernum": playernum,
        "provider": provider_name,
        "cluster": cluster,
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

    face_model_path = _ensure_face_model()
    log.info("Loading face model: %s", face_model_path)
    face_model = YOLO(str(face_model_path))

    ok = errors = 0
    for cluster_dir in cluster_dirs:
        try:
            _rebuild_cluster(cluster_dir, face_model)
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
