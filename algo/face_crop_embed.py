"""Shared face-crop landmark detection and embedding.

This module is the single source of truth for turning an *already-cropped*
face image into an embedding.  It is used both at prediction time
(``algo/facereco.py``) and when rebuilding the face database
(``RebuildFaceDB.py``) so that the two pipelines are guaranteed to behave
identically: crop → re-detect landmarks on the crop → embed from the crop.

Keeping this logic in one place is what makes a prediction-time embedding
comparable against a database embedding for the same face.
"""

from __future__ import annotations

import logging
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from algo.facereco_provider import Box as FRBox
from algo.facereco_provider import BodyRecord, FaceRecoProvider, Player
from algo.models import Box as ModelBox
from algo.models import Face, PredictedKeyPoint, Point
from algo.utils import _narrow_face_box

log = logging.getLogger("BlurPictureDetector")

_FACE_MODEL_URL = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
)
# Repository root is the parent of the ``algo`` package directory.
_FACE_MODEL_PATH = Path(__file__).resolve().parent.parent / "yolov8n-face.pt"

# Canonical 5-point face template (ArcFace / InsightFace), defined for a
# 112x112 output and ordered: left-eye, right-eye, nose, left-mouth,
# right-mouth — the same order the YOLOv8-face model emits its keypoints.
_ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
# Output size of the aligned face.  160 matches FaceNet's expected input.
_ALIGN_OUTPUT_SIZE = 160

# Cached YOLO face model (loaded lazily so importing this module stays cheap
# and does not require ultralytics until a model is actually needed).
_FACE_MODEL = None
_FACE_MODEL_LOCK = threading.Lock()


def ensure_face_model() -> Path:
    """Return the path to ``yolov8n-face.pt``, downloading it if missing."""
    if _FACE_MODEL_PATH.exists():
        log.debug("Face model present: %s", _FACE_MODEL_PATH)
        return _FACE_MODEL_PATH
    log.info("Downloading yolov8n-face.pt from %s ...", _FACE_MODEL_URL)
    urllib.request.urlretrieve(_FACE_MODEL_URL, _FACE_MODEL_PATH)
    log.info("Download complete: %s", _FACE_MODEL_PATH)
    return _FACE_MODEL_PATH


def load_face_model():
    """Return the process-wide YOLOv8-face model singleton.

    The model is loaded once on first call and reused thereafter, so repeated
    calls (across pipelines, clusters, or rebuild runs) never reload it.
    Thread-safe via double-checked locking.
    """
    global _FACE_MODEL
    if _FACE_MODEL is not None:
        return _FACE_MODEL
    with _FACE_MODEL_LOCK:
        if _FACE_MODEL is not None:  # another thread may have loaded it
            return _FACE_MODEL
        from ultralytics import YOLO  # imported lazily — heavy dependency

        path = ensure_face_model()
        log.info("Loading face model: %s", path)
        _FACE_MODEL = YOLO(str(path))
    return _FACE_MODEL


def detect_face_in_crop(
    image: np.ndarray, face_model
) -> tuple[FRBox, FRBox | None, list[tuple[float, float]]]:
    """Detect the face bbox, narrow landmark bbox, and 5 landmarks in *image*.

    Uses the same two-pass strategy as ``image_analysis.extract_faces``:
      Pass 1 — full image to find the bounding box.
      Pass 2 — padded crop of that box for refined landmark positions.

    Returns ``(face_bbox, narrow_face_bbox, landmarks_px)`` where the boxes are
    normalised :class:`FRBox` values and ``landmarks_px`` is the list of 5
    ``(x, y)`` landmark coordinates in *image* pixel space (empty when no
    landmarks were detected).  Falls back to ``Box(0, 0, 1, 1)`` with no narrow
    box and no landmarks when no face is found.
    """
    h, w = image.shape[:2]

    # --- Pass 1: detect on full image ---
    results = face_model.predict(image, verbose=False)
    if not results or len(results[0].boxes) == 0:
        log.debug("[detect] no face found in crop — using full-image fallback")
        return FRBox(0.0, 0.0, 1.0, 1.0), None, []

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
    landmarks_px: list[tuple[float, float]] = []
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
            # Landmark coordinates in the full *image* pixel space.
            landmarks_px = [
                (float(kps[i, 0]) + cx1, float(kps[i, 1]) + cy1)
                for i in range(len(kps))
            ]
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
    return fr_face, fr_narrow, landmarks_px


def align_face(
    crop: np.ndarray,
    landmarks_px: list[tuple[float, float]],
    output_size: int = _ALIGN_OUTPUT_SIZE,
) -> np.ndarray | None:
    """Similarity-align *crop* to the canonical 5-point template.

    Computes a similarity transform (rotation + uniform scale + translation)
    that maps the 5 detected landmarks onto :data:`_ARCFACE_TEMPLATE_112`
    (scaled to *output_size*) and warps the crop accordingly, producing an
    upright, canonically-framed ``output_size``x``output_size`` face.

    Returns ``None`` when fewer than 5 valid landmarks are available, so the
    caller can fall back to the unaligned crop.
    """
    if landmarks_px is None or len(landmarks_px) < 5:
        return None
    src = np.asarray(landmarks_px[:5], dtype=np.float32)
    if not np.all(np.isfinite(src)):
        return None
    dst = _ARCFACE_TEMPLATE_112 * (output_size / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        return None
    return cv2.warpAffine(
        crop, matrix, (output_size, output_size), flags=cv2.INTER_LINEAR
    )


def make_crop_body(fr_face: FRBox, fr_narrow: FRBox | None) -> BodyRecord:
    """Build a :class:`BodyRecord` describing a single isolated face crop."""
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


def embed_face_crop(
    provider: FaceRecoProvider,
    face_model,
    crop: np.ndarray,
    fallback_confidence: float | None = None,
    align: bool = False,
    collect_debug: bool = False,
):
    """Re-detect landmarks on *crop* and return the provider's :class:`Player`.

    This is the canonical "crop → embed" step shared by prediction and DB
    rebuild.  When *align* is True the crop is similarity-aligned to the
    canonical 5-point template before embedding (falling back to the unaligned
    crop if landmark detection fails); when False the provider embeds the
    crop's narrow face box directly.  ``fallback_confidence`` is used for the
    returned player's confidence when the crop's own detection does not supply
    one.

    When *collect_debug* is True the return value is ``(player, debug)`` where
    ``debug`` is ``{"landmarks_px": [...], "aligned": <ndarray|None>}`` for
    visual QA; otherwise just ``player`` is returned.
    """
    fr_face, fr_narrow, landmarks_px = detect_face_in_crop(crop, face_model)

    aligned = align_face(crop, landmarks_px) if align else None
    if aligned is not None:
        # The aligned image is the canonically-framed face itself, so the
        # provider should embed the whole image (full-image bbox).
        body = make_crop_body(FRBox(0.0, 0.0, 1.0, 1.0), None)
        player = provider.predict_player(aligned, body)
    else:
        if align:
            log.debug("[align] alignment unavailable (need 5 landmarks) — using unaligned crop")
        body = make_crop_body(fr_face, fr_narrow)
        player = provider.predict_player(crop, body)

    if player.confidence is None:
        player.confidence = fallback_confidence

    if collect_debug:
        return player, {"landmarks_px": landmarks_px, "aligned": aligned}
    return player


# Distinct BGR colours for landmarks 0..4 (eye, eye, nose, mouth, mouth).
_LANDMARK_COLORS = [
    (0, 0, 255),    # 0 — red
    (0, 165, 255),  # 1 — orange
    (0, 255, 0),    # 2 — green
    (255, 0, 0),    # 3 — blue
    (255, 0, 255),  # 4 — magenta
]


def make_alignment_debug_image(
    crop: np.ndarray,
    landmarks_px: list[tuple[float, float]],
    aligned: np.ndarray | None = None,
    output_size: int = _ALIGN_OUTPUT_SIZE,
) -> np.ndarray:
    """Build a side-by-side ``[annotated crop | aligned face]`` QA image.

    The left panel shows the source crop with each detected landmark drawn and
    numbered (matching the template order); the right panel, when *aligned* is
    supplied, shows the aligned face overlaid with the canonical template
    targets so the landmark order and alignment quality can be eyeballed.
    """
    # Left panel: crop with numbered landmarks.
    left = crop.copy()
    radius = max(2, output_size // 50)
    for i, (x, y) in enumerate(landmarks_px[:5]):
        color = _LANDMARK_COLORS[i % len(_LANDMARK_COLORS)]
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(left, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(left, str(i), (cx + radius, cy - radius),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    lh, lw = left.shape[:2]
    scale = output_size / max(1, lh)
    left = cv2.resize(left, (max(1, int(lw * scale)), output_size))

    panels = [left]
    if aligned is not None:
        right = aligned.copy()
        dst = _ARCFACE_TEMPLATE_112 * (output_size / 112.0)
        for i, (x, y) in enumerate(dst):
            color = _LANDMARK_COLORS[i % len(_LANDMARK_COLORS)]
            cv2.drawMarker(right, (int(round(x)), int(round(y))), color,
                           cv2.MARKER_CROSS, max(6, output_size // 14), 1, cv2.LINE_AA)
        panels.append(right)

    # 4px black separator between panels.
    sep = np.zeros((output_size, 4, 3), dtype=panels[0].dtype)
    stacked: list[np.ndarray] = []
    for idx, panel in enumerate(panels):
        if idx:
            stacked.append(sep)
        stacked.append(panel)
    return cv2.hconcat(stacked)
