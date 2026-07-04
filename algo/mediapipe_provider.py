"""MediaPipe-based replacement for the YOLO pose/face models.

Provides the same data contracts (:class:`~algo.models.Body` /
:class:`~algo.models.Face`) as the YOLO-based extraction functions in
``algo/stages/image_analysis.py`` and ``algo/face_crop_embed.py`` so every
downstream stage (grading, jersey counting, annotation, face recognition)
works unchanged regardless of which engine produced the detections.

Model licensing: MediaPipe Tasks models (BlazePose / BlazeFace + FaceMesh-V2)
are distributed by Google under Apache-2.0, unlike the YOLO models which are
AGPL-3.0 / GPL-3.0.  See README licensing notes.

Keypoint mapping
-----------------
BlazePose (Pose Landmarker) emits 33 landmarks; we only need the 17 COCO
keypoints the rest of the codebase already understands (see
``algo/utils.py:_HEAD_KP_INDICES`` and the ``_COCO_SKELETON`` /
``_TORSO_KP_INDICES`` constants elsewhere).  ``_COCO_FROM_MP_POSE[coco_idx]``
gives the corresponding BlazePose landmark index:

    COCO:  0 nose, 1 L-eye, 2 R-eye, 3 L-ear, 4 R-ear,
           5 L-shoulder, 6 R-shoulder, 7 L-elbow, 8 R-elbow,
           9 L-wrist, 10 R-wrist, 11 L-hip, 12 R-hip,
           13 L-knee, 14 R-knee, 15 L-ankle, 16 R-ankle

FaceMesh-V2 (Face Landmarker) emits 478 landmarks with no official "5-point"
output.  ``_MP_FACE_5PT`` is a best-effort, community-standard index mapping
to the same (L-eye, R-eye, nose, L-mouth, R-mouth) order the YOLO face model
used, chosen so the existing ArcFace alignment template keeps working.
Validate visually via ``--debug-align`` QA images; adjust here if needed.

Known limitation: unlike the YOLO face model, MediaPipe's Face Landmarker
does not expose a reliable per-landmark occlusion/visibility score in the
Python Tasks API, so mapped face landmarks are given a fixed confidence of
1.0.  This makes ``face_coverage_min_visible`` (occlusion disqualification)
effectively a pass-through under this engine — see README.
"""

from __future__ import annotations

import logging
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from algo.config import app_config
from algo.facereco_provider import Box as FRBox
from algo.models import Body, Box, Face, Point, PredictedKeyPoint
from algo.utils import _narrow_face_box

log = logging.getLogger("BlurPictureDetector")

# ---------------------------------------------------------------------------
# Model bootstrap
# ---------------------------------------------------------------------------

_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
_POSE_MODEL_PATH = Path(__file__).resolve().parent.parent / "pose_landmarker_full.task"

_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
_FACE_MODEL_PATH = Path(__file__).resolve().parent.parent / "face_landmarker.task"


def _ensure_model(url: str, path: Path) -> Path:
    if path.exists():
        log.debug("MediaPipe model present: %s", path)
        return path
    log.info("Downloading %s from %s ...", path.name, url)
    urllib.request.urlretrieve(url, path)
    log.info("Download complete: %s", path)
    return path


def ensure_pose_model() -> Path:
    """Return the path to ``pose_landmarker_full.task``, downloading it if missing."""
    return _ensure_model(_POSE_MODEL_URL, _POSE_MODEL_PATH)


def ensure_face_model() -> Path:
    """Return the path to ``face_landmarker.task``, downloading it if missing."""
    return _ensure_model(_FACE_MODEL_URL, _FACE_MODEL_PATH)


# ---------------------------------------------------------------------------
# Landmarker singletons (thread-safe, lazily constructed)
# ---------------------------------------------------------------------------

_POSE_LANDMARKER = None
_POSE_LANDMARKER_NUM_POSES: int | None = None
_POSE_LANDMARKER_LOCK = threading.Lock()

_FACE_LANDMARKERS: dict[int, object] = {}
_FACE_LANDMARKER_LOCK = threading.Lock()


def load_pose_landmarker(num_poses: int = 8):
    """Return the process-wide Pose Landmarker singleton (built for *num_poses*).

    Rebuilt only if a different ``num_poses`` is requested; otherwise the
    cached instance is reused across calls/images.
    """
    global _POSE_LANDMARKER, _POSE_LANDMARKER_NUM_POSES
    if _POSE_LANDMARKER is not None and _POSE_LANDMARKER_NUM_POSES == num_poses:
        return _POSE_LANDMARKER
    with _POSE_LANDMARKER_LOCK:
        if _POSE_LANDMARKER is not None and _POSE_LANDMARKER_NUM_POSES == num_poses:
            return _POSE_LANDMARKER
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = ensure_pose_model()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _POSE_LANDMARKER = mp_vision.PoseLandmarker.create_from_options(options)
        _POSE_LANDMARKER_NUM_POSES = num_poses
        log.info("MediaPipe Pose Landmarker loaded (num_poses=%d)", num_poses)
    return _POSE_LANDMARKER


def load_face_landmarker(num_faces: int = 8):
    """Return a process-wide Face Landmarker singleton for *num_faces*.

    Two distinct instances are typically kept alive: one with a high
    ``num_faces`` for full-image detection, and one with ``num_faces=1`` for
    the per-crop landmark-refinement pass used during embedding/alignment.
    """
    if num_faces in _FACE_LANDMARKERS:
        return _FACE_LANDMARKERS[num_faces]
    with _FACE_LANDMARKER_LOCK:
        if num_faces in _FACE_LANDMARKERS:
            return _FACE_LANDMARKERS[num_faces]
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = ensure_face_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        _FACE_LANDMARKERS[num_faces] = landmarker
        log.info("MediaPipe Face Landmarker loaded (num_faces=%d)", num_faces)
    return landmarker


def _to_mp_image(image_bgr: np.ndarray):
    import mediapipe as mp
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


# ---------------------------------------------------------------------------
# COCO-17 keypoint mapping (BlazePose 33 -> COCO 17)
# ---------------------------------------------------------------------------

_COCO_FROM_MP_POSE: tuple[int, ...] = (
    0,   # nose
    2,   # left eye
    5,   # right eye
    7,   # left ear
    8,   # right ear
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
    25,  # left knee
    26,  # right knee
    27,  # left ankle
    28,  # right ankle
)

# Best-effort 5-point mapping (L-eye, R-eye, nose, L-mouth, R-mouth) from
# FaceMesh-V2's 478 landmarks — see module docstring for caveats.
_MP_FACE_5PT: tuple[int, ...] = (468, 473, 1, 61, 291)


# ---------------------------------------------------------------------------
# Body / pose extraction
# ---------------------------------------------------------------------------

def extract_bodies_mp(image: np.ndarray, pose_landmarker, max_bodies: int = 8, pad: int = 10) -> list[Body]:
    """Run MediaPipe Pose Landmarker on *image* and return up to *max_bodies*
    Body objects (largest first), each with 17 COCO-mapped keypoints.

    Mirrors ``algo/stages/image_analysis.py:extract_bodies`` (YOLO) so callers
    can swap engines without touching downstream code.
    """
    h, w = image.shape[:2]
    mp_image = _to_mp_image(image)
    result = pose_landmarker.detect(mp_image)
    if not result.pose_landmarks:
        log.debug("[bodies:mp] pose landmarker: 0 persons detected")
        return []

    log.debug("[bodies:mp] image %dx%d — pose landmarker: %d body(ies) detected",
              w, h, len(result.pose_landmarks))

    candidates: list[tuple[float, Box, list[PredictedKeyPoint]]] = []
    for pose in result.pose_landmarks:
        xs = [lm.x for lm in pose]
        ys = [lm.y for lm in pose]
        box = Box(
            max(0.0, min(xs)), max(0.0, min(ys)),
            min(1.0, max(xs)), min(1.0, max(ys)),
        ).padded(pad, w, h)
        area = box.area * w * h

        keypoints: list[PredictedKeyPoint] = []
        for mp_idx in _COCO_FROM_MP_POSE:
            lm = pose[mp_idx]
            confidence = float(getattr(lm, "visibility", 1.0) or 0.0)
            keypoints.append(PredictedKeyPoint(Point(lm.x, lm.y), confidence))

        candidates.append((area, box, keypoints))

    candidates.sort(key=lambda c: c[0], reverse=True)
    candidates = candidates[:max_bodies]

    bodies: list[Body] = []
    for area, box, keypoints in candidates:
        px1, py1, px2, py2 = box.as_px_ints(w, h)
        log.debug("[bodies:mp]   body: bbox=(%d,%d,%d,%d) area=%.0f kps=%d",
                  px1, py1, px2, py2, area, len(keypoints))
        bodies.append(Body(
            crop=np.empty((0, 0, 3), dtype=np.uint8),
            bbox=box,
            faces=[],
            keypoints=keypoints,
        ))

    log.debug("[bodies:mp] %d body(ies) returned", len(bodies))
    return bodies


def extract_body_keypoints_for_box_mp(
    image: np.ndarray, box: Box, pose_landmarker, pad: int = 20
) -> list[PredictedKeyPoint]:
    """Run MediaPipe Pose Landmarker (num_poses=1) on a padded crop around
    *box* and return 17 COCO-mapped keypoints in full-image normalised
    coordinates (empty list if no pose is found in the crop).

    Used by the hybrid engine: a torchvision person detector supplies robust
    body boxes (see ``algo/torchvision_provider.py``), and this function
    supplies the keypoints MediaPipe's own person detector under-recalls on
    busy multi-person photos.
    """
    h, w = image.shape[:2]
    padded = box.padded(pad, w, h)
    cx1, cy1, cx2, cy2 = padded.as_px_ints(w, h)
    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return []

    crop_h, crop_w = crop.shape[:2]
    result = pose_landmarker.detect(_to_mp_image(crop))
    if not result.pose_landmarks:
        return []

    pose = result.pose_landmarks[0]
    keypoints: list[PredictedKeyPoint] = []
    for mp_idx in _COCO_FROM_MP_POSE:
        lm = pose[mp_idx]
        confidence = float(getattr(lm, "visibility", 1.0) or 0.0)
        full_x = (cx1 + lm.x * crop_w) / w
        full_y = (cy1 + lm.y * crop_h) / h
        keypoints.append(PredictedKeyPoint(Point(full_x, full_y), confidence))
    return keypoints


# ---------------------------------------------------------------------------
# Face extraction (pose-guided, per body)
# ---------------------------------------------------------------------------
#
# MediaPipe's bundled face detector (BlazeFace "short-range") is tuned for
# close-range faces (selfie/video-call distance).  Empirically it fails to
# find any faces at all when run directly on a normalised sports-photo frame
# (faces occupy only a few percent of the frame) — verified 0/N recall on
# real test images even at min_face_detection_confidence=0.1.  BlazePose's
# body/head keypoints, however, DO localise heads reliably at this range (see
# extract_bodies_mp).  So instead of running Face Landmarker on the full
# image, we crop a generous region around each body's confident head
# keypoints — bringing the face back into BlazeFace's effective operating
# range — and run Face Landmarker (num_faces=1) on that crop.

# Head-crop padding, as a fraction of the head-keypoint bounding box size,
# plus a small fixed fraction of the image to keep tiny clusters usable.
_HEAD_CROP_PAD_X_FRAC = 1.0
_HEAD_CROP_PAD_Y_FRAC = 1.2
_HEAD_CROP_PAD_MIN = 0.03


def detect_face_for_body_mp(image: np.ndarray, body: Body, face_landmarker) -> Face | None:
    """Return a :class:`Face` for *body* by running Face Landmarker on a crop
    around its confident head keypoints (nose/eyes/ears — COCO indices 0-4),
    or ``None`` if no face is found in the crop.

    Falls back to the top 40% of the body's own bounding box (a reasonable
    head-region estimate) when fewer than 2 head keypoints are confident —
    this happens when the hybrid engine's per-crop pose refinement fails to
    find a pose at all, in which case ``body.keypoints`` is empty.

    *face_landmarker* should be a ``num_faces=1`` instance (see
    :func:`load_face_landmarker`).
    """
    h, w = image.shape[:2]
    xs = [kp.point.x for kp in body.keypoints[:5] if kp.confidence >= 0.3]
    ys = [kp.point.y for kp in body.keypoints[:5] if kp.confidence >= 0.3]
    if len(xs) < 2:
        bb = body.bbox
        x1, y1, x2, y2 = bb.x1, bb.y1, bb.x2, bb.y1 + bb.height * 0.4
    else:
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    pad_x = bw * _HEAD_CROP_PAD_X_FRAC + _HEAD_CROP_PAD_MIN
    pad_y = bh * _HEAD_CROP_PAD_Y_FRAC + _HEAD_CROP_PAD_MIN

    cx1 = max(0, int((x1 - pad_x) * w))
    cy1 = max(0, int((y1 - pad_y) * h))
    cx2 = min(w, int((x2 + pad_x) * w))
    cy2 = min(h, int((y2 + pad_y) * h))
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    result = face_landmarker.detect(_to_mp_image(crop))
    if not result.face_landmarks:
        return None

    mesh = result.face_landmarks[0]
    crop_h, crop_w = crop.shape[:2]
    mesh_xs = [lm.x for lm in mesh]
    mesh_ys = [lm.y for lm in mesh]
    # Face bbox in full-image normalised coordinates.
    face_box = Box(
        max(0.0, (cx1 + min(mesh_xs) * crop_w) / w),
        max(0.0, (cy1 + min(mesh_ys) * crop_h) / h),
        min(1.0, (cx1 + max(mesh_xs) * crop_w) / w),
        min(1.0, (cy1 + max(mesh_ys) * crop_h) / h),
    )
    if face_box.width <= 0 or face_box.height <= 0:
        return None

    if app_config.face_min_size_fraction > 0:
        face_long = max(face_box.width, face_box.height)
        if face_long < app_config.face_min_size_fraction:
            log.debug("[faces:mp] face too small (min=%.3f) — skipped",
                      app_config.face_min_size_fraction)
            return None

    landmarks = [
        PredictedKeyPoint(
            Point((cx1 + mesh[idx].x * crop_w) / w, (cy1 + mesh[idx].y * crop_h) / h),
            1.0,
        )
        for idx in _MP_FACE_5PT
    ]
    return Face(bbox=face_box, confidence=1.0, landmarks=landmarks)


# ---------------------------------------------------------------------------
# Per-crop face detection (used by embedding/alignment pipeline)
# ---------------------------------------------------------------------------

def detect_face_in_crop_mp(
    image: np.ndarray, face_landmarker
) -> tuple[FRBox, FRBox | None, list[tuple[float, float]]]:
    """Detect the face bbox, narrow landmark bbox, and 5 landmarks in *image*.

    Drop-in MediaPipe replacement for
    ``algo/face_crop_embed.py:detect_face_in_crop`` (YOLO); same return
    contract (see that function's docstring).
    """
    h, w = image.shape[:2]
    mp_image = _to_mp_image(image)
    result = face_landmarker.detect(mp_image)
    if not result.face_landmarks:
        log.debug("[detect:mp] no face found in crop — using full-image fallback")
        return FRBox(0.0, 0.0, 1.0, 1.0), None, []

    # Pick the mesh closest to the crop centre (crop images should have
    # exactly one face; defensive in case more than one is ever returned).
    cx_img, cy_img = w / 2.0, h / 2.0
    best_mesh = None
    best_dist = float("inf")
    for mesh in result.face_landmarks:
        xs = [lm.x * w for lm in mesh]
        ys = [lm.y * h for lm in mesh]
        cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        dist = (cx - cx_img) ** 2 + (cy - cy_img) ** 2
        if dist < best_dist:
            best_dist = dist
            best_mesh = mesh

    xs = [lm.x for lm in best_mesh]
    ys = [lm.y for lm in best_mesh]
    face_box_norm = Box(
        max(0.0, min(xs)), max(0.0, min(ys)),
        min(1.0, max(xs)), min(1.0, max(ys)),
    )

    landmarks_px = [(best_mesh[idx].x * w, best_mesh[idx].y * h) for idx in _MP_FACE_5PT]
    landmarks = [
        PredictedKeyPoint(Point.from_px(px, py, w, h), 1.0)
        for px, py in landmarks_px
    ]

    model_face = Face(bbox=face_box_norm, confidence=1.0, landmarks=landmarks)
    narrow = _narrow_face_box(model_face, pad=0)

    fr_face = FRBox(face_box_norm.x1, face_box_norm.y1, face_box_norm.x2, face_box_norm.y2)
    fr_narrow = FRBox(narrow.x1, narrow.y1, narrow.x2, narrow.y2) if narrow else None
    return fr_face, fr_narrow, landmarks_px
