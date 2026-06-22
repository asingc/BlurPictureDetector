from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import rawpy
from ultralytics import YOLO

from algo.config import AppConfig, app_config
from algo.frame import Frame
from algo.models import Body, Box, Face, Point, PredictedKeyPoint
from algo.stage import ProcessStage
from algo.utils import _HEAD_KP_INDICES, cap_long_edge

log = logging.getLogger("BlurPictureDetector")

# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)
_RAW_EXTENSIONS: frozenset[str] = frozenset({".cr3", ".cr2"})
IMAGE_EXTENSIONS = IMAGE_EXTENSIONS | _RAW_EXTENSIONS

# ---------------------------------------------------------------------------
# Face model bootstrap
# ---------------------------------------------------------------------------

_FACE_MODEL_URL  = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
)
# Resolve relative to the repo root (three levels up from this file).
_FACE_MODEL_PATH = Path(__file__).parent.parent.parent / "yolov8n-face.pt"


def _ensure_face_model() -> Path:
    """Download yolov8n-face.pt to the repo root if not already present."""
    if _FACE_MODEL_PATH.exists():
        log.debug("Face model already present: %s", _FACE_MODEL_PATH)
        return _FACE_MODEL_PATH
    log.info("Downloading yolov8n-face.pt from %s …", _FACE_MODEL_URL)
    urllib.request.urlretrieve(_FACE_MODEL_URL, _FACE_MODEL_PATH)
    log.info("Download complete: %s", _FACE_MODEL_PATH)
    return _FACE_MODEL_PATH


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def collect_images(input_path: Path) -> list[Path]:
    """Return a sorted list of supported image files at *input_path*."""
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_EXTENSIONS else []
    if input_path.is_dir():
        return sorted(
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []


def _read_image(path: Path) -> np.ndarray | None:
    """Read *path* as a BGR numpy array.
    RAW formats (CR3, CR2) are decoded via rawpy; all others via OpenCV.
    Returns None when the file cannot be decoded.
    """
    if path.suffix.lower() in _RAW_EXTENSIONS:
        try:
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=8,
                    half_size=True,
                )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            log.debug("[read] rawpy failed for %s: %s", path.name, exc)
            return None
    return cv2.imread(str(path))


# ---------------------------------------------------------------------------
# Body / face detection helpers  (moved from 1_prep_review.py)
# ---------------------------------------------------------------------------

def extract_bodies(image: np.ndarray, pose_model: YOLO) -> list[Body]:
    """Run body/pose detection on *image* and return up to 8 Body objects
    (largest first), each with an empty faces list and a placeholder crop."""
    h, w = image.shape[:2]
    pad  = 10

    results = pose_model.predict(image, verbose=False)
    if not results or len(results[0].boxes) == 0:
        log.debug("[bodies] pose model: 0 persons detected")
        return []

    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    log.debug("[bodies] image %dx%d — pose model: %d body(ies) detected", w, h, len(boxes))

    top = list(np.argsort(areas)[::-1][:8])
    log.debug("[bodies] top-%d candidates, areas=%s",
              len(top), [f"{areas[i]:.0f}" for i in top])

    kps_data = (
        results[0].keypoints.data.cpu().numpy()
        if results[0].keypoints is not None else None
    )

    bodies: list[Body] = []
    for idx in top:
        body_box = Box.from_px(*boxes[idx], w, h).padded(pad, w, h)
        if kps_data is not None:
            kps_raw   = kps_data[idx]
            keypoints = [
                PredictedKeyPoint(
                    Point.from_px(kps_raw[i, 0], kps_raw[i, 1], w, h),
                    float(kps_raw[i, 2]),
                )
                for i in range(len(kps_raw))
            ]
        else:
            keypoints = []
        px1, py1, px2, py2 = body_box.as_px_ints(w, h)
        log.debug("[bodies]   body[%d]: bbox=(%d,%d,%d,%d) area=%.0f kps=%d",
                  idx, px1, py1, px2, py2, areas[idx], len(keypoints))
        bodies.append(Body(
            crop=np.empty((0, 0, 3), dtype=np.uint8),
            bbox=body_box,
            faces=[],
            keypoints=keypoints,
        ))

    log.debug("[bodies] %d body(ies) returned", len(bodies))
    return bodies


def extract_faces(image: np.ndarray, face_model: YOLO) -> list[Face]:
    """Run face detection in two passes and return all Face objects that pass
    the configured size filter.

    Pass 1 — full image: locate face bounding boxes.
    Pass 2 — per-face crop: re-run on a padded crop to obtain refined landmark
             positions, then transform coordinates back to full-image space.
             Falls back to pass-1 landmarks when pass 2 yields no detection.
    """
    h, w = image.shape[:2]

    face_results = face_model.predict(image, verbose=False)
    if face_results and len(face_results[0].boxes) > 0:
        fdet_boxes = face_results[0].boxes.xyxy.cpu().numpy()
        fdet_confs = face_results[0].boxes.conf.cpu().numpy()
    else:
        return []

    log.debug("[faces] face model pass-1: %d raw detection(s)", len(fdet_boxes))
    faces: list[Face] = []
    for f_idx in range(len(fdet_boxes)):
        face_box = Box.from_px(
            fdet_boxes[f_idx, 0], fdet_boxes[f_idx, 1],
            fdet_boxes[f_idx, 2], fdet_boxes[f_idx, 3],
            w, h,
        )

        if face_box.width <= 0 or face_box.height <= 0:
            continue

        if app_config.face_min_size_fraction > 0:
            face_long = max(face_box.width, face_box.height)
            if face_long < app_config.face_min_size_fraction:
                px1, py1, px2, py2 = face_box.as_px_ints(w, h)
                log.debug("[faces] face[%d]: %dx%d too small (min=%.3f) — skipped",
                          f_idx, px2 - px1, py2 - py1, app_config.face_min_size_fraction)
                continue

        landmarks: list[PredictedKeyPoint] = []
        px1, py1, px2, py2 = face_box.as_px_ints(w, h)
        crop_pad = max(10, int(max(px2 - px1, py2 - py1) * 0.2))
        crop_box = face_box.padded(crop_pad, w, h)
        cx1, cy1, cx2, cy2 = crop_box.as_px_ints(w, h)
        face_crop = image[cy1:cy2, cx1:cx2]

        if face_crop.size > 0:
            crop_results = face_model.predict(face_crop, verbose=False)
            if crop_results and len(crop_results[0].boxes) > 0 and crop_results[0].keypoints is not None:
                crop_boxes = crop_results[0].boxes.xyxy.cpu().numpy()
                crop_h, crop_w = face_crop.shape[:2]
                cx_centres = (crop_boxes[:, 0] + crop_boxes[:, 2]) / 2
                cy_centres = (crop_boxes[:, 1] + crop_boxes[:, 3]) / 2
                dists = np.sqrt((cx_centres - crop_w / 2) ** 2 + (cy_centres - crop_h / 2) ** 2)
                best = int(np.argmin(dists))

                kps_raw2 = crop_results[0].keypoints.data.cpu().numpy()[best]
                landmarks = [
                    PredictedKeyPoint(
                        Point.from_px(int(kps_raw2[i, 0]) + cx1, int(kps_raw2[i, 1]) + cy1, w, h),
                        float(kps_raw2[i, 2]),
                    )
                    for i in range(len(kps_raw2))
                ]
                log.debug("[faces] face[%d]: pass-2 landmarks from crop (%d,%d,%d,%d)",
                          f_idx, cx1, cy1, cx2, cy2)
            else:
                log.debug("[faces] face[%d]: pass-2 no detection in crop — landmarks empty", f_idx)

        faces.append(Face(
            bbox=face_box,
            confidence=float(fdet_confs[f_idx]),
            landmarks=landmarks,
        ))

    log.debug("[faces] %d face(s) after filtering", len(faces))
    return faces


def _head_region(body: Body, conf_threshold: float = 0.3) -> Box | None:
    """Return the bounding box of confident head keypoints (indices 0-4),
    or None if fewer than 2 are detected."""
    xs, ys = [], []
    for i in _HEAD_KP_INDICES:
        if i >= len(body.keypoints):
            continue
        kp = body.keypoints[i]
        if kp.confidence >= conf_threshold:
            xs.append(kp.point.x)
            ys.append(kp.point.y)
    if len(xs) < 2:
        return None
    return Box(min(xs), min(ys), max(xs), max(ys))


def match_faces_to_bodies(bodies: list[Body], faces: list[Face]) -> None:
    """Assign each face to every body whose head-keypoint region overlaps the
    face bbox.  Falls back to the body bbox when head keypoints are absent.
    Mutates *bodies* in place."""
    for face in faces:
        for body in bodies:
            region = _head_region(body) or body.bbox
            if region.overlaps(face.bbox) or body.bbox.overlaps(face.bbox):
                body.faces.append(face)
                log.debug("[match]   face conf=%.3f → body head_region=(%.3f,%.3f,%.3f,%.3f)",
                          face.confidence,
                          region.x1, region.y1, region.x2, region.y2)


def detect_qualified_persons(
    image: np.ndarray,
    pose_model: YOLO,
    face_model: YOLO,
) -> tuple[list[Body], bool]:
    """Return (bodies, had_persons).

    had_persons : True if the pose model detected at least one person body.
    bodies      : up to 8 Body objects (largest-first), with matched faces
                  and keypoints populated.  No threshold filtering here.
    """
    bodies = extract_bodies(image, pose_model)
    if not bodies:
        return [], False

    detected_faces = extract_faces(image, face_model)
    log.debug("[detect] face model: %d face(s) after filtering", len(detected_faces))

    match_faces_to_bodies(bodies, detected_faces)

    for body in bodies:
        if body.faces:
            fx1, fy1, fx2, fy2 = body.faces[0].bbox.as_px_ints(image.shape[1], image.shape[0])
            body.crop = image[fy1:fy2, fx1:fx2]
            bx1, by1, bx2, by2 = body.bbox.as_px_ints(image.shape[1], image.shape[0])
            log.debug("[detect]   body bbox=(%d,%d,%d,%d): %d face(s), crop %.0fx%.0f",
                      bx1, by1, bx2, by2,
                      len(body.faces),
                      body.faces[0].bbox.width * image.shape[1],
                      body.faces[0].bbox.height * image.shape[0])
        else:
            bx1, by1, bx2, by2 = body.bbox.as_px_ints(image.shape[1], image.shape[0])
            log.debug("[detect]   body bbox=(%d,%d,%d,%d): no matched face",
                      bx1, by1, bx2, by2)

    log.debug("[detect] result: %d body(ies) returned", len(bodies))
    return bodies, True


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class ImageAnalysisStage(ProcessStage):
    """Load every image under *input_path*, run pose and face YOLO models,
    and return a fully-populated list of Frame objects.

    The *frames* argument passed to :meth:`process` is ignored — this stage
    always constructs a fresh list from the images on disk.
    """

    def __init__(self, input_path: Path, pose_model: YOLO, face_model: YOLO) -> None:
        self.input_path = input_path
        self.pose_model = pose_model
        self.face_model = face_model

    def process(self, frames: list[Frame], config: AppConfig) -> list[Frame]:
        files = collect_images(self.input_path)
        result: list[Frame] = []
        width = len(str(len(files)))

        for idx, path in enumerate(files, 1):
            log.debug("[ImageAnalysisStage] [%*d/%d] %s", width, idx, len(files), path.name)
            image = _read_image(path)
            if image is None:
                log.error("[ImageAnalysisStage] %s — cannot read image file", path.name)
                continue

            normalized = cap_long_edge(image, config.normalized_img_max_long_edge)
            log.debug("[ImageAnalysisStage] %s — original %dx%d  normalized %dx%d",
                      path.name, image.shape[1], image.shape[0],
                      normalized.shape[1], normalized.shape[0])

            bodies, had_persons = detect_qualified_persons(normalized, self.pose_model, self.face_model)
            if not had_persons:
                log.debug("[ImageAnalysisStage] %s — no person detected", path.name)

            result.append(Frame(
                path=path,
                bodies=bodies,
                image=image,
                normalized_image=normalized,
            ))

        log.info("[ImageAnalysisStage] %d frame(s) constructed from %d file(s)",
                 len(result), len(files))
        return result
