#!/usr/bin/env python3
"""
BlurPictureDetector
-------------------
Detect blurry sport images by analysing the sharpness of the main subject.

Usage:
    python detect_blur.py <image_or_directory> [--sensitivity low|medium|high]

How it works:
    1. YOLOv8n detects persons in the image (model auto-downloaded on first run).
    2. The largest detected person is cropped out as the "main subject".
    3. Two classical sharpness metrics are computed on the greyscale crop:
       - Laplacian variance  (sensitive to fine detail / high-frequency content)
       - Tenengrad           (gradient-energy measure, robust to noise)
    4. The two metrics are combined into a single sharpness_score in [0, 1]
       where 1 = perfectly sharp and 0 = completely blurry.
    5. Images whose sharpness_score falls below the sensitivity threshold are
       flagged, given a 1-star rating, and logged in blurry.csv and blur.lst.
       Files are NOT moved — their paths are recorded instead.
    6. A copy of every processed image is saved to
       <image_dir>/annotated/ with the subject bounding box drawn (style via app_config).

If no person is detected in an image the file is left untouched and skipped.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)

# sharpness_score threshold per sensitivity level.
# A file is flagged as blurry when  sharpness_score <= threshold.
#   low    → only flag severely blurry images  (high tolerance)
#   medium → balanced default
#   high   → flag even slightly blurry images  (low tolerance)
SENSITIVITY_THRESHOLDS: dict[str, float] = {
    "low":    0.30,
    "medium": 0.55,
    "high":   0.75,
}

# Normalisation scales for the blur-score formula  1 / (1 + x / scale).
# At x == scale the component equals 0.5, placing it right on the medium boundary.
# Calibrated for typical subject crops from sports photography.
_LAP_SCALE: float = 100.0    # Laplacian variance reference
_TEN_SCALE: float = 5_000.0  # Tenengrad reference

# COCO 17-keypoint skeleton: pairs of indices to connect with a line.
# Keypoint order: 0=nose 1=L-eye 2=R-eye 3=L-ear 4=R-ear
#   5=L-shoulder 6=R-shoulder 7=L-elbow 8=R-elbow 9=L-wrist 10=R-wrist
#   11=L-hip 12=R-hip 13=L-knee 14=R-knee 15=L-ankle 16=R-ankle
_COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),   # head
    (5, 6),                             # shoulders
    (5, 7), (7, 9),                     # left arm
    (6, 8), (8, 10),                    # right arm
    (5, 11), (6, 12),                   # torso sides
    (11, 12),                           # hips
    (11, 13), (13, 15),                 # left leg
    (12, 14), (14, 16),                 # right leg
)


@dataclass
class AppConfig:
    # Bounding box and status icon drawn on annotated preview images.
    # Colors are BGR (OpenCV convention).
    annotation_box_color_pass: tuple[int, int, int] = field(default=(255, 0, 0))  # blue  – sharp
    annotation_box_color_fail: tuple[int, int, int] = field(default=(0,   0, 255))  # red   – blurry
    annotation_box_thickness:  int                  = 5
    annotation_icon_size:      int                  = 50  # px, drawn on the resized preview

    # Subject selection: prefer the largest person whose face is at least
    # half-visible, judged by how many face keypoints (out of 5) are
    # detected with confidence >= face_kp_conf_threshold.
    # Set face_kp_min_visible = 0 to disable face-visibility filtering.
    face_kp_min_visible:    int   = 3    # ≥ 3 of 5 face KPs must be confident
    face_kp_conf_threshold: float = 0.5  # per-keypoint confidence cutoff

    # Face-coverage check using the face model's 5 landmarks (eyes / nose / mouth).
    # A matched face is disqualified when fewer than face_coverage_min_visible
    # of its landmarks are confident — meaning the face is more than
    # (1 - face_coverage_min_visible/5) covered.
    # Set face_coverage_min_visible = 0 to disable the check.
    face_coverage_min_visible:    int   = 3    # ≥ 3 of 5 face-model landmarks must be confident
    face_coverage_conf_threshold: float = 0.85  # per-landmark confidence cutoff

    # Minimum face size: disqualify persons whose face bbox long edge is smaller
    # than this fraction of the image long edge (e.g. 0.04 = 4 %).
    # Filters out background spectators or faces too small to reliably analyse.
    # Set to 0 to disable.
    face_min_size_fraction: float = 0.025

    # Face bounding box drawn on annotated previews (separate from the body box).
    annotation_face_box_color:        tuple[int, int, int] = field(default=(0, 255, 255))  # yellow
    annotation_face_box_thickness:    int                  = 2
    # Blur score label drawn below each face bounding box.
    annotation_score_font_size_px:    int   = 20   # target text height in pixels
    annotation_score_font_thickness:  int   = 3
    # Face landmark circle.
    annotation_face_kp_radius:        int   = 6    # circle radius (px)
    annotation_face_kp_thickness:     int   = 2    # circle line thickness
    # Body keypoint square.
    annotation_body_kp_size:          int   = 5    # square side (px)
    annotation_body_kp_thickness:     int   = 2    # square line thickness
    # Body skeleton line.
    annotation_skeleton_thickness:    int   = 1    # line thickness (px)

    # Annotated preview / processing image scaling.
    # Images are downsized so the long edge equals normalized_img_max_long_edge.
    # Never upscales.
    normalized_img_max_long_edge: int = 1800


app_config = AppConfig()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("BlurPictureDetector")


def _setup_console_logging() -> None:
    """Configure a DEBUG-level console handler (called once at startup)."""
    log.setLevel(logging.DEBUG)
    if log.handlers:
        return
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%H:%M:%S"
    ))
    log.addHandler(ch)


def _add_file_logging(log_path: Path) -> None:
    """Attach a DEBUG-level file handler once the output directory is known."""
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s"))
    log.addHandler(fh)
    log.debug("Log file opened: %s", log_path.resolve())


# ---------------------------------------------------------------------------
# Sharpness metrics
# ---------------------------------------------------------------------------

def _laplacian_variance(gray: np.ndarray) -> float:
    """Higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _tenengrad(gray: np.ndarray) -> float:
    """Sum of squared gradient magnitudes — higher = sharper."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx ** 2 + gy ** 2))


def compute_sharpness_score(gray: np.ndarray) -> tuple[float, float, float]:
    """
    Compute a normalised sharpness score for a greyscale image patch.

    Returns
    -------
    sharpness_score   : float in [0, 1]  — 1 = perfectly sharp, 0 = completely blurry
    laplacian_variance: float            — raw Laplacian variance
    tenengrad         : float            — raw Tenengrad score
    """
    lap_var = _laplacian_variance(gray)
    ten     = _tenengrad(gray)

    # Map each raw metric to a blur component in (0, 1) using a monotone-
    # decreasing curve; sharper images yield blur components closer to 0.
    lap_component = 1.0 / (1.0 + lap_var / _LAP_SCALE)
    ten_component = 1.0 / (1.0 + ten    / _TEN_SCALE)

    # Weighted combination then invert so that 1 = sharp, 0 = blurry.
    blur = float(np.clip(0.6 * lap_component + 0.4 * ten_component, 0.0, 1.0))
    return 1.0 - blur, lap_var, ten


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _draw_status_icon(
    image: np.ndarray,
    x: int,
    y: int,
    size: int,
    color: tuple[int, int, int],
    *,
    passed: bool,
) -> None:
    """
    Draw a filled *size* x *size* badge at (x, y) with a white checkmark
    (passed=True) or white X (passed=False) on top.
    Clips safely to the image boundary.
    """
    h, w = image.shape[:2]
    x2, y2 = min(x + size, w), min(y + size, h)
    sw, sh = x2 - x, y2 - y          # actual drawn area (may be < size at edges)
    lw = max(2, size // 14)           # line width scales with icon size

    cv2.rectangle(image, (x, y), (x2, y2), color, -1)  # filled badge

    if passed:
        # Checkmark: two segments forming a ✓
        p1 = (x + int(sw * 0.15), y + int(sh * 0.50))
        p2 = (x + int(sw * 0.40), y + int(sh * 0.76))
        p3 = (x + int(sw * 0.85), y + int(sh * 0.24))
        cv2.line(image, p1, p2, (255, 255, 255), lw, cv2.LINE_AA)
        cv2.line(image, p2, p3, (255, 255, 255), lw, cv2.LINE_AA)
    else:
        # Cross: two diagonal lines forming an ✗
        p1 = (x + int(sw * 0.20), y + int(sh * 0.20))
        p2 = (x + int(sw * 0.80), y + int(sh * 0.80))
        p3 = (x + int(sw * 0.80), y + int(sh * 0.20))
        p4 = (x + int(sw * 0.20), y + int(sh * 0.80))
        cv2.line(image, p1, p2, (255, 255, 255), lw, cv2.LINE_AA)
        cv2.line(image, p3, p4, (255, 255, 255), lw, cv2.LINE_AA)

def normalize_img_size(image: np.ndarray) -> np.ndarray:
    """
    Downsize *image* so its long edge equals app_config.normalized_img_max_long_edge.
    Never upscales — returns the original array when it is already smaller.
    """
    h, w = image.shape[:2]
    scale = min(1.0, app_config.normalized_img_max_long_edge / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Point:
    """A 2-D coordinate in pixel space."""
    x: int
    y: int


@dataclass
class Box:
    """Axis-aligned bounding box: top-left (x1, y1) → bottom-right (x2, y2)."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def centre(self) -> Point:
        return Point((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def contains(self, p: Point) -> bool:
        """Return True if *p* lies inside or on the boundary of this box."""
        return self.x1 <= p.x <= self.x2 and self.y1 <= p.y <= self.y2

    def contains_xy(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Vectorised containment test — returns a boolean mask over the input arrays."""
        return (xs >= self.x1) & (xs <= self.x2) & (ys >= self.y1) & (ys <= self.y2)

    def padded(self, pad: int, max_w: int, max_h: int) -> "Box":
        """Return a copy expanded by *pad* pixels on every side, clamped to image bounds."""
        return Box(
            max(0,     self.x1 - pad),
            max(0,     self.y1 - pad),
            min(max_w, self.x2 + pad),
            min(max_h, self.y2 + pad),
        )

    def as_ints(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass
class Face:
    """A detected face: bounding box, detection confidence, and optional landmarks."""
    bbox:        Box
    confidence:  float        # detection confidence from the face model
    points:      list[Point]  # landmark coordinates (empty if model returned no keypoints)
    confidences: list[float]  # per-landmark confidence scores

    def n_visible(self, threshold: float) -> int:
        """Count landmarks whose confidence is at or above *threshold*."""
        return sum(1 for c in self.confidences if c >= threshold)


@dataclass
class Body:
    """A detected person ready for sharpness analysis."""
    crop:           np.ndarray  # face image crop (BGR) used for blur scoring
    bbox:           Box         # body bounding box (padded, clamped)
    faces:          list[Face]  # matched faces (may be empty before face matching)
    keypoints:      list[Point]  # 17 COCO body keypoints (x, y)
    kp_confidences: list[float]  # per-keypoint confidence scores


# ---------------------------------------------------------------------------
# Subject detection
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

    boxes = results[0].boxes.xyxy.cpu().numpy()          # (N, 4)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    log.debug("[bodies] image %dx%d — pose model: %d body(ies) detected", w, h, len(boxes))

    top = list(np.argsort(areas)[::-1][:8])
    log.debug("[bodies] top-%d candidates, areas=%s",
              len(top), [f"{areas[i]:.0f}" for i in top])

    kps_data = (
        results[0].keypoints.data.cpu().numpy()   # (N, 17, 3): x, y, conf
        if results[0].keypoints is not None else None
    )

    bodies: list[Body] = []
    for idx in top:
        body_box = Box(*boxes[idx].astype(int)).padded(pad, w, h)
        if kps_data is not None:
            kps_raw  = kps_data[idx]   # (17, 3)
            kp_pts   = [Point(int(kps_raw[i, 0]), int(kps_raw[i, 1])) for i in range(len(kps_raw))]
            kp_confs = [float(kps_raw[i, 2]) for i in range(len(kps_raw))]
        else:
            kp_pts, kp_confs = [], []
        log.debug("[bodies]   body[%d]: bbox=(%d,%d,%d,%d) area=%.0f kps=%d",
                  idx, body_box.x1, body_box.y1, body_box.x2, body_box.y2, areas[idx], len(kp_pts))
        bodies.append(Body(
            crop=np.empty((0, 0, 3), dtype=np.uint8),  # filled after face matching
            bbox=body_box,
            faces=[],
            keypoints=kp_pts,
            kp_confidences=kp_confs,
        ))

    log.debug("[bodies] %d body(ies) returned", len(bodies))
    return bodies


def extract_faces(image: np.ndarray, face_model: YOLO) -> list[Face]:
    """Run face detection on *image* and return all Face objects that pass the
    configured size and landmark-coverage filters."""
    h, w = image.shape[:2]
    pad  = 10

    face_results = face_model.predict(image, verbose=False)
    if face_results and len(face_results[0].boxes) > 0:
        fdet_boxes = face_results[0].boxes.xyxy.cpu().numpy()   # (M, 4)
        fdet_confs = face_results[0].boxes.conf.cpu().numpy()   # (M,)
        fdet_kps   = (
            face_results[0].keypoints.data.cpu().numpy()        # (M, 5, 3)
            if face_results[0].keypoints is not None else None
        )
    else:
        return []

    log.debug("[faces] face model: %d raw detection(s)", len(fdet_boxes))
    min_size = app_config.face_min_size_fraction * max(h, w)

    faces: list[Face] = []
    for f_idx in range(len(fdet_boxes)):
        face_box = Box(
            int(fdet_boxes[f_idx, 0]), int(fdet_boxes[f_idx, 1]),
            int(fdet_boxes[f_idx, 2]), int(fdet_boxes[f_idx, 3]),
        )

        if face_box.width <= 0 or face_box.height <= 0:
            continue

        # Size filter.
        if app_config.face_min_size_fraction > 0:
            face_long = max(face_box.width, face_box.height)
            if face_long < min_size:
                log.debug("[faces] face[%d]: %.0fx%.0f too small (min=%.1f) — skipped",
                          f_idx, face_box.width, face_box.height, min_size)
                continue

        # Build landmarks (if available) and apply coverage filter.
        points:   list[Point] = []
        kp_confs: list[float] = []
        if fdet_kps is not None:
            kps_raw  = fdet_kps[f_idx]   # (5, 3): x, y, conf
            points   = [Point(int(kps_raw[i, 0]), int(kps_raw[i, 1])) for i in range(len(kps_raw))]
            kp_confs = [float(kps_raw[i, 2]) for i in range(len(kps_raw))]
            # if app_config.face_coverage_min_visible > 0:
            #     n_visible = sum(1 for c in kp_confs if c >= app_config.face_coverage_conf_threshold)
            #     log.debug("[faces] face[%d]: landmark coverage %d/5 (min=%d, conf>=%.2f)",
            #               f_idx, n_visible,
            #               app_config.face_coverage_min_visible,
            #               app_config.face_coverage_conf_threshold)
            #     if n_visible < app_config.face_coverage_min_visible:
            #         log.debug("[faces] face[%d]: coverage too low — skipped", f_idx)
            #         continue

        faces.append(Face(
            bbox=face_box,
            confidence=float(fdet_confs[f_idx]),
            points=points,
            confidences=kp_confs,
        ))

    log.debug("[faces] %d face(s) after filtering", len(faces))
    return faces


def match_faces_to_bodies(bodies: list[Body], faces: list[Face]) -> None:
    """Assign each face to every body whose bbox contains the face's centre.
    Mutates *bodies* in place by appending to each body's faces list."""
    for face in faces:
        fc = face.bbox.centre
        for body in bodies:
            if body.bbox.contains(fc):
                body.faces.append(face)
                log.debug("[match]   face conf=%.3f → body bbox=(%d,%d,%d,%d)",
                          face.confidence,
                          body.bbox.x1, body.bbox.y1, body.bbox.x2, body.bbox.y2)


def detect_qualified_persons(
    image: np.ndarray,
    pose_model: YOLO,
    face_model: YOLO,
) -> tuple[list[Body], bool]:
    """
    Return (bodies, had_persons).

    had_persons : True if the pose model detected at least one person body.
    bodies      : up to 8 Body objects (largest-first); each has zero or more
                  matched faces.  Bodies with no matched face
                  are excluded.  May be empty even when had_persons is True.

    Per-face filters applied during matching (face skipped, not the whole body):
      - Landmark coverage  < face_coverage_min_visible confident landmarks.
      - Face size          < face_min_size_fraction × image long edge.
    """
    # ------------------------------------------------------------------
    # Phase 1 — body detection.
    # ------------------------------------------------------------------
    bodies = extract_bodies(image, pose_model)
    if not bodies:
        return [], False

    # ------------------------------------------------------------------
    # Phase 2 — face detection.
    # ------------------------------------------------------------------
    detected_faces = extract_faces(image, face_model)
    log.debug("[detect] face model: %d face(s) after filtering", len(detected_faces))

    # ------------------------------------------------------------------
    # Phase 3 — matching: associate faces with bodies.
    # ------------------------------------------------------------------
    match_faces_to_bodies(bodies, detected_faces)

    # ------------------------------------------------------------------
    # Phase 4 — finalise: assign crop; discard bodies with no face.
    # ------------------------------------------------------------------
    entries: list[Body] = []
    for body in bodies:
        if not body.faces:
            log.debug("[detect]   body bbox=(%d,%d,%d,%d): no face → disqualified",
                      body.bbox.x1, body.bbox.y1, body.bbox.x2, body.bbox.y2)
            continue
        fx1, fy1, fx2, fy2 = body.faces[0].bbox.as_ints()
        body.crop = image[fy1:fy2, fx1:fx2]
        log.debug("[detect]   body bbox=(%d,%d,%d,%d): %d face(s), crop %.0fx%.0f",
                  body.bbox.x1, body.bbox.y1, body.bbox.x2, body.bbox.y2,
                  len(body.faces), body.faces[0].bbox.width, body.faces[0].bbox.height)
        entries.append(body)

    log.debug("[detect] result: %d qualified / %d candidates", len(entries), len(bodies))
    return entries, True


# ---------------------------------------------------------------------------
# Per-image analysis
# ---------------------------------------------------------------------------

def analyse_image(
    image_path: Path,
    pose_model: YOLO,
    face_model: YOLO,
    threshold: float,
    boxes_dir: Optional[Path] = None,
) -> dict:
    """
    Analyse a single image file.

    Evaluates up to 3 qualified persons (largest face-visible body regions).
    The image is considered sharp if ANY of them is sharp.
    Reported metrics come from the sharpest (highest sharpness_score) person.

    Parameters
    ----------
    boxes_dir : if provided, annotated previews are saved here with:
                - body bounding box per person (blue = sharp, red = blurry)
                - status icon on the largest person (overall pass/fail)
                - face bounding box per person in yellow with blur score label

    Returns a dict with at minimum the keys 'file' and 'status'.
    status is one of: 'blurry', 'sharp', 'skipped', 'error'.
    """
    image_orig = cv2.imread(str(image_path))
    if image_orig is None:
        log.error("[analyse] %s — cannot read image file", image_path.name)
        return {"file": str(image_path), "status": "error", "error": "Cannot read image file", "persons_detail": []}

    log.debug("[analyse] %s — original size: %dx%d", image_path.name, image_orig.shape[1], image_orig.shape[0])
    # Keep the original for annotation; resize a working copy for processing (file untouched).
    normalized_img = normalize_img_size(image_orig)
    log.debug("[analyse] %s — processing size: %dx%d", image_path.name, normalized_img.shape[1], normalized_img.shape[0])

    persons, had_persons = detect_qualified_persons(normalized_img, pose_model, face_model)
    log.debug("[analyse] %s — had_persons=%s, qualified=%d",
              image_path.name, had_persons, len(persons))
    if not had_persons:
        log.debug("[analyse] %s — skipped: no person detected", image_path.name)
        if boxes_dir is not None:
            subdir = boxes_dir / "anno_skipped"
            subdir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(subdir / (image_path.stem + ".jpg")), image_orig, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return {"file": str(image_path), "status": "skipped", "reason": "No person detected", "persons_detail": []}
    if not persons:
        log.debug("[analyse] %s — blurry: all candidates disqualified", image_path.name)
        if boxes_dir is not None:
            subdir = boxes_dir / "anno_blur"
            subdir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(subdir / (image_path.stem + ".jpg")), image_orig, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return {
            "file":               str(image_path),
            "status":             "blurry",
            "sharpness_score":    0.0,
            "sharpness_grade":    0.0,
            "laplacian_variance": 0.0,
            "tenengrad_score":    0.0,
            "persons_detail":     [],
        }

    # Evaluate each qualified person.
    evaluated: list[dict] = []
    for person in persons:
        best_score: float       = 0.0
        best_lap:   float       = 0.0
        best_ten:   float       = 0.0
        best_face:  Face | None = None

        for face in person.faces:
            fx1, fy1, fx2, fy2 = face.bbox.as_ints()
            crop = normalized_img[fy1:fy2, fx1:fx2]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            s, lv, t = compute_sharpness_score(gray)
            log.debug("[analyse] %s — face conf=%.3f score=%.4f (lap=%.2f ten=%.2f)",
                      image_path.name, face.confidence, s, lv, t)
            if s > best_score:
                best_score, best_lap, best_ten, best_face = s, lv, t, face

        is_blurry = best_score <= threshold
        log.debug("[analyse] %s — person bbox=%s best_score=%.4f blurry=%s",
                  image_path.name, person.bbox, best_score, is_blurry)
        evaluated.append({
            "body_bbox":           person.bbox,
            "body_keypoints":      person.keypoints,
            "body_kp_confidences": person.kp_confidences,
            "face_bbox":           best_face.bbox if best_face else None,
            "face_kps":            best_face      if best_face else None,
            "sharpness_score":     best_score,
            "lap_var":             best_lap,
            "ten":                 best_ten,
            "is_blurry":           is_blurry,
        })

    # Image passes if ANY person is sharp.
    overall_blurry = all(p["is_blurry"] for p in evaluated)

    # Use the sharpest person's metrics for the CSV/log.
    best = max(evaluated, key=lambda p: p["sharpness_score"])
    log.debug("[analyse] %s — overall=%s best_score=%.4f threshold=%.2f (%d person(s))",
              image_path.name,
              "BLURRY" if overall_blurry else "SHARP",
              best["sharpness_score"], threshold, len(evaluated))

    # Build per-person detail rows for CSV output.
    persons_detail: list[dict] = []
    for p in evaluated:
        b: Box = p["body_bbox"]
        if p["face_kps"] is not None:
            kp_confs = ", ".join(f"{c:.2f}" for c in p["face_kps"].confidences)
        else:
            kp_confs = ""
        persons_detail.append({
            "verdict":      "Blur" if p["is_blurry"] else "Sharp",
            "orig_dim":     f"{b.width:.0f} x {b.height:.0f}",
            "score":        round(p["sharpness_score"], 2),
            "facial_boxes": kp_confs,
        })

    # --- annotated preview ---------------------------------------------------
    if boxes_dir is not None:
        # Annotate on the original (full-resolution) image; scale all bbox / keypoint
        # coordinates from processing space (resized) back up to original dimensions.
        h_proc, w_proc = normalized_img.shape[:2]
        annotated = image_orig.copy()
        h_out, w_out = annotated.shape[:2]
        sx = w_out / w_proc
        sy = h_out / h_proc
        ann_scale = (sx + sy) / 2  # uniform scale factor for annotation sizes

        face_thick = max(1, round(app_config.annotation_face_box_thickness * ann_scale))
        box_thick  = max(1, round(app_config.annotation_box_thickness * ann_scale))
        icon_size  = max(20, round(app_config.annotation_icon_size * ann_scale))
        kp_radius     = max(3, round(app_config.annotation_face_kp_radius    * ann_scale))
        kp_thick      = max(1, round(app_config.annotation_face_kp_thickness * ann_scale))
        body_kp_size    = max(2, round(app_config.annotation_body_kp_size         * ann_scale))
        body_kp_thick   = max(1, round(app_config.annotation_body_kp_thickness    * ann_scale))
        skeleton_thick  = max(1, round(app_config.annotation_skeleton_thickness   * ann_scale))
        font          = cv2.FONT_HERSHEY_SIMPLEX
        font_thick  = app_config.annotation_score_font_thickness
        # Compute font scale so text height == annotation_score_font_size_px scaled to image.
        (_, _base_h), _ = cv2.getTextSize("Mg", font, 1.0, font_thick)
        font_scale = app_config.annotation_score_font_size_px * ann_scale / max(_base_h, 1)

        for i, p in enumerate(evaluated):
            b: Box = p["body_bbox"]
            rbx1 = int(b.x1 * sx); rby1 = int(b.y1 * sy)
            rbx2 = int(b.x2 * sx); rby2 = int(b.y2 * sy)

            body_color = (
                app_config.annotation_box_color_fail
                if p["is_blurry"] else
                app_config.annotation_box_color_pass
            )
            cv2.rectangle(annotated, (rbx1, rby1), (rbx2, rby2),
                          body_color, box_thick)

            # Every person gets its own pass/fail status icon.
            _draw_status_icon(
                annotated, rbx1, rby1,
                icon_size,
                body_color,
                passed=not p["is_blurry"],
            )

            # Skeleton lines between connected body keypoints.
            kps = p["body_keypoints"]
            for ka, kb in _COCO_SKELETON:
                if ka >= len(kps) or kb >= len(kps):
                    continue
                pa, pb = kps[ka], kps[kb]
                if (pa.x == 0 and pa.y == 0) or (pb.x == 0 and pb.y == 0):
                    continue
                cv2.line(annotated,
                         (int(pa.x * sx), int(pa.y * sy)),
                         (int(pb.x * sx), int(pb.y * sy)),
                         body_color, skeleton_thick, cv2.LINE_AA)

            # Body keypoints as small squares (sharp/blurry color scheme).
            half = body_kp_size // 2
            for pt in p["body_keypoints"]:
                if pt.x == 0 and pt.y == 0:
                    continue  # undetected keypoint
                rkpx = int(pt.x * sx)
                rkpy = int(pt.y * sy)
                cv2.rectangle(annotated,
                              (rkpx - half, rkpy - half),
                              (rkpx + half, rkpy + half),
                              body_color, body_kp_thick, cv2.LINE_AA)

            # Face bbox + sharpness score label — color mirrors the sharpness pass/fail.
            if p["face_bbox"] is not None:
                fb: Box = p["face_bbox"]
                rfx1 = int(fb.x1 * sx); rfy1 = int(fb.y1 * sy)
                rfx2 = int(fb.x2 * sx); rfy2 = int(fb.y2 * sy)
                cv2.rectangle(annotated, (rfx1, rfy1), (rfx2, rfy2),
                              body_color, face_thick)

                label = f"{p['sharpness_score']:.2f}"
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
                text_y = min(rfy2 + th + baseline + 2, h_out - 1)
                cv2.putText(annotated, label, (rfx1, text_y),
                            font, font_scale, body_color, font_thick, cv2.LINE_AA)

            # Face model keypoint circles: diameter 12 (radius 6), line 3 px.
            # Confident points (>= threshold) use pass color; others use fail color.
            if p["face_kps"] is not None:
                kp_conf_thresh = app_config.face_coverage_conf_threshold
                ann_face: Face = p["face_kps"]
                for pt, kpc in zip(ann_face.points, ann_face.confidences):
                    rkpx = int(pt.x * sx)
                    rkpy = int(pt.y * sy)
                    kp_color = (
                        app_config.annotation_box_color_pass
                        if kpc >= kp_conf_thresh else
                        app_config.annotation_box_color_fail
                    )
                    cv2.circle(annotated, (rkpx, rkpy), kp_radius, kp_color, kp_thick, cv2.LINE_AA)

        out_name    = image_path.stem + ".jpg"
        anno_subdir = boxes_dir / ("anno_blur" if overall_blurry else "anno_sharp")
        anno_subdir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(anno_subdir / out_name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
    # -------------------------------------------------------------------------

    return {
        "file":               str(image_path),
        "status":             "blurry" if overall_blurry else "sharp",
        "sharpness_score":    round(best["sharpness_score"], 4),
        "sharpness_grade":    round(best["sharpness_score"] * 100, 1),
        "laplacian_variance": round(best["lap_var"], 2),
        "tenengrad_score":    round(best["ten"], 2),
        "persons_detail":     persons_detail,
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

_FACE_MODEL_URL  = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
)
_FACE_MODEL_PATH = Path(__file__).parent / "yolov8n-face.pt"


def _ensure_face_model() -> Path:
    """Download yolov8n-face.pt next to this script if not already present."""
    if _FACE_MODEL_PATH.exists():
        log.debug("Face model already present: %s", _FACE_MODEL_PATH)
        return _FACE_MODEL_PATH
    log.info("Downloading yolov8n-face.pt from %s …", _FACE_MODEL_URL)
    urllib.request.urlretrieve(_FACE_MODEL_URL, _FACE_MODEL_PATH)
    log.info("Download complete: %s", _FACE_MODEL_PATH)
    return _FACE_MODEL_PATH


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_EXTENSIONS else []
    if input_path.is_dir():
        return sorted(
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []


def process(
    input_path: Path, sensitivity: str, output_root: Optional[Path] = None
) -> tuple[list[dict], list[dict], Path]:
    """
    Process all images at *input_path*.

    Annotated copies are always saved, sorted into <output_dir>/anno_blur/,
    <output_dir>/anno_sharp/, and <output_dir>/anno_skipped/ sub-folders.

    output_root : if provided, use it directly as the output directory;
                  otherwise a timestamped sub-folder is created under ./output/.

    Returns (all_results, blurry_results, output_directory).
    """
    threshold  = SENSITIVITY_THRESHOLDS[sensitivity]
    files      = collect_images(input_path)
    ts         = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = output_root if output_root is not None else Path("output") / f"{ts}-{input_path.stem}"

    if not files:
        log.warning("No supported image files found in: %s", input_path)
        return [], [], output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    _add_file_logging(output_dir / "run.log")
    log.debug("Output directory: %s", output_dir.resolve())

    width     = len(str(len(files)))  # for aligned progress numbers
    boxes_dir = output_dir

    log.info("Loading models …")
    pose_model = YOLO("yolov8n-pose.pt")
    face_model = YOLO(_ensure_face_model())
    log.debug("pose model: yolov8n-pose.pt | face model: %s", _FACE_MODEL_PATH.name)

    log.info(
        "Processing %d image(s)  |  sensitivity=%s  |  blur threshold=%.2f  |  annotations → %s",
        len(files), sensitivity, threshold, boxes_dir,
    )
    log.debug("Config: face_coverage_min_visible=%d  face_coverage_conf_threshold=%.2f"
              "  face_min_size_fraction=%.3f  normalized_img_max_long_edge=%d",
              app_config.face_coverage_min_visible,
              app_config.face_coverage_conf_threshold,
              app_config.face_min_size_fraction,
              app_config.normalized_img_max_long_edge)

    all_results: list[dict] = []
    blurry:      list[dict] = []
    sharp_count = skip_count = error_count = 0

    for idx, image_path in enumerate(files, 1):
        tag = f"[{idx:>{width}}/{len(files)}] {image_path.name}"
        result = analyse_image(image_path, pose_model, face_model, threshold, boxes_dir)
        all_results.append(result)

        if result["status"] == "blurry":
            result["star_rating"] = 1
            blurry.append(result)
            log.info("%s  →  BLURRY   score=%.3f  grade=%.1f/100",
                     tag, result["sharpness_score"], result["sharpness_grade"])

        elif result["status"] == "sharp":
            sharp_count += 1
            log.info("%s  →  Sharp    score=%.3f  grade=%.1f/100",
                     tag, result["sharpness_score"], result["sharpness_grade"])

        elif result["status"] == "skipped":
            skip_count += 1
            log.info("%s  →  Skipped  (%s)", tag, result.get("reason", ""))

        else:
            error_count += 1
            log.error("%s  →  Error    (%s)", tag, result.get("error", ""))

    log.info("Summary —  Blurry: %d  |  Sharp: %d  |  Skipped: %d  |  Errors: %d",
             len(blurry), sharp_count, skip_count, error_count)
    return all_results, blurry, output_dir


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_MAX_BOXES = 8  # matches top-8 body selection in detect_qualified_persons

_CSV_FIELDS: list[str] = ["File", "Verdict", "Sharp Score", "# Boxes"] + [
    col
    for n in range(1, _MAX_BOXES + 1)
    for col in (
        f"Box {n} - Verdict",
        f"Box {n} - Orig Dimension",
        f"Box {n} - Face Box Sharp Score",
        f"Box {n} - Facial Boxes",
    )
]


def write_csv(all_results: list[dict], csv_path: Path) -> None:
    _VERDICT_MAP = {
        "sharp":   "Sharp",
        "blurry":  "Blur",
        "skipped": "Skipped",
        "error":   "Skipped",
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            verdict = _VERDICT_MAP.get(r["status"], "Skipped")
            score   = r.get("sharpness_score")
            row: dict = {
                "File":        Path(r["file"]).name,
                "Verdict":     verdict,
                "Sharp Score": f"{score:.2f}" if score is not None else "",
                "# Boxes":     len(r.get("persons_detail", [])),
            }
            for n, pd in enumerate(r.get("persons_detail", []), 1):
                row[f"Box {n} - Verdict"]              = pd["verdict"]
                row[f"Box {n} - Orig Dimension"]       = pd["orig_dim"]
                row[f"Box {n} - Face Box Sharp Score"] = f"{pd['score']:.2f}"
                row[f"Box {n} - Facial Boxes"]         = pd["facial_boxes"]
            writer.writerow(row)
    log.info("CSV report written to:    %s", csv_path)


def write_blur_lst(blurry: list[dict], lst_path: Path) -> None:
    """Write a plain-text list of blurry image filenames and their blur scores."""
    with open(lst_path, "w", encoding="utf-8") as fh:
        for entry in blurry:
            filename = Path(entry["file"]).name
            fh.write(f"{filename}\t{entry['sharpness_score']}\n")
    log.info("Blur list written to:     %s", lst_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect blurry sport images by analysing subject sharpness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        help="Path to a single image file or a directory of images.",
    )
    parser.add_argument(
        "--sensitivity",
        choices=["low", "medium", "high"],
        default="medium",
        help=(
            "Detection sensitivity (default: medium).  "
            "high = flag slightly blurry images;  "
            "low  = flag only severely blurry images."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Root directory for all output files (anno_blur/, anno_sharp/, "
            "anno_skipped/, blurry.csv, blur.lst, run.log).  "
            "Defaults to output/<timestamp>-<input_name>/."
        ),
    )
    args = parser.parse_args()

    _setup_console_logging()
    log.debug("Arguments: path=%s sensitivity=%s output=%s",
              args.path, args.sensitivity, args.output)

    input_path = Path(args.path).resolve()
    if not input_path.exists():
        log.error("Path does not exist: %s", input_path)
        sys.exit(1)

    output_root = Path(args.output).resolve() if args.output else None
    all_results, blurry, output_dir = process(input_path, args.sensitivity, output_root)

    if all_results:
        write_csv(all_results, output_dir / "blurry.csv")
    if blurry:
        write_blur_lst(blurry, output_dir / "blur.lst")
    else:
        log.info("No blurry images detected — blur.lst not written.")


if __name__ == "__main__":
    main()
