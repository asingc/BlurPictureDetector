#!/usr/bin/env python3
"""
BlurPictureDetector
-------------------
Detect blurry sport images by analysing the sharpness of the main subject.

Usage:
    python 1_prep_review.py <image_or_directory> [--sensitivity low|medium|high]

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
import json
import logging
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

from algo.config import AppConfig, app_config
from algo.frame import Frame
from algo.models import Body, Box, ColorLab, Face, Point, PredictedKeyPoint
from algo.stage import ProcessStage
from algo.stages.annotation import AnnotationStage
from algo.stages.face_reco import FaceRecoStage
from algo.stages.grading import GradingStage
from algo.stages.image_analysis import ImageAnalysisStage
from algo.stages.jersey_counting import JerseyCountingStage
from algo.scorers import (
    BodyArrayScorer,
    BodyArrayScorerBase,
    BodyHeadKPVisibilityScorer,
    BodyScorerBase,
    FaceLandmarkVisibilityScorer,
    FaceSharpnessScorer,
    FaceSizeScorer,
    JerseyColorScorer,
    MatchedFaceScorer,
)
from algo.sharpness import (
    GeometricMeanEvaluator,
    LaplacianTenengradEvaluator,
    SharpnessEvaluator,
    sharpness_evaluator,
)
from algo.utils import (
    _HEAD_KP_INDICES,
    _matches_allowed_jersey_color,
    _narrow_face_box,
    cap_long_edge,
)

import cv2
import numpy as np
import rawpy
from ultralytics import YOLO

try:
    from algo.facereco import FaceRecoConfig, FaceRecoPipeline
    _FACERECO_AVAILABLE = True
except ImportError:
    _FACERECO_AVAILABLE = False

try:
    from algo.facenet_provider import FaceNetFaceRecoProvider
    _FACENET_AVAILABLE = True
except ImportError:
    _FACENET_AVAILABLE = False

try:
    from algo.dlib_provider import DlibFaceRecoProvider
    _DLIB_AVAILABLE = True
except ImportError:
    _DLIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)

# RAW formats decoded via rawpy rather than OpenCV.
_RAW_EXTENSIONS: frozenset[str] = frozenset({".cr3", ".cr2"})
IMAGE_EXTENSIONS = IMAGE_EXTENSIONS | _RAW_EXTENSIONS

# sharpness_score threshold per sensitivity level.
# A file is flagged as blurry when  sharpness_score <= threshold.
#   low    → only flag severely blurry images  (high tolerance)
#   medium → balanced default
#   high   → flag even slightly blurry images  (low tolerance)
SENSITIVITY_THRESHOLDS: dict[str, float] = {
    "low":    0.35,
    "medium": 0.50,
    "high":   0.70,
}

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
    return cap_long_edge(image, app_config.normalized_img_max_long_edge)


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
                    half_size=True,   # 2× faster decode; still >>1800 px for Canon RAW
                )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            log.debug("[read] rawpy failed for %s: %s", path.name, exc)
            return None
    return cv2.imread(str(path))


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
        body_box = Box.from_px(*boxes[idx], w, h).padded(pad, w, h)
        if kps_data is not None:
            kps_raw   = kps_data[idx]   # (17, 3)
            keypoints = [PredictedKeyPoint(Point.from_px(kps_raw[i, 0], kps_raw[i, 1], w, h), float(kps_raw[i, 2]))
                         for i in range(len(kps_raw))]
        else:
            keypoints = []
        px1, py1, px2, py2 = body_box.as_px_ints(w, h)
        log.debug("[bodies]   body[%d]: bbox=(%d,%d,%d,%d) area=%.0f kps=%d",
                  idx, px1, py1, px2, py2, areas[idx], len(keypoints))
        bodies.append(Body(
            crop=np.empty((0, 0, 3), dtype=np.uint8),  # filled after face matching
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
    Pass 2 — per-face crop: re-run the face model on a padded crop of each
             detected face to obtain higher-quality landmark positions, then
             transform the landmark coordinates back to full-image space.
             Falls back to pass-1 landmarks when pass 2 yields no detection.
    """
    h, w = image.shape[:2]

    # ------------------------------------------------------------------
    # Pass 1: detect faces on the full image.
    # ------------------------------------------------------------------
    face_results = face_model.predict(image, verbose=False)
    if face_results and len(face_results[0].boxes) > 0:
        fdet_boxes = face_results[0].boxes.xyxy.cpu().numpy()   # (M, 4)
        fdet_confs = face_results[0].boxes.conf.cpu().numpy()   # (M,)
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

        # Size filter.
        if app_config.face_min_size_fraction > 0:
            face_long = max(face_box.width, face_box.height)
            if face_long < app_config.face_min_size_fraction:
                px1, py1, px2, py2 = face_box.as_px_ints(w, h)
                log.debug("[faces] face[%d]: %dx%d too small (min=%.3f) — skipped",
                          f_idx, px2 - px1, py2 - py1, app_config.face_min_size_fraction)
                continue

        # ------------------------------------------------------------------
        # Pass 2: re-run face model on a padded crop of this face's bbox to
        # get refined landmark positions.  If pass 2 yields no detection the
        # face is included with an empty landmarks list — FaceLandmarkVisibilityScorer
        # will disqualify it (n_visible() == 0 < min_visible).
        # ------------------------------------------------------------------
        landmarks: list[PredictedKeyPoint] = []
        px1, py1, px2, py2 = face_box.as_px_ints(w, h)
        crop_pad = max(10, int(max(px2 - px1, py2 - py1) * 0.2))
        crop_box = face_box.padded(crop_pad, w, h)
        cx1, cy1, cx2, cy2 = crop_box.as_px_ints(w, h)
        face_crop = image[cy1:cy2, cx1:cx2]

        if face_crop.size > 0:
            crop_results = face_model.predict(face_crop, verbose=False)
            if crop_results and len(crop_results[0].boxes) > 0 and crop_results[0].keypoints is not None:
                # Pick the detection whose centre is closest to the crop centre.
                crop_boxes = crop_results[0].boxes.xyxy.cpu().numpy()
                crop_h, crop_w = face_crop.shape[:2]
                cx_centres = (crop_boxes[:, 0] + crop_boxes[:, 2]) / 2
                cy_centres = (crop_boxes[:, 1] + crop_boxes[:, 3]) / 2
                dists = np.sqrt((cx_centres - crop_w / 2) ** 2 + (cy_centres - crop_h / 2) ** 2)
                best = int(np.argmin(dists))

                kps_raw2 = crop_results[0].keypoints.data.cpu().numpy()[best]  # (5, 3)
                # Translate crop-local coordinates back to full-image space.
                landmarks = [PredictedKeyPoint(Point.from_px(int(kps_raw2[i, 0]) + cx1, int(kps_raw2[i, 1]) + cy1, w, h),
                                               float(kps_raw2[i, 2]))
                             for i in range(len(kps_raw2))]
                log.debug("[faces] face[%d]: pass-2 landmarks from crop (%d,%d,%d,%d)",
                          f_idx, cx1, cy1, cx2, cy2)
            else:
                log.debug("[faces] face[%d]: pass-2 no detection in crop — landmarks empty", f_idx)

        # Note: landmark coverage is intentionally not filtered here.
        # Phase 2 (scoring) decides whether a face with few visible landmarks
        # is usable; we include all faces so it has the full picture.
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
    xs = []
    ys = []
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
    Mutates *bodies* in place by appending to each body's faces list."""
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
    """
    Return (bodies, had_persons).

    had_persons : True if the pose model detected at least one person body.
    bodies      : up to 8 Body objects (largest-first), with matched faces
                  and keypoints populated on a best-effort basis.
                  No threshold-based filtering is applied here — every detected
                  body is returned so that Phase 2 (scoring) can decide which
                  ones to use.
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
    # Phase 4 — finalise: assign face crop where available.
    # No bodies are discarded here — threshold-based filtering is
    # the responsibility of Phase 2 (scoring / analyse_image).
    # ------------------------------------------------------------------
    for body in bodies:
        if body.faces:
            fx1, fy1, fx2, fy2 = body.faces[0].bbox.as_px_ints(image.shape[1], image.shape[0])
            body.crop = image[fy1:fy2, fx1:fx2]
            bx1, by1, bx2, by2 = body.bbox.as_px_ints(image.shape[1], image.shape[0])
            log.debug("[detect]   body bbox=(%d,%d,%d,%d): %d face(s), crop %.0fx%.0f",
                      bx1, by1, bx2, by2,
                      len(body.faces), body.faces[0].bbox.width * image.shape[1], body.faces[0].bbox.height * image.shape[0])
        else:
            bx1, by1, bx2, by2 = body.bbox.as_px_ints(image.shape[1], image.shape[0])
            log.debug("[detect]   body bbox=(%d,%d,%d,%d): no matched face",
                      bx1, by1, bx2, by2)

    log.debug("[detect] result: %d body(ies) returned", len(bodies))
    return bodies, True


# ---------------------------------------------------------------------------
# Cloth colour prediction
# ---------------------------------------------------------------------------

class ClothColorPredictor:
    """
    Predicts the dominant jersey/cloth color for a sharp body.

    Strategy
    --------
    1. Crop the torso region using COCO keypoints 5/6 (shoulders) and 11/12
       (hips) when at least two are confident.  Falls back to the middle band
       of the body bbox (skip top 25 % head, bottom 20 % legs).
    2. Resize the crop to a small 24 × 24 sample grid.
    3. Convert to CIE L*a*b* (perceptually uniform) and assign each pixel to
       the nearest reference color by Euclidean distance in LAB space.
       Skin-tone pixels are skipped to avoid contaminating the vote with bare
       arms or necks.
    4. The colour with the most votes is returned.

    Returns one of: Hue, Shade labels (for example Blue, Navy),
                    or Unknown | N/A.
    """

    _TORSO_KP_INDICES: tuple[int, ...] = (5, 6, 11, 12)  # L/R shoulder, L/R hip
    _TORSO_KP_CONF:    float            = 0.30

    # Reference colours in CIE L*a*b* space.
    _COLORS: list = [
        ColorLab("Red",        "Crimson", ( 40.0,  65.0,  40.0)),
        ColorLab("Orange",     "Vivid",   ( 65.0,  35.0,  55.0)),
        ColorLab("Yellow",     "Gold",    ( 85.0,  -5.0,  75.0)),
        ColorLab("Green",      "Emerald", ( 45.0, -40.0,  25.0)),
        ColorLab("Light Blue", "Sky",     ( 70.0,  -8.0, -30.0)),
        ColorLab("Blue",       "Royal",   ( 35.0,   5.0, -55.0)),
        ColorLab("Blue",       "Navy",    ( 15.0,   5.0, -25.0)),
        ColorLab("Purple",     "Violet",  ( 30.0,  30.0, -35.0)),
        ColorLab("Pink",       "Magenta", ( 55.0,  60.0, -20.0)),
        ColorLab("White",      "Bright",  ( 95.0,   0.0,   0.0)),
        ColorLab("Gray",       "75%",     ( 75.0,   0.0,   0.0)),
        ColorLab("Gray",       "Medium",  ( 50.0,   0.0,   0.0)),
        ColorLab("Gray",       "25%",     ( 25.0,   0.0,   0.0)),
        ColorLab("Black",      "Deep",    (  8.0,   0.0,   0.0)),
    ]

    # Skin-tone cluster center in LAB — pixels within this ΔE distance are skipped.
    _SKIN_LAB:         tuple[float, float, float] = (65.0, 18.0, 22.0)
    _SKIN_DIST_THRESH: float                      = 35.0

    def predict(self, body: Body, normalized_image: np.ndarray) -> tuple[str, dict]:
        torso = self._torso_crop(body, normalized_image)
        if torso is None or torso.size == 0:
            return "N/A", {}
        sample = cv2.resize(torso, (24, 24), interpolation=cv2.INTER_AREA)
        # float32 input → OpenCV returns true CIE L*a*b* values (L: 0-100, a/b: ±127)
        lab    = cv2.cvtColor(sample.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape(-1, 3)  # (576, 3)

        colors = self._COLORS
        refs   = np.array([c.lab for c in colors], dtype=np.float32)
        skin   = np.array(self._SKIN_LAB, dtype=np.float32)

        # Nearest reference color per pixel: (576, N_colors, 3) → (576,)
        diffs       = pixels[:, None, :] - refs[None, :, :]
        nearest_idx = (diffs ** 2).sum(axis=2).argmin(axis=1)

        # Skin exclusion mask
        skin_dists = np.sqrt(((pixels - skin) ** 2).sum(axis=1))
        is_skin    = skin_dists < self._SKIN_DIST_THRESH

        votes: dict[int, int] = {}
        valid_pixels: list[np.ndarray] = []
        for i, (idx, skip) in enumerate(zip(nearest_idx.tolist(), is_skin.tolist())):
            if skip:
                continue
            votes[idx] = votes.get(idx, 0) + 1
            valid_pixels.append(pixels[i])

        if not votes:
            return "Unknown", {"votes": {}, "mean_lab": None}
        winner_idx = max(votes, key=votes.__getitem__)
        winner     = colors[winner_idx]
        mean_lab   = (
            [round(float(v), 1) for v in np.mean(valid_pixels, axis=0)]
            if valid_pixels else None
        )
        votes_by_label = {colors[k].label: v for k, v in votes.items()}
        return winner.label, {"votes": votes_by_label, "mean_lab": mean_lab}

    def _torso_crop(self, body: Body, image: np.ndarray) -> np.ndarray | None:
        h_img, w_img = image.shape[:2]
        kps  = body.keypoints
        pts = [
            (kps[i].point.x * w_img, kps[i].point.y * h_img)
            for i in self._TORSO_KP_INDICES
            if i < len(kps) and kps[i].confidence >= self._TORSO_KP_CONF
        ]
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1 = max(0, int(np.floor(min(xs))))
            y1 = max(0, int(np.floor(min(ys))))
            # Upper bounds are exclusive for numpy slicing.
            x2 = min(w_img, int(np.ceil(max(xs))))
            y2 = min(h_img, int(np.ceil(max(ys))))
        else:
            b  = body.bbox
            bh = b.y2 - b.y1
            x1 = int(np.floor(b.x1 * w_img))
            x2 = int(np.ceil(b.x2 * w_img))
            # Skip head and lower legs in normalized coordinates, then convert to px.
            y1 = int(np.floor((b.y1 + (bh * 0.25)) * h_img))
            y2 = int(np.ceil((b.y2 - (bh * 0.20)) * h_img))

        x1 = max(0, min(x1, w_img))
        x2 = max(0, min(x2, w_img))
        y1 = max(0, min(y1, h_img))
        y2 = max(0, min(y2, h_img))
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]


cloth_color_predictor = ClothColorPredictor()



# ---------------------------------------------------------------------------
# Per-image analysis
# ---------------------------------------------------------------------------

def analyse_image(
    image_path: Path,
    pose_model: YOLO,
    face_model: YOLO,
) -> dict:
    """
    Analyse a single image file.

    Evaluates up to 8 qualified persons (largest face-visible body regions).
    The image is considered sharp if ANY of them is sharp.
    Reported metrics come from the sharpest (highest sharpness_score) person.

    Returns a dict with at minimum the keys 'file', 'status', and
    '_annotation_data'.  '_annotation_data' carries the per-person bounding
    boxes and keypoints needed by annotate_image() to draw previews.
    status is one of: 'analysed', 'skipped', 'error'.
    """
    image_orig = _read_image(image_path)
    if image_orig is None:
        log.error("[analyse] %s — cannot read image file", image_path.name)
        return {"file": str(image_path), "status": "error", "error": "Cannot read image file", "persons_detail": [], "_annotation_data": None}

    log.debug("[analyse] %s — original size: %dx%d", image_path.name, image_orig.shape[1], image_orig.shape[0])
    # Keep the original for annotation; resize a working copy for processing (file untouched).
    normalized_img = normalize_img_size(image_orig)
    log.debug("[analyse] %s — processing size: %dx%d", image_path.name, normalized_img.shape[1], normalized_img.shape[0])

    persons, had_persons = detect_qualified_persons(normalized_img, pose_model, face_model)
    log.debug("[analyse] %s — had_persons=%s, bodies=%d",
              image_path.name, had_persons, len(persons))
    if not had_persons:
        log.debug("[analyse] %s — skipped: no person detected", image_path.name)
        return {"file": str(image_path), "status": "skipped", "reason": "No person detected", "persons_detail": [], "_annotation_data": None}

    # Run extraction scoring only (no jersey filtering, no blur threshold gating).
    # Threshold and jersey-based pass/fail are applied in process() phase 2.
    scorer = BodyArrayScorer([
        MatchedFaceScorer(),
        FaceSizeScorer(),
        BodyHeadKPVisibilityScorer(),
    #FaceLandmarkVisibilityScorer(),
        FaceSharpnessScorer(-1.0),
    ])
    persons = scorer.process(normalized_img, persons)

    # Predict cloth colour for ALL bodies during extraction.
    for person in persons:
        if not person.passed:
            continue

        person.cloth_color, person.cloth_color_detail = cloth_color_predictor.predict(person, normalized_img)
        bx1, by1, bx2, by2 = person.bbox.as_px_ints(normalized_img.shape[1], normalized_img.shape[0])
        log.debug("[colour] %s — body bbox=(%d,%d,%d,%d) cloth_color=%s votes=%s",
                  image_path.name,
                  bx1, by1, bx2, by2,
                  person.cloth_color, person.cloth_color_detail.get("votes", {}))
        
    evaluated: list[dict] = []
    for person in persons:
        qualified_for_sharpness = bool(person.passed)
        bx1, by1, bx2, by2 = person.bbox.as_px_ints(normalized_img.shape[1], normalized_img.shape[0])
        log.debug("[analyse] %s — person bbox=%s best_score=%.4f qualified=%s",
                  image_path.name, (bx1, by1, bx2, by2), person.sharpness_score, qualified_for_sharpness)
        evaluated.append({
            "body_bbox":       person.bbox,
            "body_keypoints":  person.keypoints,
            "face_bbox":       person.best_face.bbox if person.best_face else None,
            "narrow_face_bbox": person.best_narrow_box,
            "face_kps":        person.best_face,
            "sharpness_score": person.sharpness_score,
            "lap_var":         person.lap_var,
            "ten":             person.ten,
            "qualified_for_sharpness": qualified_for_sharpness,
            "is_blurry":       True,
            "cloth_color":     person.cloth_color,
            "cloth_color_detail": person.cloth_color_detail,
        })

    # Keep extracted metrics; final status/verdict is assigned in process() phase 2.
    best = max(evaluated, key=lambda p: p["sharpness_score"])
    log.debug("[analyse] %s — extracted best_score=%.4f (%d person(s))",
              image_path.name, best["sharpness_score"], len(evaluated))

    # Build per-person detail rows for CSV output.l
    persons_detail: list[dict] = []
    for p in evaluated:
        b: Box = p["body_bbox"]
        if p["face_kps"] is not None:
            kp_confs = ", ".join(f"{lm.confidence:.2f}" for lm in p["face_kps"].landmarks)
        else:
            kp_confs = ""
        persons_detail.append({
            "verdict":      "Pending",
            "orig_dim":     f"{b.width * normalized_img.shape[1]:.0f} x {b.height * normalized_img.shape[0]:.0f}",
            "score":        round(p["sharpness_score"], 2),
            "facial_boxes": kp_confs,
        })

    return {
        "file":               str(image_path),
        "status":             "analysed",
        "sharpness_score":    round(best["sharpness_score"], 4),
        "sharpness_grade":    round(best["sharpness_score"] * 100, 1),
        "laplacian_variance": round(best["lap_var"], 2),
        "tenengrad_score":    round(best["ten"], 2),
        "persons_detail":     persons_detail,
        "_annotation_data": {
            "evaluated":        evaluated,
            "overall_blurry":   True,
            "processing_shape": normalized_img.shape[:2],
        },
    }


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate_image(result: dict, boxes_dir: Path, jersey_colors: frozenset[str] = frozenset()) -> None:
    """
    Write an annotated preview for one image result.

    'skipped' — saves the original (unannotated) image to anno_skipped/.
    'blurry' / 'sharp' — draws body boxes, face boxes, keypoints, and
        sharpness scores, then saves to anno_blur/ or anno_sharp/.
    'error' — does nothing (no image to read).
    """
    status = result["status"]
    if status == "error":
        return

    image_path = Path(result["file"])
    image_orig = _read_image(image_path)
    if image_orig is None:
        log.warning("[annotate] cannot re-read %s — skipping annotation", image_path.name)
        return

    if status == "skipped":
        subdir = boxes_dir / "anno_skipped"
        subdir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(subdir / (image_path.stem + ".jpg")), image_orig, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return

    # status is 'blurry' or 'sharp'
    ann_data       = result["_annotation_data"]
    evaluated      = ann_data["evaluated"]
    overall_blurry = ann_data["overall_blurry"]

    # Coordinates are stored as fractions of image width/height, so convert
    # directly to the original image dimensions here.
    annotated = image_orig.copy()
    overlay   = annotated.copy()   # all drawing goes here; blended back at the end
    h_out, w_out = annotated.shape[:2]
    sx = w_out
    sy = h_out
    ann_scale = max(h_out, w_out) / app_config.normalized_img_max_long_edge

    face_thick    = max(1, round(app_config.annotation_face_box_thickness * ann_scale))
    box_thick     = max(1, round(app_config.annotation_box_thickness * ann_scale))
    icon_size     = max(20, round(app_config.annotation_icon_size * ann_scale))
    kp_radius     = max(3, round(app_config.annotation_face_kp_radius    * ann_scale))
    kp_thick      = max(1, round(app_config.annotation_face_kp_thickness * ann_scale))
    body_kp_size  = max(2, round(app_config.annotation_body_kp_size         * ann_scale))
    body_kp_thick = max(1, round(app_config.annotation_body_kp_thickness    * ann_scale))
    skeleton_thick = max(1, round(app_config.annotation_skeleton_thickness   * ann_scale))
    narrow_thick  = max(1, round(app_config.annotation_narrow_face_box_thickness * ann_scale))
    font          = cv2.FONT_HERSHEY_SIMPLEX
    font_thick    = app_config.annotation_score_font_thickness
    # Compute font scale so text height == annotation_score_font_size_px scaled to image.
    (_, _base_h), _ = cv2.getTextSize("Mg", font, 1.0, font_thick)
    font_scale = app_config.annotation_score_font_size_px * ann_scale / max(_base_h, 1)

    score_labels: list[tuple] = []  # (text, x, y, color) — drawn opaquely after blend
    for p in evaluated:
        # Requirement: only annotate bodies whose jersey color is in the allowed set.
        if jersey_colors and not _matches_allowed_jersey_color(p.get("cloth_color", "N/A"), jersey_colors):
            continue
        b: Box = p["body_bbox"]
        rbx1 = int(b.x1 * sx); rby1 = int(b.y1 * sy)
        rbx2 = int(b.x2 * sx); rby2 = int(b.y2 * sy)

        body_color = (
            app_config.annotation_box_color_fail
            if p["is_blurry"] else
            app_config.annotation_box_color_pass
        )
        cv2.rectangle(overlay, (rbx1, rby1), (rbx2, rby2), body_color, box_thick)

        # Every person gets its own pass/fail status icon.
        _draw_status_icon(overlay, rbx1, rby1, icon_size, body_color, passed=not p["is_blurry"])

        # Skeleton lines between connected body keypoints.
        kps = p["body_keypoints"]
        for ka, kb in _COCO_SKELETON:
            if ka >= len(kps) or kb >= len(kps):
                continue
            pa, pb = kps[ka].point, kps[kb].point
            if (pa.x == 0 and pa.y == 0) or (pb.x == 0 and pb.y == 0):
                continue
            cv2.line(overlay,
                     (int(pa.x * sx), int(pa.y * sy)),
                     (int(pb.x * sx), int(pb.y * sy)),
                     body_color, skeleton_thick, cv2.LINE_AA)

        # Body keypoints as small squares (sharp/blurry color scheme).
        half = body_kp_size // 2
        for kp in p["body_keypoints"]:
            if kp.point.x == 0 and kp.point.y == 0:
                continue  # undetected keypoint
            rkpx = int(kp.point.x * sx)
            rkpy = int(kp.point.y * sy)
            cv2.rectangle(overlay,
                          (rkpx - half, rkpy - half),
                          (rkpx + half, rkpy + half),
                          body_color, body_kp_thick, cv2.LINE_AA)

        # Face bbox + sharpness score label.
        if p["face_bbox"] is not None:
            fb: Box = p["face_bbox"]
            rfx1 = int(fb.x1 * sx); rfy1 = int(fb.y1 * sy)
            rfx2 = int(fb.x2 * sx); rfy2 = int(fb.y2 * sy)

            nfb: Box | None = p["narrow_face_bbox"]
            if app_config.use_narrow_face_box and nfb is not None:
                rnx1 = int(nfb.x1 * sx); rny1 = int(nfb.y1 * sy)
                rnx2 = int(nfb.x2 * sx); rny2 = int(nfb.y2 * sy)
                cv2.rectangle(overlay, (rnx1, rny1), (rnx2, rny2), body_color, narrow_thick)
                label_x, label_bottom = rnx1, rny2
            else:
                cv2.rectangle(overlay, (rfx1, rfy1), (rfx2, rfy2), body_color, face_thick)
                label_x, label_bottom = rfx1, rfy2

            label = f"{p['sharpness_score']:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
            text_y = min(label_bottom + th + baseline + 2, h_out - 1)
            score_labels.append((label, label_x, text_y, body_color))

        # Cloth colour label — top-right corner of the body box.
        cloth = p.get("cloth_color", "N/A")
        if cloth not in ("N/A", "Unknown"):
            (clw, clh), _ = cv2.getTextSize(cloth, font, font_scale, font_thick)
            cl_x = max(rbx1, rbx2 - clw - 4)
            cl_y = max(clh + 4, rby1 + clh + 4)
            score_labels.append((cloth, cl_x, cl_y, body_color))

        # Face model keypoint circles.
        if p["face_kps"] is not None:
            kp_conf_thresh = app_config.face_coverage_conf_threshold
            ann_face: Face = p["face_kps"]
            for lm in ann_face.landmarks:
                rkpx = int(lm.point.x * sx)
                rkpy = int(lm.point.y * sy)
                kp_color = (
                    app_config.annotation_box_color_pass
                    if lm.confidence >= kp_conf_thresh else
                    app_config.annotation_box_color_fail
                )
                cv2.circle(overlay, (rkpx, rkpy), kp_radius, kp_color, kp_thick, cv2.LINE_AA)

    # Blend annotations onto the clean image.
    cv2.addWeighted(overlay, app_config.annotation_alpha,
                    annotated, 1.0 - app_config.annotation_alpha, 0, annotated)

    # Draw score labels opaquely on top of the blended result.
    for label, lx, ly, lcolor in score_labels:
        cv2.putText(annotated, label, (lx, ly), font, font_scale, lcolor, font_thick, cv2.LINE_AA)

    out_name    = image_path.stem + ".jpg"
    anno_subdir = boxes_dir / ("anno_blur" if overall_blurry else "anno_sharp")
    anno_subdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(anno_subdir / out_name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])


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
    input_path: Path, sensitivity: str, output_root: Optional[Path] = None,
    jersey_colors: frozenset[str] = frozenset(),
) -> tuple[list[dict], list[dict], Path, str | None, float]:
    """
    Process all images at *input_path*.

    Annotated copies are always saved, sorted into <output_dir>/anno_blur/,
    <output_dir>/anno_sharp/, and <output_dir>/anno_skipped/ sub-folders.

    output_root : if provided, use it directly as the output directory;
                  otherwise a timestamped sub-folder is created under ./output/.

    Returns (all_results, blurry_results, output_directory).
    """
    try:
        threshold = float(sensitivity)
    except ValueError:
        threshold = SENSITIVITY_THRESHOLDS[sensitivity]
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
    skip_count = error_count = 0

    # ── Phase 1: extract data for all images ────────────────────────────────
    log.info("Phase 1/3 — extraction (detect + measure + color votes) …")
    for idx, image_path in enumerate(files, 1):
        tag = f"[{idx:>{width}}/{len(files)}] {image_path.name}"
        result = analyse_image(image_path, pose_model, face_model)
        all_results.append(result)

        if result["status"] == "skipped":
            skip_count += 1
            log.info("%s  →  Skipped  (%s)", tag, result.get("reason", ""))

        elif result["status"] == "error":
            error_count += 1
            log.error("%s  →  Error    (%s)", tag, result.get("error", ""))

        else:
            log.info("%s  →  Analysed  extracted_score=%.3f",
                     tag, result.get("sharpness_score", 0.0))

    analysed_count = sum(1 for r in all_results if r.get("status") == "analysed")
    log.info("Extraction summary —  Analysed: %d  |  Skipped: %d  |  Errors: %d",
             analysed_count, skip_count, error_count)

    jersey_color_filter = ";".join(sorted(jersey_colors)) if jersey_colors else None
    log.info("Jersey colour filter: %s", jersey_color_filter or "(none)")

    # Poll dominant jersey color between extraction and verdict phases.
    polled_jersey_color = _compute_jersey_color(all_results)
    log.info("Polled jersey colour (all evaluated bodies): %s", polled_jersey_color)

    # ── Phase 2: assign final sharp/blurry verdicts from extracted data ─────
    log.info("Phase 2/3 — verdicts from extracted data …")
    blurry = _recompute_verdicts(
        all_results,
        our_jersey_color=polled_jersey_color,
        threshold=threshold,
        jersey_colors=jersey_colors,
    )
    sharp_count = sum(1 for r in all_results if r.get("status") == "sharp")

    log.info("Summary —  Blurry: %d  |  Sharp: %d  |  Skipped: %d  |  Errors: %d",
             len(blurry), sharp_count, skip_count, error_count)

    # ── Phase 3: write annotated previews ────────────────────────────────────
    log.info("Phase 3/3 — writing annotated previews …")
    for idx, result in enumerate(all_results, 1):
        log.debug("[annotate] [%d/%d] %s", idx, len(all_results), Path(result["file"]).name)
        annotate_image(result, boxes_dir, jersey_colors)

    return all_results, blurry, output_dir, polled_jersey_color, threshold


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


def write_csv(frames: list[Frame], csv_path: Path) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for frame in frames:
            norm_img = frame.normalized_image
            img_w = norm_img.shape[1] if norm_img is not None else 1
            img_h = norm_img.shape[0] if norm_img is not None else 1
            if not frame.bodies:
                verdict, score = "Skipped", None
            elif frame.is_sharp():
                verdict = "Sharp"
                score = max(b.sharpness_score for b in frame.bodies if b.passed)
            else:
                verdict = "Blur"
                score = max(b.sharpness_score for b in frame.bodies)
            row: dict = {
                "File":        frame.path.name,
                "Verdict":     verdict,
                "Sharp Score": f"{score:.2f}" if score is not None else "",
                "# Boxes":     len(frame.bodies),
            }
            for n, body in enumerate(frame.bodies, 1):
                b = body.bbox
                kp_confs = (
                    ", ".join(f"{lm.confidence:.2f}" for lm in body.best_face.landmarks)
                    if body.best_face else ""
                )
                row[f"Box {n} - Verdict"]              = "Sharp" if body.passed else "Blur"
                row[f"Box {n} - Orig Dimension"]       = f"{b.width * img_w:.0f} x {b.height * img_h:.0f}"
                row[f"Box {n} - Face Box Sharp Score"] = f"{body.sharpness_score:.2f}"
                row[f"Box {n} - Facial Boxes"]         = kp_confs
            writer.writerow(row)
    log.info("CSV report written to:    %s", csv_path)


def write_blur_lst(frames: list[Frame], lst_path: Path) -> None:
    """Write a plain-text list of blurry image filenames and their blur scores."""
    with open(lst_path, "w", encoding="utf-8") as fh:
        for frame in frames:
            if frame.bodies and not frame.is_sharp():
                score = max(b.sharpness_score for b in frame.bodies)
                fh.write(f"{frame.path.name}\t{score}\n")
    log.info("Blur list written to:     %s", lst_path)


def _serial_box(b: Box) -> dict:
    return {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2}


def _serial_keypoint(kp: PredictedKeyPoint) -> dict:
    return {"x": kp.point.x, "y": kp.point.y, "conf": kp.confidence, "passed": kp.passed}


def _serial_face(f: Face) -> dict:
    return {
        "bbox":       _serial_box(f.bbox),
        "confidence": f.confidence,
        "landmarks":  [_serial_keypoint(lm) for lm in f.landmarks],
        "passed":     f.passed,
    }


def _serial_annotation_data(ann: dict) -> dict:
    return {
        "processing_shape": list(ann["processing_shape"]),
        "overall_blurry":   ann["overall_blurry"],
        "evaluated": [
            {
                "body_bbox":        _serial_box(p["body_bbox"]),
                "body_keypoints":   [_serial_keypoint(kp) for kp in p["body_keypoints"]],
                "face_bbox":        _serial_box(p["face_bbox"]) if p["face_bbox"] else None,
                "narrow_face_bbox": _serial_box(p["narrow_face_bbox"]) if p["narrow_face_bbox"] else None,
                "face_kps":         _serial_face(p["face_kps"]) if p["face_kps"] else None,
                "sharpness_score":  p["sharpness_score"],
                "lap_var":          p["lap_var"],
                "ten":              p["ten"],
                "is_blurry":        p["is_blurry"],
                "cloth_color":      p.get("cloth_color", "N/A"),
                "cloth_color_detail": p.get("cloth_color_detail", {}),
            }
            for p in ann["evaluated"]
        ],
    }


class _NumpyEncoder(json.JSONEncoder):
    """Encode numpy scalar types as their Python equivalents."""
    def default(self, obj: object) -> object:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_results_json(frames: list[Frame], json_path: Path, *, our_jersey_color: str | None = None) -> None:
    """Write full per-image analytic results (scores, bboxes, keypoints) to JSON."""
    serializable = []
    for frame in frames:
        norm_img = frame.normalized_image
        if not frame.bodies:
            entry: dict = {"file": str(frame.path), "status": "skipped"}
        else:
            overall_blurry = not frame.is_sharp()
            passing = [b for b in frame.bodies if b.passed]
            best = max(passing or frame.bodies, key=lambda b: b.sharpness_score)
            entry = {
                "file":               str(frame.path),
                "status":             "blurry" if overall_blurry else "sharp",
                "sharpness_score":    round(best.sharpness_score, 4),
                "sharpness_grade":    round(best.sharpness_score * 100, 1),
                "laplacian_variance": round(best.lap_var, 2),
                "tenengrad_score":    round(best.ten, 2),
                "annotation_data": {
                    "processing_shape": list(norm_img.shape[:2]) if norm_img is not None else [0, 0],
                    "overall_blurry":   overall_blurry,
                    "evaluated": [
                        {
                            "body_bbox":          _serial_box(body.bbox),
                            "body_keypoints":     [_serial_keypoint(kp) for kp in body.keypoints],
                            "face_bbox":          _serial_box(body.best_face.bbox) if body.best_face else None,
                            "narrow_face_bbox":   _serial_box(body.best_narrow_box) if body.best_narrow_box else None,
                            "face_kps":           _serial_face(body.best_face) if body.best_face else None,
                            "sharpness_score":    body.sharpness_score,
                            "lap_var":            body.lap_var,
                            "ten":                body.ten,
                            "is_blurry":          not body.passed,
                            "cloth_color":        body.cloth_color,
                            "cloth_color_detail": body.cloth_color_detail,
                        }
                        for body in frame.bodies
                    ],
                },
            }
        serializable.append(entry)
    payload = {
        "our_jersey_color": our_jersey_color,
        "results": serializable,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, cls=_NumpyEncoder)
    log.info("Results JSON written to:  %s", json_path)


def _compute_jersey_color(all_results: list[dict]) -> str:
    """
    Tally cloth colors across all evaluated bodies and return the most common one.
    Results labeled 'Unknown' or 'N/A' are excluded from the tally.
    Returns 'Unknown' when no evaluated body has a usable color.
    """
    counts: dict[str, int] = {}
    for r in all_results:
        ann = r.get("_annotation_data")
        if ann is None:
            continue
        for p in ann["evaluated"]:
            color = p.get("cloth_color", "N/A")
            if color not in ("N/A", "Unknown"):
                counts[color] = counts.get(color, 0) + 1
    if not counts:
        return "Unknown"
    summary = "  ".join(f"{c}={n}" for c, n in sorted(counts.items(), key=lambda x: -x[1]))
    log.info("Jersey colour distribution (all evaluated persons): %s", summary)
    return max(counts, key=counts.__getitem__)


def _recompute_verdicts(
    all_results: list[dict],
    our_jersey_color: str,
    *,
    threshold: float,
    jersey_colors: frozenset[str],
) -> list[dict]:
    """
    Re-evaluate image-level verdicts from extracted phase-1 data.

    A body passes only when all of these hold:
    1) it was qualified in phase 1 (face/keypoint/size checks),
    2) it matches --jerseycolor allow-list (when provided),
    3) it matches the polled jersey color (when known),
    4) its sharpness score is above threshold.
    """
    blurry: list[dict] = []
    for r in all_results:
        if r.get("status") not in ("analysed", "blurry", "sharp"):
            continue
        ann = r.get("_annotation_data")
        if ann is None:
            continue

        has_our_player = False
        best_pass_score = -1.0
        best_pass_lap = 0.0
        best_pass_ten = 0.0
        best_any_score = 0.0
        best_any_lap = 0.0
        best_any_ten = 0.0

        for p in ann["evaluated"]:
            score = float(p.get("sharpness_score", 0.0))
            if score > best_any_score:
                best_any_score = score
                best_any_lap = float(p.get("lap_var", 0.0))
                best_any_ten = float(p.get("ten", 0.0))

            passes = bool(p.get("qualified_for_sharpness", False))
            color = p.get("cloth_color", "N/A")

            if passes and jersey_colors and not _matches_allowed_jersey_color(color, jersey_colors):
                passes = False
            if passes and our_jersey_color != "Unknown" and color != our_jersey_color:
                passes = False
            if passes and score <= threshold:
                passes = False

            p["is_blurry"] = not passes
            if passes:
                has_our_player = True
                if score > best_pass_score:
                    best_pass_score = score
                    best_pass_lap = float(p.get("lap_var", 0.0))
                    best_pass_ten = float(p.get("ten", 0.0))

        new_blurry = not has_our_player
        ann["overall_blurry"] = new_blurry
        r["status"] = "blurry" if new_blurry else "sharp"

        # Keep top-line metrics aligned with final pass/fail logic.
        if new_blurry:
            r["sharpness_score"] = round(best_any_score, 4)
            r["laplacian_variance"] = round(best_any_lap, 2)
            r["tenengrad_score"] = round(best_any_ten, 2)
        else:
            r["sharpness_score"] = round(best_pass_score, 4)
            r["laplacian_variance"] = round(best_pass_lap, 2)
            r["tenengrad_score"] = round(best_pass_ten, 2)
        r["sharpness_grade"] = round(float(r["sharpness_score"]) * 100, 1)

        # Update per-person CSV verdict text for phase-2 outcome.
        for pd, p in zip(r.get("persons_detail", []), ann["evaluated"]):
            pd["verdict"] = "Blur" if p.get("is_blurry", True) else "Sharp"

        if new_blurry:
            blurry.append(r)
    return blurry


def write_info_json(
    frames: list[Frame],
    input_path: Path,
    timestamp: str,
    json_path: Path,
    our_jersey_color: str | None = None,
) -> None:
    """Write a run-summary JSON file."""
    def _entry(frame: Frame) -> dict:
        return {"src": frame.path.name, "anno": frame.path.stem + ".jpg"}

    blur_files    = [_entry(f) for f in frames if f.bodies and not f.is_sharp()]
    sharp_files   = [_entry(f) for f in frames if f.bodies and f.is_sharp()]
    skipped_files = [_entry(f) for f in frames if not f.bodies]

    payload = {
        "SrcDir":         str(input_path.resolve()),
        "SrcType":        "File" if input_path.is_file() else "Directory",
        "Timestamp":      timestamp,
        "OurJerseyColor": our_jersey_color,
        "Anno_Blur":      blur_files,
        "Anno_Sharp":     sharp_files,
        "Anno_Skipped":   skipped_files,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=4)
    log.info("Info JSON written to:     %s", json_path)


# ---------------------------------------------------------------------------
# Face Recognition
# ---------------------------------------------------------------------------

def _run_facereco(
    output_dir: Path,
    sensitivity_threshold: float,
    face_db_dir: Path | None = None,
    face_db_match_threshold: float = 0.72,
) -> None:
    """Run face recognition pipeline on the Phase 1 output."""
    if not _FACERECO_AVAILABLE:
        log.warning(
            "Face recognition disabled: facereco module not available. "
            "Install with: pip install facenet-pytorch"
        )
        return

    try:
        log.info("Starting face recognition clustering...")
        if _FACENET_AVAILABLE:
            provider = FaceNetFaceRecoProvider()
        elif _DLIB_AVAILABLE:
            log.warning("FaceNet provider unavailable; falling back to dlib for this run.")
            provider = DlibFaceRecoProvider()
        else:
            log.warning("No FaceReco provider available. Install facenet-pytorch or face-recognition + dlib.")
            return
        config = FaceRecoConfig(
            cluster_similarity_threshold=0.72,
            face_buffer_ratio=0.15,
            face_db_dir=face_db_dir,
            face_db_match_threshold=face_db_match_threshold,
        )
        pipeline = FaceRecoPipeline(provider=provider, config=config)
        facereco_dir = pipeline.run(output_dir)
        log.info("Face recognition complete: %s", facereco_dir)
    except Exception as exc:
        log.error("Face recognition failed: %s", exc, exc_info=True)


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
    def _sensitivity_type(value: str) -> str:
        if value in SENSITIVITY_THRESHOLDS:
            return value
        try:
            float(value)
            return value
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"sensitivity must be low/medium/high or a numeric threshold (0–1), got {value!r}"
            )

    parser.add_argument(
        "--sensitivity",
        type=_sensitivity_type,
        default="medium",
        metavar="low|medium|high|<threshold>",
        help=(
            "Detection sensitivity (default: medium).  "
            "Use low/medium/high, or supply a numeric threshold directly (0–1, "
            "e.g. 0.45).  Scores <= threshold are flagged as blurry."
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
    parser.add_argument(
        "--jerseycolor",
        default="blue;white;+purple;+orange;+light blue;+pink",
        metavar="COLOR[;COLOR...]",
        help=(
            "Semicolon-separated list of jersey colours that qualify for "
            "evaluation and annotation (case-insensitive).  "
            "Each entry can be a plain colour (e.g. blue) or a full hue+shade "
            "label (e.g. navy).  "
            "Prefix a colour with '+' to make it forced-include — it will always "
            "be in the filter regardless of what other colours are listed "
            "(e.g. +light blue;+pink for goalies).  "
            "Pass an empty string to disable jersey filtering."
        ),
    )
    parser.add_argument(
        "--skip-facereco",
        action="store_true",
        help=(
            "Skip face recognition clustering. By default, .FaceReco/ is generated "
            "under the output directory after preview generation completes. "
            "Use this flag to disable it (e.g. if dlib is not installed)."
        ),
    )
    parser.add_argument(
        "--face-db",
        default=None,
        metavar="DIR",
        help=(
            "Path to a face-DB directory.  Each sub-directory must represent a "
            "person and contain a face.json with positive embeddings.  "
            "Matched clusters will be stored in a folder named after that person."
        ),
    )
    parser.add_argument(
        "--face-db-match-threshold",
        type=float,
        default=0.72,
        metavar="THRESHOLD",
        help=(
            "Cosine similarity threshold for matching a cluster against the face DB "
            "(default: 0.72).  Higher = stricter matching."
        ),
    )
    parser.add_argument(
        "--noteam",
        action="store_true",
        help=(
            "Disable jersey-colour filtering. When set, bodies are never "
            "disqualified for wearing the wrong colour — all detected persons "
            "are scored regardless of jersey colour. Overrides --jerseycolor."
        ),
    )
    args = parser.parse_args()

    _setup_console_logging()
    log.debug("Arguments: path=%s sensitivity=%s output=%s jerseycolor=%s skip_facereco=%s noteam=%s face_db=%s",
              args.path, args.sensitivity, args.output, args.jerseycolor, args.skip_facereco, args.noteam, args.face_db)

    input_path = Path(args.path).resolve()
    if not input_path.exists():
        log.error("Path does not exist: %s", input_path)
        sys.exit(1)

    # Parse the semicolon-separated jersey colour list, normalise to title-case.
    # Colours prefixed with '+' are forced-include: they are always added to the
    # filter regardless of what other colours are listed (e.g. goalie colours).
    # --noteam overrides everything and disables all colour filtering.
    if args.noteam:
        jersey_colors: frozenset[str] = frozenset()
        log.info("--noteam: jersey-colour filtering disabled")
    else:
        forced_colors: frozenset[str] = frozenset(
            c.strip().lstrip("+").strip().title()
            for c in (args.jerseycolor or "").split(";")
            if c.strip().startswith("+") and c.strip().lstrip("+").strip()
        )
        regular_colors: frozenset[str] = frozenset(
            c.strip().title()
            for c in (args.jerseycolor or "").split(";")
            if c.strip() and not c.strip().startswith("+")
        )
        jersey_colors: frozenset[str] = regular_colors | forced_colors
        if forced_colors:
            log.debug("Jersey colours — regular: %s  forced (+): %s",
                      sorted(regular_colors), sorted(forced_colors))
        else:
            log.debug("Jersey colours: %s", sorted(jersey_colors))

    try:
        sensitivity_threshold = float(args.sensitivity)
    except ValueError:
        sensitivity_threshold = SENSITIVITY_THRESHOLDS[args.sensitivity]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = Path(args.output).resolve() if args.output else None
    output_dir  = output_root if output_root is not None else Path("output") / f"{ts}-{input_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _add_file_logging(output_dir / "run.log")
    log.debug("Output directory: %s", output_dir.resolve())

    log.info("Loading models …")
    pose_model = YOLO("yolov8n-pose.pt")
    face_model = YOLO(_ensure_face_model())

    face_db_dir: Path | None = None
    if args.face_db is not None:
        face_db_dir = Path(args.face_db).resolve()
        if not face_db_dir.is_dir():
            log.error("--face-db directory not found: %s", face_db_dir)
            face_db_dir = None

    jersey_stage = JerseyCountingStage(jersey_colors)
    stages: list[ProcessStage] = [
        ImageAnalysisStage(input_path, pose_model, face_model),
        GradingStage(sensitivity_threshold),
        jersey_stage,
        AnnotationStage(output_dir, jersey_colors),
    ]
    frames: list[Frame] = []
    for stage in stages:
        frames = stage.process(frames, app_config)

    our_jersey_color = jersey_stage.our_color
    write_results_json(frames, output_dir / "results.json", our_jersey_color=our_jersey_color)

    if frames:
        write_csv(frames, output_dir / "blurry.csv")
        write_info_json(frames, input_path, datetime.now().strftime("%Y%m%d-%H%M%S"),
                        output_dir / "info.json", our_jersey_color=our_jersey_color)
    if any(f.bodies and not f.is_sharp() for f in frames):
        write_blur_lst(frames, output_dir / "blur.lst")
    else:
        log.info("No blurry images detected — blur.lst not written.")

    if frames:
        log.info("")
        log.info("Annotated previews saved to:")
        log.info("  %s", output_dir / "anno_blur")
        log.info("  %s", output_dir / "anno_sharp")
        log.info("  %s", output_dir / "anno_skipped")
        log.info("")
        log.info("Review the previews, then delete images you want to override:")
        log.info("  anno_blur/   delete a preview → keep that original (not blurry after all)")
        log.info("  anno_sharp/  delete a preview → exclude that original (move to Unselected/)")
        log.info("")

    if frames and not args.skip_facereco:
        FaceRecoStage(
            output_dir,
            face_db_dir=face_db_dir,
            face_db_match_threshold=args.face_db_match_threshold,
        ).process(frames, app_config)

    if frames:
        log.info("When done, run:  python 2_apply_changes.py \"%s\"", output_dir)


if __name__ == "__main__":
    main()
