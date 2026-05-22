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
    6. When --showbox is supplied a copy of every processed image is saved to
       <image_dir>/annotated/ with the subject bounding box drawn (style via app_config).

If no person is detected in an image the file is left untouched and skipped.
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from dataclasses import dataclass, field
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
    "medium": 0.40,
    "high":   0.65,
}

# Normalisation scales for the blur-score formula  1 / (1 + x / scale).
# At x == scale the component equals 0.5, placing it right on the medium boundary.
# Calibrated for typical subject crops from sports photography.
_LAP_SCALE: float = 100.0    # Laplacian variance reference
_TEN_SCALE: float = 5_000.0  # Tenengrad reference


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
    face_coverage_conf_threshold: float = 0.8  # per-landmark confidence cutoff

    # Minimum face size: disqualify persons whose face bbox long edge is smaller
    # than this fraction of the image long edge (e.g. 0.04 = 4 %).
    # Filters out background spectators or faces too small to reliably analyse.
    # Set to 0 to disable.
    face_min_size_fraction: float = 0.04

    # Face bounding box drawn on annotated previews (separate from the body box).
    annotation_face_box_color:     tuple[int, int, int] = field(default=(0, 255, 255))  # yellow
    annotation_face_box_thickness: int                  = 2
    # Blur score label drawn below each face bounding box.
    annotation_score_font_size_px:    int   = 40   # target text height in pixels
    annotation_score_font_thickness: int   = 1

    # Annotated preview / processing image scaling.
    # Images are downsized so the long edge equals preview_max_long_edge.
    # Never upscales.
    preview_max_long_edge: int = 1800


app_config = AppConfig()


# ---------------------------------------------------------------------------
# Sharpness metrics
# ---------------------------------------------------------------------------

# COCO face keypoint indices within the 17-point YOLOv8-pose skeleton.
# 0=nose  1=left_eye  2=right_eye  3=left_ear  4=right_ear
_FACE_KP_INDICES: list[int] = [0, 1, 2, 3, 4]

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

def resize_for_preview(image: np.ndarray) -> np.ndarray:
    """
    Downsize *image* so its long edge equals app_config.preview_max_long_edge.
    Never upscales — returns the original array when it is already smaller.
    """
    h, w = image.shape[:2]
    scale = min(1.0, app_config.preview_max_long_edge / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Subject detection
# ---------------------------------------------------------------------------

# Each detected person entry: (face_crop, body_bbox, face_bbox, face_kps)
# face_bbox is always set — persons with no face model match are discarded before appending.
# face_kps: (5, 3) [x, y, conf] per face model landmark, or None if the model returned no KPs.
_PersonEntry = tuple[np.ndarray, tuple[int, int, int, int], Optional[tuple[int, int, int, int]], Optional[np.ndarray]]


def detect_qualified_persons(
    image: np.ndarray,
    pose_model: YOLO,
    face_model: YOLO,
) -> tuple[list[_PersonEntry], bool]:
    """
    Return (entries, had_persons).

    had_persons : True if the pose model detected at least one person body.
    entries     : up to 3 qualified persons (face_crop, body_bbox, face_bbox, face_kps),
                  sorted largest-first.  May be empty even when had_persons is True
                  (all candidates were disqualified by the face-model checks).

    Disqualification pipeline (any failure → person skipped):
      1. Pose-model face KP visibility (< face_kp_min_visible confident KPs).
      2. No face detected by the face model inside the body bbox.
      3. Face-model landmark coverage (< face_coverage_min_visible confident landmarks).
      4. Face size < face_min_size_fraction × image long edge.
    """
    results = pose_model.predict(image, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return [], False

    boxes   = results[0].boxes.xyxy.cpu().numpy()          # (N, 4)
    areas   = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    kps     = results[0].keypoints
    kp_data = kps.data.cpu().numpy() if kps is not None else None  # (N, 17, 3)
    h, w    = image.shape[:2]
    pad     = 10

    # Build qualifying list (face-visibility filter).
    qualifying: list[int] = []
    if kp_data is not None and app_config.face_kp_min_visible > 0:
        for i, kp in enumerate(kp_data):
            face_kps  = kp[_FACE_KP_INDICES]              # (5, 3)
            n_visible = int(np.sum(face_kps[:, 2] >= app_config.face_kp_conf_threshold))
            if n_visible >= app_config.face_kp_min_visible:
                qualifying.append(i)

    # Take up to 3 largest by body area.
    qualifying.sort(key=lambda i: areas[i], reverse=True)
    top = qualifying[:3]

    # --- run dedicated face detector once on the whole image ---------------
    face_results = face_model.predict(image, verbose=False)
    if face_results and len(face_results[0].boxes) > 0:
        fdet_boxes = face_results[0].boxes.xyxy.cpu().numpy()   # (M, 4)
        fdet_confs = face_results[0].boxes.conf.cpu().numpy()   # (M,)
        fdet_kps   = (
            face_results[0].keypoints.data.cpu().numpy()        # (M, 5, 3)
            if face_results[0].keypoints is not None else None
        )
    else:
        fdet_boxes = np.empty((0, 4), dtype=np.float32)
        fdet_confs = np.empty((0,),   dtype=np.float32)
        fdet_kps   = None
    # -----------------------------------------------------------------------

    entries: list[_PersonEntry] = []
    for idx in top:
        # --- body bbox ---
        bx1, by1, bx2, by2 = boxes[idx].astype(int)
        bx1, by1 = max(0, bx1 - pad), max(0, by1 - pad)
        bx2, by2 = min(w, bx2 + pad), min(h, by2 + pad)
        body_bbox = (bx1, by1, bx2, by2)

        # --- face bbox from face model (centre must fall inside body bbox) ---
        face_bbox: Optional[tuple[int, int, int, int]] = None
        face_kps:  Optional[np.ndarray]               = None  # (5,3) from face model
        if len(fdet_boxes) > 0:
            cx = (fdet_boxes[:, 0] + fdet_boxes[:, 2]) / 2
            cy = (fdet_boxes[:, 1] + fdet_boxes[:, 3]) / 2
            inside = np.where(
                (cx >= bx1) & (cx <= bx2) & (cy >= by1) & (cy <= by2)
            )[0]
            if len(inside) > 0:
                best_face = inside[np.argmax(fdet_confs[inside])]

                # Save keypoints and run coverage check.
                if fdet_kps is not None:
                    face_kps = fdet_kps[best_face]       # (5, 3): x, y, conf
                    if app_config.face_coverage_min_visible > 0:
                        n_visible = int(np.sum(
                            face_kps[:, 2] >= app_config.face_coverage_conf_threshold
                        ))
                        if n_visible < app_config.face_coverage_min_visible:
                            continue  # face too covered — skip this person
                fx1 = max(0, int(fdet_boxes[best_face, 0]) - pad)
                fy1 = max(0, int(fdet_boxes[best_face, 1]) - pad)
                fx2 = min(w, int(fdet_boxes[best_face, 2]) + pad)
                fy2 = min(h, int(fdet_boxes[best_face, 3]) + pad)
                if fx2 > fx1 and fy2 > fy1:
                    face_bbox = (fx1, fy1, fx2, fy2)

        # No face detected by the face model — hard disqualification.
        if face_bbox is None:
            continue

        # Size filter: disqualify person if face bbox long edge < configured fraction.
        if face_bbox is not None and app_config.face_min_size_fraction > 0:
            fw = face_bbox[2] - face_bbox[0]
            fh = face_bbox[3] - face_bbox[1]
            if max(fw, fh) < app_config.face_min_size_fraction * max(h, w):
                continue  # face too small — skip this person

        # Face crop for blur analysis (from face bbox, else body crop).
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            face_crop: np.ndarray = image[fy1:fy2, fx1:fx2]
        else:
            face_crop = image[by1:by2, bx1:bx2]

        entries.append((face_crop, body_bbox, face_bbox, face_kps))

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
    image = cv2.imread(str(image_path))
    if image is None:
        return {"file": str(image_path), "status": "error", "error": "Cannot read image file"}

    # Normalise image size in-memory before all processing (original file untouched).
    image = resize_for_preview(image)

    persons, had_persons = detect_qualified_persons(image, pose_model, face_model)
    if not had_persons:
        return {"file": str(image_path), "status": "skipped", "reason": "No person detected"}
    if not persons:
        return {
            "file":               str(image_path),
            "status":             "blurry",
            "sharpness_score":    0.0,
            "sharpness_grade":    0.0,
            "laplacian_variance": 0.0,
            "tenengrad_score":    0.0,
        }

    # Evaluate each qualified person.
    evaluated: list[dict] = []
    for face_crop, body_bbox, face_bbox, face_kps in persons:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        score, lap_var, ten = compute_sharpness_score(gray)
        evaluated.append({
            "body_bbox":        body_bbox,
            "face_bbox":        face_bbox,
            "face_kps":         face_kps,
            "sharpness_score":  score,
            "lap_var":          lap_var,
            "ten":              ten,
            "is_blurry":        score <= threshold,
        })

    # Image passes if ANY person is sharp.
    overall_blurry = all(p["is_blurry"] for p in evaluated)

    # Use the sharpest person's metrics for the CSV/log.
    best = max(evaluated, key=lambda p: p["sharpness_score"])

    # --- annotated preview ---------------------------------------------------
    if boxes_dir is not None:
        boxes_dir.mkdir(exist_ok=True)

        # Image is already normalised; annotate at native (processing) coordinates.
        annotated = image.copy()
        h_out, w_out = annotated.shape[:2]
        sx, sy = 1.0, 1.0

        face_thick  = app_config.annotation_face_box_thickness
        font        = cv2.FONT_HERSHEY_SIMPLEX
        font_thick  = app_config.annotation_score_font_thickness
        # Compute font scale so text height == annotation_score_font_size_px.
        (_, _base_h), _ = cv2.getTextSize("Mg", font, 1.0, font_thick)
        font_scale = app_config.annotation_score_font_size_px / max(_base_h, 1)

        for i, p in enumerate(evaluated):
            bx1, by1, bx2, by2 = p["body_bbox"]
            rbx1 = int(bx1 * sx); rby1 = int(by1 * sy)
            rbx2 = int(bx2 * sx); rby2 = int(by2 * sy)

            body_color = (
                app_config.annotation_box_color_fail
                if p["is_blurry"] else
                app_config.annotation_box_color_pass
            )
            cv2.rectangle(annotated, (rbx1, rby1), (rbx2, rby2),
                          body_color, app_config.annotation_box_thickness)

            # Every person gets its own pass/fail status icon.
            _draw_status_icon(
                annotated, rbx1, rby1,
                app_config.annotation_icon_size,
                body_color,
                passed=not p["is_blurry"],
            )

            # Face bbox + sharpness score label — color mirrors the sharpness pass/fail.
            if p["face_bbox"] is not None:
                fx1, fy1, fx2, fy2 = p["face_bbox"]
                rfx1 = int(fx1 * sx); rfy1 = int(fy1 * sy)
                rfx2 = int(fx2 * sx); rfy2 = int(fy2 * sy)
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
                for kp in p["face_kps"]:
                    kpx, kpy, kpc = float(kp[0]), float(kp[1]), float(kp[2])
                    rkpx = int(kpx * sx)
                    rkpy = int(kpy * sy)
                    kp_color = (
                        app_config.annotation_box_color_pass
                        if kpc >= kp_conf_thresh else
                        app_config.annotation_box_color_fail
                    )
                    cv2.circle(annotated, (rkpx, rkpy), 6, kp_color, 3, cv2.LINE_AA)

        out_name = image_path.stem + ".jpg"
        cv2.imwrite(str(boxes_dir / out_name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
    # -------------------------------------------------------------------------

    return {
        "file":               str(image_path),
        "status":             "blurry" if overall_blurry else "sharp",
        "sharpness_score":    round(best["sharpness_score"], 4),
        "sharpness_grade":    round(best["sharpness_score"] * 100, 1),
        "laplacian_variance": round(best["lap_var"], 2),
        "tenengrad_score":    round(best["ten"], 2),
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
        return _FACE_MODEL_PATH
    print(f"Downloading yolov8n-face.pt from {_FACE_MODEL_URL} …")
    urllib.request.urlretrieve(_FACE_MODEL_URL, _FACE_MODEL_PATH)
    print("Download complete.")
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
    input_path: Path, sensitivity: str, showbox: bool = False
) -> tuple[list[dict], Path]:
    """
    Process all images at *input_path*.

    Parameters
    ----------
    showbox : when True, save an annotated copy of each image (bounding box
              drawn in blue) to <base_dir>/annotated/.

    Returns (blurry_results, base_directory).
    """
    threshold = SENSITIVITY_THRESHOLDS[sensitivity]
    files     = collect_images(input_path)

    if not files:
        print("No supported image files found.")
        base_dir = input_path if input_path.is_dir() else input_path.parent
        return [], base_dir

    base_dir  = files[0].parent
    width     = len(str(len(files)))  # for aligned progress numbers
    boxes_dir = (base_dir / "annotated") if showbox else None

    print("Loading models …")
    pose_model = YOLO("yolov8n-pose.pt")
    face_model = YOLO(_ensure_face_model())

    print(
        f"\nProcessing {len(files)} image(s)"
        f"  |  sensitivity = {sensitivity}"
        f"  |  blur threshold = {threshold}"
        + (f"  |  showbox → {boxes_dir}" if showbox else "") + "\n"
    )

    blurry:      list[dict] = []
    sharp_count = skip_count = error_count = 0

    for idx, image_path in enumerate(files, 1):
        tag = f"[{idx:>{width}}/{len(files)}] {image_path.name}"
        result = analyse_image(image_path, pose_model, face_model, threshold, boxes_dir)

        if result["status"] == "blurry":
            result["star_rating"] = 1
            blurry.append(result)

            print(
                f"{tag}  →  BLURRY   "
                f"score={result['sharpness_score']:.3f}  "
                f"grade={result['sharpness_grade']:.1f}/100"
            )

        elif result["status"] == "sharp":
            sharp_count += 1
            print(
                f"{tag}  →  Sharp    "
                f"score={result['sharpness_score']:.3f}  "
                f"grade={result['sharpness_grade']:.1f}/100"
            )

        elif result["status"] == "skipped":
            skip_count += 1
            print(f"{tag}  →  Skipped  ({result.get('reason', '')})")

        else:
            error_count += 1
            print(f"{tag}  →  Error    ({result.get('error', '')})")

    print(
        f"\nSummary  —  "
        f"Blurry: {len(blurry)}  |  "
        f"Sharp: {sharp_count}  |  "
        f"Skipped (no person): {skip_count}  |  "
        f"Errors: {error_count}"
    )
    return blurry, base_dir


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_FIELDS: list[str] = [
    "file",
    "star_rating",
    "sharpness_score",
    "sharpness_grade",
    "laplacian_variance",
    "tenengrad_score",
]


def write_csv(blurry: list[dict], csv_path: Path) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(blurry)
    print(f"CSV report written to:    {csv_path}")


def write_blur_lst(blurry: list[dict], lst_path: Path) -> None:
    """Write a plain-text list of blurry image filenames and their blur scores."""
    with open(lst_path, "w", encoding="utf-8") as fh:
        for entry in blurry:
            filename = Path(entry["file"]).name
            fh.write(f"{filename}\t{entry['sharpness_score']}\n")
    print(f"Blur list written to:     {lst_path}")


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
        "--showbox",
        action="store_true",
        default=False,
        help=(
            "For each image with a detected person, save an annotated copy with "
            "the subject bounding box drawn in blue (4 px) to <img_dir>/annotated/."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.path).resolve()
    if not input_path.exists():
        print(f"Error: '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    blurry, base_dir = process(input_path, args.sensitivity, showbox=args.showbox)

    if blurry:
        write_csv(blurry, base_dir / "blurry.csv")
        write_blur_lst(blurry, base_dir / "blur.lst")
    else:
        print("No blurry images detected — blurry.csv and blur.lst not written.")


if __name__ == "__main__":
    main()
