from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from algo.config import AppConfig
from algo.frame import Frame
from algo.models import Box, Face
from algo.stage import ProcessStage
from algo.utils import THUMBNAIL_SIZE, THUMBNAILS_SUBDIR, apply_auto_adjustment, write_cover_thumbnail

log = logging.getLogger("BlurPictureDetector")

# COCO 17-keypoint skeleton: pairs of indices to connect with a line.
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


def _draw_status_icon(
    image: np.ndarray,
    x: int,
    y: int,
    size: int,
    color: tuple[int, int, int],
    *,
    passed: bool,
) -> None:
    """Draw a filled *size* × *size* badge with a white ✓ or ✗ at (x, y)."""
    h, w = image.shape[:2]
    x2, y2 = min(x + size, w), min(y + size, h)
    sw, sh = x2 - x, y2 - y
    lw = max(2, size // 14)

    cv2.rectangle(image, (x, y), (x2, y2), color, -1)

    if passed:
        p1 = (x + int(sw * 0.15), y + int(sh * 0.50))
        p2 = (x + int(sw * 0.40), y + int(sh * 0.76))
        p3 = (x + int(sw * 0.85), y + int(sh * 0.24))
        cv2.line(image, p1, p2, (255, 255, 255), lw, cv2.LINE_AA)
        cv2.line(image, p2, p3, (255, 255, 255), lw, cv2.LINE_AA)
    else:
        p1 = (x + int(sw * 0.20), y + int(sh * 0.20))
        p2 = (x + int(sw * 0.80), y + int(sh * 0.80))
        p3 = (x + int(sw * 0.80), y + int(sh * 0.20))
        p4 = (x + int(sw * 0.20), y + int(sh * 0.80))
        cv2.line(image, p1, p2, (255, 255, 255), lw, cv2.LINE_AA)
        cv2.line(image, p3, p4, (255, 255, 255), lw, cv2.LINE_AA)


def _annotate_frame(
    frame: Frame,
    output_dir: Path,
    config: AppConfig,
) -> None:
    """Write an annotated preview for one frame into the shared ``previews/``
    sub-folder.

    - No image data  → do nothing (read error upstream).
    - No bodies      → no person detected; save original as-is.
    - Otherwise      → draw boxes, keypoints, scores.
    """
    if frame.image is None:
        return

    if not frame.bodies:
        subdir = output_dir / "previews"
        subdir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(subdir / (frame.key_stem + ".jpg")),
            frame.image,
            [cv2.IMWRITE_JPEG_QUALITY, 60],
        )
        write_cover_thumbnail(frame.image, output_dir / THUMBNAILS_SUBDIR / (frame.key_stem + ".jpg"), THUMBNAIL_SIZE)
        return

    annotated = apply_auto_adjustment(frame.image, frame.auto_adjustment).copy()
    overlay   = annotated.copy()
    h_out, w_out = annotated.shape[:2]
    sx = w_out
    sy = h_out
    ann_scale = max(h_out, w_out) / config.normalized_img_max_long_edge

    face_thick     = max(1, round(config.annotation_face_box_thickness    * ann_scale))
    box_thick      = max(1, round(config.annotation_box_thickness          * ann_scale))
    icon_size      = max(20, round(config.annotation_icon_size             * ann_scale))
    kp_radius      = max(3, round(config.annotation_face_kp_radius         * ann_scale))
    kp_thick       = max(1, round(config.annotation_face_kp_thickness      * ann_scale))
    body_kp_size   = max(2, round(config.annotation_body_kp_size           * ann_scale))
    body_kp_thick  = max(1, round(config.annotation_body_kp_thickness      * ann_scale))
    skeleton_thick = max(1, round(config.annotation_skeleton_thickness      * ann_scale))
    narrow_thick   = max(1, round(config.annotation_narrow_face_box_thickness * ann_scale))
    font           = cv2.FONT_HERSHEY_SIMPLEX
    font_thick     = config.annotation_score_font_thickness
    (_, _base_h), _ = cv2.getTextSize("Mg", font, 1.0, font_thick)
    font_scale           = config.annotation_score_font_size_px            * ann_scale / max(_base_h, 1)
    rejection_font_scale = config.annotation_rejection_reason_font_size_px * ann_scale / max(_base_h, 1)

    score_labels: list[tuple] = []

    # Bodies to annotate: top-N by bbox area ∪ all passed bodies.
    # Sort largest-first so bigger boxes render behind smaller ones.
    sorted_by_size = sorted(frame.bodies, key=lambda b: b.bbox.area, reverse=True)
    top_n_ids = {id(b) for b in sorted_by_size[:config.annotation_top_n_bodies]}
    bodies_to_render = [b for b in sorted_by_size if id(b) in top_n_ids or b.passed]

    for body in bodies_to_render:
        b = body.bbox
        rbx1 = int(b.x1 * sx); rby1 = int(b.y1 * sy)
        rbx2 = int(b.x2 * sx); rby2 = int(b.y2 * sy)

        body_color = (
            config.annotation_box_color_fail if not body.passed
            else config.annotation_box_color_pass
        )
        cv2.rectangle(overlay, (rbx1, rby1), (rbx2, rby2), body_color, box_thick)
        _draw_status_icon(overlay, rbx1, rby1, icon_size, body_color, passed=body.passed)

        # Skeleton
        kps = body.keypoints
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

        # Body keypoints
        half = body_kp_size // 2
        for kp in kps:
            if kp.point.x == 0 and kp.point.y == 0:
                continue
            rkpx = int(kp.point.x * sx)
            rkpy = int(kp.point.y * sy)
            cv2.rectangle(overlay,
                          (rkpx - half, rkpy - half),
                          (rkpx + half, rkpy + half),
                          body_color, body_kp_thick, cv2.LINE_AA)

        # Face bbox + sharpness score label
        if body.best_face is not None:
            fb: Box = body.best_face.bbox
            rfx1 = int(fb.x1 * sx); rfy1 = int(fb.y1 * sy)
            rfx2 = int(fb.x2 * sx); rfy2 = int(fb.y2 * sy)

            nfb: Box | None = body.best_narrow_box
            if config.use_narrow_face_box and nfb is not None:
                rnx1 = int(nfb.x1 * sx); rny1 = int(nfb.y1 * sy)
                rnx2 = int(nfb.x2 * sx); rny2 = int(nfb.y2 * sy)
                cv2.rectangle(overlay, (rnx1, rny1), (rnx2, rny2), body_color, narrow_thick)
                label_x, label_bottom = rnx1, rny2
            else:
                cv2.rectangle(overlay, (rfx1, rfy1), (rfx2, rfy2), body_color, face_thick)
                label_x, label_bottom = rfx1, rfy2

            label = f"{body.sharpness_score:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
            text_y = min(label_bottom + th + baseline + 2, h_out - 1)
            score_labels.append((label, label_x, text_y, body_color, font_scale))

        # Cloth colour label (passed) or rejection reason (failed)
        if body.passed:
            cloth = body.cloth_color
            if cloth not in ("N/A", "Unknown"):
                (clw, clh), _ = cv2.getTextSize(cloth, font, font_scale, font_thick)
                cl_x = max(rbx1, rbx2 - clw - 4)
                cl_y = max(clh + 4, rby1 + clh + 4)
                score_labels.append((cloth, cl_x, cl_y, body_color, font_scale))
        else:
            reason = body.rejection_reason or "rejected (no reason given)"
            (rw, rh), _ = cv2.getTextSize(reason, font, rejection_font_scale, font_thick)
            r_x = max(rbx1, rbx2 - rw - 4)
            r_y = max(rh + 4, rby1 + rh + 4)
            score_labels.append((reason, r_x, r_y, body_color, rejection_font_scale))

        # Face landmark circles
        if body.best_face is not None:
            ann_face: Face = body.best_face
            for lm in ann_face.landmarks:
                rkpx = int(lm.point.x * sx)
                rkpy = int(lm.point.y * sy)
                kp_color = (
                    config.annotation_box_color_pass
                    if lm.confidence >= config.face_coverage_conf_threshold
                    else config.annotation_box_color_fail
                )
                cv2.circle(overlay, (rkpx, rkpy), kp_radius, kp_color, kp_thick, cv2.LINE_AA)

    cv2.addWeighted(overlay, config.annotation_alpha,
                    annotated, 1.0 - config.annotation_alpha, 0, annotated)

    for label, lx, ly, lcolor, fscale in score_labels:
        cv2.putText(annotated, label, (lx, ly), font, fscale, lcolor, font_thick, cv2.LINE_AA)

    out_name    = frame.key_stem + ".jpg"
    anno_subdir = output_dir / "previews"
    anno_subdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(anno_subdir / out_name), annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
    write_cover_thumbnail(annotated, output_dir / THUMBNAILS_SUBDIR / out_name, THUMBNAIL_SIZE)


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class AnnotationStage(ProcessStage):
    """Annotate every frame and write previews to the output directory.

    Every frame (sharp, blurry, or no-person) → ``<output_dir>/previews/``.

    The frame list is returned unchanged.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def process(self, frames: list[Frame], config: AppConfig) -> list[Frame]:
        for idx, frame in enumerate(frames, 1):
            log.debug("[AnnotationStage] [%d/%d] %s", idx, len(frames), frame.path.name)
            _annotate_frame(frame, self.output_dir, config)
        log.info("[AnnotationStage] annotated %d frame(s) → %s", len(frames), self.output_dir)
        return frames
