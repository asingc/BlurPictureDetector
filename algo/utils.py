from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from algo.models import AutoAdjustment, Box, ColorLab, Face


# COCO 17-keypoint head indices: nose, left-eye, right-eye, left-ear, right-ear.
_HEAD_KP_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4)


def apply_auto_adjustment(image: np.ndarray, adjustment: AutoAdjustment | None) -> np.ndarray:
    """Apply a simple EV (exposure/brightness) correction to a BGR *image*.

    ``adjustment.ev`` multiplies pixel values by ``2 ** ev`` (a stop-based
    exposure compensation). Returns *image* unchanged (same array, not a
    copy) when there is nothing to apply.
    """
    if adjustment is None or adjustment.is_noop:
        return image
    out = image.astype(np.float32)
    out *= 2.0 ** adjustment.ev
    return np.clip(out, 0, 255).astype(np.uint8)


def cap_long_edge(image: np.ndarray, max_long_edge: float) -> np.ndarray:
    """
    Downsize *image* so its long edge is at most *max_long_edge* pixels.
    Never upscales — returns the original array when it is already smaller.
    """
    h, w = image.shape[:2]
    scale = min(1.0, max_long_edge / max(h, w))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _narrow_face_box(
    face: Face,
    conf_threshold: float = 0.3,
    pad: int = 0,
    img_w: Optional[int] = None,
    img_h: Optional[int] = None,
) -> Box | None:
    """Return the minimal bounding box enclosing the confident face landmarks,
    expanded by *pad* pixels on every side and clamped to the image bounds.
    Returns None if fewer than 2 landmarks are detected."""
    xs = [lm.point.x for lm in face.landmarks if lm.confidence >= conf_threshold]
    ys = [lm.point.y for lm in face.landmarks if lm.confidence >= conf_threshold]
    if len(xs) < 2:
        return None
    if img_w is not None and img_h is not None:
        pad_x = pad / img_w
        pad_y = pad / img_h
        x1 = max(0.0, min(xs) - pad_x)
        y1 = max(0.0, min(ys) - pad_y)
        x2 = min(1.0, max(xs) + pad_x)
        y2 = min(1.0, max(ys) + pad_y)
    else:
        x1 = min(xs) - pad
        y1 = min(ys) - pad
        x2 = max(xs) + pad
        y2 = max(ys) + pad
    return Box(x1, y1, x2, y2)


def _color_from_label(label: str) -> "ColorLab | None":
    """Parse a 'Hue:Shade' label into a ColorLab, or return None for N/A / Unknown."""
    if not label or label in ("N/A", "Unknown"):
        return None
    hue, _, shade = label.partition(":")
    return ColorLab(hue.strip(), shade.strip())


def _matches_allowed_jersey_color(color: "ColorLab | None", allowed_colors: frozenset[str]) -> bool:
    """Match *color* against the allow-list.

    Shade match is preferred; hue match is accepted as fallback.
    Allow-list entries may be plain hue names, shade names, or "Hue:Shade" labels.
    """
    if color is None:
        return False
    c_hue_l   = color.hue.strip().lower()
    c_shade_l = color.shade.strip().lower()
    for allowed in allowed_colors:
        a = allowed.strip().lower()
        if not a:
            continue
        if ":" in a:
            a_hue, _, a_shade = a.partition(":")
            if a_shade == c_shade_l or a_hue == c_hue_l:
                return True
        else:
            if a == c_shade_l or a == c_hue_l:
                return True
    return False


def _colors_match(color: "ColorLab | None", reference: "ColorLab | None") -> bool:
    """Return True if *color* matches *reference*.

    Shade equality is preferred; hue equality is accepted as fallback.
    """
    if color is None or reference is None:
        return False
    if color.shade.strip().lower() == reference.shade.strip().lower():
        return True
    return color.hue.strip().lower() == reference.hue.strip().lower()
