from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Point:
    """A 2-D coordinate stored as a normalised fraction of image width/height."""

    x: float
    y: float

    @staticmethod
    def from_px(x: float, y: float, img_w: int, img_h: int) -> "Point":
        return Point(x / img_w, y / img_h)

    def as_px(self, img_w: int, img_h: int) -> tuple[int, int]:
        return int(round(self.x * img_w)), int(round(self.y * img_h))


@dataclass
class Box:
    """Axis-aligned bounding box stored as normalised fractions of image width/height."""
    x1: float
    y1: float
    x2: float
    y2: float

    @staticmethod
    def from_px(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> "Box":
        return Box(x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> Point:
        return Point((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def contains(self, p: Point) -> bool:
        """Return True if *p* lies inside or on the boundary of this box."""
        return self.x1 <= p.x <= self.x2 and self.y1 <= p.y <= self.y2

    def contains_xy(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Vectorised containment test — returns a boolean mask over the input arrays."""
        return (xs >= self.x1) & (xs <= self.x2) & (ys >= self.y1) & (ys <= self.y2)

    def padded(self, pad: int, max_w: int, max_h: int) -> "Box":
        """Return a copy expanded by *pad* pixels on every side, clamped to image bounds."""
        pad_x = pad / max_w
        pad_y = pad / max_h
        return Box(
            max(0.0, self.x1 - pad_x),
            max(0.0, self.y1 - pad_y),
            min(1.0,  self.x2 + pad_x),
            min(1.0,  self.y2 + pad_y),
        )

    def as_px_ints(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        return (
            int(round(self.x1 * img_w)),
            int(round(self.y1 * img_h)),
            int(round(self.x2 * img_w)),
            int(round(self.y2 * img_h)),
        )

    def overlaps(self, other: "Box") -> bool:
        """Return True if this box and *other* share any area."""
        return (
            self.x1 < other.x2 and self.x2 > other.x1 and
            self.y1 < other.y2 and self.y2 > other.y1
        )


@dataclass
class PredictedKeyPoint:
    """A detected keypoint with its confidence score and pass/fail verdict."""
    point:      Point
    confidence: float
    passed:     bool = True   # scorers set this to False to disqualify


@dataclass
class Face:
    """A detected face: bounding box, detection confidence, and optional landmarks."""
    bbox:       Box
    confidence: float               # detection confidence from the face model
    landmarks:  list[PredictedKeyPoint]  # 5 face-model landmarks (empty if unavailable)
    passed:     bool = True          # scorers set this to False to disqualify

    def n_visible(self) -> int:
        """Count landmarks that have passed classification."""
        return sum(1 for lm in self.landmarks if lm.passed)


@dataclass
class Body:
    """A detected person ready for sharpness analysis."""
    crop:              np.ndarray                # face image crop (BGR) used for blur scoring
    bbox:              Box                       # body bounding box (padded, clamped)
    faces:             list[Face]                # matched faces (may be empty before face matching)
    keypoints:         list[PredictedKeyPoint]   # 17 COCO body keypoints
    passed:            bool        = True        # scorers set this to False to disqualify
    rejection_reason:  str         = ""          # reason why this body was rejected (empty if accepted)
    sharpness_score:   float       = 0.0         # best face sharpness score (set by FaceSharpnessScorer)
    best_face:         Face | None = None        # face that yielded sharpness_score
    best_narrow_box:   Box  | None = None        # narrow landmark bbox for best_face
    lap_var:           float       = 0.0         # Laplacian variance of best face crop
    ten:               float       = 0.0         # Tenengrad of best face crop
    cloth_color:       str         = "N/A"       # predicted jersey/cloth color
    cloth_color_detail: dict       = field(default_factory=dict)  # votes + mean LAB


@dataclass
class ColorLab:
    """A named reference colour in CIE L*a*b* space."""
    hue:   str                                          # broad category: "Red", "Blue", "Gray" …
    shade: str                                          # precise variant: "Crimson", "Royal", "75%" …
    lab:   tuple[float, float, float] = (0.0, 0.0, 0.0)  # L*, a*, b* reference values (optional for label-only instances)

    @property
    def label(self) -> str:
        """Combined label used for cloth_color: 'Hue:Shade'."""
        return f"{self.hue}:{self.shade}"


@dataclass
class AutoAdjustment:
    """A simple, discretised auto exposure (brightness) correction prescription.

    Deliberately kept coarse (e.g. EV +0.5, not EV +0.4231) so the correction
    stays easy to reason about, log, and re-apply later from JSON.
    """
    ev: float = 0.0  # exposure compensation in stops; output *= 2**ev

    @property
    def is_noop(self) -> bool:
        return self.ev == 0.0
