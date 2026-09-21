from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Encode numpy scalar types as their Python equivalents."""

    def default(self, obj: object) -> object:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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

    def to_wire(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @staticmethod
    def from_wire(data: dict | None) -> "Box | None":
        if not data:
            return None
        return Box(float(data["x1"]), float(data["y1"]), float(data["x2"]), float(data["y2"]))


@dataclass
class PredictedKeyPoint:
    """A detected keypoint with its confidence score and pass/fail verdict."""
    point:      Point
    confidence: float
    passed:     bool = True   # scorers set this to False to disqualify

    def to_wire(self) -> dict:
        return {"x": self.point.x, "y": self.point.y, "conf": self.confidence, "passed": self.passed}

    @staticmethod
    def from_wire(data: dict) -> "PredictedKeyPoint":
        return PredictedKeyPoint(
            Point(data["x"], data["y"]), data["conf"], bool(data.get("passed", True))
        )


@dataclass
class Face:
    """A detected face: bounding box, detection confidence, and optional landmarks."""
    bbox:       Box
    confidence: float               # detection confidence from the face model
    landmarks:  list[PredictedKeyPoint]  # 5 face-model landmarks (empty if unavailable)
    passed:     bool = True          # scorers set this to False to disqualify

    # Computed during analysis (while the frame is still decoded) so the
    # sharpness scorer can score from the cached crop alone. Transient:
    # body.best_narrow_box is what gets persisted.
    narrow_box: "Box | None" = None

    # Scratch, per-run pixel cache (see algo/image_cache.py). Populated during
    # analysis while the source frame is still decoded, so later stages (and
    # FaceRecoStage) never need to keep a whole album's images in memory at
    # once -- None until written. Never persisted to album.json.
    cache_normalized_path: str | None = None  # face-bbox crop from the normalized frame
    cache_original_path:   str | None = None  # face-bbox crop (padded) from the native source image
    _normalized_crop: "np.ndarray | None" = field(default=None, repr=False, compare=False)
    _original_crop:   "np.ndarray | None" = field(default=None, repr=False, compare=False)

    def set_normalized_crop(self, crop: "np.ndarray | None") -> None:
        """Attach an already-decoded crop in memory (e.g. a single-image
        regrade) instead of persisting/reading it from the scratch cache."""
        self._normalized_crop = crop

    def get_normalized_crop(self) -> "np.ndarray | None":
        """Face-bbox crop of the normalized frame, or None if never populated."""
        if self._normalized_crop is None and self.cache_normalized_path:
            self._normalized_crop = np.load(self.cache_normalized_path, allow_pickle=False)
        return self._normalized_crop

    def set_original_crop(self, crop: "np.ndarray | None") -> None:
        self._original_crop = crop

    def get_original_crop(self) -> "np.ndarray | None":
        """Padded face-bbox crop of the native source image (for FaceReco), or None."""
        if self._original_crop is None and self.cache_original_path:
            self._original_crop = np.load(self.cache_original_path, allow_pickle=False)
        return self._original_crop

    def n_visible(self) -> int:
        """Count landmarks that have passed classification."""
        return sum(1 for lm in self.landmarks if lm.passed)

    def to_wire(self) -> dict:
        return {
            "bbox":       self.bbox.to_wire(),
            "confidence": self.confidence,
            "landmarks":  [lm.to_wire() for lm in self.landmarks],
            "passed":     self.passed,
        }

    @staticmethod
    def from_wire(data: dict | None) -> "Face | None":
        if not data:
            return None
        bbox = Box.from_wire(data.get("bbox"))
        if bbox is None:
            return None
        return Face(
            bbox=bbox,
            confidence=float(data.get("confidence", 0.0)),
            landmarks=[PredictedKeyPoint.from_wire(lm) for lm in data.get("landmarks", [])],
            passed=bool(data.get("passed", True)),
        )


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
    # Mean grey level [0,1] of this body's face crop, measured during analysis
    # so AutoAdjustStage never needs the pixels again.
    crop_brightness:   float       = 0.0

    # Scratch, per-run pixel cache (see algo/image_cache.py) -- same contract
    # as Face's cache_normalized_path above.
    cache_normalized_path: str | None = None  # body-bbox crop from the normalized frame
    _normalized_crop: "np.ndarray | None" = field(default=None, repr=False, compare=False)

    def set_normalized_crop(self, crop: "np.ndarray | None") -> None:
        """Attach an already-decoded crop in memory instead of the scratch cache."""
        self._normalized_crop = crop

    def get_normalized_crop(self) -> "np.ndarray | None":
        """Body-bbox crop of the normalized frame, or None if never populated."""
        if self._normalized_crop is None and self.cache_normalized_path:
            self._normalized_crop = np.load(self.cache_normalized_path, allow_pickle=False)
        return self._normalized_crop


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
