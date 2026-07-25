from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Box:
    """Simple axis-aligned bounding box stored as normalised fractions."""

    x1: float
    y1: float
    x2: float
    y2: float

    @staticmethod
    def from_px(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> "Box":
        return Box(x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h)

    @property
    def width(self) -> float:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0, self.y2 - self.y1)

    def as_px_ints(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        return (
            int(round(self.x1 * img_w)),
            int(round(self.y1 * img_h)),
            int(round(self.x2 * img_w)),
            int(round(self.y2 * img_h)),
        )


@dataclass
class BodyRecord:
    """Serializable body record loaded from album.json annotation_data."""

    orig_filename: str
    body_index: int
    body_bbox: Box
    face_bbox: Box | None
    narrow_face_bbox: Box | None
    cloth_color: str
    qualified_for_sharpness: bool
    is_blurry: bool
    confidence: float | None
    raw_body: dict


@dataclass
class Player:
    """Face recognition output for a single body record."""

    name: str = ""
    jersey_number: int | None = None
    confidence: float | None = None
    internal: dict = field(default_factory=dict)


class FaceRecoProvider(ABC):
    """Abstract face recognition provider API."""

    @abstractmethod
    def provider_name(self) -> str:
        """Return a short provider identifier (for example: dlib)."""

    @abstractmethod
    def predict_player(self, image_bgr: np.ndarray, body: BodyRecord) -> Player:
        """Infer player identity for one body and return a Player object."""
