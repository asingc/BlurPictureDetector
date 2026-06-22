from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from algo.models import Body


@dataclass
class Frame:
    """A single image frame with all detected bodies and optional image data."""
    path:             Path
    bodies:           list[Body]           = field(default_factory=list)
    image:            np.ndarray | None    = None  # original image (BGR)
    normalized_image: np.ndarray | None    = None  # normalised image (BGR)

    def is_sharp(self) -> bool:
        """Return True if at least one body in this frame passed all scorers."""
        return any(body.passed for body in self.bodies)
