from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from algo.models import AutoAdjustment, Body


@dataclass
class Frame:
    """A single image frame with all detected bodies and optional image data."""
    path:             Path
    bodies:           list[Body]           = field(default_factory=list)
    image:            np.ndarray | None    = None  # original image (BGR)
    normalized_image: np.ndarray | None    = None  # normalised image (BGR)
    auto_adjustment:  AutoAdjustment | None = None  # exposure/WB correction prescription
    # Disambiguated bookkeeping key (preview filename, album.json/info.json
    # entries, FaceReco origFilename) -- see algo/utils.py::make_unique_import_key.
    # Empty means "use path.stem" (single-source-dir album, the common case).
    output_key:       str = ""

    @property
    def key_stem(self) -> str:
        """The stem used for output artifact filenames (previews, etc.)."""
        return Path(self.output_key).stem if self.output_key else self.path.stem

    def is_sharp(self) -> bool:
        """Return True if at least one body in this frame passed all scorers."""
        return any(body.passed for body in self.bodies)
