from __future__ import annotations

from abc import ABC, abstractmethod

from algo.config import AppConfig
from algo.frame import Frame


class ProcessStage(ABC):
    """Abstract base class for a single stage in the processing pipeline.

    Concrete subclasses implement :meth:`process` to transform (or filter)
    the list of frames and return the result to the next stage.
    """

    @abstractmethod
    def process(self, frames: list[Frame], config: AppConfig) -> list[Frame]:
        """Process *frames* and return the (possibly modified) list.

        Args:
            frames: The full set of image frames to operate on.
            config: Application-wide configuration.

        Returns:
            The processed list of frames, passed to the next stage.
        """
