from __future__ import annotations

import logging
from pathlib import Path

from algo.config import AppConfig
from algo.frame import Frame
from algo.stage import ProcessStage

log = logging.getLogger("BlurPictureDetector")

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


class FaceRecoStage(ProcessStage):
    """Run face-recognition clustering on the sharp bodies in *output_dir*.

    Iterates through all frames and logs which bodies qualify (``passed=True``).
    The heavy lifting is delegated to :class:`FaceRecoPipeline`, which reads
    ``results.json`` from *output_dir* and writes its output under
    ``<output_dir>/.FaceReco/``.

    *output_dir* must therefore contain a valid ``results.json`` before this
    stage runs (written by an upstream output step).

    Parameters
    ----------
    output_dir:
        Directory produced by the prep/grading stage.
    face_db_dir:
        Optional path to a face-database directory.  Each sub-directory must
        represent a person and contain a ``face.json`` with embeddings.
        Clusters matching a DB entry will be named after that person.
    """

    def __init__(
        self,
        output_dir: Path,
        face_db_dir: Path | None = None,
        face_db_match_threshold: float = 0.72,
    ) -> None:
        self.output_dir = output_dir
        self.face_db_dir = face_db_dir
        self.face_db_match_threshold = face_db_match_threshold

    def process(self, frames: list[Frame], config: AppConfig) -> list[Frame]:
        sharp_body_count = sum(
            1 for frame in frames for body in frame.bodies if body.passed
        )
        log.info("[FaceRecoStage] %d sharp body(ies) across %d frame(s)",
                 sharp_body_count, len(frames))

        if not _FACERECO_AVAILABLE:
            log.warning(
                "[FaceRecoStage] facereco module not available — skipping. "
                "Install with: pip install facenet-pytorch"
            )
            return frames

        try:
            if _FACENET_AVAILABLE:
                provider = FaceNetFaceRecoProvider()
            elif _DLIB_AVAILABLE:
                log.warning("[FaceRecoStage] FaceNet unavailable; falling back to dlib.")
                provider = DlibFaceRecoProvider(
                    face_db_dir=self.face_db_dir,
                    face_db_match_threshold=self.face_db_match_threshold,
                )
            
            else:
                log.warning("[FaceRecoStage] No FaceReco provider available. "
                            "Install facenet-pytorch or face-recognition + dlib.")
                return frames

            facereco_config = FaceRecoConfig(
                face_db_dir=self.face_db_dir,
                face_db_match_threshold=self.face_db_match_threshold,
            )
            pipeline = FaceRecoPipeline(provider=provider, config=facereco_config)
            facereco_dir = pipeline.run(self.output_dir)
            log.info("[FaceRecoStage] face recognition complete: %s", facereco_dir)
        except Exception as exc:
            log.error("[FaceRecoStage] face recognition failed: %s", exc, exc_info=True)

        return frames
