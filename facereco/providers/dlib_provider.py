from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .base import BodyRecord, FaceRecoProvider, Player

try:
    import face_recognition
except ImportError:  # pragma: no cover - optional dependency at runtime
    face_recognition = None


@dataclass
class DlibFaceRecoProvider(FaceRecoProvider):
    """dlib-based provider using the face_recognition package.

    This provider currently focuses on embedding extraction and storage. It
    returns an unnamed Player with the embedding in player.internal so another
    phase can do supervised identity assignment.
    """

    model: str = "small"
    num_jitters: int = 1

    def provider_name(self) -> str:
        return "dlib"

    def predict_player(self, image_bgr: np.ndarray, body: BodyRecord) -> Player:
        if face_recognition is None:
            raise RuntimeError(
                "face_recognition is not installed. Install with: "
                "pip install face-recognition dlib"
            )

        face_box = body.narrow_face_bbox or body.face_bbox
        if face_box is None:
            return Player(name="", jersey_number=None, confidence=0.0, internal={"embedding": None})

        h, w = image_bgr.shape[:2]
        left, top, right, bottom = face_box.as_px_ints(w, h)
        left = max(0, min(w - 1, left))
        right = max(0, min(w - 1, right))
        top = max(0, min(h - 1, top))
        bottom = max(0, min(h - 1, bottom))
        if right <= left or bottom <= top:
            return Player(name="", jersey_number=None, confidence=0.0, internal={"embedding": None})

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(
            rgb,
            known_face_locations=[(top, right, bottom, left)],
            num_jitters=self.num_jitters,
            model=self.model,
        )
        if not encodings:
            return Player(name="", jersey_number=None, confidence=0.0, internal={"embedding": None})

        embedding = np.asarray(encodings[0], dtype=np.float32)
        return Player(
            name="",
            jersey_number=None,
            confidence=body.confidence,
            internal={
                "embedding": embedding,
                "provider": self.provider_name(),
            },
        )
