from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import cv2

try:
    import torch
    import torch.nn.functional as F
    from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
except ImportError:  # pragma: no cover - optional dependency at runtime
    torch = None
    F = None
    InceptionResnetV1 = None
    fixed_image_standardization = None

from .facereco_provider import BodyRecord, FaceRecoProvider, Player


_MODEL: InceptionResnetV1 | None = None
_DEVICE: "torch.device" | None = None
_MODEL_DEVICE_NAME: str | None = None


@dataclass
class FaceNetFaceRecoProvider(FaceRecoProvider):
    """FaceNet-based provider using facenet-pytorch embeddings."""

    model_name: str = "vggface2"
    device: str | None = None
    image_size: int = 160
    _runtime_info: dict = field(default_factory=dict, init=False, repr=False)

    def provider_name(self) -> str:
        return "facenet"

    def predict_player(self, image_bgr: np.ndarray, body: BodyRecord) -> Player:
        if torch is None or F is None or InceptionResnetV1 is None or fixed_image_standardization is None:
            raise RuntimeError(
                "facenet-pytorch is not installed. Install with: pip install facenet-pytorch"
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

        crop = image_bgr[top:bottom, left:right]
        if crop.size == 0:
            return Player(name="", jersey_number=None, confidence=0.0, internal={"embedding": None})

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)
        tensor = F.interpolate(tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        tensor = fixed_image_standardization(tensor)

        model = self._get_model()
        with torch.no_grad():
            embedding = model(tensor.to(_DEVICE)).cpu().numpy()[0].astype(np.float32)

        return Player(
            name="",
            jersey_number=None,
            confidence=body.confidence,
            internal={
                "embedding": embedding,
                "provider": self.provider_name(),
            },
        )

    def _get_model(self) -> InceptionResnetV1:
        global _MODEL, _DEVICE, _MODEL_DEVICE_NAME

        if torch is None or InceptionResnetV1 is None:
            raise RuntimeError(
                "facenet-pytorch is not installed. Install with: pip install facenet-pytorch"
            )

        device_name = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if _MODEL is not None and _MODEL_DEVICE_NAME == device_name:
            return _MODEL

        _DEVICE = torch.device(device_name)
        _MODEL = InceptionResnetV1(pretrained=self.model_name).eval().to(_DEVICE)
        _MODEL_DEVICE_NAME = device_name
        return _MODEL
