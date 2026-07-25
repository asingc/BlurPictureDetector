"""Torchvision-based person detector (hybrid engine, body-box stage only).

MediaPipe's Pose Landmarker under-recalls badly on busy multi-person sports
photos (verified: ~1 body/image vs YOLOv8-pose's ~4.2 bodies/image on the same
732-image test set — see /memories/repo/python-notes.md for the full
investigation). Torchvision's COCO-pretrained Faster R-CNN person detector
recovers most of that recall (5-11 persons/image on the same test images) and
is permissively licensed (BSD-3-Clause, torchvision/PyTorch).

Used only for STAGE 1 (find body bounding boxes). Stage 2 (33 body keypoints)
still runs MediaPipe Pose Landmarker, but on a close-range per-person crop
instead of the full image — the same two-pass philosophy already used for
faces (see detect_face_for_body_mp in mediapipe_provider.py).
"""

from __future__ import annotations

import logging
import threading

import cv2
import numpy as np
import torch

from algo.models import Box

log = logging.getLogger("BlurPictureDetector")

_COCO_PERSON_LABEL = 1

_PERSON_DETECTOR = None
_PERSON_DETECTOR_DEVICE: str | None = None
_PERSON_DETECTOR_LOCK = threading.Lock()


def load_person_detector(force_cpu: bool = False):
    """Return the process-wide torchvision person-detector singleton
    (Faster R-CNN ResNet50-FPN-v2, COCO-pretrained, eval mode).

    Runs on CUDA automatically when available (much faster than CPU for this
    model — see /memories/repo/python-notes.md), falling back to CPU
    otherwise. Pass ``force_cpu=True`` (e.g. from ``--cpu-only``) to pin it to
    CPU regardless of GPU availability. Re-loaded (and moved) if a different
    device is requested than the cached singleton's.
    """
    global _PERSON_DETECTOR, _PERSON_DETECTOR_DEVICE
    requested_device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if _PERSON_DETECTOR is not None and _PERSON_DETECTOR_DEVICE == requested_device:
        return _PERSON_DETECTOR
    with _PERSON_DETECTOR_LOCK:
        if _PERSON_DETECTOR is not None and _PERSON_DETECTOR_DEVICE == requested_device:
            return _PERSON_DETECTOR
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn_v2,
            FasterRCNN_ResNet50_FPN_V2_Weights,
        )
        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            box_score_thresh=0.5,
        )
        model.eval()
        model.to(requested_device)
        _PERSON_DETECTOR = model
        _PERSON_DETECTOR_DEVICE = requested_device
        log.info(
            "Torchvision Faster R-CNN (ResNet50-FPN-v2) person detector loaded on %s",
            requested_device,
        )
    return _PERSON_DETECTOR


def extract_person_boxes_tv(image: np.ndarray, model, max_bodies: int = 8) -> list[Box]:
    """Return up to *max_bodies* person bounding boxes (largest-first),
    normalised to [0, 1] fractions of *image*'s width/height."""
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    device = next(model.parameters()).device
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().to(device) / 255.0

    with torch.no_grad():
        result = model([tensor])[0]

    boxes_px = result["boxes"].cpu().numpy()
    labels = result["labels"].cpu().numpy()
    scores = result["scores"].cpu().numpy()

    person_idx = [i for i, lbl in enumerate(labels) if lbl == _COCO_PERSON_LABEL]
    if not person_idx:
        log.debug("[bodies:tv] 0 persons detected")
        return []

    areas = [
        (boxes_px[i, 2] - boxes_px[i, 0]) * (boxes_px[i, 3] - boxes_px[i, 1])
        for i in person_idx
    ]
    order = sorted(range(len(person_idx)), key=lambda k: areas[k], reverse=True)[:max_bodies]

    boxes: list[Box] = []
    for k in order:
        i = person_idx[k]
        x1, y1, x2, y2 = boxes_px[i]
        boxes.append(Box(
            max(0.0, x1 / w), max(0.0, y1 / h),
            min(1.0, x2 / w), min(1.0, y2 / h),
        ))
        log.debug("[bodies:tv]   person bbox_px=(%.0f,%.0f,%.0f,%.0f) score=%.3f",
                  x1, y1, x2, y2, scores[i])

    log.debug("[bodies:tv] %d person(s) returned (of %d raw detections)",
              len(boxes), len(person_idx))
    return boxes
