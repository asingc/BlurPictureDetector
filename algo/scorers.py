from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

from algo.config import app_config
from algo.models import Body, Box, Face
from algo.sharpness import sharpness_evaluator
from algo.utils import _HEAD_KP_INDICES, _color_from_label, _matches_allowed_jersey_color, _narrow_face_box, cap_long_edge

log = logging.getLogger("BlurPictureDetector")


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class BodyArrayScorerBase(ABC):
    """
    Pipeline stage that operates on the full list of detected bodies for one
    image.  Implementations may score, filter, rank, or augment the list.

    Parameters
    ----------
    normalized_image : normalised BGR image used as context (e.g. for cropping).
    bodies           : body objects as produced by the detection phase.

    Returns the (possibly modified or filtered) body list.
    """

    @abstractmethod
    def process(self, normalized_image: np.ndarray, bodies: list[Body]) -> list[Body]:
        ...


class BodyScorerBase(ABC):
    """
    Scorer that evaluates a single Body and returns an updated Body with
    the pass/fail verdict written to body.passed (and optionally to its
    faces / keypoints).  Scorers use a turn-off strategy: set passed=False
    to disqualify; leave it True to keep the default passing state.
    """

    @abstractmethod
    def binary_classify(self, body: Body, normalized_image: np.ndarray) -> Body:
        ...


# ---------------------------------------------------------------------------
# Concrete scorer implementations
# ---------------------------------------------------------------------------

class MatchedFaceScorer(BodyScorerBase):
    """Rule 5 — body must have at least one matched face.
    Should run first: bodies with no face cannot be sharpness-scored."""

    def binary_classify(self, body: Body, normalized_image: np.ndarray) -> Body:
        if not body.faces:
            body.passed = False
            body.rejection_reason = "no matched face"
            bx1, by1, bx2, by2 = body.bbox.as_px_ints(normalized_image.shape[1], normalized_image.shape[0])
            log.debug("[scorer:matched_face] body bbox=(%d,%d,%d,%d): no matched face → fail",
                      bx1, by1, bx2, by2)
        return body


class FaceSizeScorer(BodyScorerBase):
    """Rule 1 — every face bbox long edge must be >= min_fraction × image long edge.
    Faces that are too small are disqualified; if all faces on a body fail,
    the body is also disqualified."""

    def __init__(self, min_fraction: float | None = None) -> None:
        self.min_fraction = min_fraction if min_fraction is not None \
                            else app_config.face_min_size_fraction

    def binary_classify(self, body: Body, normalized_image: np.ndarray) -> Body:
        if self.min_fraction <= 0:
            return body
        h, w   = normalized_image.shape[:2]
        for face in body.faces:
            face_long = max(face.bbox.width, face.bbox.height)
            if face_long < self.min_fraction:
                face.passed = False
                fx1, fy1, fx2, fy2 = face.bbox.as_px_ints(w, h)
                log.debug("[scorer:face_size] face bbox=(%d,%d,%d,%d): long_edge=%.3f < min=%.3f → fail",
                          fx1, fy1, fx2, fy2, face_long, self.min_fraction)
        if body.faces and not any(f.passed for f in body.faces):
            body.passed = False
            body.rejection_reason = "all faces too small"
        return body


class BodyHeadKPVisibilityScorer(BodyScorerBase):
    """Rule 3 — the pose model must see at least min_visible of the 5 head
    keypoints (nose, eyes, ears) with confidence >= conf_threshold.
    Individual keypoints below the threshold are marked passed=False."""

    def __init__(
        self,
        min_visible:    int   | None = None,
        conf_threshold: float | None = None,
    ) -> None:
        self.min_visible    = min_visible    if min_visible    is not None \
                              else app_config.face_kp_min_visible
        self.conf_threshold = conf_threshold if conf_threshold is not None \
                              else app_config.face_kp_conf_threshold

    def binary_classify(self, body: Body, normalized_image: np.ndarray) -> Body:
        if self.min_visible <= 0:
            return body
        for i in _HEAD_KP_INDICES:
            if i < len(body.keypoints) and \
               body.keypoints[i].confidence < self.conf_threshold:
                body.keypoints[i].passed = False
        n_vis = sum(
            1 for i in _HEAD_KP_INDICES
            if i < len(body.keypoints) and body.keypoints[i].passed
        )
        if n_vis < self.min_visible:
            body.passed = False
            body.rejection_reason = f"only {n_vis}/{len(_HEAD_KP_INDICES)} head keypoints visible (need {self.min_visible})"
            bx1, by1, bx2, by2 = body.bbox.as_px_ints(normalized_image.shape[1], normalized_image.shape[0])
            log.debug("[scorer:head_kp] body bbox=(%d,%d,%d,%d): %d/%d head KPs visible (need %d) → fail",
                      bx1, by1, bx2, by2,
                      n_vis, len(_HEAD_KP_INDICES), self.min_visible)
        return body


class FaceLandmarkVisibilityScorer(BodyScorerBase):
    """Rule 4 — for each face, at least min_visible of the 5 face-model landmarks
    must have confidence >= conf_threshold (i.e. the face must not be covered).
    Individual landmarks and faces below the threshold are marked passed=False.
    If all faces on a body fail, the body is also disqualified."""

    def __init__(
        self,
        min_visible:    int   | None = None,
        conf_threshold: float | None = None,
    ) -> None:
        self.min_visible    = min_visible    if min_visible    is not None \
                              else app_config.face_coverage_min_visible
        self.conf_threshold = conf_threshold if conf_threshold is not None \
                              else app_config.face_coverage_conf_threshold

    def binary_classify(self, body: Body, normalized_image: np.ndarray) -> Body:
        if self.min_visible <= 0:
            return body
        for face in body.faces:
            if not face.passed:
                continue
            for lm in face.landmarks:
                if lm.confidence < self.conf_threshold:
                    lm.passed = False
            n_vis = face.n_visible()
            if n_vis < self.min_visible:
                face.passed = False
                fx1, fy1, fx2, fy2 = face.bbox.as_px_ints(normalized_image.shape[1], normalized_image.shape[0])
                log.debug("[scorer:face_landmark] face bbox=(%d,%d,%d,%d): %d/%d landmarks visible (need %d) → fail",
                          fx1, fy1, fx2, fy2,
                          n_vis, len(face.landmarks), self.min_visible)
        if body.faces and not any(f.passed for f in body.faces):
            body.passed = False
            body.rejection_reason = "all faces failed landmark visibility"
        return body


class FaceSharpnessScorer(BodyScorerBase):
    """Rule 2 — the best sharpness score across all passing faces must exceed
    the threshold.  Uses the narrow landmark bbox when available, falling back
    to the full face bbox."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def binary_classify(self, body: Body, normalized_image: np.ndarray) -> Body:
        h, w        = normalized_image.shape[:2]
        narrow_pad  = round(0.005 * max(h, w))
        best_score  = 0.0
        best_face:   Face | None = None
        best_narrow: Box  | None = None
        best_lap    = 0.0
        best_ten    = 0.0

        for face in body.faces:
            if not face.passed:
                continue
            narrow_box = _narrow_face_box(face, pad=narrow_pad, img_w=w, img_h=h) \
                         if app_config.use_narrow_face_box else None
            score_box  = narrow_box or face.bbox
            fx1, fy1, fx2, fy2 = score_box.as_px_ints(w, h)
            crop = normalized_image[fy1:fy2, fx1:fx2]
            if crop.size == 0:
                continue
            crop = cap_long_edge(crop, max(h, w) * 0.04)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            s, lv, t = sharpness_evaluator.score(gray)
            log.debug("[scorer:sharpness] face conf=%.3f score=%.4f (lap=%.2f ten=%.2f)",
                      face.confidence, s, lv, t)
            if s > best_score:
                best_score   = s
                best_face    = face
                best_narrow  = narrow_box
                best_lap     = lv
                best_ten     = t
        body.sharpness_score = best_score
        body.best_face       = best_face
        body.best_narrow_box = best_narrow
        body.lap_var         = best_lap
        body.ten             = best_ten
        if best_score <= self.threshold:
            body.passed = False
            body.rejection_reason = f"sharpness score {best_score:.4f} <= threshold {self.threshold:.2f}"
            bx1, by1, bx2, by2 = body.bbox.as_px_ints(normalized_image.shape[1], normalized_image.shape[0])
            log.debug("[scorer:sharpness] body bbox=(%d,%d,%d,%d): best_score=%.4f <= threshold=%.2f → fail",
                      bx1, by1, bx2, by2,
                      best_score, self.threshold)
        return body


class JerseyColorScorer(BodyScorerBase):
    """Disqualify bodies whose jersey/cloth color is not in the allowed colour set.

    Must run *after* cloth color has already been predicted for each body.
    When *allowed_colors* is empty this scorer is a no-op (all bodies pass).
    """

    def __init__(self, allowed_colors: frozenset[str]) -> None:
        self.allowed_colors = allowed_colors

    def binary_classify(self, body: Body, normalized_image: np.ndarray) -> Body:
        if not self.allowed_colors:
            return body
        color = _color_from_label(body.cloth_color)
        if not _matches_allowed_jersey_color(color, self.allowed_colors):
            body.passed = False
            body.rejection_reason = f"jersey color '{color}' not in allowed set {sorted(self.allowed_colors)}"
            log.debug(
                "[scorer:jersey_color] body bbox=(%d,%d,%d,%d): cloth_color=%s not in %s → fail",
                    *body.bbox.as_px_ints(normalized_image.shape[1], normalized_image.shape[0]),
                color, sorted(self.allowed_colors),
            )
        return body


class BodyArrayScorer(BodyArrayScorerBase):
    """Runs a sequence of BodyScorerBase scorers over every body in the list.

    normalized_image is forwarded to every binary_classify call so scorers
    that need to crop from it receive it directly.

    Short-circuits per body: once a body is marked passed=False no further
    scorers are called on it (avoids expensive work on already-failed bodies).

    Returns all bodies with their passed flags updated — callers decide
    whether to keep or discard failed bodies.
    """

    def __init__(self, scorers: list[BodyScorerBase]) -> None:
        self._scorers = scorers

    def process(self, normalized_image: np.ndarray, bodies: list[Body]) -> list[Body]:
        for body in bodies:
            for scorer in self._scorers:
                body = scorer.binary_classify(body, normalized_image)
                if not body.passed:
                    break  # short-circuit: no point scoring a disqualified body
        return bodies
