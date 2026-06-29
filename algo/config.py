from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AppConfig:
    # Bounding box and status icon drawn on annotated preview images.
    # Colors are BGR (OpenCV convention).
    annotation_box_color_pass: tuple[int, int, int] = field(default=(255, 0, 0))  # blue  – sharp
    annotation_box_color_fail: tuple[int, int, int] = field(default=(0,   0, 255))  # red   – blurry
    annotation_box_thickness:  int                  = 3
    annotation_icon_size:      int                  = 50  # px, drawn on the resized preview

    # Subject selection: prefer the largest person whose face is at least
    # half-visible, judged by how many face keypoints (out of 5) are
    # detected with confidence >= face_kp_conf_threshold.
    # Set face_kp_min_visible = 0 to disable face-visibility filtering.
    face_kp_min_visible:    int   = 2    # ≥ 3 of 5 face KPs must be confident
    face_kp_conf_threshold: float = 0.5  # per-keypoint confidence cutoff

    # Face-coverage check using the face model's 5 landmarks (eyes / nose / mouth).
    # A matched face is disqualified when fewer than face_coverage_min_visible
    # of its landmarks are confident — meaning the face is more than
    # (1 - face_coverage_min_visible/5) covered.
    # Set face_coverage_min_visible = 0 to disable the check.
    face_coverage_min_visible:    int   = 2    # ≥ 3 of 5 face-model landmarks must be confident
    face_coverage_conf_threshold: float = 0.75  # per-landmark confidence cutoff

    # Minimum face size: disqualify persons whose face bbox long edge is smaller
    # than this fraction of the image long edge (e.g. 0.04 = 4 %).
    # Filters out background spectators or faces too small to reliably analyse.
    # Set to 0 to disable.
    face_min_size_fraction: float = 0.025

    # If True, sharpness is scored on the minimal bbox enclosing the 5 face
    # landmarks rather than the face model's full detection bbox.
    # Falls back to the full bbox when fewer than 2 landmarks are detected.
    use_narrow_face_box: bool = True

    # Face bounding box drawn on annotated previews (separate from the body box).
    annotation_face_box_color:        tuple[int, int, int] = field(default=(0, 255, 255))  # yellow
    annotation_face_box_thickness:    int                  = 1
    # Narrow face bbox: minimal box enclosing the 5 face landmarks.
    annotation_narrow_face_box_color:     tuple[int, int, int] = field(default=(0, 255, 0))  # green
    annotation_narrow_face_box_thickness: int                  = 1
    # Blur score label drawn below each face bounding box.
    annotation_score_font_size_px:          int   = 20   # target text height in pixels
    annotation_score_font_thickness:        int   = 3
    # Rejection reason label drawn on failed bodies instead of jersey colour.
    annotation_rejection_reason_font_size_px: int = 16   # target text height in pixels
    # Face landmark circle.
    annotation_face_kp_radius:        int   = 3    # circle radius (px)
    annotation_face_kp_thickness:     int   = 1    # circle line thickness
    # Body keypoint square.
    annotation_body_kp_size:          int   = 3    # square side (px)
    annotation_body_kp_thickness:     int   = 1    # square line thickness
    # Body skeleton line.
    annotation_skeleton_thickness:    int   = 1    # line thickness (px)
    # Annotation opacity: 1.0 = fully opaque, 0.0 = invisible.
    annotation_alpha:                 float = 0.25

    # Number of largest bodies (by bbox area) always annotated regardless of
    # pass/fail.  All passed bodies are annotated on top of this set.
    annotation_top_n_bodies:          int   = 5

    # Annotated preview / processing image scaling.
    # Images are downsized so the long edge equals normalized_img_max_long_edge.
    # Never upscales.
    normalized_img_max_long_edge: int = 1800


app_config = AppConfig()
