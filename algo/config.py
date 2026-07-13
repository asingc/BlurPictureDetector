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

    # Jersey team-colour matching mode.
    # When True, the polled team-colour check collapses every jersey into one of
    # two lightness buckets — "Light" (white / pale) vs "Dark" (non-white /
    # coloured) — instead of comparing exact Hue:Shade labels.  This makes team
    # filtering robust to the large brightness swings caused by light and shadow,
    # which would otherwise read the same real-world jersey as different shades.
    # Set to False to fall back to exact shade/hue matching.
    jersey_binary_lightness: bool = True
    # A jersey is classified "Light" only when its mean L* is at least
    # jersey_light_l_min AND its chroma (sqrt(a*^2 + b*^2)) is at most
    # jersey_light_chroma_max.  The chroma gate keeps brightly-lit *coloured*
    # jerseys (e.g. yellow, light blue) in the "Dark" bucket.  Only used when
    # jersey_binary_lightness is True.
    jersey_light_l_min:      float = 55.0
    jersey_light_chroma_max: float = 20.0

    # Jersey team-colour matching by weighted L*a*b* distance (preferred).
    # When True (default) the polled team-colour check compares each body's
    # measured L*a*b* against the team's target colour using a distance
    # decomposed into Lightness (L*), Chroma (C*, saturation) and Hue (H*)
    # components — the same decomposition CIE94/CIEDE2000 use for perceptual
    # colour differences:
    #     dist = sqrt(l_weight*dL^2 + c_weight*dC^2 + h_weight*dH^2)
    # Hue is what makes a jersey "yellow" vs. "green" and is largely
    # invariant to brightness/shadow, so it stays at full weight.  Lightness
    # and Chroma both swing heavily with lighting/shadow (shadows both darken
    # *and* desaturate a jersey) so they are down-weighted.  This tolerates
    # brightness/shadow variation while still rejecting an actual hue change.
    # Takes precedence over jersey_binary_lightness; set both to False to
    # fall back to exact shade/hue matching.
    jersey_lab_match:    bool  = True
    # Weight applied to the L* (brightness) squared difference.
    # Smaller = more forgiving of brightness.  At 0.0 brightness is ignored
    # entirely (white and black would then be indistinguishable, which is why
    # a small non-zero weight is kept so achromatic jerseys still split by
    # lightness).
    jersey_lab_l_weight: float = 0.15
    # Weight applied to the C* (chroma / saturation) squared difference.
    # Smaller = more forgiving of a jersey looking more washed-out or more
    # vivid depending on lighting (shadow desaturates; direct sun saturates).
    jersey_lab_c_weight: float = 0.30
    # Weight applied to the H* (hue) squared difference.  Kept at full weight
    # by default since hue is the primary perceptual "which colour is this"
    # signal and should stay discriminative even when brightness/chroma are
    # forgiven.
    jersey_lab_h_weight: float = 1.0
    # Maximum weighted LCh distance for a body to match the team colour.
    jersey_lab_max_dist: float = 22.0

    # ------------------------------------------------------------------
    # Auto adjustment (exposure / brightness)
    # ------------------------------------------------------------------
    # Target mean brightness (0-1 gray level) the auto-exposure correction
    # aims for, measured on a 50/50 blend of the whole image and the main
    # subject's face crop.
    auto_adjust_target_brightness: float = 0.45
    # EV correction is rounded to the nearest multiple of this step
    # (e.g. 0.5 → corrections read as EV +0.5, EV -1.0, …).
    auto_adjust_ev_step:           float = 0.5
    # Clamp on the magnitude of the EV correction, in stops.
    auto_adjust_max_ev:            float = 2.0


app_config = AppConfig()
