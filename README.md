# BlurPictureDetector

Detects blurry sport photos for photographers who unethically abused the burst mode.

Runs two YOLO models (body pose + face detection) on a folder of images, scores the sharpness of each person's face, and produces annotated previews so you can review the results before any files are moved.

---

## How it works

```
Image → normalize size (long edge = 1800 px)
      → YOLOv8n-pose  → body bounding boxes + 17 COCO keypoints
      → YOLOv8n-face  → face bounding boxes + 5 landmarks
      → match faces to bodies (head-keypoint overlap)
      → compute sharpness score on the narrow face crop
      → annotate preview image
      → write info.json / blurry.csv / blur.lst
```

Sharpness is a weighted combination of two classical metrics:

| Metric | Weight | What it measures |
|---|---|---|
| Laplacian variance | 60 % | Fine detail / high-frequency content |
| Tenengrad | 40 % | Gradient energy — robust to noise |

The two raw values are normalized to a **sharpness score in [0 – 1]** (1 = sharp, 0 = blurry).

---

## Workflow

The tool is split into two steps so you can review annotations before committing to any file moves.

### Step 1 — Review

```
python 1_prep_review.py <image_or_directory> [--sensitivity low|medium|high] [--output <dir>]
```

Reads images, scores each one, and writes an output folder containing:

```
<output_dir>/
    anno_blur/        ← annotated copies of blurry images
    anno_sharp/       ← annotated copies of sharp images
    anno_skipped/     ← annotated copies where no person was detected
    info.json         ← full classification results (used by step 2)
    blurry.csv        ← one row per blurry image with score details
    blur.lst          ← plain list of blurry file paths
    run.log           ← full debug log
```

**Sensitivity thresholds** (score ≤ threshold → blurry):

| Level | Threshold | Use when |
|---|---|---|
| `low` | 0.30 | Flag only severely blurry images |
| `medium` *(default)* | 0.55 | Balanced |
| `high` | 0.75 | Flag even slightly soft images |

**Annotated preview legend:**

- **Blue box / ✓** — person scored sharp
- **Red box / ✗** — person scored blurry
- **Green box** — narrow face crop used for scoring (minimal bbox around the 5 face landmarks + 0.5 % padding)
- **Circles** — face landmarks (blue = confident, red = low confidence)
- **Score label** — sharpness score printed below the face box
- All annotation is semi-transparent (`annotation_alpha = 0.35`); score text is opaque

### Step 2 — Apply

Once you are happy with the review:

```
python 2_apply_changes.py <ref_dir>
```

Reads `info.json` from `<ref_dir>` and moves original source files:

| Classification | Action |
|---|---|
| Blurry | Move original → `<SrcDir>/Blur/` |
| Sharp (annotated copy found) | Leave original in place |
| Sharp (annotated copy missing) | Move original → `<SrcDir>/Blur/` |
| Skipped (annotated copy found) | Move original → `<SrcDir>/Skipped/` |
| Skipped (annotated copy missing) | Leave original in place |

Writes an `apply.log` to `<ref_dir>` when done.

---

## Installation

### Requirements

- Python 3.10+
- PyTorch with CUDA (optional but strongly recommended for speed)

### Setup

```bash
# 1. Install PyTorch — CUDA 12.x example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2. Install remaining dependencies:
pip install -r requirements.txt
```

`yolov8n-pose.pt` is downloaded automatically by Ultralytics on first run.  
`yolov8n-face.pt` is downloaded automatically from the
[akanametov/yolo-face](https://github.com/akanametov/yolo-face) release on first run.

---

## Key configuration (`AppConfig` in `1_prep_review.py`)

| Setting | Default | Description |
|---|---|---|
| `use_narrow_face_box` | `True` | Score and annotate on the minimal landmark bbox rather than the full face detection bbox |
| `face_min_size_fraction` | `0.025` | Discard faces whose long edge is < 2.5 % of the image long edge |
| `face_coverage_min_visible` | `3` | Require ≥ 3 of 5 face landmarks to be confident |
| `normalized_img_max_long_edge` | `1800` | Downscale input to this long edge before processing (never upscales) |
| `annotation_alpha` | `0.35` | Annotation translucency (0 = invisible, 1 = opaque) |

