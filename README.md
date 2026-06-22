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

Sharpness is computed by a pluggable `SharpnessEvaluator`. The active implementation is `GeometricMeanEvaluator`, which uses two classical metrics:

| Metric | What it measures |
|---|---|
| Laplacian variance | Fine detail / high-frequency content |
| Tenengrad | Gradient energy — robust to noise |

Each metric is normalised to a sharpness component in [0 – 1], then combined as a **geometric mean**:

```
score = sqrt(lap_sharp × ten_sharp)
```

The geometric mean only scores high when *both* metrics agree, which avoids false "sharp" results caused by noise, compression artefacts, or over-sharpening that inflate one metric while the other stays low.

The raw values are still available in the CSV output. To switch back to the previous weighted-average formula, change the active evaluator in `1_prep_review.py` to `LaplacianTenengradEvaluator`.

---

## Workflow

The tool is split into two steps so you can review and override the automatic classifications before any files are moved.

### Step 1 — Score and annotate

```
python 1_prep_review.py <image_or_directory> [--sensitivity low|medium|high] [--output <dir>] [--skip-facereco]
```

Scores every image and writes an output folder:

```
<output_dir>/
    anno_blur/        ← annotated previews of images scored as blurry
    anno_sharp/       ← annotated previews of images scored as sharp
    anno_skipped/     ← annotated previews where no person was detected
    info.json         ← full classification results (used by step 2)
    blurry.csv        ← one row per blurry image with score details
    blur.lst          ← plain list of blurry file paths
    run.log           ← full debug log
    .FaceReco/        ← face recognition clusters (unless --skip-facereco)
```

**Face Recognition (optional):**

By default, after annotation previews are generated, face recognition automatically clusters faces from sharp, in-team bodies using the Facenet provider. The output is stored in `.FaceReco/` within the same output directory. You can switch back to dlib with `4_face_reco.py --provider dlib`, or disable the auto-run with `--skip-facereco`.

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

### Step 2 — Review

Open the `anno_blur/` and `anno_sharp/` folders in your photo viewer and **delete any preview images you want to override**:

| Folder | Delete a preview when… | Effect on the original |
|---|---|---|
| `anno_blur/` | The photo isn't actually blurry — you want to keep it | Left in place |
| `anno_sharp/` | The photo is sharp but you don't want it | Moved to `Blur/` |
| `anno_skipped/` | You want to leave the original untouched | Left in place |

Leave previews you agree with untouched.

### Step 3 — Apply

```
python 2_apply_changes.py <ref_dir>
```

Compares what previews remain against `info.json` and moves original source files accordingly:

| Preview in | Preview present | Preview deleted |
|---|---|---|
| `anno_blur/` | Move original → `<SrcDir>/Blur/` | Leave original in place |
| `anno_sharp/` | Leave original in place | Move original → `<SrcDir>/Blur/` |
| `anno_skipped/` | Move original → `<SrcDir>/Skipped/` | Leave original in place |

No files are ever deleted. Writes an `apply.log` to `<ref_dir>` when done.

### Step 4 - Face Recognition Clustering (Integrated into Step 1)

Face recognition now runs automatically at the end of Step 1 unless `--skip-facereco` is used. It processes all bodies with `sharpness_score >= min(sensitivity_threshold, 0.03)`, regardless of blur status or team. If you want to run it independently with different parameters:

```
python 4_face_reco.py <ref_dir> [--provider dlib] [--cluster-threshold 0.68] [--face-buffer-ratio 0.20] [--min-sharpness 0.03]
```

**Filtering logic:**
- Keeps only bodies with `sharpness_score >= min(sensitivity_threshold, 0.03)`
- Processes all bodies regardless of blur status or team (allows reviewing even blurry faces)
- Default minimum sharpness is 0.03 (very permissive)

**What it does:**
1. Loads per-body data from `results.json`
2. Extracts dlib face embeddings from qualifying bodies
3. Clusters likely same-person faces by cosine similarity (default threshold: 0.68)
4. Writes face crops and metadata to `<ref_dir>/.FaceReco/`

Output layout (within Phase 1 output directory):

```
<ref_dir>/.FaceReco/
    0000/
        Face/           ← qualified body faces
        Negative/       ← faces from other clusters
        face.json       ← editable metadata
    0001/
        Face/
        Negative/
        face.json
```

Each `face.json` stores editable person metadata (`name`, `playernum`) and a `faces` list with:

- original filename
- original body JSON payload  
- crop file name
- confidence
- embedding payload (float32, base64-encoded)

Provider API and implementations live under the `facereco/` module:

- `facereco/providers/base.py` - abstract provider interface
- `facereco/providers/dlib_provider.py` - dlib implementation
- `facereco/providers/facenet_provider.py` - FaceNet (facenet-pytorch / VGGFace2) implementation

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

