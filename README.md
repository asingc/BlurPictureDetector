# BlurPictureDetector

Detects blurry sport photos for photographers who unethically abused the burst mode.

Runs two YOLO models (body pose + face detection) on a folder of images, filters subjects down to your team by jersey colour, scores the sharpness of each person's face, and produces annotated previews so you can review the results before any files are moved.

---

## How it works

```
Image → normalize size (long edge = 1800 px)
      → YOLOv8n-pose  → body bounding boxes + 17 COCO keypoints
      → YOLOv8n-face  → face bounding boxes + 5 landmarks
      → match faces to bodies (head-keypoint overlap)
      → grade sharpness of each person's face crop
      → poll jersey colours → keep only your team (+ forced colours)
      → annotate preview image
      → write results.json / info.json / blurry.csv / blur.lst
      → (optional) cluster faces into .FaceReco/
```

The pipeline is a sequence of stages (`algo/stages/`):

| Stage | Responsibility |
|---|---|
| `ImageAnalysisStage` | Run both YOLO models, match faces to bodies |
| `GradingStage` | Compute the sharpness score and pass/fail per body |
| `JerseyCountingStage` | Detect the dominant team colour and disqualify off-team bodies |
| `AnnotationStage` | Draw and save annotated previews into `anno_*/` |
| `FaceRecoStage` | Cluster qualifying faces into `.FaceReco/` (optional) |

### Sharpness scoring

Sharpness is computed by a pluggable `SharpnessEvaluator`. The active implementation is `GeometricMeanEvaluator`, which uses two classical metrics:

| Metric | What it measures |
|---|---|
| Laplacian variance | Fine detail / high-frequency content |
| Tenengrad | Gradient energy — robust to noise |

Each metric is normalised to a sharpness component in [0 – 1], then combined as a **geometric mean**:

```
score = sqrt(lap_sharp × ten_sharp)
```

The geometric mean only scores high when *both* metrics agree, which avoids false "sharp" results caused by noise, compression artefacts, or over-sharpening that inflate one metric while the other stays low. The raw values are still available in the CSV output. To switch back to the previous weighted-average formula, change the active evaluator in `algo/sharpness.py` to `LaplacianTenengradEvaluator`.

### Team / jersey filtering

`JerseyCountingStage` measures each body's jersey colour in CIE L\*a\*b\*, polls the dominant colour across all sharp bodies in a frame to identify "your" team, and disqualifies bodies wearing a different colour so the tool only scores and annotates your players. Matching is brightness-forgiving (a jersey in shadow still matches the same jersey in sunlight). Pass `--jerseycolor` to bias the team colour, `+`-prefix a colour to force-include it (e.g. goalie kit), or `--noteam` to disable filtering entirely.

---

## Workflow

The tool is split into steps so you can review and override the automatic classifications before any files are moved.

### Step 1 — Score and annotate

```
python 1_prep_review.py <image_or_directory> [options]
```

| Option | Default | Description |
|---|---|---|
| `--sensitivity low\|medium\|high\|<n>` | `medium` | Blur threshold; or pass a numeric value (0–1) directly |
| `--jerseycolor "COLOR[;COLOR...]"` | `blue;white;+purple;+orange;+light blue;+pink` | Team colours; `+` = forced-include; empty string disables filtering |
| `--noteam` | off | Disable jersey filtering (score every detected person) |
| `--output <dir>` | `output/<timestamp>-<input>/` | Root output directory |
| `--skip-facereco` | off | Don't run face-recognition clustering |
| `--face-db <dir>` | none | Match clusters against an existing face DB |
| `--face-db-match-threshold <n>` | `0.72` | Cosine similarity required for a face-DB match |
| `--align-faces` | off | Similarity-align faces to a 5-point template before embedding |
| `--debug-align` | off | Write alignment QA images to `.FaceReco/.debug` |

Scores every image and writes an output folder:

```
<output_dir>/
    anno_blur/        ← annotated previews of images scored as blurry
    anno_sharp/       ← annotated previews of images scored as sharp
    anno_skipped/     ← annotated previews where no person was detected
    results.json      ← full per-body data (used by face recognition)
    info.json         ← classification results (used by step 2)
    blurry.csv        ← one row per blurry image with score details
    blur.lst          ← plain list of blurry file paths
    run.log           ← full debug log
    .FaceReco/        ← face recognition clusters (unless --skip-facereco)
```

**Face Recognition (optional):**

By default, after annotation previews are generated, face recognition automatically clusters qualifying faces using the Facenet provider. The output is stored in `.FaceReco/` within the same output directory. You can re-run independently with a different provider via `4_face_reco.py`, or disable the auto-run with `--skip-facereco`.

**Sensitivity thresholds** (score ≤ threshold → blurry):

| Level | Threshold | Use when |
|---|---|---|
| `low` | 0.35 | Flag only severely blurry images |
| `medium` *(default)* | 0.50 | Balanced |
| `high` | 0.70 | Flag even slightly soft images |

You can also pass any numeric value directly, e.g. `--sensitivity 0.45`.

**Annotated preview legend:**

- **Blue box / ✓** — person scored sharp
- **Red box / ✗** — person scored blurry
- **Green box** — narrow face crop used for scoring (minimal bbox around the 5 face landmarks)
- **Yellow box** — full face-detection bbox
- **Circles** — face landmarks (blue = confident, red = low confidence)
- **Score label** — sharpness score printed below the face box
- All annotation is semi-transparent (`annotation_alpha = 0.25`); score text is opaque

### Step 2 — Review

Open the `anno_*` folders in your photo viewer and **delete any preview images you want to override**:

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

### Step 4 — Sync to sibling folders (optional)

```
python 3_sync_results.py <target_dir>
```

Propagates `Blur/` and `Skipped/` decisions from one already-applied directory to all of its sibling directories. For each image whose filename stem matches a file already sorted into `<target_dir>/blur/` or `<target_dir>/skipped/`, the matching original in each sibling folder is moved into that sibling's own `blur/` or `skipped/` sub-folder. Useful when the same shoot was exported into multiple format folders (e.g. JPG and RAW). No files are ever deleted; a `sync_results.log` is written to `target_dir`.

---

## Face recognition

Face recognition runs automatically at the end of Step 1 (unless `--skip-facereco`). To run it independently with different parameters:

```
python 4_face_reco.py <prep_output_dir> [options]
```

| Option | Default | Description |
|---|---|---|
| `--provider dlib\|facenet` | `facenet` | Embedding provider |
| `--cluster-threshold <n>` | `0.72` | Cosine similarity for clustering (higher = tighter clusters) |
| `--face-buffer-ratio <n>` | `0.15` | Extra crop padding around the face box (per side) |
| `--face-db <dir>` | none | Match faces against an existing face DB |
| `--face-db-match-threshold <n>` | `0.80` | Per-face cosine similarity required for a DB match |
| `--align-faces` | off | Similarity-align faces before embedding |
| `--debug-align` | off | Write alignment QA images to `.FaceReco/.debug` |
| `--open-viewer` | off | Open the generated `.FaceReco/` folder afterwards |

**What it does:**

1. Loads per-body data from `results.json`.
2. Extracts face embeddings from qualifying bodies (via the selected provider).
3. Clusters likely same-person faces by cosine similarity (agglomerative, average linkage).
4. Optionally matches faces against a face DB and names matched clusters.
5. Writes face crops and metadata to `<prep_output_dir>/.FaceReco/`.

Output layout:

```
<prep_output_dir>/.FaceReco/
    0000/
        Face/           ← qualified body faces
        Negative/       ← reserved for manually curated DB counter-examples
        face.json       ← editable metadata
    0001/
        Face/
        Negative/
        face.json
```

Each `face.json` stores editable person metadata (`name`, `playernum`) and a `faces` list with the original filename, original body JSON payload, crop file name, confidence, an `aligned` flag, and a base64-encoded float32 embedding.

### Provider modules (`algo/`)

- `algo/facereco_provider.py` — abstract provider interface
- `algo/dlib_provider.py` — dlib implementation
- `algo/facenet_provider.py` — FaceNet (facenet-pytorch / VGGFace2) implementation
- `algo/face_crop_embed.py` — shared crop → detect → (optional align) → embed pipeline used by both prediction and DB rebuild

### Rebuilding a face DB

```
python RebuildFaceDB.py <facedb_dir> [--align-faces]
```

Recomputes every embedding in a `.FaceReco/` directory from its saved crop images. Use `--align-faces` only if the database was originally built with alignment — the rebuild and prediction must use the same flag so embeddings live in the same space.

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

## Key configuration (`AppConfig` in `algo/config.py`)

| Setting | Default | Description |
|---|---|---|
| `use_narrow_face_box` | `True` | Score and annotate on the minimal landmark bbox rather than the full face detection bbox |
| `face_min_size_fraction` | `0.025` | Discard faces whose long edge is < 2.5 % of the image long edge |
| `face_coverage_min_visible` | `2` | Require this many of 5 face landmarks to be confident |
| `normalized_img_max_long_edge` | `1800` | Downscale input to this long edge before processing (never upscales) |
| `annotation_alpha` | `0.25` | Annotation translucency (0 = invisible, 1 = opaque) |
| `annotation_top_n_bodies` | `5` | Always annotate the N largest bodies regardless of pass/fail |
| `jersey_lab_match` | `True` | Match team colour by weighted L\*a\*b\* distance (brightness-forgiving) |
| `jersey_lab_l_weight` | `0.15` | Weight on the L\* (brightness) axis when matching jersey colour |
| `jersey_lab_max_dist` | `22.0` | Max weighted L\*a\*b\* distance to count as the team colour |
| `jersey_binary_lightness` | `True` | Fallback Light/Dark bucketing when `jersey_lab_match` is off |

