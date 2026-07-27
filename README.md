# BlurPictureDetector

Detects blurry sport photos for photographers who unethically abused the burst mode.

Runs two YOLO models (body pose + face detection) on a folder of images, filters subjects down to your team by jersey colour, scores the sharpness of each person's face, and produces annotated previews so you can review the results before any files are moved.

---

## How it works

```
Image → normalize size (long edge = 1800 px)
      → Pose model      → body bounding boxes + 17 COCO keypoints
      → Face model       → face bounding boxes + 5 landmarks
      → match faces to bodies (head-keypoint overlap)
      → grade sharpness of each person's face crop
      → poll jersey colours → keep only your team (+ forced colours)
      → annotate preview image
      → write album.json / info.json / blurry.csv / blur.lst
      → (optional) cluster faces into .FaceReco/
```

Detection runs on a pluggable **engine** (`--engine mediapipe` default, or `--engine yolo`):

| Engine | Pose | Face | License |
|---|---|---|---|
| `mediapipe` *(default)* | BlazePose (Pose Landmarker) | BlazeFace + FaceMesh-V2 (Face Landmarker) | Apache-2.0 (permissive) |
| `yolo` | YOLOv8n-pose | YOLOv8n-face | AGPL-3.0 / GPL-3.0 (copyleft — see licensing note below) |

The pipeline is a sequence of stages (`algo/stages/`):

| Stage | Responsibility |
|---|---|
| `ImageAnalysisStage` | Run both YOLO models, match faces to bodies |
| `GradingStage` | Compute the sharpness score and pass/fail per body |
| `JerseyCountingStage` | Detect the dominant team colour and disqualify off-team bodies |
| `AnnotationStage` | Draw and save annotated previews into `previews/` |
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
| `--engine mediapipe\|yolo` | `mediapipe` | Detection/pose/face-landmark engine. `mediapipe` is Apache-2.0 licensed (default); `yolo` is the legacy engine, kept for comparison/rollback (AGPL-3.0/GPL-3.0 — see Licensing below) |
| `--jerseycolor "COLOR[;COLOR...]"` | `blue;white;+purple;+orange;+light blue;+pink` | Team colours; `+` = forced-include; empty string disables filtering |
| `--noteam` | off | Disable jersey filtering (score every detected person) |
| `--output <dir>` | `albums/<timestamp>-<input>/` | Root output directory |
| `--skip-facereco` | off | Don't run face-recognition clustering |
| `--face-db <dir>` | none | Match clusters against an existing face DB |
| `--face-db-match-threshold <n>` | `0.72` | Cosine similarity required between a face and its closest-matching person prototype |
| `--face-db-match-margin <n>` | `0.05` | Minimum similarity gap required over the best-matching different person before accepting a match |
| `--face-db-prototype-threshold <n>` | `0.62` | Cosine similarity used to split each DB person's own photos into visual sub-clusters |
| `--min-face-crop-px <n>` | `32` | Minimum short-edge crop size (pixels) trusted for matching/clustering |
| `--align-faces` | off | Similarity-align faces to a 5-point template before embedding |
| `--debug-align` | off | Write alignment QA images to `.FaceReco/.debug` |

Scores every image and writes an output folder:

```
<output_dir>/
    previews/         ← annotated previews for every image (blur/sharp/skipped alike)
    album.json      ← full per-body data (used by face recognition); each result also carries a "preview_path" pointing into previews/
    info.json         ← classification results (used by step 2)
    blurry.csv        ← one row per blurry image with score details
    blur.lst          ← plain list of blurry file paths
    run.log           ← full debug log
    .FaceReco/        ← face recognition clusters (unless --skip-facereco)
```

**Face Recognition (optional):**

By default, after annotation previews are generated, face recognition automatically clusters qualifying faces using the Facenet provider. The output is stored in `.FaceReco/` within the same output directory. Disable the auto-run with `--skip-facereco` if needed.

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

Open the `previews/` folder in your photo viewer and **delete any preview images you want to override**:

| Preview is for… | Delete a preview when… | Effect on the original |
|---|---|---|
| a blurry image | The photo isn't actually blurry — you want to keep it | Left in place |
| a sharp image | The photo is sharp but you don't want it | Moved to `Blur/` |
| a skipped image | You want to leave the original untouched | Left in place |

Leave previews you agree with untouched.

Then use the web app (`culling_app.py`) to review, tag faces, and export the kept photos — see below.

---

## Face recognition

Face recognition runs automatically at the end of Step 1 (unless `--skip-facereco`).

| Option | Default | Description |
|---|---|---|
| `--cluster-threshold <n>` | `0.72` | Cosine similarity for clustering (higher = tighter clusters) |
| `--face-buffer-ratio <n>` | `0.15` | Extra crop padding around the face box (per side) |
| `--face-db <dir>` | none | Match faces against an existing face DB |
| `--face-db-match-threshold <n>` | `0.72` | Cosine similarity required between a face and its closest-matching person prototype |
| `--face-db-match-margin <n>` | `0.05` | Minimum similarity gap required over the best-matching different person before accepting a match |
| `--face-db-prototype-threshold <n>` | `0.62` | Cosine similarity used to split each DB person's own photos into visual sub-clusters ("prototypes") |
| `--min-face-crop-px <n>` | `32` | Minimum short-edge crop size (pixels) trusted for matching/clustering; smaller crops are skipped |
| `--align-faces` | off | Similarity-align faces before embedding |
| `--debug-align` | off | Write alignment QA images to `.FaceReco/.debug` |
| `--open-viewer` | off | Open the generated `.FaceReco/` folder afterwards |

**What it does:**

1. Loads per-body data from `album.json`.
2. Extracts face embeddings from qualifying bodies (via the selected provider), skipping crops smaller than `--min-face-crop-px`.
3. Optionally matches faces against a face DB and names matched clusters (see **How face-DB matching works** below).
4. Clusters the remaining (unmatched) faces by cosine similarity (agglomerative, average linkage).
5. Writes face crops and metadata to `<prep_output_dir>/.FaceReco/`.

### How face-DB matching works

A real person's reference photos rarely form one tight blob in embedding space - glasses on/off, lighting, angle, and expression all shift the embedding. Loading the face DB therefore splits **each person's own positive embeddings** into visually-cohesive **prototypes** (sub-clusters, via the same cosine agglomerative clustering used elsewhere) instead of treating a person as one blended average. A query face is scored against every person's single *closest* prototype.

A face is only assigned to a person when **all** of the following hold:

1. **Absolute floor** - the best prototype similarity meets `--face-db-match-threshold`.
2. **Margin over the runner-up** - the best-matching person must beat the best-matching *different* person by at least `--face-db-match-margin`. Two near-tied candidates is exactly the failure mode that mixes up similar-looking people; when the margin isn't met the face is left unmatched (and falls into ordinary clustering) rather than guessed.
3. **Not vetoed** - the face isn't at least as similar to that person's curated `Negative/` examples as it is to the match.

Each matched face's `matchScore`, `matchMargin`, and `matchRunnerUp` are written into `face.json` so a match can be audited after the fact.

Don't guess `--face-db-match-threshold` / `--face-db-match-margin` - calibrate them from your own face DB with `RebuildFaceDB.py` (below), which calibrates automatically every time it rebuilds.

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

Each `face.json` stores editable person metadata (`name`, `playernum`) and a `faces` list with the original filename, original body JSON payload, crop file name, confidence, match diagnostics (`matchScore`/`matchMargin`/`matchRunnerUp`, set only for face-DB matches), an `aligned` flag, and a base64-encoded float32 embedding.

> **Note:** a *rebuilt* face DB (see below) uses a different, flatter `faces` schema — just a list of `{dtype, shape, encoding, value}` embeddings with no wrapper. That's the schema `--face-db` / `FaceDb.load` actually reads; the richer per-face metadata above is only present in the unlabeled `.FaceReco/` output of a fresh run, before `RebuildFaceDB.py` strips it down.

### Provider modules (`algo/`)

- `algo/facereco_provider.py` — abstract provider interface
- `algo/dlib_provider.py` — dlib implementation
- `algo/facenet_provider.py` — FaceNet (facenet-pytorch / VGGFace2) implementation
- `algo/face_crop_embed.py` — shared crop → detect → (optional align) → embed pipeline used by both prediction and DB rebuild

### Rebuilding (and calibrating) a face DB

```
python RebuildFaceDB.py <facedb_dir> [--align-faces] [--prototype-threshold 0.62] [--skip-calibration]
```

One command does both steps:

1. **Rebuild** — recomputes every embedding in a `.FaceReco/` directory from its saved crop images. Use `--align-faces` only if the database was originally built with alignment — the rebuild and prediction must use the same flag so embeddings live in the same space.
2. **Calibrate** (runs automatically afterwards; pass `--skip-calibration` to skip it) — runs leave-one-out cross-validation over the just-rebuilt DB: every photo is temporarily held out, scored against its own person's remaining photos (genuine similarity) and against every other person (impostor similarity) — exactly the comparison a real query faces. It reports:
   - A recommended `--face-db-match-threshold` (nearest the equal-error-rate point) and `--face-db-match-margin`.
   - **Confusions** — specific photos where a *different* person scored as high or higher than the true person. This is the most actionable signal for "faces are getting mixed up": it names the exact photo and the exact other person it's confused with, which is almost always either a genuine look-alike pair or a mislabeled crop sitting in the wrong person's `Face/` folder.
   - Pairwise prototype overlap — which people are closest to each other in embedding space overall.

Run it whenever the face DB changes meaningfully (new people, new photos) rather than reusing the default thresholds blindly — it only takes one command.

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

Models are downloaded automatically on first run:

- **mediapipe engine (default):** `pose_landmarker_full.task` and `face_landmarker.task`, downloaded from Google's
  [MediaPipe model zoo](https://storage.googleapis.com/mediapipe-models/) (Apache-2.0 licensed).
- **yolo engine (`--engine yolo`):** `yolov8n-pose.pt` via Ultralytics, and `yolov8n-face.pt` from the
  [akanametov/yolo-face](https://github.com/akanametov/yolo-face) release.

### Licensing

**Permissive, no copyleft obligations** — everything used by the default `mediapipe` engine and default face
recognition path:

| Component | Used for | License |
|---|---|---|
| mediapipe (BlazePose / BlazeFace / FaceMesh-V2) | default pose/face engine | Apache-2.0 |
| torch / torchvision | hybrid engine's person-box detector (`fasterrcnn_resnet50_fpn_v2`) | BSD-3-Clause |
| facenet-pytorch | face embedding (default provider) | MIT |
| dlib | optional face embedding provider | Boost Software License 1.0 |
| face-recognition | wraps dlib's face models | MIT |
| numpy, scikit-learn | array ops, clustering | BSD-3-Clause |
| opencv-python | image I/O and processing | MIT (wrapper) + Apache-2.0 (OpenCV itself) |

**Weak copyleft (LGPL), used only as dynamically-linked binaries** — no source-disclosure obligation for this
project's own code, but keep in mind if you redistribute the binaries themselves:

- `rawpy` (MIT) wraps **LibRaw**, dual-licensed LGPL-2.1 or CDDL (or a paid commercial license). The PyPI wheel
  links it dynamically, which is the standard LGPL-compatible way to use it from a closed-source app.
- `opencv-python`'s prebuilt wheels bundle **FFmpeg** (LGPLv2.1); non-headless Linux wheels additionally bundle
  **Qt5** (LGPLv3). Not applicable to Windows/macOS wheels or `opencv-python-headless`.

**Copyleft — opt-in only, not the default:** the `--engine yolo` path uses Ultralytics YOLO models, which are
**AGPL-3.0** (or a paid Enterprise license) for the pose model, and **GPL-3.0** for the `akanametov/yolo-face`
model. These carry source-disclosure obligations for derivative/networked works — review them before enabling
`--engine yolo` in any distributed or hosted use of this tool.

All model weight files (`*.pt`, `*.task`) are `.gitignore`d and downloaded at runtime — none are committed to
or redistributed with this repository.

### Known limitation (mediapipe engine)

MediaPipe's Face Landmarker does not expose a reliable per-landmark occlusion/visibility score the way the YOLO
face model does, so `face_coverage_min_visible` (the check that disqualifies a face with too few visible
landmarks) is effectively a pass-through under `--engine mediapipe`.

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

