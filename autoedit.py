#!/usr/bin/env python3
"""autoedit.py — Configurable auto-editing pipeline for photographs.

Runs a user-selected set of editing pipelines over one image, a directory of
images, or a list file of image paths. Pipelines are composed in a fixed,
code-defined order (never the order the user typed them) because they have
tonal interdependencies; the result of each pipeline feeds the next, producing
one output file per input image.

Usage:
    python autoedit.py photo.jpg
    python autoedit.py C:\\shoot\\keepers -pipelines "brightness level"
    python autoedit.py keepers.lst -pipelines "all openai_ig" --openaikey sk-...
    python autoedit.py photo.cr3 -o ./edited -ext png

Available pipelines (see PIPELINE_ORDER for the canonical execution order):
    level            Auto black/white point correction.
    brightness       Auto exposure correction (highlight-safe gamma).
    aisharpen        Real-ESRGAN (realesr-general-x4v3, compact) detail
                     restoration at input resolution. Optional GFPGAN
                     face-enhance pass via --face-enhance (off by default).
    openai_autoedit  Generative retouch via the OpenAI image model.
    openai_ig        Generative heroic 4:5 Instagram reframe.

"all" expands to every pipeline that does NOT start with "openai", so the
generative (billed) stages are always opt-in by name.

This script is deliberately standalone: the image-reading and extension
helpers are duplicated from algo/stages/image_analysis.py rather than imported,
so running a brightness-only edit doesn't drag in YOLO/ultralytics. Real-ESRGAN
weights are auto-downloaded to the repo root on first use, mirroring
esrsharpen.py, which this script is intended to eventually replace.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import math
import queue
import sys
import threading
import types
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# facexlib (a GFPGAN dependency, only imported when --face-enhance is used)
# still loads its ResNet50 backbone via torchvision's old pretrained=True API.
warnings.filterwarnings("ignore", message=r".*'pretrained'.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*for 'weights' are deprecated.*", category=UserWarning)

log = logging.getLogger("autoedit")

_REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Supported input formats
# ---------------------------------------------------------------------------

_RAW_EXTENSIONS: frozenset[str] = frozenset({".cr3", ".cr2"})
IMAGE_EXTENSIONS: frozenset[str] = (
    frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}) | _RAW_EXTENSIONS
)

OUTPUT_EXTENSIONS: frozenset[str] = frozenset({"jpg", "png"})

DEFAULT_OUTPUT_DIR = Path("output_img")
DEFAULT_COMPRESSION = 95
DEFAULT_WORKERS = 4

# ---------------------------------------------------------------------------
# Tonal analysis tuning
# ---------------------------------------------------------------------------

# Tonal statistics are measured on a downscaled copy: it is both much faster
# than sorting a full-resolution frame and acts as a low-pass filter so hot
# pixels and specular pinpoints can't define the black/white points.
_ANALYSIS_LONG_EDGE = 512

# Weighting runs from 1.0 at the frame centre down to this at the corners.
# Deliberately never metered off a detected face/subject: dark-skinned players
# drag a subject-metered reading down and the whole frame gets blown out.
_CENTER_WEIGHT_FLOOR = 0.25
_CENTER_WEIGHT_SIGMA = 0.40

_LEVEL_LOW_PERCENTILE = 0.5
_LEVEL_HIGH_PERCENTILE = 99.5
# Skip levelling entirely when the histogram already spans this much range.
_LEVEL_SPAN_SKIP = 0.95
# Caps how hard a low-contrast (hazy, backlit) frame gets pushed.
_LEVEL_MAX_GAIN = 1.6

_BRIGHTNESS_TARGET = 0.45
_BRIGHTNESS_MIN_GAMMA = 0.45
_BRIGHTNESS_MAX_GAMMA = 2.20

# ---------------------------------------------------------------------------
# AI sharpening
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SharpenModelSpec:
    """One selectable Real-ESRGAN checkpoint + the net architecture it needs."""

    key: str
    label: str
    model_name: str
    model_url: str
    kind: str  # "compact" (SRVGGNetCompact) or "rrdbnet" (RRDBNet)
    num_feat: int = 64
    num_conv: int = 32     # compact only
    num_block: int = 23    # rrdbnet only
    num_grow_ch: int = 32  # rrdbnet only


# Two switchable models for A/B testing (-sharpen-model), both restoring detail
# at outscale=1 (no resolution change):
# - "compact" (default): realesr-general-x4v3, SRVGGNetCompact — a fraction of
#   x4plus's size/VRAM/compute, meant for general photo restoration.
# - "x4plus": the original RealESRGAN_x4plus RRDBNet (23 residual blocks) —
#   much heavier but restores more/stronger detail; also used by esrsharpen.py.
SHARPEN_MODELS: dict[str, SharpenModelSpec] = {
    "compact": SharpenModelSpec(
        key="compact", label="realesr-general-x4v3 (compact)",
        model_name="realesr-general-x4v3.pth",
        model_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        kind="compact", num_feat=64, num_conv=32,
    ),
    "x4plus": SharpenModelSpec(
        key="x4plus", label="RealESRGAN_x4plus (RRDBNet)",
        model_name="RealESRGAN_x4plus.pth",
        model_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        kind="rrdbnet", num_feat=64, num_block=23, num_grow_ch=32,
    ),
}
DEFAULT_SHARPEN_MODEL = "x4plus"

# Peak-VRAM-per-output-pixel at tile=0/half precision, measured on a real GPU
# (Titan V) across 512-1600px square images: compact ~407-494 B/px, rrdbnet
# ~6970-7015 B/px (both stable once the fixed CUDA/model overhead amortizes).
# Padded up for safety margin/other GPUs' larger fixed overhead.
_VRAM_BYTES_PER_PIXEL: dict[str, int] = {"compact": 700, "rrdbnet": 9000}
# Only budget this fraction of currently-free VRAM for a tile, leaving
# headroom for allocator fragmentation and the next job's ramp-up.
_VRAM_SAFETY_FACTOR = 0.7
_TILE_MIN_PX = 256
_TILE_MAX_PX = 2048

# CPU has no cheap equivalent of torch.cuda.mem_get_info, so it keeps the old
# fixed-threshold behaviour: above this long edge, tile at a safe fixed size.
_AUTO_TILE_THRESHOLD_PX = 1600
_AUTO_TILE_SIZE_PX = 512

# Optional GFPGAN face-restoration pass (--face-enhance, off by default),
# same checkpoint esrsharpen.py already downloads.
_GFPGAN_MODEL_NAME = "GFPGANv1.4.pth"
_GFPGAN_MODEL_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
)


# ---------------------------------------------------------------------------
# OpenAI image model
# ---------------------------------------------------------------------------

_OPENAI_IMAGE_MODEL = "gpt-image-2"
_OPENAI_QUALITY = "high"
_OPENAI_TIMEOUT_SECONDS = 300.0

# gpt-image-2 accepts arbitrary sizes within these constraints.
_GPT_IMAGE_MAX_EDGE = 3840
_GPT_IMAGE_EDGE_MULTIPLE = 16
_GPT_IMAGE_MIN_PIXELS = 655_360
_GPT_IMAGE_MAX_PIXELS = 8_294_400
_GPT_IMAGE_MAX_ASPECT = 3.0
_GPT_IMAGE_TARGET_PIXELS = 4_000_000

# Exact 4:5, both edges multiples of 16, comfortably inside the pixel cap.
# Downsampled locally afterwards to Instagram's native feed size, which cannot
# be requested directly (1080 and 1350 are not multiples of 16).
_IG_REQUEST_SIZE = (1728, 2160)
_IG_FINAL_SIZE = (1080, 1350)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_IDENTITY_AND_KIT_RULES = """\
Hard constraints — these override every other instruction:
- Do not change the identity of any face. Every person must remain immediately
  recognisable as the same individual: preserve bone structure, facial
  proportions, the shape and spacing of eyes, nose and mouth, skin tone and
  undertone, hairline, hair texture and facial expression.
- Do not slim, reshape, age, de-age, or otherwise "beautify" any face or body
  beyond the mild blemish cleanup described above.
- Do not alter clothing, kit or equipment design in any way. Letters, numbers,
  player names, team crests, sponsor logos, manufacturer marks, stripes,
  patterns, trim, kit colours and fabric texture must be reproduced exactly as
  they appear in the source image. Do not invent, correct, straighten or
  re-letter any text or number, even if it is partially obscured or distorted.
- Do not add, remove, duplicate or reposition any person, limb, or piece of
  equipment.
- Do not change the pose or the action taking place.
- Do not add text, captions, borders, frames, watermarks or graphics.
- The result must read as an unretouched-looking photograph, not an
  illustration, painting or render."""

_AUTOEDIT_PROMPT = f"""\
Retouch this photograph to professional sports-publication quality, as a
skilled photo editor would in Lightroom and Photoshop.

You may:
- Correct exposure, white balance, contrast, saturation and colour so the
  subject reads cleanly and the skin tones look natural and accurate.
- Recover detail in blown highlights and blocked-up shadows.
- Reduce sensor noise and mild compression artefacts, and recover fine detail.
- Gently clean transient skin blemishes, stray flyaway hairs, sweat glare and
  sensor dust.
- Deepen and tidy the background so the subject separates from it, without
  changing what is actually in that background.

Preserve the original composition, framing and aspect ratio exactly.

{_IDENTITY_AND_KIT_RULES}"""

_IG_PROMPT = f"""\
Reframe this photograph into a single dramatic, heroic vertical image with a
4:5 aspect ratio, intended for an Instagram feed post.

Composition:
- Identify the main subject: the person most central to the action and most in
  focus. Build the crop around them so they dominate the frame.
- Crop in tighter than the original for impact, but keep the action readable —
  retain the ball or key object if it is part of the moment, leave natural
  headroom, and do not cut limbs at a joint.
- Place the subject using strong, intentional composition and give the frame a
  cinematic, poster-like sense of scale and energy.
- Apply the same professional colour, contrast and tonal grading you would
  apply to a published hero shot, keeping skin tones natural.

Framing constraint: build the result only from content already present in the
source photograph. Do not outpaint, extend, or invent any new background,
scenery, crowd or ground beyond the edges of the original frame.

{_IDENTITY_AND_KIT_RULES}"""


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def read_image(path: Path) -> np.ndarray | None:
    """Read *path* as a BGR uint8 array, RAW decoded at full resolution.

    Uses imdecode over a byte read rather than cv2.imread so non-ASCII paths
    work on Windows.
    """
    if path.suffix.lower() in _RAW_EXTENSIONS:
        import rawpy
        try:
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8, half_size=False)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            log.error("rawpy failed to decode %s: %s", path.name, exc)
            return None
    try:
        buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError as exc:
        log.error("Could not read %s: %s", path.name, exc)
        return None
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        log.error("Could not decode %s", path.name)
    return image


def write_image(image: np.ndarray, path: Path, ext: str, compression: int) -> bool:
    if ext == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(compression, 1, 100))]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, buf = cv2.imencode(f".{ext}", image, params)
    if not ok:
        log.error("Encoding failed for %s", path.name)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())
    return True


def collect_inputs(input_path: Path) -> list[Path]:
    """Resolve *input_path* into the list of images to process.

    A directory is enumerated non-recursively; a supported image resolves to
    itself; anything else is treated as a newline-delimited list file whose
    blank and '#'-prefixed lines are ignored.
    """
    if input_path.is_dir():
        return sorted(
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not input_path.is_file():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if input_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [input_path]

    images: list[Path] = []
    for lineno, raw_line in enumerate(
        input_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip().strip('"')
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = (input_path.parent / candidate).resolve()
        if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            log.warning("List line %d: unsupported image type, skipping: %s", lineno, line)
            continue
        if not candidate.is_file():
            log.warning("List line %d: file not found, skipping: %s", lineno, line)
            continue
        images.append(candidate)
    return images


# ---------------------------------------------------------------------------
# Tonal analysis helpers
# ---------------------------------------------------------------------------

def _analysis_view(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = _ANALYSIS_LONG_EDGE / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(
        image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
    )


def _luminance(image: np.ndarray) -> np.ndarray:
    """BT.709 luma of a BGR uint8 image, as float in [0, 1]."""
    bgr = image.astype(np.float32) / 255.0
    return bgr[:, :, 0] * 0.0722 + bgr[:, :, 1] * 0.7152 + bgr[:, :, 2] * 0.2126


def _center_weights(height: int, width: int) -> np.ndarray:
    """Gaussian falloff from 1.0 at the frame centre to _CENTER_WEIGHT_FLOOR."""
    ys = (np.arange(height, dtype=np.float32) / max(height - 1, 1)) - 0.5
    xs = (np.arange(width, dtype=np.float32) / max(width - 1, 1)) - 0.5
    dist_sq = ys[:, None] ** 2 + xs[None, :] ** 2
    falloff = np.exp(-dist_sq / (2.0 * _CENTER_WEIGHT_SIGMA ** 2))
    return _CENTER_WEIGHT_FLOOR + (1.0 - _CENTER_WEIGHT_FLOOR) * falloff


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, axis=None)
    sorted_values = values.ravel()[order]
    cumulative = np.cumsum(weights.ravel()[order])
    return float(sorted_values[int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])


def _apply_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return cv2.LUT(image, lut)


def auto_level(image: np.ndarray) -> np.ndarray:
    """Stretch the luminance histogram's black and white points to full range.

    The same scale and offset are applied to all three channels so the
    correction can't introduce a colour cast.
    """
    lum = _luminance(_analysis_view(image))
    low, high = np.percentile(lum, [_LEVEL_LOW_PERCENTILE, _LEVEL_HIGH_PERCENTILE])
    span = float(high - low)
    if span <= 1e-6:
        log.debug("level: degenerate histogram, skipped")
        return image
    if span >= _LEVEL_SPAN_SKIP:
        log.debug("level: histogram already spans %.3f, skipped", span)
        return image

    gain = min(1.0 / span, _LEVEL_MAX_GAIN)
    ramp = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip((ramp - float(low)) * gain, 0.0, 1.0) * 255.0
    log.debug("level: black=%.3f white=%.3f gain=%.3f", low, high, gain)
    return _apply_lut(image, lut.round().astype(np.uint8))


def auto_brightness(image: np.ndarray) -> np.ndarray:
    """Correct exposure toward a target midtone using a gamma curve.

    Gamma is used rather than a linear gain because it maps 0 to 0 and 1 to 1,
    so it cannot clip highlights -- which matters directly for white kit and
    bright pitches. The reading is a centre-weighted MEDIAN so a bright sky or
    a large shadow mass can't drag it.
    """
    small = _analysis_view(image)
    lum = _luminance(small)
    median = _weighted_median(lum, _center_weights(*lum.shape[:2]))
    if not 1e-3 < median < 1.0 - 1e-3:
        log.debug("brightness: median %.4f out of correctable range, skipped", median)
        return image

    gamma = float(
        np.clip(
            math.log(_BRIGHTNESS_TARGET) / math.log(median),
            _BRIGHTNESS_MIN_GAMMA,
            _BRIGHTNESS_MAX_GAMMA,
        )
    )
    if abs(gamma - 1.0) < 0.01:
        log.debug("brightness: already on target (median %.3f), skipped", median)
        return image

    ramp = np.arange(256, dtype=np.float32) / 255.0
    lut = np.power(ramp, gamma) * 255.0
    log.debug("brightness: median=%.3f gamma=%.3f", median, gamma)
    return _apply_lut(image, lut.round().astype(np.uint8))


# ---------------------------------------------------------------------------
# AI sharpening: two-tier dispatch
# ---------------------------------------------------------------------------

def _patch_basicsr_torchvision_compat() -> None:
    """basicsr imports ``torchvision.transforms.functional_tensor``, removed in
    torchvision >= 0.17 (this repo pins 0.17.2). Must run before ANY
    basicsr/realesrgan import."""
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    import torchvision.transforms.functional as _functional
    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = _functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = shim


def _ensure_sharpen_weights(spec: SharpenModelSpec) -> str:
    """Download *spec*'s weight file up front, on one thread, so parallel
    processors can't race each other into a half-written checkpoint."""
    _patch_basicsr_torchvision_compat()
    from basicsr.utils.download_util import load_file_from_url

    return load_file_from_url(
        url=spec.model_url, model_dir=str(_REPO_ROOT), progress=True,
        file_name=spec.model_name,
    )


def _ensure_gfpgan_weights() -> str:
    """Download the GFPGAN checkpoint up front (--face-enhance only)."""
    _patch_basicsr_torchvision_compat()
    from basicsr.utils.download_util import load_file_from_url

    return load_file_from_url(
        url=_GFPGAN_MODEL_URL, model_dir=str(_REPO_ROOT), progress=True,
        file_name=_GFPGAN_MODEL_NAME,
    )


def _pick_tile_size(image: np.ndarray, device: str, spec: SharpenModelSpec) -> int:
    """Return the RealESRGANer tile size to use for *image* on *device*.

    GPU: sized from CURRENTLY FREE VRAM (queried live, so it adapts to other
    processes/jobs sharing the card) rather than a fixed pixel threshold — a
    whole image is processed in one shot (tile=0, fastest, no tile seams)
    whenever it fits the VRAM budget, only falling back to the largest tile
    that does fit otherwise. CPU has no cheap equivalent of
    torch.cuda.mem_get_info, so it keeps the old fixed-threshold behaviour.
    """
    h, w = image.shape[:2]
    if not device.startswith("cuda"):
        long_edge = max(h, w)
        return _AUTO_TILE_SIZE_PX if long_edge > _AUTO_TILE_THRESHOLD_PX else 0

    import torch
    free_bytes, _total = torch.cuda.mem_get_info(device)
    budget = free_bytes * _VRAM_SAFETY_FACTOR
    bytes_per_px = _VRAM_BYTES_PER_PIXEL[spec.kind]
    if w * h * bytes_per_px <= budget:
        return 0
    tile_edge = int((budget / bytes_per_px) ** 0.5)
    tile_edge = max(_TILE_MIN_PX, min(_TILE_MAX_PX, tile_edge))
    return tile_edge - (tile_edge % 32)


class SharpenProcessor(ABC):
    """A single unit of sharpening capacity.

    Implementations own whatever resources they need exclusively, so the
    dispatcher never has to lock around a shared model. A web-service backed
    processor can be added later behind this same interface.
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Return *image* sharpened, at the same resolution."""

    @property
    @abstractmethod
    def label(self) -> str:
        ...


class LocalSharpenProcessor(SharpenProcessor):
    """Real-ESRGAN, optionally with a GFPGAN face-enhance pass, pinned to one
    local device.

    Models are built lazily and are never shared with another processor, which
    is what makes inference thread-safe without locking.
    """

    # Construction is serialised across processors so two processors building
    # at once can't race into a half-written checkpoint read.
    _build_lock = threading.Lock()

    def __init__(
        self, device: str, realesrgan_path: str, spec: SharpenModelSpec,
        gfpgan_path: str | None = None,
    ) -> None:
        self._device = device
        self._realesrgan_path = realesrgan_path
        self._spec = spec
        self._gfpgan_path = gfpgan_path
        self._upsampler = None
        self._face_enhancer = None

    @property
    def label(self) -> str:
        return self._device

    def _build(self):
        import torch
        from realesrgan import RealESRGANer

        device = torch.device(self._device)
        spec = self._spec
        if spec.kind == "compact":
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            net = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=spec.num_feat,
                num_conv=spec.num_conv, upscale=4, act_type="prelu",
            )
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            net = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=spec.num_feat,
                num_block=spec.num_block, num_grow_ch=spec.num_grow_ch, scale=4,
            )
        upsampler = RealESRGANer(
            scale=4,
            model_path=self._realesrgan_path,
            model=net,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=self._device.startswith("cuda"),
            device=device,
        )
        if self._gfpgan_path is None:
            return upsampler, None

        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path=self._gfpgan_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
            device=device,
        )
        return upsampler, face_enhancer

    def process(self, image: np.ndarray) -> np.ndarray:
        if self._upsampler is None:
            with LocalSharpenProcessor._build_lock:
                if self._upsampler is None:
                    log.info("[%s] loading Real-ESRGAN (%s)%s …",
                             self._device, self._spec.label,
                             " + GFPGAN face-enhance" if self._gfpgan_path else "")
                    self._upsampler, self._face_enhancer = self._build()

        # RealESRGANer's tiling knob is `tile_size`, not `tile` (its constructor
        # arg is named `tile` but stores it as `self.tile_size`) -- setting the
        # wrong attribute here would silently no-op and always process the
        # whole image untiled, regardless of size.
        self._upsampler.tile_size = _pick_tile_size(image, self._device, self._spec)
        if self._face_enhancer is not None:
            _, _, output = self._face_enhancer.enhance(
                image, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            output, _ = self._upsampler.enhance(image, outscale=1)
        if output.shape[:2] != image.shape[:2]:
            # outscale=1 should already guarantee this; resize defensively so a
            # library-version quirk can never silently change dimensions.
            output = cv2.resize(
                output, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4
            )
        return output


@dataclass
class _SharpenJob:
    image: np.ndarray
    done: threading.Event = field(default_factory=threading.Event)
    result: np.ndarray | None = None
    error: BaseException | None = None


class SharpenDispatcher:
    """Blocking work queue in front of a pool of :class:`SharpenProcessor`.

    File workers submit and block; processors run independently. One CPU
    processor exists as spillover capacity and only takes a job when every GPU
    processor is already busy -- CPU inference is roughly 30-40x slower, so an
    ungated CPU worker would grab jobs and straggle at the end of a batch. With
    no GPU present the "all GPUs busy" test is vacuously true, so the same rule
    makes the CPU processor the sole worker.
    """

    _CPU_POLL_SECONDS = 0.05

    def __init__(
        self, model_spec: SharpenModelSpec | None = None, face_enhance: bool = False,
    ) -> None:
        self._model_spec = model_spec or SHARPEN_MODELS[DEFAULT_SHARPEN_MODEL]
        self._face_enhance = face_enhance
        self._queue: queue.Queue[_SharpenJob | None] = queue.Queue()
        self._lock = threading.Lock()
        self._gpu_busy = 0
        self._gpu_count = 0
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def start(self) -> None:
        log.info("Sharpen model: %s%s", self._model_spec.label,
                  " + GFPGAN face-enhance" if self._face_enhance else "")
        realesrgan_path = _ensure_sharpen_weights(self._model_spec)
        gfpgan_path = _ensure_gfpgan_weights() if self._face_enhance else None

        import torch
        self._gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        for index in range(self._gpu_count):
            device = f"cuda:{index}"
            log.info("Sharpen processor: %s (%s)", device, torch.cuda.get_device_name(index))
            self._spawn(
                LocalSharpenProcessor(device, realesrgan_path, self._model_spec, gfpgan_path),
                self._gpu_loop,
                f"sharpen-{device}",
            )

        log.info(
            "Sharpen processor: cpu (%s)",
            "sole worker" if self._gpu_count == 0 else "spillover only",
        )
        self._spawn(
            LocalSharpenProcessor("cpu", realesrgan_path, self._model_spec, gfpgan_path),
            self._cpu_loop,
            "sharpen-cpu",
        )

    def _spawn(self, processor: SharpenProcessor, loop, name: str) -> None:
        thread = threading.Thread(target=loop, args=(processor,), name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def submit(self, image: np.ndarray) -> np.ndarray:
        job = _SharpenJob(image=image)
        self._queue.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        assert job.result is not None
        return job.result

    def shutdown(self) -> None:
        self._stop.set()
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=10.0)

    @staticmethod
    def _run(processor: SharpenProcessor, job: _SharpenJob) -> None:
        try:
            job.result = processor.process(job.image)
        except BaseException as exc:  # surfaced to the waiting file worker
            job.error = exc
        finally:
            job.done.set()

    def _gpu_loop(self, processor: SharpenProcessor) -> None:
        while not self._stop.is_set():
            job = self._queue.get()
            if job is None:
                return
            with self._lock:
                self._gpu_busy += 1
            try:
                self._run(processor, job)
            finally:
                with self._lock:
                    self._gpu_busy -= 1

    def _cpu_loop(self, processor: SharpenProcessor) -> None:
        while not self._stop.is_set():
            with self._lock:
                all_gpus_busy = self._gpu_busy >= self._gpu_count
            if not all_gpus_busy:
                self._stop.wait(self._CPU_POLL_SECONDS)
                continue
            try:
                job = self._queue.get(timeout=self._CPU_POLL_SECONDS)
            except queue.Empty:
                continue
            if job is None:
                return
            self._run(processor, job)


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

def _snap_to_multiple(value: float) -> int:
    return max(
        _GPT_IMAGE_EDGE_MULTIPLE,
        int(round(value / _GPT_IMAGE_EDGE_MULTIPLE)) * _GPT_IMAGE_EDGE_MULTIPLE,
    )


def gpt_image_size(width: int, height: int) -> tuple[int, int]:
    """Nearest size to *width* x *height* that gpt-image-2 will accept.

    Both edges must be multiples of 16, neither may exceed 3840, the long:short
    ratio may not exceed 3:1, and the total pixel count must fall between
    655,360 and 8,294,400.
    """
    aspect = width / height
    clamped = min(max(aspect, 1.0 / _GPT_IMAGE_MAX_ASPECT), _GPT_IMAGE_MAX_ASPECT)
    if abs(clamped - aspect) > 1e-6:
        log.warning(
            "Aspect ratio %.2f exceeds the model's 3:1 limit; requesting %.2f instead",
            aspect, clamped,
        )
    aspect = clamped

    pixels = float(np.clip(width * height, _GPT_IMAGE_MIN_PIXELS, _GPT_IMAGE_TARGET_PIXELS))
    for _ in range(8):
        edge_h = math.sqrt(pixels / aspect)
        edge_w = aspect * edge_h
        overshoot = max(edge_w, edge_h) / _GPT_IMAGE_MAX_EDGE
        if overshoot > 1.0:
            edge_w /= overshoot
            edge_h /= overshoot
        out_w, out_h = _snap_to_multiple(edge_w), _snap_to_multiple(edge_h)
        total = out_w * out_h
        if total < _GPT_IMAGE_MIN_PIXELS:
            pixels *= _GPT_IMAGE_MIN_PIXELS / total * 1.02
            continue
        if total > _GPT_IMAGE_MAX_PIXELS:
            pixels *= _GPT_IMAGE_MAX_PIXELS / total * 0.98
            continue
        return out_w, out_h
    return 1024, 1024


def openai_edit(
    client,
    image: np.ndarray,
    prompt: str,
    size: tuple[int, int],
) -> np.ndarray:
    """Send *image* to the OpenAI image model and return the edited result."""
    request_w, request_h = size
    # Resized before encoding so the upload stays well inside the API's size
    # limit and matches the resolution actually being asked for.
    payload = cv2.resize(image, (request_w, request_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", payload)
    if not ok:
        raise RuntimeError("Could not PNG-encode the image for upload")

    response = client.images.edit(
        model=_OPENAI_IMAGE_MODEL,
        image=("input.png", io.BytesIO(buf.tobytes()), "image/png"),
        prompt=prompt,
        size=f"{request_w}x{request_h}",
        quality=_OPENAI_QUALITY,
        output_format="png",
    )
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(response.data[0].b64_json), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if decoded is None:
        raise RuntimeError("Could not decode the image returned by the API")
    return decoded


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

@dataclass
class EditContext:
    """Services shared by every stage across every worker thread."""

    source: Path
    dispatcher: SharpenDispatcher | None = None
    openai_client: object | None = None


class EditPipeline(ABC):
    name: str = ""

    @abstractmethod
    def apply(self, image: np.ndarray, ctx: EditContext) -> np.ndarray:
        ...


class LevelPipeline(EditPipeline):
    name = "level"

    def apply(self, image: np.ndarray, ctx: EditContext) -> np.ndarray:
        return auto_level(image)


class BrightnessPipeline(EditPipeline):
    name = "brightness"

    def apply(self, image: np.ndarray, ctx: EditContext) -> np.ndarray:
        return auto_brightness(image)


class AiSharpenPipeline(EditPipeline):
    name = "aisharpen"

    def apply(self, image: np.ndarray, ctx: EditContext) -> np.ndarray:
        assert ctx.dispatcher is not None
        return ctx.dispatcher.submit(image)


class OpenAiAutoEditPipeline(EditPipeline):
    name = "openai_autoedit"

    def apply(self, image: np.ndarray, ctx: EditContext) -> np.ndarray:
        height, width = image.shape[:2]
        size = gpt_image_size(width, height)
        log.info("[%s] openai_autoedit at %dx%d", ctx.source.name, *size)
        return openai_edit(ctx.openai_client, image, _AUTOEDIT_PROMPT, size)


class OpenAiInstagramPipeline(EditPipeline):
    name = "openai_ig"

    def apply(self, image: np.ndarray, ctx: EditContext) -> np.ndarray:
        log.info("[%s] openai_ig at %dx%d", ctx.source.name, *_IG_REQUEST_SIZE)
        result = openai_edit(ctx.openai_client, image, _IG_PROMPT, _IG_REQUEST_SIZE)
        return cv2.resize(result, _IG_FINAL_SIZE, interpolation=cv2.INTER_LANCZOS4)


# Canonical execution order, independent of the order the user lists them.
# Levels sets the black/white endpoints first and brightness then corrects the
# midtone against an already-normalised histogram (Photoshop Levels semantics);
# sharpening follows all tonal work; the generative stages run last so they see
# the best possible local edit.
PIPELINE_ORDER: tuple[type[EditPipeline], ...] = (
    LevelPipeline,
    BrightnessPipeline,
    AiSharpenPipeline,
    OpenAiAutoEditPipeline,
    OpenAiInstagramPipeline,
)

PIPELINES_BY_NAME: dict[str, type[EditPipeline]] = {cls.name: cls for cls in PIPELINE_ORDER}


def resolve_pipelines(tokens: list[str]) -> list[EditPipeline]:
    """Expand and de-duplicate user pipeline names into ordered instances."""
    requested: set[str] = set()
    for token in tokens:
        for name in token.split():
            key = name.strip().lower()
            if not key:
                continue
            if key == "all":
                requested.update(
                    n for n in PIPELINES_BY_NAME if not n.startswith("openai")
                )
            elif key in PIPELINES_BY_NAME:
                requested.add(key)
            else:
                log.warning("Unknown pipeline '%s', ignoring.", name)
    return [cls() for cls in PIPELINE_ORDER if cls.name in requested]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def output_path_for(source: Path, output_dir: Path, ext: str) -> Path:
    """Timestamped auto-name for *source*'s output, disambiguated with a
    '_n' suffix (first available n, starting at 1) if that exact path is
    already taken — e.g. two sources sharing a stem processed in the same
    second. An explicit --out-file path bypasses this entirely and always
    overwrites (handled by the caller/write_image)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{source.stem}_processed_{stamp}"
    candidate = output_dir / f"{base}.{ext}"
    n = 1
    while candidate.exists():
        candidate = output_dir / f"{base}_{n}.{ext}"
        n += 1
    return candidate


def process_one(
    source: Path,
    pipelines: list[EditPipeline],
    output_dir: Path,
    ext: str,
    compression: int,
    dispatcher: SharpenDispatcher | None,
    openai_client,
    out_file: Path | None = None,
) -> Path | None:
    image = read_image(source)
    if image is None:
        return None
    log.info("[%s] %dx%d", source.name, image.shape[1], image.shape[0])

    ctx = EditContext(source=source, dispatcher=dispatcher, openai_client=openai_client)
    for pipeline in pipelines:
        try:
            image = pipeline.apply(image, ctx)
        except Exception as exc:
            log.error("[%s] pipeline '%s' failed: %s", source.name, pipeline.name, exc)
            return None

    # out_file may equal source (re-editing an already-edited image in
    # place) — safe since read_image() fully decodes into memory up front,
    # nothing streams from disk during pipeline processing.
    destination = out_file if out_file is not None else output_path_for(source, output_dir, ext)
    if not write_image(image, destination, ext, compression):
        return None
    log.info("[%s] wrote %s (%dx%d)", source.name, destination, image.shape[1], image.shape[0])
    return destination


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputPath", type=Path,
        help="An image, a directory of images (non-recursive), or a newline-delimited list file.",
    )
    parser.add_argument(
        "-pipelines", "--pipelines", nargs="*", default=["all"], metavar="NAME",
        help=(
            "Space-separated pipelines to run: "
            + ", ".join(PIPELINES_BY_NAME)
            + ". 'all' (default) selects every non-openai pipeline. Unknown names are "
              "warned about and ignored; order is decided by this script, not by you."
        ),
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT_DIR, metavar="DIR",
        help=f"Output directory (default: ./{DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "-ext", "--ext", default="jpg", choices=sorted(OUTPUT_EXTENSIONS),
        help="Output file format (default: jpg).",
    )
    parser.add_argument(
        "-compression", "--compression", type=int, default=DEFAULT_COMPRESSION, metavar="1-100",
        help=f"JPEG quality when -ext is jpg (default: {DEFAULT_COMPRESSION}). Ignored for png.",
    )
    parser.add_argument(
        "--openaikey", default=None, metavar="KEY",
        help="OpenAI API key. Without it, the openai_* pipelines are skipped.",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, metavar="N",
        help=(
            f"Images to process concurrently (default: {DEFAULT_WORKERS}). Each image still "
            "runs its pipelines strictly in sequence."
        ),
    )
    parser.add_argument(
        "--out-file", type=Path, default=None, metavar="PATH",
        help=(
            "Exact output file path, overriding -o/-ext's timestamped naming. Only valid "
            "with a single input image. May equal inputPath to overwrite it in place "
            "(re-editing an already-edited image). If this path already exists it is "
            "overwritten; auto-named output (no --out-file) instead gets a '_n' suffix."
        ),
    )
    parser.add_argument(
        "--sharpen-model", choices=sorted(SHARPEN_MODELS), default=DEFAULT_SHARPEN_MODEL,
        metavar="NAME",
        help=(
            "Real-ESRGAN checkpoint used by the aisharpen pipeline, to A/B test detail "
            f"restoration strength (default: {DEFAULT_SHARPEN_MODEL}). Choices: "
            + ", ".join(f"{k} ({v.label})" for k, v in SHARPEN_MODELS.items())
        ),
    )
    parser.add_argument(
        "--face-enhance", action="store_true",
        help="Run a GFPGAN face-restoration pass after Real-ESRGAN in the aisharpen "
             "pipeline (off by default).",
    )
    args = parser.parse_args()

    try:
        sources = collect_inputs(args.inputPath)
    except (FileNotFoundError, OSError, UnicodeDecodeError) as exc:
        log.error("%s", exc)
        sys.exit(1)
    if not sources:
        log.error("No supported images found at: %s", args.inputPath)
        sys.exit(1)
    if args.out_file is not None and len(sources) != 1:
        log.error("--out-file requires exactly one input image (got %d).", len(sources))
        sys.exit(1)

    pipelines = resolve_pipelines(args.pipelines)
    if not pipelines:
        log.error("No valid pipelines selected.")
        sys.exit(1)

    openai_client = None
    if any(p.name.startswith("openai") for p in pipelines):
        if args.openaikey:
            import openai
            openai_client = openai.OpenAI(
                api_key=args.openaikey, timeout=_OPENAI_TIMEOUT_SECONDS
            )
        else:
            log.warning("No --openaikey supplied; skipping the openai_* pipelines.")
            pipelines = [p for p in pipelines if not p.name.startswith("openai")]
            if not pipelines:
                log.error("Nothing left to run.")
                sys.exit(1)

    dispatcher = None
    if any(p.name == "aisharpen" for p in pipelines):
        dispatcher = SharpenDispatcher(SHARPEN_MODELS[args.sharpen_model], args.face_enhance)
        dispatcher.start()

    log.info(
        "Processing %d image(s) through: %s",
        len(sources), " -> ".join(p.name for p in pipelines),
    )

    succeeded = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            results = pool.map(
                lambda source: process_one(
                    source, pipelines, args.output, args.ext, args.compression,
                    dispatcher, openai_client, out_file=args.out_file,
                ),
                sources,
            )
            succeeded = sum(1 for result in results if result is not None)
    finally:
        if dispatcher is not None:
            dispatcher.shutdown()

    failed = len(sources) - succeeded
    log.info("Done. %d succeeded, %d failed. Output: %s", succeeded, failed, args.output)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
