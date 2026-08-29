#!/usr/bin/env python3
"""esrsharpen.py — Sharpen a single image using Real-ESRGAN (+ optional GFPGAN
face restoration), writing the result back out at the SAME resolution as the
input.

Real-ESRGAN's native network is a fixed 4x super-resolution model; this
script asks RealESRGANer to enhance at ``outscale=1`` so its own built-in
Lanczos resize step brings the 4x output back down to the original
dimensions regardless of how large or small the input is. The net effect is
a denoise + detail-restoration pass at the input's own resolution rather
than an upscale.

Usage:
    python esrsharpen.py input.jpg -o output.jpg
    python esrsharpen.py input.jpg -o output.jpg --disable-face-enhance
    python esrsharpen.py input.cr3 -o output.jpg --tile 400 --cpu-only
    python esrsharpen.py input.jpg   # writes to ./output/input.jpg

Output is always written as a JPEG (regardless of the extension given in
-o/--output) — this mirrors the rest of this repo's photo tooling, which
works with JPEG originals almost exclusively. -o/--output is optional;
when omitted, the result is written to ./output/<input file name>.

Model weights (RealESRGAN_x4plus.pth, GFPGANv1.4.pth) are auto-downloaded to
the repo root on first use, mirroring 1_prep_review.py's yolov8n-face.pt /
face_landmarker.task bootstrap pattern — both are listed in .gitignore.
"""

from __future__ import annotations

import argparse
import logging
import sys
import types
import warnings
from pathlib import Path

import cv2
import numpy as np

# facexlib (a GFPGAN dependency) still loads its ResNet50 backbone via
# torchvision's old pretrained=True API, which just warns rather than
# breaking on the torchvision version this repo pins -- harmless noise on
# every run, so silence it here rather than at every call site.
warnings.filterwarnings("ignore", message=r".*'pretrained'.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*for 'weights' are deprecated.*", category=UserWarning)

log = logging.getLogger("esrsharpen")

_MODEL_DIR = Path(__file__).resolve().parent

_REALESRGAN_MODEL_NAME = "RealESRGAN_x4plus.pth"
_REALESRGAN_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)
_GFPGAN_MODEL_NAME = "GFPGANv1.4.pth"
_GFPGAN_MODEL_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
)

_RAW_EXTENSIONS = frozenset({".cr3", ".cr2"})

# Images with a long edge above this get tiled (see --tile) so the 4x
# super-resolution pass doesn't try to allocate one enormous intermediate
# tensor and OOM a modest GPU. Below it, the whole image is processed in
# one shot (faster, and quality is very slightly better with no tile seams).
_AUTO_TILE_THRESHOLD_PX = 1600
_AUTO_TILE_SIZE_PX = 512


def _patch_basicsr_torchvision_compat() -> None:
    """basicsr (a realesrgan/gfpgan dependency) imports
    ``torchvision.transforms.functional_tensor``, which was removed in
    torchvision >= 0.17 (its contents moved into
    ``torchvision.transforms.functional``). This repo pins torchvision
    0.17.2, so importing basicsr fails outright without this shim. Must run
    before ANY basicsr/realesrgan/gfpgan import. No-op on torchvision
    versions that still have the module."""
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    import torchvision.transforms.functional as _functional
    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = _functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = shim


_patch_basicsr_torchvision_compat()

from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: E402
from basicsr.utils.download_util import load_file_from_url  # noqa: E402
from realesrgan import RealESRGANer  # noqa: E402


def _read_image(path: Path) -> np.ndarray | None:
    """Read *path* as a BGR uint8 array. RAW formats decoded via rawpy at
    half-size 8-bit (matches algo/stages/image_analysis.py's _read_image;
    duplicated here rather than imported so this standalone script doesn't
    also pull in ultralytics/YOLO as a hard dependency)."""
    if path.suffix.lower() in _RAW_EXTENSIONS:
        import rawpy
        try:
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8, half_size=True)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            log.error("rawpy failed to decode %s: %s", path.name, exc)
            return None
    return cv2.imread(str(path))


def _ensure_model(name: str, url: str) -> str:
    log.info("Ensuring model weights: %s", name)
    return load_file_from_url(url=url, model_dir=str(_MODEL_DIR), progress=True, file_name=name)


def _pick_tile(image: np.ndarray, requested_tile: int) -> int:
    if requested_tile > 0:
        return requested_tile
    if requested_tile < 0:
        return 0  # explicit opt-out
    long_edge = max(image.shape[0], image.shape[1])
    return _AUTO_TILE_SIZE_PX if long_edge > _AUTO_TILE_THRESHOLD_PX else 0


def sharpen_image(
    image: np.ndarray,
    *,
    tile: int = 0,
    cpu_only: bool = False,
    face_enhance: bool = True,
) -> np.ndarray:
    """Run Real-ESRGAN (+ optional GFPGAN) on *image*, returning a BGR uint8
    array at the SAME resolution as the input."""
    import torch

    if cpu_only:
        # RealESRGANer/GFPGANer pick their device from torch.cuda.is_available()
        # internally with no plain "force CPU" constructor arg in every
        # released version, so this is the reliable way to force it.
        torch.cuda.is_available = lambda: False  # type: ignore[method-assign]

    use_half = torch.cuda.is_available()
    model_path = _ensure_model(_REALESRGAN_MODEL_NAME, _REALESRGAN_MODEL_URL)
    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=net,
        tile=_pick_tile(image, tile),
        tile_pad=10,
        pre_pad=0,
        half=use_half,
    )

    if not face_enhance:
        output, _ = upsampler.enhance(image, outscale=1)
        return output

    from gfpgan import GFPGANer

    gfpgan_model_path = _ensure_model(_GFPGAN_MODEL_NAME, _GFPGAN_MODEL_URL)
    face_enhancer = GFPGANer(
        model_path=gfpgan_model_path,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler,
    )
    _, _, output = face_enhancer.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
    return output


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Path to the input image.")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Path to write the sharpened JPEG to. Defaults to ./output/<input file name>.",
    )
    parser.add_argument(
        "--disable-face-enhance", action="store_true",
        help="Skip the GFPGAN face-restoration pass (on by default) and only run Real-ESRGAN's general enhancement.",
    )
    parser.add_argument(
        "--tile", type=int, default=0, metavar="N",
        help=(
            "Tile size in pixels for the super-resolution pass, to bound GPU memory use. "
            "0 (default) = auto (tiles large images, processes small ones whole); "
            "negative = force no tiling."
        ),
    )
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU inference even if a GPU is available.")
    parser.add_argument("--quality", type=int, default=95, metavar="1-100", help="Output JPEG quality (default: 95).")
    args = parser.parse_args()

    if not args.input.is_file():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    output_path = args.output if args.output is not None else Path("output") / args.input.name

    image = _read_image(args.input)
    if image is None:
        log.error("Could not read input image: %s", args.input)
        sys.exit(1)
    log.info("Input: %s (%dx%d)", args.input.name, image.shape[1], image.shape[0])

    output = sharpen_image(
        image,
        tile=args.tile,
        cpu_only=args.cpu_only,
        face_enhance=not args.disable_face_enhance,
    )
    if output.shape[:2] != image.shape[:2]:
        # outscale=1 should already guarantee this; resize defensively so a
        # library-version quirk can never silently change output dimensions.
        output = cv2.resize(output, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
    if not ok:
        log.error("JPEG encoding failed.")
        sys.exit(1)
    output_path.write_bytes(buf.tobytes())
    log.info("Wrote: %s (%dx%d)", output_path, output.shape[1], output.shape[0])


if __name__ == "__main__":
    main()
