"""Experimental: render each detected photo "burst" in an album into an MP4.

A burst is any run of photos whose capture timestamps are each within
BURST_GAP_SECONDS of the previous one -- the same definition
culling_app.py's Culling-page nav-pane grouping uses -- except every photo
in the album is considered here regardless of its blur/sharp/skipped
verdict (the whole point is to keep the discarded frames too).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import rawpy

from algo.album import album_for
from algo.utils import cap_long_edge, image_capture_timestamp

log = logging.getLogger("BlurPictureDetector")

# Mirrors culling_app.py's BURST_GAP_SECONDS so both features agree on what
# a "burst" is.
BURST_GAP_SECONDS = 1.0

DEFAULT_MIN_FRAMES = 3
DEFAULT_FPS = 6.0
# Floor for "timestamp" mode so two near-simultaneous shots never collapse
# to a zero-length frame. No ceiling is needed -- burst membership itself
# already caps every inter-frame gap at BURST_GAP_SECONDS.
MIN_FRAME_SECONDS = 1.0 / 30
# Long-edge cap for rendered frames: big enough to look sharp on a 4K TV
# without the file size/encode time ballooning for full-resolution originals.
MAX_LONG_EDGE = 3840

_RAW_EXTENSIONS = frozenset({".cr3", ".cr2", ".nef", ".arw"})


@dataclass
class BurstFrame:
    key: str
    path: Path  # frame to render: accepted AI edit if any, else the original
    timestamp: float
    name_stem: str  # original photo's filename stem, used to name the output video


@dataclass
class Burst:
    index: int
    frames: list[BurstFrame] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return len(self.frames)


def detect_bursts(album_path: Path) -> list[Burst]:
    """Group every photo in *album_path* (any status) into bursts by
    capture-time gap, ordered chronologically."""
    album = album_for(album_path)
    items: list[BurstFrame] = []
    for key, image in album.images.items():
        frame_path = image.image_path
        if frame_path is None or not frame_path.is_file():
            continue
        original = image.original_path
        ts_source = original if original and original.is_file() else frame_path
        items.append(BurstFrame(
            key=key,
            path=frame_path,
            timestamp=image_capture_timestamp(ts_source),
            name_stem=(original or frame_path).stem,
        ))
    items.sort(key=lambda it: (it.timestamp, it.key))

    bursts: list[Burst] = []
    for item in items:
        if bursts and item.timestamp - bursts[-1].frames[-1].timestamp <= BURST_GAP_SECONDS:
            bursts[-1].frames.append(item)
        else:
            bursts.append(Burst(index=len(bursts), frames=[item]))
    return bursts


def summarize_bursts(album_path: Path, min_frames: int = DEFAULT_MIN_FRAMES) -> dict:
    bursts = detect_bursts(album_path)
    qualifying = [b for b in bursts if b.frame_count >= min_frames]
    return {
        "totalBursts": len(bursts),
        "qualifyingBursts": len(qualifying),
        "qualifyingFrames": sum(b.frame_count for b in qualifying),
        "ffmpegAvailable": shutil.which("ffmpeg") is not None,
    }


def _log(on_line: Optional[Callable[[str], None]], msg: str) -> None:
    if on_line:
        on_line(msg)
    log.info("[burst-video] %s", msg)


def _decode_frame(path: Path) -> Optional[np.ndarray]:
    """BGR array for *path* -- RAW formats decoded via rawpy, same as the
    rest of the pipeline (algo/stages/image_analysis.py::_read_image)."""
    if path.suffix.lower() in _RAW_EXTENSIONS:
        try:
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8, half_size=True)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            log.warning("[burst-video] rawpy failed for %s: %s", path.name, exc)
            return None
    return cv2.imread(str(path))


def _fit_to_canvas(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Scale *image* to fit within *size* preserving its own aspect ratio,
    letterboxing only the shortfall (a no-op when *image* already matches
    *size*, which is the common case within one burst)."""
    tw, th = size
    h, w = image.shape[:2]
    if (w, h) == (tw, th):
        return image
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x0, y0 = (tw - nw) // 2, (th - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _unique_path(directory: Path, stem: str, suffix: str = ".mp4") -> Path:
    candidate = directory / f"{stem}{suffix}"
    n = 2
    while candidate.exists():
        candidate = directory / f"{stem}_{n}{suffix}"
        n += 1
    return candidate


def _output_path_for_burst(bursts_dir: Path, burst: Burst) -> Path:
    return _unique_path(bursts_dir, burst.frames[0].name_stem)


_INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize_filename(name: str) -> str:
    cleaned = _INVALID_FS_CHARS.sub("_", name).strip(" .")
    return cleaned or "Unnamed"


def _render_with_ffmpeg(
    frame_dir: Path, frame_count: int, durations: Optional[list[float]], fps: float, out_path: Path,
) -> None:
    if durations is None:
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(frame_dir / "frame_%06d.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
            "-movflags", "+faststart", str(out_path),
        ]
    else:
        list_path = frame_dir / "concat.txt"
        with open(list_path, "w", encoding="utf-8") as fh:
            for i, dur in enumerate(durations):
                fh.write(f"file 'frame_{i:06d}.jpg'\n")
                fh.write(f"duration {dur:.6f}\n")
            # The concat demuxer ignores the duration on the final entry
            # unless the same file is repeated once more without one.
            fh.write(f"file 'frame_{frame_count - 1:06d}.jpg'\n")
        cmd = [
            "ffmpeg", "-y", "-safe", "0", "-f", "concat", "-i", str(list_path),
            "-vsync", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "20", "-preset", "medium", "-movflags", "+faststart", str(out_path),
        ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg exited {result.returncode}: {result.stderr[-2000:]}")


def _render_with_opencv(
    frame_dir: Path, frame_count: int, durations: Optional[list[float]], fps: float, out_path: Path,
    size: tuple[int, int],
) -> None:
    # cv2.VideoWriter only supports a constant frame rate -- "timestamp"
    # mode is approximated by repeating each frame proportionally to how
    # long it should be held, at a fine-grained base rate.
    base_fps = fps if durations is None else 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, base_fps, size)
    try:
        for i in range(frame_count):
            img = cv2.imread(str(frame_dir / f"frame_{i:06d}.jpg"))
            repeat = 1 if durations is None else max(1, round(durations[i] * base_fps))
            for _ in range(repeat):
                writer.write(img)
    finally:
        writer.release()


def _render_burst(
    burst: Burst, out_path: Path, mode: str, fps: float, tmp_root: Path,
    on_line: Optional[Callable[[str], None]],
) -> bool:
    frame_dir = tmp_root / f"burst_{burst.index:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    try:
        target_size: Optional[tuple[int, int]] = None
        durations: list[float] = []
        written = 0
        for i, bf in enumerate(burst.frames):
            img = _decode_frame(bf.path)
            if img is None:
                _log(on_line, f"  skipping unreadable frame: {bf.path.name}")
                continue
            img = cap_long_edge(img, MAX_LONG_EDGE)
            h, w = img.shape[:2]
            if target_size is None:
                target_size = (w - w % 2, h - h % 2)
            img = _fit_to_canvas(img, target_size)
            cv2.imwrite(str(frame_dir / f"frame_{written:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
            if mode == "timestamp" and i + 1 < len(burst.frames):
                gap = burst.frames[i + 1].timestamp - bf.timestamp
                durations.append(max(MIN_FRAME_SECONDS, gap))

        if written < 2 or target_size is None:
            _log(on_line, f"  burst {burst.index}: fewer than 2 readable frames, skipping")
            return False
        if mode == "timestamp":
            durations.append(durations[-1] if durations else 1.0 / fps)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame_durations = durations if mode == "timestamp" else None
        if shutil.which("ffmpeg"):
            _render_with_ffmpeg(frame_dir, written, frame_durations, fps, out_path)
        else:
            _render_with_opencv(frame_dir, written, frame_durations, fps, out_path, target_size)
        return True
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def render_album_bursts(
    album_path: Path,
    *,
    min_frames: int = DEFAULT_MIN_FRAMES,
    mode: str = "fps",
    fps: float = DEFAULT_FPS,
    on_line: Optional[Callable[[str], None]] = None,
) -> dict:
    """Render every burst in *album_path* with at least *min_frames* photos
    to ``<album_path>/bursts/<first-frame-name>.mp4``."""
    bursts = [b for b in detect_bursts(album_path) if b.frame_count >= min_frames]
    bursts_dir = album_path / "bursts"
    rendered = skipped = 0

    _log(on_line, f"Found {len(bursts)} burst(s) with >= {min_frames} frame(s).")
    if shutil.which("ffmpeg") is None:
        _log(on_line, "ffmpeg not found on PATH -- falling back to a lower-quality built-in encoder.")

    with tempfile.TemporaryDirectory(prefix="burst_video_") as tmp:
        tmp_root = Path(tmp)
        for n, burst in enumerate(bursts, 1):
            out_path = _output_path_for_burst(bursts_dir, burst)
            _log(on_line, f"Rendering burst {n}/{len(bursts)} ({burst.frame_count} frames) -> {out_path.name}")
            try:
                ok = _render_burst(burst, out_path, mode, fps, tmp_root, on_line)
            except Exception as exc:
                _log(on_line, f"  failed: {exc}")
                ok = False
            rendered += 1 if ok else 0
            skipped += 0 if ok else 1

    _log(on_line, f"Done. Rendered {rendered} video(s), skipped {skipped}.")
    return {"burstsFound": len(bursts), "rendered": rendered, "skipped": skipped, "outputDir": str(bursts_dir)}


def _render_frame_sequence(
    frame_paths: list[Path], out_path: Path, fps: float, tmp_root: Path,
    on_line: Optional[Callable[[str], None]],
) -> bool:
    """Render an already-ordered list of photos into one constant-fps MP4 --
    the per-player counterpart of _render_burst, minus the "timestamp" mode
    (a player's photos span the whole album, so matching real capture gaps
    would make for a mostly-frozen video)."""
    frame_dir = Path(tempfile.mkdtemp(prefix="seq_", dir=tmp_root))
    try:
        target_size: Optional[tuple[int, int]] = None
        written = 0
        for path in frame_paths:
            img = _decode_frame(path)
            if img is None:
                _log(on_line, f"  skipping unreadable frame: {path.name}")
                continue
            img = cap_long_edge(img, MAX_LONG_EDGE)
            h, w = img.shape[:2]
            if target_size is None:
                target_size = (w - w % 2, h - h % 2)
            img = _fit_to_canvas(img, target_size)
            cv2.imwrite(str(frame_dir / f"frame_{written:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1

        if written < 1 or target_size is None:
            _log(on_line, "  no readable frames, skipping")
            return False

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("ffmpeg"):
            _render_with_ffmpeg(frame_dir, written, None, fps, out_path)
        else:
            _render_with_opencv(frame_dir, written, None, fps, out_path, target_size)
        return True
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def render_player_videos(
    player_frames: dict[str, list[Path]],
    output_dir: Path,
    *,
    fps: float = DEFAULT_FPS,
    on_line: Optional[Callable[[str], None]] = None,
) -> dict:
    """Render one highlight MP4 per player to *output_dir*/<player name>.mp4.

    *player_frames* maps a tagged player's name to every distinct photo
    they appear in (any order -- sorted here by capture time). Callers
    resolve the FaceReco-tagged photos to filesystem paths; this function
    only knows how to turn an ordered photo list into a video."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(player_frames.keys())
    rendered = skipped = 0

    _log(on_line, f"Found {len(names)} player(s) with tagged faces.")
    if shutil.which("ffmpeg") is None:
        _log(on_line, "ffmpeg not found on PATH -- falling back to a lower-quality built-in encoder.")

    with tempfile.TemporaryDirectory(prefix="player_video_") as tmp:
        tmp_root = Path(tmp)
        for n, name in enumerate(names, 1):
            frames = sorted(player_frames[name], key=image_capture_timestamp)
            out_path = _unique_path(output_dir, _sanitize_filename(name))
            _log(on_line, f"Rendering {n}/{len(names)}: {name} ({len(frames)} photo(s)) -> {out_path.name}")
            try:
                ok = _render_frame_sequence(frames, out_path, fps, tmp_root, on_line)
            except Exception as exc:
                _log(on_line, f"  failed: {exc}")
                ok = False
            rendered += 1 if ok else 0
            skipped += 0 if ok else 1

    _log(on_line, f"Done. Rendered {rendered} video(s), skipped {skipped}.")
    return {"playersFound": len(names), "rendered": rendered, "skipped": skipped, "outputDir": str(output_dir)}

