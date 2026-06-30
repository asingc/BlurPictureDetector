#!/usr/bin/env python3
"""
4_face_reco.py
Generate face-recognition clusters from 1_prep_review.py output.

Usage:
    python 4_face_reco.py <prep_output_dir> [--provider facenet|dlib]

Expected input folder contents:
    <prep_output_dir>/results.json
    <prep_output_dir>/info.json

Output:
    <SrcDir>/.FaceReco/
        0000/
            Face/
            Negative/
            face.json
        0001/
            ...
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from algo.facereco import FaceRecoConfig, FaceRecoPipeline

try:
    from algo.facenet_provider import FaceNetFaceRecoProvider
    _FACENET_AVAILABLE = True
except ImportError:
    FaceNetFaceRecoProvider = None  # type: ignore[assignment]
    _FACENET_AVAILABLE = False

try:
    from algo.dlib_provider import DlibFaceRecoProvider
    _DLIB_AVAILABLE = True
except ImportError:
    DlibFaceRecoProvider = None  # type: ignore[assignment]
    _DLIB_AVAILABLE = False

log = logging.getLogger("4_face_reco")


def _setup_logging() -> None:
    log.setLevel(logging.DEBUG)
    if log.handlers:
        return
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(ch)


def _build_provider(name: str):
    if name == "facenet":
        if not _FACENET_AVAILABLE:
            raise RuntimeError(
                "FaceNet provider is unavailable. Install facenet-pytorch and try again, "
                "or use --provider dlib."
            )
        return FaceNetFaceRecoProvider()
    if name == "dlib":
        if not _DLIB_AVAILABLE:
            raise RuntimeError(
                "dlib provider is unavailable. Install face-recognition and dlib, or use --provider facenet."
            )
        return DlibFaceRecoProvider()
    raise ValueError(f"Unsupported provider: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create .FaceReco clusters from sharp, in-team bodies.",
    )
    parser.add_argument(
        "prep_output_dir",
        help="Folder produced by 1_prep_review.py containing results.json and info.json.",
    )
    parser.add_argument(
        "--provider",
        default="facenet",
        choices=["dlib", "facenet"],
        help="Face recognition provider (default: facenet).",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.72,
        help="Cosine similarity threshold for DBSCAN clustering (eps = 1 - threshold). Higher = tighter clusters.",
    )
    parser.add_argument(
        "--face-buffer-ratio",
        type=float,
        default=0.15,
        help="Extra crop padding ratio around face box (0.15 = 15%% per side).",
    )
    parser.add_argument(
        "--face-db",
        default=None,
        metavar="DIR",
        help=(
            "Path to a face-DB directory.  Each sub-directory must represent a "
            "person and contain a face.json with positive (and optionally negative) "
            "embeddings.  Clusters whose centroid matches a DB entry above the "
            "match threshold will be stored in a folder named after that person."
        ),
    )
    parser.add_argument(
        "--face-db-match-threshold",
        type=float,
        default=0.80,
        help=(
            "Cosine similarity threshold for matching a single face directly "
            "against the face DB (default: 0.80).  Each face is matched "
            "independently before clustering, so this is intentionally strict; "
            "higher = fewer false matches, more faces left to cluster."
        ),
    )
    parser.add_argument(
        "--align-faces",
        action="store_true",
        help=(
            "Similarity-align each face to a canonical 5-point template before "
            "computing its embedding.  Must match the setting used to build the "
            "face DB (RebuildFaceDB.py --align-faces).  Use for alignment A/B "
            "comparison."
        ),
    )
    parser.add_argument(
        "--debug-align",
        action="store_true",
        help=(
            "Write per-face alignment QA images (annotated crop + aligned face) "
            "to <output>/.FaceReco/.debug for visual inspection of landmark "
            "order and alignment quality."
        ),
    )
    parser.add_argument(
        "--open-viewer",
        action="store_true",
        help="Open the generated .FaceReco folder in the system file explorer.",
    )
    args = parser.parse_args()

    _setup_logging()

    prep_output_dir = Path(args.prep_output_dir).resolve()
    if not prep_output_dir.is_dir():
        log.error("prep_output_dir not found: %s", prep_output_dir)
        sys.exit(1)

    face_db_dir: Path | None = None
    if args.face_db is not None:
        face_db_dir = Path(args.face_db).resolve()
        if not face_db_dir.is_dir():
            log.error("--face-db directory not found: %s", face_db_dir)
            sys.exit(1)

    provider = _build_provider(args.provider)
    config = FaceRecoConfig(
        cluster_similarity_threshold=args.cluster_threshold,
        face_buffer_ratio=args.face_buffer_ratio,
        face_db_dir=face_db_dir,
        face_db_match_threshold=args.face_db_match_threshold,
        align_faces=args.align_faces,
        debug_align=args.debug_align,
    )
    pipeline = FaceRecoPipeline(provider=provider, config=config)

    out_root = pipeline.run(prep_output_dir)
    log.info("Generated FaceReco folder: %s", out_root)

    if args.open_viewer:
        os.startfile(str(out_root))


if __name__ == "__main__":
    main()
