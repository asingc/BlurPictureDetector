from __future__ import annotations

import base64
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rawpy
from sklearn.cluster import DBSCAN

from .facereco_provider import BodyRecord, Box, FaceRecoProvider, Player

log = logging.getLogger("BlurPictureDetector")


@dataclass
class FaceRecoConfig:
    # Cosine similarity floor for two faces to be placed in the same cluster.
    # Higher = tighter clusters (fewer faces per cluster, more clusters overall).
    cluster_similarity_threshold: float = 0.68
    face_buffer_ratio: float = 0.15
    output_dir_name: str = ".FaceReco"
    # Optional path to a face-DB directory.  Each subdirectory must contain
    # a face.json produced by a previous FaceReco run.
    face_db_dir: Path | None = None
    # Minimum cosine similarity for a cluster centroid to be matched against
    # a face-DB entry.
    face_db_match_threshold: float = 0.72


@dataclass
class FaceSample:
    body: BodyRecord
    embedding: np.ndarray
    confidence: float | None
    crop_file_name: str
    crop_image: np.ndarray | None


@dataclass
class Cluster:
    cluster_id: int
    samples: list[FaceSample]


@dataclass
class FaceDbEntry:
    """One person loaded from the face database."""

    name: str
    playernum: int | None
    provider: str
    embeddings: list[np.ndarray]          # positive embeddings
    negative_embeddings: list[np.ndarray]  # negative embeddings (may be empty)


def _decode_embedding(face_data: dict) -> np.ndarray | None:
    """Decode a base64-encoded embedding stored in a face.json entry."""
    try:
        dtype = np.dtype(face_data.get("dtype", "float32"))
        raw = base64.b64decode(face_data["value"])
        return np.frombuffer(raw, dtype=dtype).copy()
    except Exception:  # noqa: BLE001
        return None


class FaceDb:
    """Loaded face database: a collection of known people with their embeddings."""

    def __init__(self, entries: list[FaceDbEntry]) -> None:
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, db_dir: Path) -> "FaceDb":
        """Walk *db_dir* and load every ``face.json`` found in a sub-directory."""
        entries: list[FaceDbEntry] = []
        if not db_dir.is_dir():
            raise FileNotFoundError(f"Face-DB directory not found: {db_dir}")
        for person_dir in sorted(db_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            face_json = person_dir / "face.json"
            if not face_json.exists():
                log.debug("FaceDB [load]: %s — no face.json, skipped", person_dir.name)
                continue
            try:
                with open(face_json, encoding="utf-8-sig") as fh:
                    data = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                log.warning("FaceDB [load]: %s — failed to parse face.json: %s", person_dir.name, exc)
                continue
            name = data.get("name") or person_dir.name
            playernum = data.get("playernum")
            provider = str(data.get("provider", ""))
            embeddings = [
                e for e in (_decode_embedding(f) for f in data.get("faces", []))
                if e is not None
            ]
            neg_embeddings = [
                e for e in (_decode_embedding(f) for f in data.get("negative_faces", []))
                if e is not None
            ]
            if not embeddings:
                log.debug("FaceDB [load]: %s — no valid embeddings, skipped", name)
                continue
            entries.append(FaceDbEntry(
                name=name,
                playernum=playernum,
                provider=provider,
                embeddings=embeddings,
                negative_embeddings=neg_embeddings,
            ))
            log.debug("FaceDB [load]: %s  playernum=%s  provider=%s  embeddings=%d  negatives=%d",
                      name, playernum, provider, len(embeddings), len(neg_embeddings))
        total_faces = sum(len(e.embeddings) for e in entries)
        log.info(
            "FaceDB [load]: loaded %d cluster(s) / %d face(s) from %s",
            len(entries), total_faces, db_dir,
        )
        for entry in entries:
            log.info(
                "FaceDB [load]:   %-20s  playernum=%-4s  faces=%d  negatives=%d",
                entry.name, entry.playernum, len(entry.embeddings), len(entry.negative_embeddings),
            )
        return cls(entries)


@dataclass
class _QualifiedBody:
    image_path: Path
    body: BodyRecord


class FaceRecoPipeline:
    def __init__(self, provider: FaceRecoProvider, config: FaceRecoConfig) -> None:
        self.provider = provider
        self.config = config

    def run(self, prep_output_dir: Path) -> Path:
        prep_output_dir = prep_output_dir.resolve()
        results_path = prep_output_dir / "results.json"
        info_path = prep_output_dir / "info.json"

        log.info("FaceReco: starting  prep_dir=%s", prep_output_dir)
        log.debug(
            "FaceReco config: provider=%s  cluster_threshold=%.3f  eps=%.3f  "
            "face_buffer_ratio=%.2f  output_dir=%s",
            self.provider.provider_name(),
            self.config.cluster_similarity_threshold,
            1.0 - self.config.cluster_similarity_threshold,
            self.config.face_buffer_ratio,
            self.config.output_dir_name,
        )

        payload = self._load_json(results_path)
        info = self._load_json(info_path)
        src_dir = Path(info.get("SrcDir", "")).resolve() if info.get("SrcDir") else prep_output_dir
        log.debug("FaceReco: src_dir=%s", src_dir)

        qualified = self._collect_qualified_bodies(payload)

        out_root = prep_output_dir / self.config.output_dir_name
        out_root.mkdir(parents=True, exist_ok=True)
        log.debug("FaceReco: output root=%s", out_root)

        if not qualified:
            log.warning("FaceReco: no qualified (sharp) bodies found — empty folder created: %s", out_root)
            return out_root

        # Load face DB before clustering so identity info is available for matching.
        face_db: FaceDb | None = None
        if self.config.face_db_dir is not None:
            face_db = FaceDb.load(self.config.face_db_dir)
            log.info("FaceReco: face DB loaded — %d person(s) from %s",
                     len(face_db), self.config.face_db_dir)

        log.info("FaceReco: %d qualified bodies collected", len(qualified))
        samples = self._predict_samples(qualified)
        log.info("FaceReco: %d/%d embeddings extracted successfully", len(samples), len(qualified))
        clusters = self._cluster_samples(samples)
        log.info("FaceReco: %d cluster(s) identified  (sizes: %s)",
                 len(clusters), [len(c.samples) for c in clusters])

        cluster_name_map: dict[int, FaceDbEntry] = {}
        if face_db is not None:
            cluster_name_map = self._match_clusters_to_facedb(clusters, face_db)
            log.info("FaceReco: %d/%d cluster(s) matched to face DB",
                     len(cluster_name_map), len(clusters))

        self._write_all_faces(out_root, clusters)
        self._write_cluster_outputs(out_root, clusters, src_dir, cluster_name_map)
        log.info("FaceReco completed: %d qualified bodies → %d clusters  output=%s",
                 len(samples), len(clusters), out_root)
        return out_root

    def _load_json(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _collect_qualified_bodies(self, payload: dict) -> list[_QualifiedBody]:
        qualified: list[_QualifiedBody] = []
        total_results = len(payload.get("results", []))
        total_bodies = skipped_blurry = skipped_no_ann = 0
        log.debug("FaceReco [collect]: scanning %d result entries", total_results)
        for result in payload.get("results", []):
            image_path = Path(result.get("file", ""))
            ann = result.get("annotation_data")
            if ann is None:
                skipped_no_ann += 1
                log.debug("FaceReco [collect]: %s — no annotation_data, skipped", image_path.name)
                continue
            evaluated = ann.get("evaluated", [])
            log.debug("FaceReco [collect]: %s — %d evaluated body/bodies", image_path.name, len(evaluated))
            for idx, body_data in enumerate(evaluated):
                total_bodies += 1
                is_blurry = body_data.get("is_blurry", True)
                sharpness = body_data.get("sharpness_score", 0.0)
                cloth = body_data.get("cloth_color", "N/A")
                if is_blurry:
                    skipped_blurry += 1
                    log.debug(
                        "FaceReco [collect]: %s body#%d  score=%.3f  color=%s  → SKIP (blurry)",
                        image_path.name, idx, sharpness, cloth,
                    )
                    continue
                log.debug(
                    "FaceReco [collect]: %s body#%d  score=%.3f  color=%s  → QUALIFIED",
                    image_path.name, idx, sharpness, cloth,
                )
                body = self._parse_body_record(image_path.name, idx, body_data)
                qualified.append(_QualifiedBody(image_path=image_path, body=body))
        log.info(
            "FaceReco [collect]: total_results=%d  total_bodies=%d  "
            "qualified=%d  skipped_blurry=%d  skipped_no_annotation=%d",
            total_results, total_bodies, len(qualified), skipped_blurry, skipped_no_ann,
        )
        return qualified

    def _parse_body_record(self, orig_filename: str, body_index: int, body_data: dict) -> BodyRecord:
        def _box(box_data: dict | None) -> Box | None:
            if box_data is None:
                return None
            return Box(
                float(box_data["x1"]),
                float(box_data["y1"]),
                float(box_data["x2"]),
                float(box_data["y2"]),
            )

        face_kps = body_data.get("face_kps") or {}
        return BodyRecord(
            orig_filename=orig_filename,
            body_index=body_index,
            body_bbox=_box(body_data["body_bbox"]),
            face_bbox=_box(body_data.get("face_bbox")),
            narrow_face_bbox=_box(body_data.get("narrow_face_bbox")),
            cloth_color=str(body_data.get("cloth_color", "N/A")),
            qualified_for_sharpness=bool(body_data.get("qualified_for_sharpness", False)),
            is_blurry=bool(body_data.get("is_blurry", True)),
            confidence=face_kps.get("confidence"),
            raw_body=body_data,
        )

    # RAW extensions that require rawpy instead of OpenCV.
    _RAW_EXTENSIONS: frozenset[str] = frozenset({".cr3", ".cr2"})

    def _load_image(self, path: Path) -> np.ndarray | None:
        """Load the original source image as a full-resolution BGR array.
        RAW files (.cr3/.cr2) are decoded with rawpy at full resolution.
        """
        if path.suffix.lower() in self._RAW_EXTENSIONS:
            try:
                with rawpy.imread(str(path)) as raw:
                    rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
                img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                log.debug("FaceReco [load]: RAW %s  decoded %dx%d", path.name, img.shape[1], img.shape[0])
                return img
            except Exception as exc:
                log.warning("FaceReco [load]: rawpy failed for %s: %s", path.name, exc)
                return None
        img = cv2.imread(str(path))
        if img is not None:
            log.debug("FaceReco [load]: %s  loaded %dx%d", path.name, img.shape[1], img.shape[0])
        return img

    def _predict_samples(self, qualified: list[_QualifiedBody]) -> list[tuple[_QualifiedBody, Player, np.ndarray]]:
        predicted: list[tuple[_QualifiedBody, Player, np.ndarray]] = []
        total = len(qualified)
        skipped_load = skipped_embedding = skipped_crop = 0
        log.debug("FaceReco [embed]: processing %d qualified body/bodies", total)
        for index, item in enumerate(qualified, start=1):
            tag = f"{item.image_path.name} body#{item.body.body_index}"
            image = self._load_image(item.image_path)
            if image is None:
                skipped_load += 1
                log.warning("FaceReco [embed]: %s — cannot read image, skipped", tag)
                continue
            player = self.provider.predict_player(image, item.body)
            embedding = player.internal.get("embedding")
            if embedding is None:
                skipped_embedding += 1
                log.debug("FaceReco [embed]: %s — no embedding returned by provider, skipped", tag)
                continue
            emb_arr = np.asarray(embedding, dtype=np.float32)
            log.debug(
                "FaceReco [embed]: %s — embedding dim=%d  norm=%.4f  provider_conf=%s",
                tag, emb_arr.shape[0], float(np.linalg.norm(emb_arr)),
                f"{player.confidence:.3f}" if player.confidence is not None else "n/a",
            )
            crop = self._crop_face_with_buffer(image, item.body)
            if crop is None:
                skipped_crop += 1
                log.debug("FaceReco [embed]: %s — face crop returned None, skipped", tag)
                continue
            log.debug("FaceReco [embed]: %s — crop %dx%d  ✓", tag, crop.shape[1], crop.shape[0])
            predicted.append((item, player, crop))
            if index == 1 or index % 50 == 0 or index == total:
                log.info("FaceReco [embed]: %d/%d processed  embedded=%d", index, total, len(predicted))
        log.info(
            "FaceReco [embed]: done  embedded=%d  skipped_load=%d  "
            "skipped_embedding=%d  skipped_crop=%d",
            len(predicted), skipped_load, skipped_embedding, skipped_crop,
        )
        return predicted

    def _cluster_samples(self, predicted: list[tuple[_QualifiedBody, Player, np.ndarray]]) -> list[Cluster]:
        """Cluster face embeddings with DBSCAN using pairwise cosine distance.

        Every sample is always assigned to a cluster (min_samples=1).
        eps = 1 - cluster_similarity_threshold converts the similarity
        threshold to a cosine *distance* threshold.
        """
        if not predicted:
            log.warning("FaceReco [cluster]: no samples to cluster")
            return []

        # Build unit-normalised embedding matrix.
        embeddings: list[np.ndarray] = []
        for _item, player, _crop in predicted:
            emb = np.asarray(player.internal["embedding"], dtype=np.float32)
            embeddings.append(self._normalize(emb))
        matrix = np.array(embeddings, dtype=np.float32)

        eps = max(1e-6, 1.0 - self.config.cluster_similarity_threshold)
        log.info(
            "FaceReco [cluster]: running DBSCAN  n_samples=%d  eps=%.4f  "
            "(similarity_threshold=%.3f)  metric=cosine",
            len(embeddings), eps, self.config.cluster_similarity_threshold,
        )
        labels: np.ndarray = DBSCAN(
            eps=eps,
            min_samples=1,
            metric="cosine",
            n_jobs=1,
        ).fit_predict(matrix)

        unique_labels = sorted(set(labels.tolist()))
        log.info("FaceReco [cluster]: DBSCAN produced %d cluster(s)  labels=%s",
                 len(unique_labels), unique_labels)

        cluster_map: dict[int, Cluster] = {}
        for i, (label, (item, player, crop)) in enumerate(zip(labels.tolist(), predicted)):
            if label not in cluster_map:
                cluster_map[label] = Cluster(cluster_id=label, samples=[])
            sample = FaceSample(
                body=item.body,
                embedding=embeddings[i],
                confidence=player.confidence,
                crop_file_name="",
                crop_image=crop,
            )
            cluster_map[label].samples.append(sample)
            log.debug(
                "FaceReco [cluster]: %s body#%d → cluster %d",
                item.body.orig_filename, item.body.body_index, label,
            )

        clusters = sorted(cluster_map.values(), key=lambda c: c.cluster_id)
        for c in clusters:
            log.debug(
                "FaceReco [cluster]: cluster %04d  size=%d  files=[%s]",
                c.cluster_id, len(c.samples),
                ", ".join(s.body.orig_filename for s in c.samples),
            )
        return clusters

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return vector
        return vector / norm

    def _match_clusters_to_facedb(
        self, clusters: list[Cluster], face_db: FaceDb,
    ) -> dict[int, FaceDbEntry]:
        """Match clusters to face-DB entries using per-sample majority voting.

        For each sample in a cluster, the best-matching DB entry (highest
        cosine similarity to any of that person's known embeddings) is found.
        If that similarity meets ``config.face_db_match_threshold`` the sample
        casts a vote for that person.  The cluster is assigned to the person
        with the most votes, provided they hold a strict majority (> 50 % of
        all samples in the cluster).

        Voting per-sample — rather than comparing a single cluster centroid —
        handles the common case where a person's face embeddings are split
        across multiple DBSCAN clusters: even a cluster whose centroid has
        drifted away from the DB embeddings will still be correctly identified
        if most of its individual samples match that person.

        Only DB entries produced by the same provider as the current pipeline
        are considered, so that embedding spaces are compatible.
        """
        provider_name = self.provider.provider_name()
        compatible = [e for e in face_db.entries if e.provider == provider_name]
        if not compatible:
            log.warning(
                "FaceReco [match]: no face-DB entries for provider '%s' — skipping DB matching",
                provider_name,
            )
            return {}

        # Pre-normalise all DB embeddings once to avoid repeated work.
        normalised_db: list[tuple[FaceDbEntry, list[np.ndarray]]] = [
            (entry, [self._normalize(emb.astype(np.float32)) for emb in entry.embeddings])
            for entry in compatible
        ]

        result: dict[int, FaceDbEntry] = {}
        for cluster in clusters:
            if not cluster.samples:
                continue

            # Per-sample voting: each sample independently finds its best DB match.
            votes: dict[str, int] = {}          # person name → vote count
            best_sim_for: dict[str, float] = {} # person name → highest sim seen

            for sample in cluster.samples:
                emb = self._normalize(sample.embedding.astype(np.float32))
                sample_best_score = -1.0
                sample_best_entry: FaceDbEntry | None = None

                for entry, normed_embs in normalised_db:
                    sim = max(float(np.dot(emb, ne)) for ne in normed_embs)
                    if sim > sample_best_score:
                        sample_best_score = sim
                        sample_best_entry = entry

                if (sample_best_entry is not None
                        and sample_best_score >= self.config.face_db_match_threshold):
                    name = sample_best_entry.name
                    votes[name] = votes.get(name, 0) + 1
                    best_sim_for[name] = max(best_sim_for.get(name, -1.0), sample_best_score)

            n_samples = len(cluster.samples)
            if not votes:
                log.debug(
                    "FaceReco [match]: cluster %04d (%d sample(s)) — no votes above threshold %.3f",
                    cluster.cluster_id, n_samples, self.config.face_db_match_threshold,
                )
                continue

            winner_name = max(votes, key=lambda n: votes[n])
            winner_votes = votes[winner_name]
            vote_ratio = winner_votes / n_samples

            log.debug(
                "FaceReco [match]: cluster %04d (%d sample(s)) — votes=%s  winner=%s (%d/%d = %.0f%%)",
                cluster.cluster_id, n_samples, dict(votes),
                winner_name, winner_votes, n_samples, vote_ratio * 100,
            )

            if vote_ratio > 0.5:
                winner_entry = next(e for e in compatible if e.name == winner_name)
                result[cluster.cluster_id] = winner_entry
                log.info(
                    "FaceReco [match]: cluster %04d → %s (playernum=%s  votes=%d/%d=%.0f%%  best_sim=%.4f)",
                    cluster.cluster_id, winner_name, winner_entry.playernum,
                    winner_votes, n_samples, vote_ratio * 100, best_sim_for[winner_name],
                )
            else:
                log.debug(
                    "FaceReco [match]: cluster %04d — no majority  (winner=%s  %d/%d=%.0f%%)",
                    cluster.cluster_id, winner_name, winner_votes, n_samples, vote_ratio * 100,
                )
        return result

    def _write_all_faces(self, out_root: Path, clusters: list[Cluster]) -> None:
        """Write every processed face crop and a combined face.json to
        ``<out_root>/.AllFaces/``.

        This is a flat dump of the entire run — one image per face, one JSON
        entry per face — independent of how the faces are later clustered.
        The directory is written *before* :meth:`_write_cluster_outputs` so
        that ``crop_image`` is still in memory for each sample.
        """
        all_faces_dir = out_root / ".AllFaces"
        all_faces_dir.mkdir(parents=True, exist_ok=True)

        face_entries: list[dict] = []
        written = 0

        for cluster in clusters:
            cluster_num = f"{cluster.cluster_id:04d}"
            for sample in cluster.samples:
                crop = sample.crop_image
                if crop is None:
                    log.debug(
                        "FaceReco [allfaces]: %s body#%d — no crop, skipped",
                        sample.body.orig_filename, sample.body.body_index,
                    )
                    continue

                crop_name = (
                    f"{cluster_num}-"
                    f"{Path(sample.body.orig_filename).stem}-"
                    f"b{sample.body.body_index}.jpg"
                )
                cv2.imwrite(
                    str(all_faces_dir / crop_name), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )
                written += 1

                emb = sample.embedding.astype(np.float32)
                face_entries.append({
                    "origFilename": sample.body.orig_filename,
                    "cluster": cluster_num,
                    "cropFileName": crop_name,
                    "confidence": sample.confidence,
                    "Body": sample.body.raw_body,
                    "embedding": {
                        "dtype": "float32",
                        "shape": [int(emb.shape[0])],
                        "encoding": "base64",
                        "value": base64.b64encode(emb.tobytes()).decode("ascii"),
                    },
                })

        payload = {
            "provider": self.provider.provider_name(),
            "total": written,
            "faces": face_entries,
        }
        with open(all_faces_dir / "face.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        log.info("FaceReco [allfaces]: %d face(s) written to %s", written, all_faces_dir)

    def _write_cluster_outputs(
        self,
        out_root: Path,
        clusters: list[Cluster],
        src_dir: Path,
        cluster_name_map: dict[int, FaceDbEntry] | None = None,
    ) -> None:
        # Maps cluster_id → actual output directory (may be named after a person).
        cluster_id_to_dir: dict[int, Path] = {}
        log.info("FaceReco [write]: writing %d cluster(s) to %s", len(clusters), out_root)
        for cluster_index, cluster in enumerate(clusters, start=1):
            # The 4-digit numeric name is always used for internal file naming.
            cluster_num = f"{cluster.cluster_id:04d}"
            db_entry = (cluster_name_map or {}).get(cluster.cluster_id)
            # Use the person's name as the directory name when matched.
            dir_name = db_entry.name if db_entry is not None else cluster_num
            cluster_dir = out_root / dir_name
            face_dir = cluster_dir / "Face"
            negative_dir = cluster_dir / "Negative"
            face_dir.mkdir(parents=True, exist_ok=True)
            negative_dir.mkdir(parents=True, exist_ok=True)
            cluster_id_to_dir[cluster.cluster_id] = cluster_dir

            written = 0
            for sample in cluster.samples:
                crop = sample.crop_image
                if crop is None:
                    log.debug("FaceReco [write]: cluster %s — %s body#%d has no crop image",
                              cluster_num, sample.body.orig_filename, sample.body.body_index)
                    continue

                crop_name = f"{cluster_num}-{Path(sample.body.orig_filename).stem}-b{sample.body.body_index}.jpg"
                cv2.imwrite(str(face_dir / crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                sample.crop_file_name = crop_name
                sample.crop_image = None
                written += 1
                log.debug("FaceReco [write]: cluster %s — saved crop %s  (%dx%d)",
                          cluster_num, crop_name, crop.shape[1], crop.shape[0])

            self._write_face_json(
                cluster_dir, cluster_num, cluster.samples,
                name=db_entry.name if db_entry is not None else "",
                playernum=db_entry.playernum if db_entry is not None else None,
            )
            label = f"{cluster_num} ({dir_name})" if db_entry is not None else cluster_num
            log.info(
                "FaceReco [write]: cluster %s  (%d/%d)  samples=%d  crops_written=%d",
                label, cluster_index, len(clusters), len(cluster.samples), written,
            )

        log.info("FaceReco [write]: populating Negative folders for %d cluster(s)", len(clusters))
        for target_index, target_cluster in enumerate(clusters, start=1):
            target_dir = cluster_id_to_dir[target_cluster.cluster_id]
            negative_dir = target_dir / "Negative"
            target_num = f"{target_cluster.cluster_id:04d}"
            neg_count = 0
            for other_cluster in clusters:
                if other_cluster.cluster_id == target_cluster.cluster_id:
                    continue
                # When a face DB was used, only matched clusters are reliable
                # negative sources; skip unmatched clusters entirely.
                if cluster_name_map and other_cluster.cluster_id not in cluster_name_map:
                    continue
                source_face_dir = cluster_id_to_dir[other_cluster.cluster_id] / "Face"
                for sample in other_cluster.samples:
                    if not sample.crop_file_name:
                        continue
                    src = source_face_dir / sample.crop_file_name
                    if not src.exists():
                        log.debug("FaceReco [neg]: source crop missing: %s", src)
                        continue
                    neg_name = f"{target_num}-{sample.crop_file_name.split('-', 1)[1]}"
                    self._link_or_copy_file(src, negative_dir / neg_name)
                    neg_count += 1
            log.debug("FaceReco [neg]: cluster %s — %d negative sample(s) linked", target_num, neg_count)
            if target_index == 1 or target_index % 25 == 0 or target_index == len(clusters):
                log.info("FaceReco [neg]: populated Negative folder %d/%d", target_index, len(clusters))

    def _resolve_image_path(self, src_dir: Path, orig_filename: str) -> Path | None:
        direct = src_dir / orig_filename
        if direct.exists():
            return direct
        return None

    def _link_or_copy_file(self, src: Path, dst: Path) -> None:
        if dst.exists():
            return
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copyfile(src, dst)

    def _crop_face_with_buffer(self, image: np.ndarray, body: BodyRecord) -> np.ndarray | None:
        # Use the full face-detector bbox (bigger box); fall back to narrow only
        # if the full bbox is unavailable.
        box = body.face_bbox or body.narrow_face_bbox
        if box is None:
            log.debug("FaceReco [crop]: %s body#%d — no face bbox available", body.orig_filename, body.body_index)
            return None

        h, w = image.shape[:2]
        bbox_src = "face_bbox" if body.face_bbox else "narrow_face_bbox"
        pad_x = box.width * self.config.face_buffer_ratio
        pad_y = box.height * self.config.face_buffer_ratio
        x1 = max(0, int(round((box.x1 - pad_x) * w)))
        y1 = max(0, int(round((box.y1 - pad_y) * h)))
        x2 = min(w, int(round((box.x2 + pad_x) * w)))
        y2 = min(h, int(round((box.y2 + pad_y) * h)))
        if x2 <= x1 or y2 <= y1:
            log.debug(
                "FaceReco [crop]: %s body#%d — degenerate crop (%d,%d,%d,%d), skipped",
                body.orig_filename, body.body_index, x1, y1, x2, y2,
            )
            return None
        log.debug(
            "FaceReco [crop]: %s body#%d — src=%s  box=(%.3f,%.3f,%.3f,%.3f)  "
            "pad=%.2f  px=(%d,%d,%d,%d)  crop=%dx%d  img=%dx%d",
            body.orig_filename, body.body_index, bbox_src,
            box.x1, box.y1, box.x2, box.y2,
            self.config.face_buffer_ratio,
            x1, y1, x2, y2, x2 - x1, y2 - y1, w, h,
        )
        return image[y1:y2, x1:x2]

    def _write_face_json(
        self,
        cluster_dir: Path,
        cluster_name: str,
        samples: list[FaceSample],
        name: str = "",
        playernum: int | None = None,
    ) -> None:
        faces = []
        for sample in samples:
            emb = sample.embedding.astype(np.float32)
            emb_b64 = base64.b64encode(emb.tobytes()).decode("ascii")
            faces.append(
                {
                    "origFilename": sample.body.orig_filename,
                    "Body": sample.body.raw_body,
                    "cropFileName": sample.crop_file_name,
                    "confidence": sample.confidence,
                    "embedding": {
                        "dtype": "float32",
                        "shape": [int(emb.shape[0])],
                        "encoding": "base64",
                        "value": emb_b64,
                    },
                }
            )

        payload = {
            "name": name,
            "playernum": playernum,
            "cluster": cluster_name,
            "provider": self.provider.provider_name(),
            "faces": faces,
        }
        with open(cluster_dir / "face.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
