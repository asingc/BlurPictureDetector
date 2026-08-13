from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import rawpy
from sklearn.cluster import AgglomerativeClustering

from .face_crop_embed import annotate_face_crop, embed_face_crop, load_face_model, make_alignment_debug_image
from .facereco_provider import BodyRecord, Box, FaceRecoProvider, Player

log = logging.getLogger("BlurPictureDetector")


@dataclass
class FaceRecoConfig:
    # Cosine similarity floor for two faces to be placed in the same cluster.
    # Higher = tighter clusters (fewer faces per cluster, more clusters overall).
    cluster_similarity_threshold: float = 0.65
    face_buffer_ratio: float = 0.15
    output_dir_name: str = ".FaceReco"
    # When True, each face crop is similarity-aligned to a canonical 5-point
    # template before its embedding is computed.  Must match the setting used
    # to build the face DB (RebuildFaceDB.py --align-faces) for matching to
    # work.  Provided as an on/off switch for alignment A/B comparison.
    align_faces: bool = True
    # When True, write per-face alignment QA images (annotated crop + aligned
    # face) to ``<output_dir>/.FaceReco/.debug`` so the landmark order and
    # alignment quality can be inspected visually.
    debug_align: bool = False
    # Optional path to a face-DB directory.  Each subdirectory must contain
    # a face.json produced by a previous FaceReco run.
    face_db_dir: Path | None = None
    # Optional allow-list of person names (e.g. an album's team roster) to
    # restrict face-DB matching to. When set, face-DB entries whose name
    # isn't in this set (case-insensitive) are skipped entirely at load
    # time, so their faces never become match candidates — bodies that
    # would've matched a filtered-out person fall through to residual
    # (unnamed/pending) clustering instead. None/empty disables filtering
    # (every face-DB entry is loaded, the previous behavior).
    face_db_allowed_names: frozenset[str] | None = None
    # Minimum cosine similarity between a query face and the closest-matching
    # PROTOTYPE (visual sub-cluster — see ``Prototype``/``build_prototypes``)
    # of a face-DB person, required to accept that person as a candidate
    # match.  Each face is matched independently against the DB before any
    # clustering -- there is no majority-vote safety net -- so this should be
    # calibrated empirically with ``RebuildFaceDB.py`` (it calibrates by default
    # after rebuilding) rather than guessed.
    face_db_match_threshold: float | None = None
    # Minimum cosine-similarity gap required between the best-matching person
    # and the best-matching *different* person.  A face whose top-2
    # candidates are nearly tied is exactly the failure mode that causes
    # "faces mixed up between people" -- accepting the #1 candidate anyway is
    # a coin flip dressed up as a confident match.  When the margin isn't met
    # the face is left unmatched (sent to residual clustering instead of a
    # guessed identity).  Calibrate alongside ``face_db_match_threshold``
    # with ``RebuildFaceDB.py``.
    face_db_match_margin: float | None = None
    # Cosine-similarity threshold used to split EACH PERSON's own positive
    # embeddings into visually-cohesive "prototypes" (sub-clusters) when the
    # face DB is loaded.  A real person's photos often form more than one
    # visual group (glasses on/off, indoor/outdoor lighting, angle, squint,
    # etc.) -- averaging all of a person's embeddings together would smear
    # those groups into an unrepresentative centroid.  Matching instead scores
    # a query against the SINGLE closest prototype of each person, so it only
    # has to resemble one genuine "look" rather than the blended average of
    # all of them.
    face_db_prototype_threshold: float | None = None
    # When True and ``face_db_dir/calibration.json`` exists, any face-DB
    # matching parameter left as ``None`` above is filled from that file for
    # the active provider (facenet/dlib). This turns "rebuild + calibrate"
    # into a real one-step workflow: the next recognition run automatically
    # consumes the calibrated values without manual copy/paste.
    use_face_db_calibration: bool = True
    calibration_file_name: str = "calibration.json"
    # Minimum short-edge size (pixels) of a face crop for its embedding to be
    # trusted at all.  Tiny / heavily-upscaled crops produce noisy embeddings
    # that are a common source of confident-looking false matches; crops
    # below this size are routed straight to skipped/unmatched instead of
    # being compared against the DB or clustered.
    min_face_crop_px: int = 32
    # Which detector/landmark engine to use for the per-crop face-landmark
    # refinement pass (see algo/face_crop_embed.py:load_face_model).
    # "mediapipe" (default, Apache-2.0) or "yolo" (legacy, AGPL-3.0/GPL-3.0).
    engine: str = "mediapipe"
    # The album's overall blur-sensitivity threshold (see GradingStage /
    # FaceSharpnessScorer) -- a body already had to score > this value to be
    # marked "sharp" for review purposes. Face-crop qualification uses
    # min(sensitivity_threshold, 0.5) instead of this value directly (see
    # FaceRecoPipeline._collect_qualified_bodies) so a strict album setting
    # (e.g. "high") doesn't also start discarding faces that are perfectly
    # recognizable but merely a bit softer than that stricter overall bar.
    sensitivity_threshold: float = 0.4
@dataclass
class FaceSample:
    body: BodyRecord
    embedding: np.ndarray
    confidence: float | None
    crop_file_name: str
    crop_image: np.ndarray | None
    annotated_image: np.ndarray | None = None
    # Diagnostics populated only for face-DB matches (see
    # _match_samples_to_facedb) -- kept on the sample so they can be written
    # into face.json for human auditing of *why* a face was matched.
    match_score: float | None = None
    match_margin: float | None = None
    match_runner_up: str | None = None


@dataclass
class Cluster:
    cluster_id: int
    samples: list[FaceSample]


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    """L2-normalise *vector* to a unit cosine-similarity-ready vector.

    Shared by DB loading (prototype building) and prediction-time matching so
    every embedding comparison in this module -- clustering, prototype
    construction, and face-DB matching -- happens in the exact same space.
    """
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        log.warning("FaceReco [normalize]: zero/near-zero norm vector detected (norm=%.2e), treating as unit vector", norm)
        return np.ones_like(vector) / np.sqrt(vector.shape[0])
    return vector / norm


@dataclass
class Prototype:
    """One visually-cohesive sub-cluster of a single person's embeddings.

    A real person's reference photos rarely form one tight blob in embedding
    space -- glasses on/off, indoor/outdoor lighting, camera angle, and
    expression all shift FaceNet's output meaningfully.  Treating a person as
    a single average vector blends these sub-groups together into a centroid
    that may not closely resemble any individual photo, which both misses
    genuine matches (the blended centroid is too far from any real look) and
    invites false matches (the blend drifts toward whichever look-alike
    happens to pull the average that way).  build_prototypes() instead splits
    a person's embeddings into these tighter groups up front, and matching
    scores a query against the single closest one.
    """

    centroid: np.ndarray        # unit vector -- mean of members, re-normalised
    members: list[np.ndarray]   # unit vectors belonging to this sub-cluster
    cohesion: float              # mean cosine(member, centroid); 1.0 = identical members


def build_prototypes(embeddings: list[np.ndarray], similarity_threshold: float) -> list[Prototype]:
    """Split one person's positive *embeddings* into visually-cohesive prototypes.

    Uses the same average-linkage cosine agglomerative clustering as residual
    face clustering (see FaceRecoPipeline._cluster_samples) -- average linkage
    resists chaining unrelated-looking photos of the same person into one
    over-broad prototype, while still merging genuinely similar shots.

    Returns prototypes sorted largest-first (purely so log output reads with
    the most representative "look" first); a person with zero embeddings
    returns an empty list, and one embedding returns a single 1-member
    prototype with cohesion 1.0.
    """
    if not embeddings:
        return []
    normed = [normalize_embedding(e.astype(np.float32)) for e in embeddings]
    n = len(normed)
    if n == 1:
        labels = [0]
    else:
        eps = max(1e-6, 1.0 - similarity_threshold)
        matrix = np.array(normed, dtype=np.float32)
        labels = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=eps,
        ).fit_predict(matrix).tolist()

    groups: dict[int, list[np.ndarray]] = {}
    for label, vec in zip(labels, normed):
        groups.setdefault(int(label), []).append(vec)

    prototypes: list[Prototype] = []
    for members in groups.values():
        centroid = normalize_embedding(np.mean(members, axis=0))
        if len(members) > 1:
            cohesion = float(np.mean([float(np.dot(m, centroid)) for m in members]))
        else:
            cohesion = 1.0
        prototypes.append(Prototype(centroid=centroid, members=members, cohesion=cohesion))

    prototypes.sort(key=lambda p: len(p.members), reverse=True)
    return prototypes


def select_diverse_subset(
    embeddings: list[np.ndarray],
    max_count: int,
    prototype_threshold: float = 0.62,
) -> list[int]:
    """Pick up to *max_count* indices into *embeddings* that maximise visual
    coverage while minimising near-duplicate repeats of the same "look".

    Intended for curating a face database: a burst of 100 near-identical
    crops of one person should contribute only a handful of genuinely
    different examples (angle/lighting/expression), not clog the database
    with near-duplicates that don't meaningfully improve recognition.

    Strategy:
      1. Group embeddings into prototypes (visually-cohesive sub-clusters —
         the same average-linkage cosine agglomerative clustering used by
         :func:`build_prototypes`), each representing one distinct "look".
      2. Pick each prototype's *medoid* (the member closest to its centroid)
         first, largest prototypes first — this guarantees every distinct
         look is represented at least once before any look gets a second
         slot.
      3. If budget remains, repeatedly add whichever remaining candidate has
         the LOWEST maximum cosine similarity to everything already picked
         (greedy farthest-point sampling) — this fills the rest of the
         budget with the most additionally-informative examples rather than
         more near-duplicates of an already-covered look.

    Returns every index (order preserved) unchanged when
    ``len(embeddings) <= max_count`` — nothing needs to be trimmed.
    """
    n = len(embeddings)
    if max_count <= 0:
        return []
    if n <= max_count:
        return list(range(n))

    normed = [normalize_embedding(e.astype(np.float32)) for e in embeddings]
    eps = max(1e-6, 1.0 - prototype_threshold)
    matrix = np.array(normed, dtype=np.float32)
    labels = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=eps,
    ).fit_predict(matrix).tolist()

    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(int(label), []).append(idx)

    def medoid(indices: list[int]) -> int:
        centroid = normalize_embedding(np.mean([normed[i] for i in indices], axis=0))
        return max(indices, key=lambda i: float(np.dot(normed[i], centroid)))

    # Round 1: one medoid per prototype (largest prototypes first, so a
    # budget smaller than the number of prototypes still favours the most
    # common looks).
    ordered_groups = sorted(groups.values(), key=len, reverse=True)
    selected: list[int] = []
    remaining: set[int] = set(range(n))
    for members in ordered_groups:
        if len(selected) >= max_count:
            break
        pick = medoid(members)
        selected.append(pick)
        remaining.discard(pick)

    # Round 2: greedy farthest-point fill from whatever's left.
    while len(selected) < max_count and remaining:
        best_idx, best_score = None, float("inf")
        for i in remaining:
            sim = max(float(np.dot(normed[i], normed[j])) for j in selected)
            if sim < best_score:
                best_score, best_idx = sim, i
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


def select_useful_subset(
    embeddings: list[np.ndarray],
    redundancy_threshold: float = 0.93,
    prototype_threshold: float = 0.62,
) -> list[int]:
    """Pick which indices into *embeddings* are worth keeping in a face
    database, so the rest can be retired as redundant.

    Unlike :func:`select_diverse_subset` (which trims down to a fixed target
    *count*), this has no target size -- it keeps growing the kept set for
    as long as a candidate exists that isn't a near-duplicate (cosine
    similarity >= *redundancy_threshold*) of something already kept, i.e.
    for as long as a candidate would still meaningfully expand this person's
    visual coverage. Everything left over contributes nothing new and can be
    safely retired without hurting recognition accuracy.

    Every prototype (visually-cohesive sub-cluster, see :func:`build_prototypes`)
    contributes its medoid unconditionally first, so no distinct "look" is
    ever entirely lost regardless of the redundancy threshold.

    Returns every index (order preserved) unchanged when there are 0 or 1
    embeddings -- nothing to compare, nothing to retire.
    """
    n = len(embeddings)
    if n <= 1:
        return list(range(n))

    normed = [normalize_embedding(e.astype(np.float32)) for e in embeddings]
    eps = max(1e-6, 1.0 - prototype_threshold)
    matrix = np.array(normed, dtype=np.float32)
    labels = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=eps,
    ).fit_predict(matrix).tolist()

    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(int(label), []).append(idx)

    def medoid(indices: list[int]) -> int:
        centroid = normalize_embedding(np.mean([normed[i] for i in indices], axis=0))
        return max(indices, key=lambda i: float(np.dot(normed[i], centroid)))

    # Round 1: one medoid per prototype (largest prototypes first) -- always
    # kept, so every distinct look survives no matter how redundant.
    kept: list[int] = []
    remaining: set[int] = set(range(n))
    for members in sorted(groups.values(), key=len, reverse=True):
        pick = medoid(members)
        kept.append(pick)
        remaining.discard(pick)

    # Round 2: keep greedily adding whichever remaining candidate is LEAST
    # similar to everything already kept, stopping as soon as the best
    # remaining candidate is itself a near-duplicate (>= redundancy_threshold)
    # of something already kept -- i.e. nothing left would add real coverage.
    while remaining:
        best_idx, best_score = None, 1.0
        for i in remaining:
            sim = max(float(np.dot(normed[i], normed[j])) for j in kept)
            if sim < best_score:
                best_score, best_idx = sim, i
        if best_idx is None or best_score >= redundancy_threshold:
            break
        kept.append(best_idx)
        remaining.discard(best_idx)

    return kept


@dataclass
class FaceDbEntry:
    """One person loaded from the face database."""

    name: str
    playernum: int | None
    provider: str
    embeddings: list[np.ndarray]          # positive embeddings
    negative_embeddings: list[np.ndarray]  # negative embeddings (may be empty)
    prototypes: list[Prototype] = field(default_factory=list)  # sub-clustered positives, see build_prototypes


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
    def load(
        cls,
        db_dir: Path,
        prototype_similarity_threshold: float = 0.62,
        allowed_names: frozenset[str] | None = None,
    ) -> "FaceDb":
        """Walk *db_dir* and load every ``face.json`` found in a sub-directory.

        Each person's positive embeddings are split into visually-cohesive
        *prototypes* (see :func:`build_prototypes`) using
        *prototype_similarity_threshold* -- this is what lets a person's
        photos span multiple visual sub-clusters (lighting, angle, glasses,
        etc.) without diluting matching into one unrepresentative average.

        *allowed_names*, if given, restricts loading to entries whose name
        matches (case-insensitive); everyone else is skipped entirely so
        their faces never become match candidates.
        """
        entries: list[FaceDbEntry] = []
        if not db_dir.is_dir():
            raise FileNotFoundError(f"Face-DB directory not found: {db_dir}")
        allowed_lower = {n.casefold() for n in allowed_names} if allowed_names else None
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
            if allowed_lower is not None and name.casefold() not in allowed_lower:
                log.debug("FaceDB [load]: %s — not in roster allow-list, skipped", name)
                continue
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
            prototypes = build_prototypes(embeddings, prototype_similarity_threshold)
            entries.append(FaceDbEntry(
                name=name,
                playernum=playernum,
                provider=provider,
                embeddings=embeddings,
                negative_embeddings=neg_embeddings,
                prototypes=prototypes,
            ))
            log.debug("FaceDB [load]: %s  playernum=%s  provider=%s  embeddings=%d  negatives=%d  prototypes=%d",
                      name, playernum, provider, len(embeddings), len(neg_embeddings), len(prototypes))
        total_faces = sum(len(e.embeddings) for e in entries)
        log.info(
            "FaceDB [load]: loaded %d cluster(s) / %d face(s) from %s  (prototype_threshold=%.3f)",
            len(entries), total_faces, db_dir, prototype_similarity_threshold,
        )
        for entry in entries:
            proto_sizes = [len(p.members) for p in entry.prototypes]
            log.info(
                "FaceDB [load]:   %-20s  playernum=%-4s  faces=%d  negatives=%d  prototypes=%d %s",
                entry.name, entry.playernum, len(entry.embeddings), len(entry.negative_embeddings),
                len(entry.prototypes), proto_sizes,
            )
        return cls(entries)


# ---------------------------------------------------------------------------
# Manual-override persistence (face_tag_ui.py delete/assign actions surviving
# a full re-cluster on incremental "import more images" runs -- see
# 1_prep_review.py's merge-into-existing-album support). FaceRecoPipeline.run
# reclusters the WHOLE album's qualified bodies on every run (simplest,
# explicitly chosen over incremental clustering), so without this, a full
# re-run would forget any face a human deleted, and could revert any face a
# human manually re-assigned to a different person than automatic
# similarity-matching would pick.
# ---------------------------------------------------------------------------

MANUAL_OVERRIDES_FILENAME = "manual_overrides.json"


def _bbox_key(box: dict | None) -> tuple[float, float, float, float] | None:
    """Hashable identity for a body_bbox dict, rounded to tolerate the tiny
    float round-tripping noise JSON (de)serialization can introduce."""
    if not box:
        return None
    return tuple(round(float(box.get(k, 0.0)), 6) for k in ("x1", "y1", "x2", "y2"))


def load_manual_overrides(out_root: Path) -> dict:
    """Load ``<out_root>/manual_overrides.json`` (deleted/assigned face
    identities recorded by face_tag_ui.py), defaulting to an empty payload
    when absent or unreadable."""
    path = out_root / MANUAL_OVERRIDES_FILENAME
    if not path.is_file():
        return {"deleted": [], "assigned": []}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data.setdefault("deleted", [])
            data.setdefault("assigned", [])
            return data
    except Exception as exc:  # noqa: BLE001
        log.warning("FaceReco: failed to read %s: %s", path, exc)
    return {"deleted": [], "assigned": []}


def record_manual_override(out_root: Path, *, deleted: dict | None = None, assigned: dict | None = None) -> None:
    """Append one deletion and/or one assignment record to
    ``<out_root>/manual_overrides.json``. Called by face_tag_ui.py whenever a
    commit deletes or (re-)assigns a face, so a future FaceRecoPipeline.run
    (triggered by importing more images) preserves that human decision
    instead of reverting it via fresh automatic clustering/matching.

    ``deleted``/``assigned`` are ``{"file":, "body_bbox": {...}}`` (assigned
    additionally carries ``"name"``/``"playernum"``).
    """
    payload = load_manual_overrides(out_root)
    if deleted is not None:
        payload["deleted"].append(deleted)
    if assigned is not None:
        # An assignment supersedes any earlier assignment of the same face.
        key = (assigned.get("file"), _bbox_key(assigned.get("body_bbox")))
        payload["assigned"] = [
            a for a in payload["assigned"]
            if (a.get("file"), _bbox_key(a.get("body_bbox"))) != key
        ]
        payload["assigned"].append(assigned)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / MANUAL_OVERRIDES_FILENAME, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _load_named_cluster_entries(clusters_dir: Path, prototype_similarity_threshold: float) -> list[FaceDbEntry]:
    """Load only the already-NAMED (human-tagged) person sub-directories of
    an album's own ``.FaceReco`` output as :class:`FaceDbEntry` objects,
    skipping pending numeric clusters and internal dot-prefixed folders
    (``.AllFaces``, ``.debug``).

    Feeding these back into face-DB matching on a re-run lets an album's own
    previously-tagged people be re-matched by similarity to the SAME name
    (and therefore the same output folder) even without an external
    ``--face-db``, so tagging survives "recluster whole album" re-runs.
    """
    entries: list[FaceDbEntry] = []
    if not clusters_dir.is_dir():
        return entries
    for person_dir in sorted(clusters_dir.iterdir()):
        if not person_dir.is_dir() or person_dir.name.startswith(".") or person_dir.name.isdigit():
            continue
        face_json = person_dir / "face.json"
        if not face_json.exists():
            continue
        try:
            with open(face_json, encoding="utf-8-sig") as fh:
                data = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            log.warning("FaceReco [local-db]: %s — failed to parse face.json: %s", person_dir.name, exc)
            continue
        name = data.get("name") or person_dir.name
        embeddings = [e for e in (_decode_embedding(f) for f in data.get("faces", [])) if e is not None]
        if not embeddings:
            continue
        entries.append(FaceDbEntry(
            name=name,
            playernum=data.get("playernum"),
            provider=str(data.get("provider", "")),
            embeddings=embeddings,
            negative_embeddings=[],
            prototypes=build_prototypes(embeddings, prototype_similarity_threshold),
        ))
    return entries


@dataclass
class _QualifiedBody:
    image_path: Path
    body: BodyRecord


class FaceRecoPipeline:
    def __init__(self, provider: FaceRecoProvider, config: FaceRecoConfig, cpu_only: bool) -> None:
        self.provider = provider
        self.config = config
        self.cpu_only = cpu_only
        self._effective_face_db_match_threshold = (
            config.face_db_match_threshold if config.face_db_match_threshold is not None else 0.72
        )
        self._effective_face_db_match_margin = (
            config.face_db_match_margin if config.face_db_match_margin is not None else 0.05
        )

    def _load_calibration(self, db_dir: Path) -> dict:
        path = db_dir / self.config.calibration_file_name
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                return payload
        except Exception as exc:  # noqa: BLE001
            log.warning("FaceReco: failed to read calibration file %s: %s", path, exc)
        return {}

    def run(self, prep_output_dir: Path) -> Path:
        prep_output_dir = prep_output_dir.resolve()
        results_path = prep_output_dir / "album.json"
        info_path = prep_output_dir / "info.json"

        log.info("FaceReco: starting  prep_dir=%s", prep_output_dir)
        log.info("FaceReco: face alignment %s", "ENABLED" if self.config.align_faces else "disabled")
        log.debug(
            "FaceReco config: provider=%s  cluster_threshold=%.3f  eps=%.3f  "
            "face_buffer_ratio=%.2f  align_faces=%s  output_dir=%s",
            self.provider.provider_name(),
            self.config.cluster_similarity_threshold,
            1.0 - self.config.cluster_similarity_threshold,
            self.config.face_buffer_ratio,
            self.config.align_faces,
            self.config.output_dir_name,
        )

        payload = self._load_json(results_path)
        info = self._load_json(info_path)
        src_dir = Path(info.get("SrcDir", "")).resolve() if info.get("SrcDir") else prep_output_dir
        log.debug("FaceReco: src_dir=%s", src_dir)

        out_root = prep_output_dir / self.config.output_dir_name
        out_root.mkdir(parents=True, exist_ok=True)
        log.debug("FaceReco: output root=%s", out_root)

        # Every run reclusters the WHOLE album's qualified bodies from
        # scratch (simplest strategy, chosen so "import more images" doesn't
        # need incremental-clustering logic) -- these previously-recorded
        # human decisions (face_tag_ui.py delete/assign) are replayed on top
        # of that fresh clustering so they're never silently undone.
        overrides = load_manual_overrides(out_root)
        deleted_keys = {
            (o.get("file"), _bbox_key(o.get("body_bbox")))
            for o in overrides.get("deleted", []) if o.get("file")
        }
        assigned_overrides = {
            (o.get("file"), _bbox_key(o.get("body_bbox"))): {"name": o.get("name"), "playernum": o.get("playernum")}
            for o in overrides.get("assigned", []) if o.get("file") and o.get("name")
        }
        if deleted_keys:
            log.info("FaceReco: %d previously-deleted face(s) excluded from this run", len(deleted_keys))
        if assigned_overrides:
            log.info("FaceReco: %d manually-assigned face(s) will be pinned to their tagged person", len(assigned_overrides))

        qualified = self._collect_qualified_bodies(payload, excluded=deleted_keys)

        debug_dir: Path | None = None
        if self.config.debug_align:
            debug_dir = out_root / ".debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            log.info("FaceReco: alignment debug images → %s", debug_dir)

        if not qualified:
            log.warning("FaceReco: no qualified (sharp) bodies found — empty folder created: %s", out_root)
            return out_root

        # Load face DB before clustering so identity info is available for matching.
        face_db: FaceDb | None = None
        effective_match_threshold = (
            self.config.face_db_match_threshold
            if self.config.face_db_match_threshold is not None
            else 0.72
        )
        effective_match_margin = (
            self.config.face_db_match_margin
            if self.config.face_db_match_margin is not None
            else 0.05
        )
        effective_proto_threshold = (
            self.config.face_db_prototype_threshold
            if self.config.face_db_prototype_threshold is not None
            else 0.62
        )
        if self.config.face_db_dir is not None:
            provider_name = self.provider.provider_name()
            cal_provider: dict | None = None
            if self.config.use_face_db_calibration:
                cal = self._load_calibration(self.config.face_db_dir)
                providers = cal.get("providers") if isinstance(cal, dict) else None
                if isinstance(providers, dict):
                    candidate = providers.get(provider_name)
                    if isinstance(candidate, dict):
                        # Auto-apply calibration only when the calibration run
                        # itself indicates usable separation. A very high EER
                        # or huge confusion rate usually means the DB is
                        # contaminated/mislabeled; blindly importing those
                        # numbers can make matching worse.
                        eer = candidate.get("equal_error_rate")
                        confusions = candidate.get("confusions_count")
                        usable = candidate.get("usable_samples")
                        quality_ok = True
                        if isinstance(eer, (int, float)) and float(eer) > 0.25:
                            quality_ok = False
                        if isinstance(confusions, int) and isinstance(usable, int) and usable > 0:
                            if (confusions / usable) > 0.25:
                                quality_ok = False

                        if quality_ok:
                            cal_provider = candidate
                            if self.config.face_db_prototype_threshold is None and isinstance(candidate.get("prototype_threshold"), (int, float)):
                                effective_proto_threshold = float(candidate["prototype_threshold"])
                            if self.config.face_db_match_threshold is None and isinstance(candidate.get("recommended_match_threshold"), (int, float)):
                                effective_match_threshold = float(candidate["recommended_match_threshold"])
                            if self.config.face_db_match_margin is None and isinstance(candidate.get("recommended_match_margin"), (int, float)):
                                effective_match_margin = float(candidate["recommended_match_margin"])
                        else:
                            log.warning(
                                "FaceReco: calibration for provider=%s looks unreliable "
                                "(eer=%s confusions=%s usable=%s) — ignoring it; "
                                "using explicit/default thresholds instead",
                                provider_name, eer, confusions, usable,
                            )

            face_db = FaceDb.load(
                self.config.face_db_dir,
                prototype_similarity_threshold=effective_proto_threshold,
                allowed_names=self.config.face_db_allowed_names,
            )
            log.info("FaceReco: face DB loaded — %d person(s) from %s",
                     len(face_db), self.config.face_db_dir)
            if cal_provider is not None:
                log.info(
                    "FaceReco: using calibration for provider=%s  match_threshold=%.3f  match_margin=%.3f  prototype_threshold=%.3f",
                    provider_name, effective_match_threshold, effective_match_margin, effective_proto_threshold,
                )

        # Store effective thresholds resolved above (defaults and optional
        # calibration import) so matcher code stays focused on matching logic.
        self._effective_face_db_match_threshold = effective_match_threshold
        self._effective_face_db_match_margin = effective_match_margin

        # Feed the album's OWN already-tagged (named) clusters back into
        # matching -- this is what lets manual face_tag_ui.py tagging survive
        # a full recluster even when no external --face-db is configured, by
        # re-matching those faces to the same name (and therefore the same
        # output folder) via ordinary similarity matching.
        local_entries = _load_named_cluster_entries(out_root, effective_proto_threshold)
        if local_entries:
            log.info(
                "FaceReco: %d already-tagged person(s) found in this album's own %s — "
                "matching against them too so existing tags are preserved",
                len(local_entries), self.config.output_dir_name,
            )
            face_db = FaceDb((face_db.entries if face_db is not None else []) + local_entries)

        log.info("FaceReco: %d qualified bodies collected", len(qualified))
        samples = self._predict_samples(qualified, debug_dir)
        log.info("FaceReco: %d/%d embeddings extracted successfully", len(samples), len(qualified))

        # Strategy: match each face directly against the face DB first, then
        # cluster only the faces that matched nobody.  This avoids letting a
        # clustering mistake break DB matching for a whole bucket.
        cluster_name_map: dict[int, FaceDbEntry] = {}
        matched_clusters: list[Cluster] = []
        unmatched = samples
        if face_db is not None:
            matched_clusters, cluster_name_map, unmatched = self._match_samples_to_facedb(
                samples, face_db,
            )
            log.info(
                "FaceReco: face-DB direct match — %d face(s) matched to %d person(s), %d unmatched",
                sum(len(c.samples) for c in matched_clusters), len(matched_clusters), len(unmatched),
            )

        next_id = max((c.cluster_id for c in matched_clusters), default=-1) + 1
        residual_clusters = self._cluster_samples(unmatched, start_id=next_id)

        # Sort unmatched clusters by size (most faces first) and renumber
        # them accordingly, so the biggest new-person buckets get the
        # lowest cluster ids/folder names — easiest to spot when reviewing.
        residual_clusters.sort(key=lambda c: len(c.samples), reverse=True)
        for offset, cluster in enumerate(residual_clusters):
            cluster.cluster_id = next_id + offset

        clusters = matched_clusters + residual_clusters
        log.info(
            "FaceReco: %d cluster(s) total — %d matched person bucket(s) + %d new cluster(s)  (sizes: %s)",
            len(clusters), len(matched_clusters), len(residual_clusters),
            [len(c.samples) for c in clusters],
        )

        clusters, cluster_name_map = self._apply_manual_overrides(clusters, cluster_name_map, assigned_overrides)

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

    def _collect_qualified_bodies(
        self,
        payload: dict,
        excluded: set[tuple] | None = None,
    ) -> list[_QualifiedBody]:
        qualified: list[_QualifiedBody] = []
        excluded = excluded or set()
        excluded_count = 0
        total_results = len(payload.get("results", []))
        total_bodies = skipped_blurry = skipped_no_ann = 0
        # Capped at 0.5 regardless of how strict the album's own sensitivity
        # setting is -- see FaceRecoConfig.sensitivity_threshold.
        min_face_sharpness = min(self.config.sensitivity_threshold, 0.4)
        log.debug("FaceReco [collect]: scanning %d result entries  (min_face_sharpness=%.3f)",
                  total_results, min_face_sharpness)
        for result in payload.get("results", []):
            image_path = Path(result.get("file", ""))
            # Disambiguated bookkeeping key (see algo/utils.py::make_unique_import_key)
            # -- falls back to the plain filename for older albums written
            # before multi-source-directory import support existed.
            image_key = result.get("key") or image_path.name
            ann = result.get("annotation_data")
            if ann is None:
                skipped_no_ann += 1
                log.debug("FaceReco [collect]: %s — no annotation_data, skipped", image_key)
                continue
            evaluated = ann.get("evaluated", [])
            log.debug("FaceReco [collect]: %s — %d evaluated body/bodies", image_key, len(evaluated))
            for idx, body_data in enumerate(evaluated):
                total_bodies += 1
                if (image_key, _bbox_key(body_data.get("body_bbox"))) in excluded:
                    excluded_count += 1
                    log.debug("FaceReco [collect]: %s body#%d — excluded (previously deleted), skipped", image_key, idx)
                    continue
                sharpness = body_data.get("sharpness_score", 0.0)
                cloth = body_data.get("cloth_color", "N/A")
                if sharpness <= min_face_sharpness:
                    skipped_blurry += 1
                    log.debug(
                        "FaceReco [collect]: %s body#%d  score=%.3f  color=%s  → SKIP (<= min_face_sharpness %.3f)",
                        image_key, idx, sharpness, cloth, min_face_sharpness,
                    )
                    continue
                log.debug(
                    "FaceReco [collect]: %s body#%d  score=%.3f  color=%s  → QUALIFIED",
                    image_key, idx, sharpness, cloth,
                )
                body = self._parse_body_record(image_key, idx, body_data)
                qualified.append(_QualifiedBody(image_path=image_path, body=body))
        if excluded_count:
            log.info("FaceReco [collect]: %d previously-deleted face(s) excluded", excluded_count)
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

    def _predict_samples(
        self,
        qualified: list[_QualifiedBody],
        debug_dir: Path | None = None,
    ) -> list[tuple[_QualifiedBody, Player, np.ndarray, np.ndarray]]:
        predicted: list[tuple[_QualifiedBody, Player, np.ndarray, np.ndarray]] = []
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
            # Crop the face area first (same "face preview" rules used for the
            # saved crop), then re-detect landmarks on the crop and embed from
            # it.  Embedding from the exact crop we save guarantees that a
            # later RebuildFaceDB run reproduces this embedding, so prediction
            # and face-DB embeddings live in the same space.
            crop = self._crop_face_with_buffer(image, item.body)
            if crop is None:
                skipped_crop += 1
                log.debug("FaceReco [embed]: %s — face crop returned None, skipped", tag)
                continue
            # Quality gate: a crop too small to trust is excluded entirely --
            # never matched against the DB and never folded into residual
            # clustering -- rather than letting a noisy embedding produce a
            # confident-looking wrong answer either way.
            short_edge = min(crop.shape[0], crop.shape[1])
            if short_edge < self.config.min_face_crop_px:
                skipped_crop += 1
                log.debug(
                    "FaceReco [embed]: %s — crop too small (%dpx < %dpx min), skipped",
                    tag, short_edge, self.config.min_face_crop_px,
                )
                continue
            result = embed_face_crop(
                self.provider,
                load_face_model(force_cpu=self.cpu_only, engine=self.config.engine),
                crop,
                fallback_confidence=item.body.confidence,
                align=self.config.align_faces,
                collect_debug=True,
            )
            player, debug = result
            if debug_dir is not None:
                self._write_align_debug(debug_dir, item, crop, debug)
            embedding = player.internal.get("embedding")
            if embedding is None:
                skipped_embedding += 1
                log.debug("FaceReco [embed]: %s — no embedding returned by provider, skipped", tag)
                continue
            annotated = annotate_face_crop(
                crop,
                debug.get("face_bbox"),
                debug.get("narrow_face_bbox"),
                debug.get("landmarks_px", []),
                player.confidence,
            )
            emb_arr = np.asarray(embedding, dtype=np.float32)
            log.debug(
                "FaceReco [embed]: %s — embedding dim=%d  norm=%.4f  provider_conf=%s",
                tag, emb_arr.shape[0], float(np.linalg.norm(emb_arr)),
                f"{player.confidence:.3f}" if player.confidence is not None else "n/a",
            )
            log.debug("FaceReco [embed]: %s — crop %dx%d  ✓", tag, crop.shape[1], crop.shape[0])
            predicted.append((item, player, crop, annotated))
            if index == 1 or index % 50 == 0 or index == total:
                log.info("FaceReco [embed]: %d/%d processed  embedded=%d", index, total, len(predicted))
        log.info(
            "FaceReco [embed]: done  embedded=%d  skipped_load=%d  "
            "skipped_embedding=%d  skipped_crop=%d",
            len(predicted), skipped_load, skipped_embedding, skipped_crop,
        )
        return predicted

    def _write_align_debug(
        self,
        debug_dir: Path,
        item: _QualifiedBody,
        crop: np.ndarray,
        debug: dict,
    ) -> None:
        """Write one alignment QA image (annotated crop + aligned face)."""
        try:
            vis = make_alignment_debug_image(
                crop, debug.get("landmarks_px", []), debug.get("aligned")
            )
            name = f"{Path(item.image_path.name).stem}-b{item.body.body_index}.png"
            cv2.imwrite(str(debug_dir / name), vis, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        except Exception as exc:  # noqa: BLE001 — debug output must never break the run
            log.debug("FaceReco [debug]: failed to write alignment debug for %s body#%d: %s",
                      item.image_path.name, item.body.body_index, exc)

    def _cluster_samples(
        self,
        predicted: list[tuple[_QualifiedBody, Player, np.ndarray, np.ndarray]],
        start_id: int = 0,
    ) -> list[Cluster]:
        """Cluster the *residual* (DB-unmatched) face embeddings by cosine distance.

        Runs only on the faces that did not match any face-DB person, grouping
        the genuinely unknown faces into new clusters.  Cluster ids start at
        *start_id* so they do not collide with the matched person buckets that
        were numbered before this call.

        Uses **average-linkage** agglomerative clustering rather than DBSCAN.
        DBSCAN with ``min_samples=1`` performs single-link clustering, which is
        prone to *chaining*: two unrelated people get merged into one cluster as
        soon as a single intermediate "bridge" face sits within ``eps`` of both.
        On a roster of similar-looking athletes this collapses most faces into a
        single giant bucket.  Average linkage requires the *mean* pairwise
        distance between two groups to fall below the threshold before they
        merge, which resists chaining while still assigning every face to a
        cluster.

        ``distance_threshold = 1 - cluster_similarity_threshold`` converts the
        configured cosine *similarity* threshold into a cosine *distance*
        threshold.
        """
        if not predicted:
            log.info("FaceReco [cluster]: no residual faces to cluster")
            return []

        # Single, intentional normalisation step.  FaceNet (InceptionResnetV1)
        # already L2-normalises its output, so this is effectively a no-op for
        # those embeddings, but it guarantees unit vectors regardless of the
        # provider -- every downstream comparison (clustering here AND face-DB
        # matching) then operates on unit vectors.  There is no distorting
        # "double normalisation": re-normalising a unit vector is mathematically
        # a no-op, and cosine distance is scale-invariant either way.
        embeddings: list[np.ndarray] = [
            self._normalize(np.asarray(player.internal["embedding"], dtype=np.float32))
            for _item, player, _crop, _annotated in predicted
        ]
        matrix = np.array(embeddings, dtype=np.float32)
        n_samples = len(embeddings)

        eps = max(1e-6, 1.0 - self.config.cluster_similarity_threshold)
        if n_samples < 2:
            # AgglomerativeClustering needs >= 2 samples; trivially one cluster.
            labels = np.zeros(n_samples, dtype=int)
            log.info("FaceReco [cluster]: %d sample(s) -- single cluster", n_samples)
        else:
            log.info(
                "FaceReco [cluster]: agglomerative  n_samples=%d  distance_threshold=%.4f  "
                "(similarity_threshold=%.3f)  metric=cosine  linkage=average",
                n_samples, eps, self.config.cluster_similarity_threshold,
            )
            labels = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=eps,
            ).fit_predict(matrix)

        unique_labels = sorted(set(labels.tolist()))
        log.info("FaceReco [cluster]: produced %d cluster(s)  labels=%s",
                 len(unique_labels), unique_labels)

        cluster_map: dict[int, Cluster] = {}
        for i, (label, (item, player, crop, annotated)) in enumerate(zip(labels.tolist(), predicted)):
            cid = int(label) + start_id
            if cid not in cluster_map:
                cluster_map[cid] = Cluster(cluster_id=cid, samples=[])
            sample = FaceSample(
                body=item.body,
                embedding=embeddings[i],
                confidence=player.confidence,
                crop_file_name="",
                crop_image=crop,
                annotated_image=annotated,
            )
            cluster_map[cid].samples.append(sample)
            log.debug(
                "FaceReco [cluster]: %s body#%d → cluster %d",
                item.body.orig_filename, item.body.body_index, cid,
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
        return normalize_embedding(vector)

    def _best_prototype_match(self, emb: np.ndarray, entry: FaceDbEntry) -> float:
        """Best cosine similarity between *emb* and any of *entry*'s prototypes.

        Each person's positive embeddings were split into visually-cohesive
        prototypes when the DB was loaded (see :func:`build_prototypes`).
        Scoring against the single closest prototype -- rather than averaging
        across all of a person's embeddings -- means a query only has to
        resemble ONE genuine sub-cluster (e.g. "with glasses"), and is never
        dragged down (or up) by averaging with a visually unrelated sub-
        cluster of the same person.
        """
        if not entry.prototypes:
            return -1.0
        return max(float(np.dot(emb, proto.centroid)) for proto in entry.prototypes)

    def _match_samples_to_facedb(
        self,
        predicted: list[tuple[_QualifiedBody, Player, np.ndarray, np.ndarray]],
        face_db: FaceDb,
    ) -> tuple[list[Cluster], dict[int, FaceDbEntry], list[tuple[_QualifiedBody, Player, np.ndarray, np.ndarray]]]:
        """Assign each face directly to a face-DB person, before any clustering.

        Every face is compared independently against every person's closest
        *prototype* (visual sub-cluster -- see :func:`build_prototypes`).  A
        face is matched to the best-scoring person only when ALL of the
        following hold:

          1. ``best_score >= config.face_db_match_threshold`` -- the absolute
             similarity floor.
          2. ``best_score - second_best_score >= config.face_db_match_margin``
             -- the winner must clearly beat the best-scoring *different*
             person, not just edge them out.  A near-tie between two
             candidates is precisely the situation that causes faces to get
             mixed up between similar-looking people; when the margin isn't
             met the face is left unmatched rather than guessed.
          3. Not vetoed by the winning person's curated negative examples --
             a face resembling those non-match examples at least as strongly
             as its positive match is a known look-alike.

        Matched faces are grouped into one :class:`Cluster` per person
        (numbered ``0..M-1`` in discovery order); faces that match nobody --
        or that fail the margin check, or are vetoed -- are returned in
        *unmatched* so they can be clustered afterwards.

        Only DB entries produced by the same provider as the current pipeline
        are considered, so that embedding spaces are compatible.

        Returns ``(matched_clusters, cluster_name_map, unmatched)``.
        """
        provider_name = self.provider.provider_name()
        compatible = [e for e in face_db.entries if e.provider == provider_name]
        if not compatible:
            log.warning(
                "FaceReco [match]: no face-DB entries for provider '%s' — all faces will be clustered",
                provider_name,
            )
            return [], {}, predicted

        # Pre-normalise negative embeddings once to avoid repeated work.
        # Positive embeddings are already unit vectors inside each entry's
        # prototypes (built at DB-load time), so only negatives need it here.
        normalised_negs: dict[str, list[np.ndarray]] = {
            entry.name: [self._normalize(emb.astype(np.float32)) for emb in entry.negative_embeddings]
            for entry in compatible
        }

        threshold = self._effective_face_db_match_threshold
        margin_floor = self._effective_face_db_match_margin
        # Preserve discovery order of people for stable cluster numbering.
        person_samples: dict[str, list[FaceSample]] = {}
        person_entry: dict[str, FaceDbEntry] = {}
        unmatched: list[tuple[_QualifiedBody, Player, np.ndarray, np.ndarray]] = []

        for item, player, crop, annotated in predicted:
            emb = self._normalize(np.asarray(player.internal["embedding"], dtype=np.float32))
            tag = f"{item.body.orig_filename} body#{item.body.body_index}"

            # Score against EVERY person, not just track a running best — the
            # gap between the #1 and #2 candidate (the margin) matters as
            # much as the absolute score for telling similar people apart.
            scored = sorted(
                ((self._best_prototype_match(emb, entry), entry) for entry in compatible),
                key=lambda t: t[0], reverse=True,
            )
            best_score, best_entry = scored[0]
            second_score, second_entry = scored[1] if len(scored) > 1 else (-1.0, None)
            margin = best_score - second_score

            if best_score < threshold:
                unmatched.append((item, player, crop, annotated))
                log.debug("FaceReco [match]: %s — no DB match (best_sim=%.4f < %.3f)",
                          tag, best_score, threshold)
                continue

            if margin < margin_floor:
                unmatched.append((item, player, crop, annotated))
                log.debug(
                    "FaceReco [match]: %s — ambiguous: %s=%.4f vs %s=%.4f (margin=%.4f < %.3f) → unmatched",
                    tag, best_entry.name, best_score,
                    second_entry.name if second_entry is not None else "n/a", second_score,
                    margin, margin_floor,
                )
                continue

            # Negative veto: a face resembling the person's curated non-match
            # examples at least as much as its positive match is a look-alike.
            best_negs = normalised_negs.get(best_entry.name, [])
            neg_sim = max((float(np.dot(emb, ne)) for ne in best_negs), default=-1.0)
            if neg_sim >= best_score:
                unmatched.append((item, player, crop, annotated))
                log.debug(
                    "FaceReco [match]: %s — vetoed by negative of %s (pos_sim=%.4f neg_sim=%.4f) → unmatched",
                    tag, best_entry.name, best_score, neg_sim,
                )
                continue

            sample = FaceSample(
                body=item.body,
                embedding=emb,
                confidence=player.confidence,
                crop_file_name="",
                crop_image=crop,
                annotated_image=annotated,
                match_score=best_score,
                match_margin=margin,
                match_runner_up=second_entry.name if second_entry is not None else None,
            )
            person_samples.setdefault(best_entry.name, []).append(sample)
            person_entry.setdefault(best_entry.name, best_entry)
            log.debug("FaceReco [match]: %s → %s (sim=%.4f margin=%.4f)", tag, best_entry.name, best_score, margin)

        matched_clusters: list[Cluster] = []
        cluster_name_map: dict[int, FaceDbEntry] = {}
        for cid, (name, samples) in enumerate(person_samples.items()):
            entry = person_entry[name]
            matched_clusters.append(Cluster(cluster_id=cid, samples=samples))
            cluster_name_map[cid] = entry
            log.info(
                "FaceReco [match]: person %04d → %-20s (playernum=%s) — %d face(s)",
                cid, name, entry.playernum, len(samples),
            )
        return matched_clusters, cluster_name_map, unmatched

    def _apply_manual_overrides(
        self,
        clusters: list[Cluster],
        cluster_name_map: dict[int, FaceDbEntry],
        assigned: dict[tuple, dict],
    ) -> tuple[list[Cluster], dict[int, FaceDbEntry]]:
        """Force any face_tag_ui.py-assigned face into its pinned person's
        cluster, regardless of what this run's automatic matching/clustering
        decided -- see the ``manual_overrides.json`` design notes above
        :data:`MANUAL_OVERRIDES_FILENAME`.
        """
        if not assigned:
            return clusters, cluster_name_map

        remaining: dict[int, Cluster] = {c.cluster_id: c for c in clusters}
        by_target_name: dict[str, list[FaceSample]] = {}
        name_playernum: dict[str, int | None] = {}
        for cluster in remaining.values():
            kept: list[FaceSample] = []
            for sample in cluster.samples:
                key = (sample.body.orig_filename, _bbox_key(sample.body.raw_body.get("body_bbox")))
                override = assigned.get(key)
                if override is not None:
                    by_target_name.setdefault(override["name"], []).append(sample)
                    name_playernum[override["name"]] = override.get("playernum")
                else:
                    kept.append(sample)
            cluster.samples = kept

        if not by_target_name:
            return list(remaining.values()), cluster_name_map

        next_id = max((c.cluster_id for c in remaining.values()), default=-1) + 1
        for name, samples in by_target_name.items():
            target_cid = next(
                (cid for cid, entry in cluster_name_map.items() if entry.name == name and cid in remaining),
                None,
            )
            if target_cid is not None:
                remaining[target_cid].samples.extend(samples)
                log.info("FaceReco [override]: pinned %d face(s) into existing cluster for %s", len(samples), name)
            else:
                cid = next_id
                next_id += 1
                entry = FaceDbEntry(
                    name=name, playernum=name_playernum.get(name), provider=self.provider.provider_name(),
                    embeddings=[], negative_embeddings=[], prototypes=[],
                )
                remaining[cid] = Cluster(cluster_id=cid, samples=samples)
                cluster_name_map[cid] = entry
                log.info("FaceReco [override]: pinned %d face(s) into a new cluster for %s", len(samples), name)

        clusters_out = [c for c in remaining.values() if c.samples]
        return clusters_out, cluster_name_map

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

                # Lossless PNG: the saved crop must reproduce this run's
                # embedding exactly when the DB is later rebuilt from it.
                # Deliberately independent of cluster_num (unstable across
                # reruns -- see FaceRecoPipeline.run's manual-overrides notes)
                # so a full recluster overwrites the same file in place instead
                # of leaving the previous run's crop behind as an orphan.
                crop_name = f"{Path(sample.body.orig_filename).stem}-b{sample.body.body_index}.png"
                cv2.imwrite(
                    str(all_faces_dir / crop_name), crop,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3],
                )
                written += 1

                emb = sample.embedding.astype(np.float32)
                face_entries.append({
                    "origFilename": sample.body.orig_filename,
                    "cluster": cluster_num,
                    "cropFileName": crop_name,
                    "confidence": sample.confidence,
                    "matchScore": sample.match_score,
                    "matchMargin": sample.match_margin,
                    "matchRunnerUp": sample.match_runner_up,
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
            "aligned": self.config.align_faces,
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
        log.info("FaceReco [write]: writing %d cluster(s) to %s", len(clusters), out_root)
        for cluster_index, cluster in enumerate(clusters, start=1):
            # The 4-digit numeric name is always used for internal file naming.
            cluster_num = f"{cluster.cluster_id:04d}"
            db_entry = (cluster_name_map or {}).get(cluster.cluster_id)
            # Use the person's name as the directory name when matched.
            dir_name = db_entry.name if db_entry is not None else cluster_num
            cluster_dir = out_root / dir_name
            face_dir = cluster_dir / "Face"
            face_annotated_dir = cluster_dir / "Face.annotated"
            negative_dir = cluster_dir / "Negative"
            face_dir.mkdir(parents=True, exist_ok=True)
            negative_dir.mkdir(parents=True, exist_ok=True)

            written = 0
            for sample in cluster.samples:
                crop = sample.crop_image
                annotated = sample.annotated_image
                if crop is None:
                    log.debug("FaceReco [write]: cluster %s — %s body#%d has no crop image",
                              cluster_num, sample.body.orig_filename, sample.body.body_index)
                    continue

                # Lossless PNG: this crop becomes the face-DB image, so it must
                # reproduce the prediction embedding exactly on rebuild.
                # Deliberately independent of cluster_num (unstable across
                # reruns) so a full recluster overwrites the same file in
                # place instead of leaving the previous run's crop behind.
                crop_name = f"{Path(sample.body.orig_filename).stem}-b{sample.body.body_index}.png"
                cv2.imwrite(str(face_dir / crop_name), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                if annotated is not None:
                    face_annotated_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(face_annotated_dir / crop_name),
                        annotated,
                        [cv2.IMWRITE_PNG_COMPRESSION, 3],
                    )
                sample.crop_file_name = crop_name
                sample.crop_image = None
                sample.annotated_image = None
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

        # Negative folders are left empty here.  They are reserved for manually
        # curated counter-examples in the face DB; prediction must not fill them
        # with crops from other clusters (the FaceDB contains no such images, so
        # auto-populating them would make prediction output inconsistent with a
        # rebuilt DB and pollute the negative set with mislabelled faces).

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
                    "matchScore": sample.match_score,
                    "matchMargin": sample.match_margin,
                    "matchRunnerUp": sample.match_runner_up,
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
            "aligned": self.config.align_faces,
            "faces": faces,
        }
        with open(cluster_dir / "face.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
