"""Hash-chained atomic commit ledger for interruption-safe model training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Optional, Sequence


ATOMIC_RESUME_VERSION = "atomic-training-resume-v1"
_STAGES = {
    "training_episode", "validation_batch", "model_update",
    "rung", "trial",
}


def _canonical_json(document) -> str:
    return json.dumps(
        document, ensure_ascii=True, allow_nan=False,
        sort_keys=True, separators=(",", ":"))


def _sha256_document(document) -> str:
    return hashlib.sha256(_canonical_json(document).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def atomic_json(path: Path, document) -> None:
    """Durably replace one JSON document; partial temp files never commit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (_canonical_json(document)+"\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", prefix=path.name+".tmp-", dir=path.parent,
                delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ResumeUnit:
    order: int
    stage: str
    policy_id: str
    trial_id: str
    rung_index: int
    round_index: int
    scenario: Optional[str] = None
    seed: Optional[int] = None

    def __post_init__(self):
        integers = (self.order, self.rung_index, self.round_index)
        if (any(isinstance(value, bool) or not isinstance(value, int) or
                value < 0 for value in integers) or
                self.stage not in _STAGES or
                not isinstance(self.policy_id, str) or not self.policy_id or
                not isinstance(self.trial_id, str) or not self.trial_id or
                (self.scenario is not None and
                 self.scenario not in {"A", "B", "C"}) or
                (self.seed is not None and
                 (isinstance(self.seed, bool) or
                  not isinstance(self.seed, int) or self.seed < 0))):
            raise ValueError("invalid resume unit")
        if self.stage == "training_episode" and (
                self.scenario is None or self.seed is None):
            raise ValueError("training episode requires scenario and seed")

    def canonical(self) -> dict:
        return {"version": ATOMIC_RESUME_VERSION, **asdict(self)}

    @property
    def sha256(self) -> str:
        return _sha256_document(self.canonical())


class AtomicResumeLedger:
    """Validate and atomically extend the committed prefix of a fixed plan."""

    def __init__(self, root: Path, contract: Mapping[str, object]):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if not isinstance(contract, Mapping):
            raise ValueError("resume contract must be a mapping")
        self.contract = dict(contract)
        self.contract_sha256 = _sha256_document(self.contract)
        self.contract_path = self.root/"resume_contract.json"
        existing = self._read_json(self.contract_path)
        document = {
            "version": ATOMIC_RESUME_VERSION,
            "contract": self.contract,
            "contract_sha256": self.contract_sha256,
        }
        if existing is None:
            atomic_json(self.contract_path, document)
        elif existing != document:
            raise ValueError("resume contract changed for existing output root")
        self.commits = self.root/"commits"
        self.commits.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read_json(path: Path):
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None

    def marker_path(self, unit: ResumeUnit) -> Path:
        return self.commits/f"{unit.order:08d}_{unit.sha256[:16]}.json"

    def _artifact_record(self, name: str, path: Path) -> dict:
        if not isinstance(name, str) or not name:
            raise ValueError("artifact name must be nonempty")
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("committed artifact is outside ledger root") from exc
        if not resolved.is_file():
            raise ValueError(f"committed artifact is missing: {name}")
        return {
            "path": str(resolved),
            "size_bytes": resolved.stat().st_size,
            "sha256": file_sha256(resolved),
        }

    @staticmethod
    def _validate_metadata(unit: ResumeUnit, metadata) -> dict:
        if not isinstance(metadata, Mapping):
            raise ValueError("commit metadata must be a mapping")
        result = dict(metadata)
        # This also rejects NaN/Inf before any marker is written.
        _canonical_json(result)
        if result.get("status") != "complete":
            raise ValueError("only a complete unit can be committed")
        if unit.stage == "training_episode":
            if (result.get("duration_seconds") != 1800.0 or
                    result.get("fixed_horizon") is not True or
                    result.get("horizon_finalized") is not True or
                    result.get("termination_reason") != "episode_horizon" or
                    result.get("model_round_trip_valid") is not True):
                raise ValueError("training episode completion contract failed")
        if unit.stage == "model_update" and (
                result.get("model_round_trip_valid") is not True):
            raise ValueError("model update did not pass round-trip validation")
        return result

    def _build_document(self, unit: ResumeUnit, artifacts, metadata,
                        predecessor_commit_sha256) -> dict:
        if (predecessor_commit_sha256 is not None and
                not _is_sha256(predecessor_commit_sha256)):
            raise ValueError("predecessor commit hash is invalid")
        if not isinstance(artifacts, Mapping) or not artifacts:
            raise ValueError("a committed unit requires artifacts")
        records = {
            name: self._artifact_record(name, path)
            for name, path in artifacts.items()}
        document = {
            "version": ATOMIC_RESUME_VERSION,
            "contract_sha256": self.contract_sha256,
            "unit": unit.canonical(),
            "unit_sha256": unit.sha256,
            "predecessor_commit_sha256": predecessor_commit_sha256,
            "artifacts": records,
            "metadata": self._validate_metadata(unit, metadata),
        }
        document["commit_sha256"] = _sha256_document(document)
        return document

    def validate(self, unit: ResumeUnit,
                 predecessor_commit_sha256: Optional[str]) -> Optional[dict]:
        document = self._read_json(self.marker_path(unit))
        if not isinstance(document, dict):
            return None
        commit_hash = document.get("commit_sha256")
        unhashed = {key: value for key, value in document.items()
                    if key != "commit_sha256"}
        if (not _is_sha256(commit_hash) or
                _sha256_document(unhashed) != commit_hash or
                document.get("version") != ATOMIC_RESUME_VERSION or
                document.get("contract_sha256") != self.contract_sha256 or
                document.get("unit") != unit.canonical() or
                document.get("unit_sha256") != unit.sha256 or
                document.get("predecessor_commit_sha256") !=
                predecessor_commit_sha256):
            return None
        artifacts = document.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            return None
        for name, record in artifacts.items():
            if (not isinstance(name, str) or not isinstance(record, dict) or
                    not isinstance(record.get("path"), str) or
                    not isinstance(record.get("size_bytes"), int) or
                    not _is_sha256(record.get("sha256"))):
                return None
            path = Path(record["path"]).resolve()
            try:
                path.relative_to(self.root)
            except ValueError:
                return None
            if (not path.is_file() or path.stat().st_size !=
                    record["size_bytes"] or
                    file_sha256(path) != record["sha256"]):
                return None
        try:
            self._validate_metadata(unit, document.get("metadata"))
        except (TypeError, ValueError):
            return None
        return document

    def commit(self, unit: ResumeUnit, artifacts: Mapping[str, Path],
               metadata: Mapping[str, object],
               predecessor_commit_sha256: Optional[str]) -> dict:
        existing = self.validate(unit, predecessor_commit_sha256)
        if existing is not None:
            return {**existing, "reused": True}
        document = self._build_document(
            unit, artifacts, metadata, predecessor_commit_sha256)
        atomic_json(self.marker_path(unit), document)
        verified = self.validate(unit, predecessor_commit_sha256)
        if verified is None:
            raise RuntimeError("atomic unit commit failed post-write validation")
        return {**verified, "reused": False}

    def first_incomplete(self, units: Sequence[ResumeUnit]
                         ) -> Optional[ResumeUnit]:
        if (not isinstance(units, Sequence) or
                any(not isinstance(unit, ResumeUnit) for unit in units)):
            raise ValueError("resume plan must be a sequence")
        orders = [unit.order for unit in units]
        if (orders != sorted(orders) or len(orders) != len(set(orders))):
            raise ValueError("resume plan order must be unique and increasing")
        predecessor = None
        for unit in units:
            document = self.validate(unit, predecessor)
            if document is None:
                return unit
            predecessor = document["commit_sha256"]
        return None

    def committed_prefix(self, units: Sequence[ResumeUnit]) -> int:
        incomplete = self.first_incomplete(units)
        return len(units) if incomplete is None else units.index(incomplete)
