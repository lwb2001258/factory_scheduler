"""Preservation, review gates and resumable protocol for the isolated SARSA v3 run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "results/sarsa_v3_20261026"
OLD_RUN = ROOT / "results/all_scheduler_model_remediation/formal_1800_dual_v6"
VERSION = "sarsa-v3-workflow-v1"
STEPS = (
    "preservation_and_workflow",
    "complete_pair_features_and_traffic",
    "shared_value_sarsa_lambda",
    "runtime_events_wait_and_frozen_evaluation",
    "training_and_physical_workflow",
    "standalone_training",
    "webots_fine_tuning",
    "six_seed_evaluation_and_preservation_audit",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def confined(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"output must be inside the isolated run: {path}")
    return path


def replace_with_retry(source, destination):
    """Readers/virus scanners may briefly deny Windows atomic replacement."""
    for attempt in range(10):
        try:
            Path(source).replace(destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(.05 * 2 ** attempt, .5))


def write(root, path, value):
    path = confined(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    replace_with_retry(temporary, path)


def validate_root(root):
    root = Path(root).resolve()
    if root == OLD_RUN.resolve() or root.is_relative_to(OLD_RUN.resolve()):
        raise ValueError("the original result tree is read-only")
    protocol = read(root / "workflow.json")
    if not protocol or protocol.get("version") != VERSION:
        raise ValueError("initialize the isolated SARSA workflow first")
    if protocol.get("run_root") != str(root):
        raise ValueError("workflow belongs to a different output directory")
    return root


def initialize(root=DEFAULT_RUN):
    root = Path(root).resolve()
    if root.parent != (ROOT / "results").resolve():
        raise ValueError("new runs must be direct children of this project's results directory")
    if root.exists():
        return read(validate_root(root) / "workflow.json")
    root.mkdir(parents=True, exist_ok=False)
    source_files = sorted({p for directory in (ROOT / "controllers", ROOT / "scripts")
                           for p in directory.rglob("*.py") if "__pycache__" not in p.parts})
    sources = {}
    for path in source_files:
        relative = path.relative_to(ROOT).as_posix()
        destination = confined(root, root / "baseline/source" / relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        sources[relative] = sha256(path)
    # Inventory all pre-existing results without touching their contents. Selected
    # SARSA baselines and final result files additionally receive content hashes.
    inventory = []
    protected = {}
    for path in (ROOT / "results").rglob("*"):
        if not path.is_file() or path.is_relative_to(root):
            continue
        stat = path.stat()
        relative = path.relative_to(ROOT).as_posix()
        inventory.append({"path": relative, "size": stat.st_size,
                          "mtime_ns": stat.st_mtime_ns})
        if "SARSA_" in relative and (path.name in {
                "strict_1800_model.json", "webots_selected_model.json",
                "webots_finetuned_model.json"} or
                ("/evaluation/episodes/" in relative and path.name == "result.json")):
            protected[relative] = sha256(path)
    write(root, root / "audit/original_files.json", inventory)
    write(root, root / "audit/protected_sha256.json", protected)
    write(root, root / "audit/original_source_sha256.json", sources)
    protocol = {
        "version": VERSION, "created_at": now(), "run_root": str(root),
        "objectives": ["count", "utility_v2"], "scenarios": ["A", "B", "C"],
        "episode_seconds": 1800, "max_robots": 8, "max_candidate_tasks": 20,
        "training_seeds": list(range(21000, 21200)),
        "reserved_validation_seeds": list(range(31000, 31006)),
        "webots_fine_tune_seeds": list(range(41000, 41010)),
        "evaluation_seeds": list(range(61000, 61006)),
        "intermediate_nonregression": {
            "enabled": False, "reason": "explicit user instruction 2026-10-26"},
        "evaluation_learning": False, "evaluation_fallback": False,
        "review_rounds_required": 3,
        "review_policy": "At least three documented code reviews per step; fixes require re-review.",
        "review_dimensions": ["algorithm_and_reward_semantics", "runtime_and_resume_safety",
                              "tests_and_independent_final_diff_read"],
        "preservation": "All outputs isolated; original checkpoints and results remain in place.",
        "steps": [{"step": i, "name": name, "status": "pending"}
                  for i, name in enumerate(STEPS, 1)],
    }
    write(root, root / "workflow.json", protocol)
    return protocol


def require_reviews(root, through_step, *, verify_sources=True):
    root = validate_root(root)
    latest = {}
    for step in range(1, through_step + 1):
        receipt = read(root / "audit" / f"step_{step:02d}_reviews.json") or {}
        reviews = receipt.get("reviews", [])
        if len(reviews) < 3 or any(row.get("status") != "pass" for row in reviews[-3:]):
            raise RuntimeError(f"step {step} requires three passing review rounds")
        if len({row.get("round") for row in reviews}) != len(reviews):
            raise RuntimeError("duplicate review round")
        for row in reviews[-3:]:
            if not row.get("checks") or not row.get("findings"):
                raise RuntimeError("review evidence is missing")
            if row.get("files") != receipt.get("files"):
                raise RuntimeError("three reviews must cover the same final source version")
        latest.update(receipt.get("files", {}))
    if verify_sources:
        for name, expected in latest.items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or sha256(path) != expected:
                raise RuntimeError(f"reviewed file changed: {name}")


def record_review(root, step, findings, checks, files, *, status="pass"):
    """Record an actually performed review; this function performs no review itself."""
    root = validate_root(root)
    if not 1 <= step <= len(STEPS) or status not in {"pass", "changes_required"}:
        raise ValueError("invalid review step/status")
    require_reviews(root, step - 1, verify_sources=False)
    if not findings or not checks or not files:
        raise ValueError("review requires findings, checks and source files")
    hashes = {}
    for name in files:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError("review source is outside the project or missing")
        hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
    destination = root / "audit" / f"step_{step:02d}_reviews.json"
    receipt = read(destination) or {"step": step, "reviews": []}
    receipt["reviews"].append({"round": len(receipt["reviews"]) + 1,
        "reviewed_at": now(), "reviewer": "Codex sequential self-review",
        "status": status, "findings": findings, "checks": checks, "files": hashes})
    receipt["files"] = hashes
    write(root, destination, receipt)
    return receipt


def verify_preservation(root):
    root = validate_root(root)
    protected = read(root / "audit/protected_sha256.json")
    changed_hashes = [name for name, expected in protected.items()
                      if not (ROOT / name).is_file() or sha256(ROOT / name) != expected]
    changed_metadata = []
    for row in read(root / "audit/original_files.json"):
        path = ROOT / row["path"]
        if not path.is_file():
            changed_metadata.append({"path": row["path"], "change": "missing"})
        else:
            stat = path.stat()
            if stat.st_size != row["size"] or stat.st_mtime_ns != row["mtime_ns"]:
                changed_metadata.append({"path": row["path"], "change": "metadata_changed"})
    result = {"checked_at": now(), "protected_count": len(protected),
              "changed_hashes": changed_hashes, "changed_metadata": changed_metadata}
    write(root, root / "audit/preservation_latest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "status", "verify"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    if args.command == "init":
        result = initialize(args.run_root)
        print(json.dumps({"run_root": result["run_root"], "steps": result["steps"]}))
    elif args.command == "verify":
        print(json.dumps(verify_preservation(args.run_root)))
    else:
        root = validate_root(args.run_root)
        print(json.dumps(read(root / "workflow.json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
