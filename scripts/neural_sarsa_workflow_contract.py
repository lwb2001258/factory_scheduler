"""Review gates, source pinning, and resume rules for neural SARSA v4."""
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
DEFAULT_RUN = ROOT / "results/sarsa_v4_neural_20261028"
VERSION = "sarsa-v4-neural-workflow-v1"
STEPS = (
    "baseline_and_two_design_reviews",
    "shared_graph_traffic_features",
    "masked_dueling_expected_sarsa_agent",
    "commit_aware_runtime_and_supervisor",
    "resumable_training_evaluation_workflow",
    "focused_and_full_validation",
    "standalone_training",
    "webots_fine_tuning",
    "six_seed_frozen_evaluation",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    return (json.loads(path.read_text(encoding="utf-8"))
            if path.exists() else None)


def confined(root, path):
    root = Path(root).resolve()
    path = Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"output must be inside the isolated run: {path}")
    return path


def replace_with_retry(source, destination):
    for attempt in range(10):
        try:
            Path(source).replace(destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 0.5))


def write(root, path, value):
    path = confined(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    replace_with_retry(temporary, path)


def validate_root(root):
    root = Path(root).resolve()
    if root.parent != (ROOT / "results").resolve():
        raise ValueError("run must be a direct child of the results directory")
    protocol = read(root / "workflow.json")
    if (not protocol or protocol.get("version") != VERSION or
            protocol.get("run_root") != str(root)):
        raise ValueError("initialize the neural SARSA workflow first")
    return root


def initialize(root=DEFAULT_RUN):
    root = Path(root).resolve()
    if root.parent != (ROOT / "results").resolve():
        raise ValueError("new runs must be direct children of results")
    if root.exists():
        return read(validate_root(root) / "workflow.json")
    root.mkdir(parents=True, exist_ok=False)
    source_files = sorted({
        path for directory in (ROOT / "controllers", ROOT / "scripts")
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts})
    sources = {}
    for path in source_files:
        relative = path.relative_to(ROOT).as_posix()
        destination = confined(root, root / "baseline/source" / relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        sources[relative] = sha256(path)
    protocol = {
        "version": VERSION,
        "created_at": now(),
        "run_root": str(root),
        "objectives": ["count", "utility_v2"],
        "scenarios": ["A", "B", "C"],
        "episode_seconds": 1800,
        "training_seeds": list(range(21000, 21060)),
        "webots_fine_tune_seeds": list(range(41000, 41010)),
        "evaluation_seeds": list(range(61000, 61006)),
        "nonregression_validation": False,
        "rollback": False,
        "selection": "latest cumulative checkpoint",
        "evaluation_learning": False,
        "evaluation_fallback": False,
        "review_rounds_required": 3,
        "source_sha256": sources,
        "steps": [
            {"step": index, "name": name, "status": "pending"}
            for index, name in enumerate(STEPS, 1)],
    }
    write(root, root / "workflow.json", protocol)
    return protocol


def source_hashes(files):
    result = {}
    for name in files:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"review source is missing: {name}")
        result[path.relative_to(ROOT).as_posix()] = sha256(path)
    return result


def record_review(root, step, findings, checks, files, *, status="pass"):
    root = validate_root(root)
    if (not 1 <= step <= len(STEPS) or
            status not in {"pass", "changes_required"} or
            not findings or not checks or not files):
        raise ValueError("invalid review evidence")
    if step > 1:
        require_reviews(root, step - 1)
    hashes = source_hashes(files)
    destination = root / "audit" / f"step_{step:02d}_reviews.json"
    receipt = read(destination) or {"step": step, "reviews": []}
    if receipt.get("files") not in (None, hashes):
        receipt = {"step": step, "reviews": []}
    receipt["files"] = hashes
    receipt["reviews"].append({
        "round": len(receipt["reviews"]) + 1,
        "reviewed_at": now(),
        "reviewer": "Codex sequential self-review",
        "status": status,
        "findings": findings,
        "checks": checks,
        "files": hashes,
    })
    write(root, destination, receipt)
    return receipt


def require_reviews(root, through_step, *, verify_sources=True):
    root = validate_root(root)
    for step in range(1, through_step + 1):
        receipt = read(root / "audit" / f"step_{step:02d}_reviews.json") or {}
        reviews = receipt.get("reviews", [])
        if (len(reviews) < 3 or
                any(row.get("status") != "pass" for row in reviews[-3:])):
            raise RuntimeError(f"step {step} requires three passing reviews")
        files = receipt.get("files", {})
        if any(row.get("files") != files for row in reviews[-3:]):
            raise RuntimeError("reviews do not cover identical source hashes")
        if verify_sources:
            for name, expected in files.items():
                path = (ROOT / name).resolve()
                if not path.is_file() or sha256(path) != expected:
                    raise RuntimeError(f"reviewed source changed: {name}")


def update_step(root, step, status, **values):
    root = validate_root(root)
    protocol = read(root / "workflow.json")
    row = protocol["steps"][step - 1]
    row.update(status=status, updated_at=now(), **values)
    write(root, root / "workflow.json", protocol)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "status", "verify"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    if args.command == "init":
        print(json.dumps(initialize(args.run_root), ensure_ascii=False))
    elif args.command == "verify":
        root = validate_root(args.run_root)
        require_reviews(root, 6)
        print(json.dumps({"run_root": str(root), "reviews_through": 6}))
    else:
        root = validate_root(args.run_root)
        print(json.dumps(read(root / "workflow.json"),
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
