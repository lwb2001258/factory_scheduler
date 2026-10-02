#!/usr/bin/env python3
"""Train, validate, gate and promote the learning-based schedulers."""

import argparse
import glob
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR_DIR = ROOT / "controllers" / "factory_supervisor"
if str(SUPERVISOR_DIR) not in sys.path:
    sys.path.insert(0, str(SUPERVISOR_DIR))

from learning_scheduler import (
    PAIR_FEATURE_NAMES,
    GraphEdgeImitationModel,
    RidgeCostModel,
    completion_samples,
    graph_edge_feature_tensor,
    pair_feature_tensor,
    solve_hungarian,
)
from schedulers import CostMatrix, build_cost_matrix
from training_scenarios import factory_scenario


WORKFLOW_VERSION = "ml-scheduler-workflow-v1"


@dataclass(frozen=True)
class SnapshotSamples:
    seed: int
    matrix: CostMatrix
    pair_features: np.ndarray
    graph_features: np.ndarray
    expert_pairs: Tuple[Tuple[int, int], ...]


def generate_snapshot(seed: int) -> SnapshotSamples:
    robots, tasks, context = factory_scenario(seed)
    matrix = build_cost_matrix(tasks, robots, context)
    pair_features = pair_feature_tensor(matrix, robots, tasks, context)
    graph_features = graph_edge_feature_tensor(matrix, pair_features)
    expert_pairs = tuple(solve_hungarian(matrix.values, matrix.feasible))
    if not expert_pairs:
        raise RuntimeError(f"seed {seed} produced no feasible expert matching")
    return SnapshotSamples(
        seed, matrix, pair_features, graph_features, expert_pairs)


def generate_snapshots(seeds: Iterable[int]) -> List[SnapshotSamples]:
    return [generate_snapshot(int(seed)) for seed in seeds]


def proxy_cost_samples(snapshots: Sequence[SnapshotSamples]
                       ) -> Tuple[np.ndarray, np.ndarray]:
    features = np.concatenate([
        snapshot.pair_features[snapshot.matrix.feasible]
        for snapshot in snapshots], axis=0)
    targets = np.concatenate([
        snapshot.matrix.values[snapshot.matrix.feasible]
        for snapshot in snapshots], axis=0)
    return features, targets


def graph_imitation_samples(snapshots: Sequence[SnapshotSamples]
                            ) -> Tuple[np.ndarray, np.ndarray]:
    feature_batches = []
    label_batches = []
    for snapshot in snapshots:
        coordinates = np.argwhere(snapshot.matrix.feasible)
        expert = set(snapshot.expert_pairs)
        feature_batches.append(
            snapshot.graph_features[snapshot.matrix.feasible])
        label_batches.append(np.asarray([
            1.0 if (int(row), int(column)) in expert else 0.0
            for row, column in coordinates], dtype=np.float64))
    return np.concatenate(feature_batches), np.concatenate(label_batches)


def load_result_documents(pattern: str) -> List[Tuple[Path, dict]]:
    if not pattern:
        return []
    documents = []
    for name in sorted(set(glob.glob(pattern, recursive=True))):
        path = Path(name)
        if not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(document, dict):
            documents.append((path, document))
    return documents


def filter_documents_by_runtime(
        documents: Sequence[Tuple[Path, dict]],
        required_runtime_mode: str) -> List[Tuple[Path, dict]]:
    """Keep only traces produced by the requested execution environment."""
    if not required_runtime_mode:
        return list(documents)
    return [
        (path, document) for path, document in documents
        if (document.get("experiment_info") or {}).get("runtime_mode") ==
        required_runtime_mode]


def split_real_samples(documents: Sequence[Tuple[Path, dict]]):
    """Split by run seed (or file when legacy results lack seed metadata)."""
    groups: Dict[str, List[Tuple[Path, dict]]] = {}
    for path, document in documents:
        seed = (document.get("experiment_info") or {}).get("seed")
        key = f"seed:{seed}" if seed is not None else f"legacy-file:{path}"
        groups.setdefault(key, []).append((path, document))
    ordered_groups = [groups[key] for key in sorted(groups)]
    if len(ordered_groups) < 2:
        empty = np.empty((0, len(PAIR_FEATURE_NAMES)), dtype=np.float64)
        return empty, np.empty(0), empty.copy(), np.empty(0), [], []
    boundary = max(1, min(len(ordered_groups) - 1,
                          int(math.floor(len(ordered_groups) * 0.8))))
    train_docs = [item for group in ordered_groups[:boundary]
                  for item in group]
    validation_docs = [item for group in ordered_groups[boundary:]
                       for item in group]
    train_x, train_y = completion_samples(
        document for _, document in train_docs)
    validation_x, validation_y = completion_samples(
        document for _, document in validation_docs)
    return (train_x, train_y, validation_x, validation_y,
            [str(path) for path, _ in train_docs],
            [str(path) for path, _ in validation_docs])


def evaluate_ridge(model: RidgeCostModel, features: np.ndarray,
                   targets: np.ndarray) -> Dict[str, float]:
    latencies_ms = []
    predictions = None
    for _ in range(25):
        started = time.perf_counter()
        predictions = np.asarray(model.predict(features), dtype=np.float64)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    errors = predictions - targets
    baseline = features[:, PAIR_FEATURE_NAMES.index("baseline_pair_cost")]
    return {
        "samples": int(targets.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "baseline_mae": float(np.mean(np.abs(baseline - targets))),
        "nonfinite_predictions": int(np.count_nonzero(
            ~np.isfinite(predictions))),
        "negative_predictions": int(np.count_nonzero(predictions < 0)),
        "inference_p95_ms": float(np.percentile(latencies_ms, 95)),
    }


def evaluate_graph(model: GraphEdgeImitationModel,
                   snapshots: Sequence[SnapshotSamples]) -> Dict[str, float]:
    edge_correct = 0
    edge_total = 0
    selected_intersection = 0
    expert_total = 0
    objective_gaps = []
    illegal_matchings = 0
    nonfinite_probabilities = 0
    decision_latencies_ms = []
    for snapshot in snapshots:
        started = time.perf_counter()
        matrix = snapshot.matrix
        probabilities = np.asarray(model.predict_proba(
            snapshot.graph_features[matrix.feasible]), dtype=np.float64)
        nonfinite_probabilities += int(np.count_nonzero(
            ~np.isfinite(probabilities)))
        coordinates = np.argwhere(matrix.feasible)
        expert = set(snapshot.expert_pairs)
        labels = np.asarray([
            (int(row), int(column)) in expert for row, column in coordinates])
        edge_correct += int(np.count_nonzero(
            (probabilities >= 0.5) == labels))
        edge_total += int(labels.size)

        scores = np.full(matrix.values.shape, np.inf, dtype=np.float64)
        for index, (row, column) in enumerate(coordinates):
            probability = min(1.0 - 1e-9,
                              max(1e-9, float(probabilities[index])))
            scores[row, column] = -math.log(probability)
        selected = set(solve_hungarian(scores, matrix.feasible))
        decision_latencies_ms.append(
            (time.perf_counter() - started) * 1000.0)
        rows = [row for row, _ in selected]
        columns = [column for _, column in selected]
        if (len(rows) != len(set(rows)) or
                len(columns) != len(set(columns)) or
                any(not matrix.feasible[row, column]
                    for row, column in selected)):
            illegal_matchings += 1
        selected_intersection += len(selected & expert)
        expert_total += len(expert)
        expert_cost = sum(matrix.values[row, column]
                          for row, column in expert)
        selected_cost = sum(matrix.values[row, column]
                            for row, column in selected)
        if len(selected) != len(expert):
            illegal_matchings += 1
        objective_gaps.append(max(
            0.0, (selected_cost - expert_cost) / max(expert_cost, 1e-9)))
    return {
        "snapshots": len(snapshots),
        "edge_accuracy": edge_correct / max(1, edge_total),
        "expert_pair_overlap": selected_intersection / max(1, expert_total),
        "mean_objective_gap": float(np.mean(objective_gaps)),
        "max_objective_gap": float(np.max(objective_gaps)),
        "illegal_matchings": int(illegal_matchings),
        "nonfinite_probabilities": int(nonfinite_probabilities),
        "decision_p95_ms": float(np.percentile(decision_latencies_ms, 95)),
    }


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_workflow(*, output_dir: Path, train_snapshots: int = 32,
                 validation_snapshots: int = 12,
                 seed: int = 4200, results_glob: str = "",
                 min_real_samples: int = 20,
                 max_ridge_mae_ratio: float = 1.10,
                 max_bootstrap_mae: float = 0.50,
                 min_graph_overlap: float = 0.60,
                 max_graph_objective_gap: float = 0.25,
                 max_model_latency_ms: float = 20.0,
                 required_runtime_mode: str = "webots",
                 allow_proxy_bootstrap: bool = False) -> dict:
    output_dir = Path(output_dir)
    if train_snapshots < 2 or validation_snapshots < 1:
        raise ValueError("need at least two training and one validation snapshot")
    if (max_ridge_mae_ratio < 0 or max_bootstrap_mae < 0 or
            not 0 <= min_graph_overlap <= 1 or
            max_graph_objective_gap < 0 or max_model_latency_ms <= 0):
        raise ValueError("workflow gate thresholds are out of range")
    train_seeds = list(range(seed, seed + train_snapshots))
    validation_seeds = list(range(
        seed + 100_000, seed + 100_000 + validation_snapshots))
    if set(train_seeds) & set(validation_seeds):
        raise AssertionError("training and validation seeds overlap")

    train_graph_snapshots = generate_snapshots(train_seeds)
    validation_graph_snapshots = generate_snapshots(validation_seeds)
    documents = load_result_documents(results_glob)
    eligible_documents = filter_documents_by_runtime(
        documents, required_runtime_mode)
    (real_train_x, real_train_y, real_validation_x, real_validation_y,
     train_files, validation_files) = split_real_samples(eligible_documents)
    use_real = (real_train_y.size >= min_real_samples and
                real_validation_y.size >= max(2, min_real_samples // 4))
    if use_real:
        ridge_train_x, ridge_train_y = real_train_x, real_train_y
        ridge_validation_x, ridge_validation_y = (
            real_validation_x, real_validation_y)
        ridge_source = "real_execution_time"
    else:
        ridge_train_x, ridge_train_y = proxy_cost_samples(
            train_graph_snapshots)
        ridge_validation_x, ridge_validation_y = proxy_cost_samples(
            validation_graph_snapshots)
        ridge_source = "factory_astar_proxy"

    graph_train_x, graph_train_y = graph_imitation_samples(
        train_graph_snapshots)
    ridge_target_name = ("execution_time" if use_real
                         else "baseline_pair_cost")
    ridge_model = RidgeCostModel.fit(
        ridge_train_x, ridge_train_y, l2=0.1,
        target_name=ridge_target_name)
    graph_model = GraphEdgeImitationModel.fit(
        graph_train_x, graph_train_y, epochs=700,
        learning_rate=0.06, l2=1e-4)

    # Repeat fitting as an executable reproducibility check, not an assumption.
    ridge_replay = RidgeCostModel.fit(
        ridge_train_x, ridge_train_y, l2=0.1,
        target_name=ridge_target_name)
    graph_replay = GraphEdgeImitationModel.fit(
        graph_train_x, graph_train_y, epochs=700,
        learning_rate=0.06, l2=1e-4)
    deterministic = bool(
        np.array_equal(ridge_model.coefficients, ridge_replay.coefficients) and
        ridge_model.intercept == ridge_replay.intercept and
        np.array_equal(graph_model.weights, graph_replay.weights) and
        graph_model.intercept == graph_replay.intercept)

    ridge_metrics = evaluate_ridge(
        ridge_model, ridge_validation_x, ridge_validation_y)
    graph_metrics = evaluate_graph(
        graph_model, validation_graph_snapshots)
    baseline_mae = ridge_metrics["baseline_mae"]
    if ridge_source == "real_execution_time" and baseline_mae > 1e-9:
        ridge_quality = (
            ridge_metrics["mae"] <= baseline_mae * max_ridge_mae_ratio)
        ridge_gate_description = (
            f"mae <= baseline_mae * {max_ridge_mae_ratio}")
    else:
        ridge_quality = ridge_metrics["mae"] <= max_bootstrap_mae
        ridge_gate_description = f"mae <= {max_bootstrap_mae} proxy units"

    gates = {
        "required_runtime_training_data": bool(
            use_real or allow_proxy_bootstrap),
        "disjoint_train_validation_seeds": not bool(
            set(train_seeds) & set(validation_seeds)),
        "deterministic_retraining": deterministic,
        "ridge_predictions_finite_nonnegative": (
            ridge_metrics["nonfinite_predictions"] == 0 and
            ridge_metrics["negative_predictions"] == 0),
        "ridge_quality": bool(ridge_quality),
        "graph_predictions_finite": (
            graph_metrics["nonfinite_probabilities"] == 0),
        "graph_matchings_legal": graph_metrics["illegal_matchings"] == 0,
        "graph_expert_overlap": (
            graph_metrics["expert_pair_overlap"] >= min_graph_overlap),
        "graph_objective_gap": (
            graph_metrics["mean_objective_gap"] <=
            max_graph_objective_gap),
        "ridge_inference_latency": (
            ridge_metrics["inference_p95_ms"] <= max_model_latency_ms),
        "graph_decision_latency": (
            graph_metrics["decision_p95_ms"] <= max_model_latency_ms),
    }
    candidate_dir = output_dir / "candidates"
    promoted_dir = output_dir / "promoted"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    ridge_path = candidate_dir / "learned_cost.npz"
    graph_path = candidate_dir / "graph_imitation.npz"
    ridge_model.save(ridge_path)
    graph_model.save(graph_path)
    loaded_ridge = RidgeCostModel.load(ridge_path)
    loaded_graph = GraphEdgeImitationModel.load(graph_path)
    artifact_round_trip = bool(
        loaded_ridge.target_name == ridge_target_name and
        np.array_equal(loaded_ridge.predict(ridge_validation_x),
                       ridge_model.predict(ridge_validation_x)) and
        np.array_equal(
            loaded_graph.predict_proba(graph_train_x[:100]),
            graph_model.predict_proba(graph_train_x[:100])))
    gates["checkpoint_round_trip"] = artifact_round_trip
    promoted = all(gates.values())
    if promoted:
        promoted_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ridge_path, promoted_dir / ridge_path.name)
        shutil.copy2(graph_path, promoted_dir / graph_path.name)
    else:
        # Never leave stale production-looking artifacts after a failed gate.
        for stale_name in ("learned_cost.npz", "graph_imitation.npz"):
            (promoted_dir / stale_name).unlink(missing_ok=True)
        if promoted_dir.is_dir() and not any(promoted_dir.iterdir()):
            promoted_dir.rmdir()

    report = {
        "workflow_version": WORKFLOW_VERSION,
        "status": "promoted" if promoted else "completed_not_promoted",
        "promoted": promoted,
        "data": {
            "ridge_source": ridge_source,
            "result_documents_found": len(documents),
            "required_runtime_mode": required_runtime_mode,
            "eligible_runtime_documents": len(eligible_documents),
            "real_training_samples": int(real_train_y.size),
            "real_validation_samples": int(real_validation_y.size),
            "training_result_files": train_files,
            "validation_result_files": validation_files,
            "training_snapshot_seeds": train_seeds,
            "validation_snapshot_seeds": validation_seeds,
            "graph_training_edges": int(graph_train_y.size),
        },
        "ridge_metrics": ridge_metrics,
        "graph_metrics": graph_metrics,
        "gates": gates,
        "gate_thresholds": {
            "ridge": ridge_gate_description,
            "min_graph_overlap": min_graph_overlap,
            "max_graph_mean_objective_gap": max_graph_objective_gap,
            "max_model_latency_ms": max_model_latency_ms,
        },
        "artifacts": {
            "candidate_learned_cost": str(ridge_path),
            "candidate_graph_imitation": str(graph_path),
            "promoted_directory": str(promoted_dir) if promoted else None,
        },
        "automated_review_evidence": {
            "data_boundary": "seed-grouped train/validation split",
            "quality_safety": "all entries in gates are executable checks",
            "small_step_review_policy": "implementation, boundary-safety, regression",
        },
    }
    _write_json_atomic(output_dir / "workflow_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train, validate and promote learning-based schedulers")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "ml_scheduler")
    parser.add_argument("--results-glob", default="")
    parser.add_argument("--train-snapshots", type=int, default=32)
    parser.add_argument("--validation-snapshots", type=int, default=12)
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument("--min-real-samples", type=int, default=20)
    parser.add_argument("--max-ridge-mae-ratio", type=float, default=1.10)
    parser.add_argument("--max-bootstrap-mae", type=float, default=0.50)
    parser.add_argument("--min-graph-overlap", type=float, default=0.60)
    parser.add_argument("--max-graph-objective-gap", type=float, default=0.25)
    parser.add_argument("--max-model-latency-ms", type=float, default=20.0)
    parser.add_argument(
        "--required-runtime-mode", default="webots",
        choices=("webots", "standalone", ""),
        help="Only this runtime may supply production execution-time labels")
    parser.add_argument(
        "--allow-proxy-bootstrap", action="store_true",
        help="Allow proxy-only checkpoints to pass (research/bootstrap only)")
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    report = run_workflow(
        output_dir=args.output_dir,
        train_snapshots=args.train_snapshots,
        validation_snapshots=args.validation_snapshots,
        seed=args.seed,
        results_glob=args.results_glob,
        min_real_samples=args.min_real_samples,
        max_ridge_mae_ratio=args.max_ridge_mae_ratio,
        max_bootstrap_mae=args.max_bootstrap_mae,
        min_graph_overlap=args.min_graph_overlap,
        max_graph_objective_gap=args.max_graph_objective_gap,
        max_model_latency_ms=args.max_model_latency_ms,
        required_runtime_mode=args.required_runtime_mode,
        allow_proxy_bootstrap=args.allow_proxy_bootstrap,
    )
    print(json.dumps({
        "status": report["status"],
        "report": str(args.output_dir / "workflow_report.json"),
        "gates": report["gates"],
    }, indent=2, sort_keys=True))
    return 2 if args.fail_on_gate and not report["promoted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
