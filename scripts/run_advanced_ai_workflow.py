"""Train and validate additional AI schedulers without production promotion."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "controllers" / "factory_supervisor"
if str(SUPERVISOR) not in sys.path:
    sys.path.insert(0, str(SUPERVISOR))

from advanced_ai_common import masked_argmax  # noqa: E402
from advanced_rl_agents import (  # noqa: E402
    CQLAgent, CQLConfig, QRDQNAgent, QRDQNConfig, RainbowConfig,
    RainbowDQNAgent,
)
from advanced_rl_training import (  # noqa: E402
    collect_cql_dataset, train_online_value_agent,
)
from bandit_scheduler import LinUCBModel, LinUCBScheduler, train_linucb  # noqa: E402
from graph_ppo_scheduler import (  # noqa: E402
    GraphPPOConfig, GraphPPOModel, GraphPPOScheduler, train_graph_ppo,
)
from rl_environment import SchedulingEnvironment  # noqa: E402
from schedulers import HungarianScheduler, validate_assignments  # noqa: E402
from training_scenarios import factory_scenario  # noqa: E402


WORKFLOW_VERSION = "advanced-ai-scheduler-workflow-v1"


def _strict_seeds(values, name):
    seeds = tuple(values)
    if (not seeds or len(seeds) != len(set(seeds)) or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in seeds)):
        raise ValueError(f"{name} seeds must be unique integers")
    return seeds


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_value_agent(agent, seeds, max_steps: int) -> dict:
    environment = SchedulingEnvironment(simulation_mode="abstract")
    returns = []
    latencies = []
    invalid_actions = 0
    decisions = 0
    for seed in seeds:
        robots, tasks, context = factory_scenario(seed)
        state, _ = environment.reset(robots, tasks, context, seed=seed)
        episode_return = 0.0
        for step in range(max_steps):
            mask = environment.get_action_mask()
            started = time.perf_counter()
            try:
                action = masked_argmax(agent.action_values(state), mask)
            except Exception:
                invalid_actions += 1
                break
            latencies.append((time.perf_counter()-started)*1000.0)
            if not mask[action]:
                invalid_actions += 1
                break
            next_state, reward, terminated, truncated, info = (
                environment.step(action))
            if info.get("invalid_action"):
                invalid_actions += 1
            decisions += 1
            episode_return += float(reward)
            state = next_state
            if terminated or truncated or step+1 >= max_steps:
                break
        returns.append(episode_return)
    return {
        "episodes": len(returns),
        "decisions": decisions,
        "invalid_actions": invalid_actions,
        "mean_return": float(np.mean(returns)),
        "decision_p95_ms": float(np.percentile(latencies, 95))
        if latencies else None,
    }


def evaluate_matching_scheduler(scheduler, seeds) -> dict:
    invalid_matchings = 0
    decisions = 0
    latency = []
    cost_ratios = []
    for seed in seeds:
        robots, tasks, context = factory_scenario(seed)
        visible_tasks = [
            task for task in tasks
            if float(task.arrival_time) <= float(context.current_time)+1e-9]
        result = scheduler.assign(visible_tasks, robots, context)
        latency.append(result.computation_time*1000.0)
        valid, _ = validate_assignments(
            result.assignments, visible_tasks, robots, context)
        if not result.is_feasible or not valid:
            invalid_matchings += 1
            continue
        decisions += 1
        baseline = HungarianScheduler().assign(
            visible_tasks, robots, context)
        if (baseline.objective_value is not None and
                result.objective_value is not None and
                baseline.objective_value > 1e-9 and
                len(result.assignments) == len(baseline.assignments)):
            cost_ratios.append(
                result.objective_value/baseline.objective_value)
    return {
        "snapshots": len(seeds),
        "legal_decisions": decisions,
        "invalid_matchings": invalid_matchings,
        "mean_equal_cardinality_cost_ratio": (
            float(np.mean(cost_ratios)) if cost_ratios else None),
        "decision_p95_ms": float(np.percentile(latency, 95))
        if latency else None,
    }


def run_workflow(*, output_dir: Path, train_seeds, validation_seeds,
                 max_steps: int = 32, offline_updates: int = 32) -> dict:
    output_dir = Path(output_dir)
    train_seeds = _strict_seeds(train_seeds, "training")
    validation_seeds = _strict_seeds(validation_seeds, "validation")
    if set(train_seeds) & set(validation_seeds):
        raise ValueError("training and validation seeds must be disjoint")
    if max_steps <= 0 or offline_updates <= 0:
        raise ValueError("max_steps and offline_updates must be positive")
    candidate_dir = output_dir/"candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    environment = SchedulingEnvironment(simulation_mode="abstract")
    expected_rows = len(train_seeds)*max_steps
    batch_size = max(2, min(32, max(2, expected_rows//2)))
    capacity = max(128, expected_rows*3)

    rainbow = RainbowDQNAgent(
        environment.observation_dim, environment.action_dim,
        environment.no_op_action,
        RainbowConfig(
            hidden_size=32, atoms=21, n_step=2,
            replay_capacity=capacity, batch_size=batch_size,
            warmup_steps=batch_size, target_update_interval=50),
        seed=train_seeds[0])
    rainbow_training = train_online_value_agent(
        rainbow, train_seeds, max_steps=max_steps)

    qrdqn = QRDQNAgent(
        environment.observation_dim, environment.action_dim,
        environment.no_op_action,
        QRDQNConfig(
            hidden_size=32, quantiles=16, risk_fraction=0.5, n_step=2,
            replay_capacity=capacity, batch_size=batch_size,
            warmup_steps=batch_size, target_update_interval=50),
        seed=train_seeds[0]+1)
    qrdqn_training = train_online_value_agent(
        qrdqn, train_seeds, max_steps=max_steps)

    cql_dataset = collect_cql_dataset(train_seeds, max_steps=max_steps)
    cql = CQLAgent(
        environment.observation_dim, environment.action_dim,
        environment.no_op_action,
        CQLConfig(
            hidden_size=32, replay_capacity=max(capacity, len(cql_dataset.states)),
            batch_size=min(batch_size, len(cql_dataset.states)),
            warmup_steps=min(batch_size, len(cql_dataset.states)),
            target_update_interval=50),
        seed=train_seeds[0]+2)
    cql_dataset.add_to(cql)
    cql_losses = [cql.train_step() for _ in range(offline_updates)]
    cql_losses = [float(loss) for loss in cql_losses if loss is not None]
    if not cql_losses or not np.all(np.isfinite(cql_losses)):
        raise ValueError("CQL produced no finite updates")
    cql_training = {
        "dataset_rows": len(cql_dataset.states),
        "behavior_policy": cql_dataset.behavior_policy,
        "dataset_seeds": list(cql_dataset.seeds),
        "updates": len(cql_losses),
        "mean_loss": float(np.mean(cql_losses)),
    }

    graph_ppo = GraphPPOModel(
        GraphPPOConfig(hidden_size=24, update_epochs=2),
        seed=train_seeds[0]+3)
    graph_training = train_graph_ppo(
        graph_ppo, train_seeds, max_steps=max_steps)

    linucb, linucb_training = train_linucb(
        train_seeds, alpha=0.35)

    paths = {
        "RainbowDQN": candidate_dir/"rainbow_dqn.npz",
        "QRDQN": candidate_dir/"qrdqn.npz",
        "CQL": candidate_dir/"cql.npz",
        "CQLDataset": candidate_dir/"cql_dataset.npz",
        "GraphPPO": candidate_dir/"graph_ppo.npz",
        "LinUCB": candidate_dir/"linucb.npz",
    }
    rainbow.save(paths["RainbowDQN"])
    qrdqn.save(paths["QRDQN"])
    cql.save(paths["CQL"])
    cql_dataset.save(paths["CQLDataset"])
    graph_ppo.save(paths["GraphPPO"])
    linucb.save(paths["LinUCB"])

    # Checkpoint round trips are executable compatibility gates.
    loaded_rainbow = RainbowDQNAgent.load(
        paths["RainbowDQN"], environment.observation_dim,
        environment.action_dim, environment.no_op_action)
    loaded_qrdqn = QRDQNAgent.load(
        paths["QRDQN"], environment.observation_dim,
        environment.action_dim, environment.no_op_action)
    loaded_cql = CQLAgent.load(
        paths["CQL"], environment.observation_dim,
        environment.action_dim, environment.no_op_action)
    GraphPPOModel.load(paths["GraphPPO"])
    LinUCBModel.load(paths["LinUCB"])

    value_validation = {
        "RainbowDQN": evaluate_value_agent(
            loaded_rainbow, validation_seeds, max_steps),
        "QRDQN": evaluate_value_agent(
            loaded_qrdqn, validation_seeds, max_steps),
        "CQL": evaluate_value_agent(
            loaded_cql, validation_seeds, max_steps),
    }
    graph_validation = evaluate_matching_scheduler(
        GraphPPOScheduler(paths["GraphPPO"]), validation_seeds)
    linucb_validation = evaluate_matching_scheduler(
        LinUCBScheduler(paths["LinUCB"]), validation_seeds)
    gates = {
        "disjoint_train_validation_seeds": not bool(
            set(train_seeds) & set(validation_seeds)),
        "checkpoint_round_trip": True,
        "value_actions_legal": all(
            metrics["invalid_actions"] == 0 and metrics["decisions"] > 0
            for metrics in value_validation.values()),
        "graph_matchings_legal": (
            graph_validation["invalid_matchings"] == 0 and
            graph_validation["legal_decisions"] == len(validation_seeds)),
        "linucb_matchings_legal": (
            linucb_validation["invalid_matchings"] == 0 and
            linucb_validation["legal_decisions"] == len(validation_seeds)),
        "training_updates_completed": (
            rainbow.training_step > 0 and qrdqn.training_step > 0 and
            cql.training_step > 0 and graph_ppo.training_step > 0 and
            linucb.training_samples > 0),
    }
    research_ready = all(gates.values())
    report = {
        "workflow_version": WORKFLOW_VERSION,
        "status": ("research_candidates_ready" if research_ready
                   else "research_gates_failed"),
        "research_candidates_ready": research_ready,
        "production_promotion": False,
        "production_blocker": (
            "Requires paired unseen-seed Webots evaluation with no safety, "
            "latency, fallback or task-flow regression."),
        "data": {
            "training_seeds": list(train_seeds),
            "validation_seeds": list(validation_seeds),
            "max_steps_per_episode": max_steps,
            "offline_cql_updates": offline_updates,
        },
        "training": {
            "RainbowDQN": rainbow_training,
            "QRDQN": qrdqn_training,
            "CQL": cql_training,
            "GraphPPO": graph_training,
            "LinUCB": linucb_training,
        },
        "validation": {
            "value_agents": value_validation,
            "GraphPPO": graph_validation,
            "LinUCB": linucb_validation,
        },
        "gates": gates,
        "artifacts": {name: str(path) for name, path in paths.items()},
        "review_policy": {
            "workflow_reviews_required": 2,
            "code_reviews_per_step": 3,
            "protected_motion_files_check": "external_hash_review_required",
        },
    }
    # Validate strict JSON before the atomic write.
    json.dumps(report, allow_nan=False)
    _write_json_atomic(output_dir/"workflow_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train additional AI task schedulers in the abstract model")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT/"results"/"advanced_ai_scheduler")
    parser.add_argument(
        "--train-seeds", nargs="+", type=int,
        default=list(range(9200, 9232)))
    parser.add_argument(
        "--validation-seeds", nargs="+", type=int,
        default=list(range(109200, 109208)))
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--offline-updates", type=int, default=32)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    report = run_workflow(
        output_dir=args.output_dir,
        train_seeds=args.train_seeds,
        validation_seeds=args.validation_seeds,
        max_steps=args.max_steps,
        offline_updates=args.offline_updates)
    print(json.dumps({
        "status": report["status"],
        "report": str(args.output_dir/"workflow_report.json"),
        "gates": report["gates"],
    }, indent=2, sort_keys=True))
    return (2 if args.fail_on_gate and
            not report["research_candidates_ready"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
