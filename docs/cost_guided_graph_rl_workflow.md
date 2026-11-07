# Cost-guided Graph RL workflow

## 1. Scope and invariants

This workflow implements and evaluates four policies:

| Policy | Objective | Primary frozen cost prior | Secondary frozen cost prior |
| --- | --- | --- | --- |
| `CostGraphPPO_count` | `count` | `GraphImitation_count` | `LearnedHungarian_count` |
| `CostGraphPPO_utility_v2` | `utility_v2` | `GraphImitation_utility_v2` | `LearnedHungarian_utility_v2` |
| `GraphRainbow_count` | `count` | `LearnedHungarian_count` | `GraphImitation_count` |
| `GraphRainbow_utility_v2` | `utility_v2` | `LearnedHungarian_utility_v2` | `GraphImitation_utility_v2` |

The selected coefficient file is copied into the run directory and pinned by
SHA-256 before any model code runs. Coefficients are frozen observations,
teacher signals, and action-logit priors. They do not modify the formal reward,
physical feasibility, collision handling, or fallback policy.

The expert cost must be computed once per decision snapshot, before any
Hungarian non-negative flooring:

```text
travel
- priority_bonus * (priority - 1)
- waiting_bonus * waiting
+ congestion_penalty * congestion
- deadline_pressure_bonus * weighted_deadline_pressure
+ traffic_expected_wait_penalty * expected_wait
+ traffic_recent_wait_penalty * exposure * peer_recent_wait
+ traffic_stop_event_penalty * exposure * peer_stop_events
```

Raw costs keep their physical units. Robust normalized costs and ranks may be
added as model features, but normalization must never replace the raw cost used
for identity checks or teacher matching. Traffic terms must not be applied a
second time outside this formula.

## 2. Data boundary and seed protocol

The partitions are immutable and pairwise disjoint:

- standalone training: `72000..72005`;
- Webots fine-tuning: `73000..73001`;
- final Webots evaluation: the same batch `74000..74005` for every policy,
  objective, and scenario A/B/C.

Final comparison therefore contains `4 policies x 3 scenarios x 6 seeds = 72`
episodes. Evaluation seeds and manifests are never used for gradient updates,
normalizer fitting, threshold selection, checkpoint selection, or retries with
different behavior. A failed evaluation episode remains a failed episode.

## 3. Step state machine

Work proceeds through these small, ordered steps:

1. workflow contract;
2. cost-prior identity and cost math;
3. cost-prior graph observation;
4. CostGraphPPO model;
5. CostGraphPPO standalone training path;
6. variable-edge GraphRainbow model;
7. GraphRainbow replay and standalone training path;
8. runtime and scheduler integration;
9. standalone workflow and training run;
10. Webots fine-tune workflow and run;
11. same-batch six-seed Webots evaluation;
12. formal gate report and reproducibility audit.

Each step is closed only after three sequential reviews over an identical set
of file hashes:

1. algorithm and numerical semantics;
2. safety, model identity, and data boundaries;
3. integration, regression, and resumability.

If a reviewed file changes, its receipt becomes invalid and all three reviews
for that step must be repeated. Step N cannot be reviewed until every review of
steps 1..N-1 still validates. Review findings, checks, timestamps, file hashes,
and the receipt hash are written below the run's `audit/` directory.

## 4. Model requirements

CostGraphPPO uses a variable-edge graph observation, a shared edge encoder,
masked action logits, an explicit no-op action, and a learned residual on top of
the primary normalized-cost prior. Its checkpoint embeds objective, feature
schema/version, reward profile, primary and secondary canonical bundles and
hashes, seeds, and training phase. Legacy GraphPPO checkpoints remain loadable
only through their existing legacy path; they must not silently enter this
workflow.

GraphRainbow is a variable-edge implementation, not the existing fixed-vector
RainbowDQN adapter. It requires Double DQN selection/evaluation, dueling
distributional C51 heads, prioritized replay with importance weights, n-step
returns, masked legal actions, explicit no-op, and deterministic inference.
Replay rows must carry graph observations and model identity; cross-objective or
cross-schema replay is rejected.

Both schedulers build a decision snapshot once and reuse it for observation,
mask, decision, logging, and physical rollout. The safety wrapper must proxy
traffic and coefficient identity so the supervisor supplies a matching traffic
snapshot. Formal reward conservation is checked independently of cost priors.

## 5. Training, fine-tuning, and evaluation

Standalone training uses only the standalone partition and persists resumable
checkpoints plus optimizer/replay state. Checkpoint selection uses standalone
validation only. Webots fine-tuning starts from the selected standalone
checkpoint, uses only the fine-tune partition, preserves safety fallback, and
writes a new phase-labelled checkpoint rather than overwriting the source.

Evaluation loads the frozen fine-tuned checkpoint, disables exploration and
updates, and runs all 72 canonical episodes. Every episode records policy and
bundle hashes, objective, reward profile, scenario, seed, manifest hash,
checkpoint hash, action counts, fallback/invalid/timeout counts, collisions,
reward-conservation error, and scheduler latency distribution.

## 6. Formal gates

The workflow is complete only when all artifacts validate and all required
episodes exist. Per-policy formal gates are:

- six identical evaluation seeds for each scenario;
- formal reward-conservation absolute error `<= 1e-9`;
- zero collision, invalid action, scheduler timeout, and safety fallback;
- deterministic checkpoint/model identity with no evaluation updates;
- scheduler decision latency P99 below `0.040 s` and configured timeout above
  the measured P99 with an explicit margin;
- no NaN/Inf in observations, costs, logits, distributions, losses, gradients,
  replay priorities, or reports.

Performance comparisons are reported with paired per-seed differences and
confidence intervals. They are secondary to safety and protocol validity: a
policy that fails a formal gate is reported as failed, not promoted based on its
mean reward or throughput.

## 7. Failure and resume rules

All manifests and reports are written atomically. Resume validates the workflow
identity, frozen coefficient hashes, checkpoint phase, seed partition, and
existing artifact hashes before skipping work. Partial/corrupt artifacts are
quarantined by the runner and regenerated only from the same seed and manifest.
No automatic retry may change a final-evaluation action, configuration, seed,
checkpoint, or manifest. Any contract change requires a new versioned run root.
