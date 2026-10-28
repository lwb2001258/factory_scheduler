# Neural SARSA v4 design and two-round review

Date: 2026-10-28

## Scope

Replace the legacy/linear SARSA learning path with an opt-in Masked Dueling
n-step Expected SARSA implementation while retaining SARSA v3 checkpoints and
runtime behavior for reproducibility.  The new policy must consume the same
frozen, decision-time traffic forecast and conflict penalty as the current
traffic-aware GraphImitation model.

The implementation does not add an intermediate non-regression comparison or
rollback gate.  After code completion and review it proceeds through standalone
training, Webots fine tuning, and frozen evaluation on scenarios A/B/C with the
same final seeds 61000--61005.

## Review round 1: algorithm and traffic semantics

Status: passed after design corrections.

Reviewed items:

- action-mask semantics, including ties and the WAIT action;
- dueling value/advantage identifiability with a changing feasible set;
- Expected SARSA policy probabilities and n-step return boundaries;
- delayed event rewards and actual commit/rejection attribution;
- GraphImitation traffic-feature, conflict-penalty, and endpoint-load parity;
- preservation of the Count and Utility V2 objective definitions.

Findings and resolutions:

1. A generic replay buffer would make the proposed method off-policy after the
   behavior network changes.  It is excluded.  Training uses contiguous
   on-policy trajectories.  The epsilon-greedy probability vector used by an
   Expected SARSA bootstrap is captured when that next decision is made.
2. The dueling advantage mean must not include impossible robot-task edges.
   `Q(s,a)=V(s)+A(s,a)-mean_legal(A)` uses only the current legal mask.  Illegal
   values are set to negative infinity for selection and never receive a loss.
3. WAIT has structurally different fields from a dispatch edge.  It receives a
   separate advantage head while sharing the state value encoder.
4. A five-step return is used to carry sparse task-completion rewards backward.
   Terminal episodes flush shorter returns.  `gamma=1.0` preserves the finite
   horizon cumulative objective.
5. A slowly synchronized target network evaluates the bootstrap.  The online
   network defines the behavior probabilities.  This stabilization is recorded
   explicitly in model metadata rather than being presented as plain one-step
   SARSA.
6. Traffic information is read from the current immutable `TrafficSnapshot`.
   GraphImitation and SARSA call one public edge-evaluation helper for the nine
   traffic fields, expert conflict penalty, adjusted route cost, pickup load,
   and delivery load.
7. The expert penalty is an observation/cost feature only.  It is not inserted
   into the reward and no GraphImitation labels or actions supervise v4.
8. State context is encoded once per decision.  The 160 dispatch edges share an
   action scorer, avoiding repeated encoding of the 385 context fields and
   reducing sensitivity to fixed robot/task slot identities.

## Review round 2: integration, persistence, and evaluation

Status: passed after workflow corrections.

Reviewed items:

- compatibility with the Supervisor commit/rejection callbacks;
- standalone/Webots feature and event-order parity;
- safe, device-neutral checkpoints and exact episode-boundary resume;
- physical fine-tune isolation and frozen evaluation;
- seed/manifests partitions and six-seed pairing;
- failure handling, latency, and source-contract pinning.

Findings and resolutions:

1. SARSA v3 JSON checkpoints and linear prediction-change statistics cannot be
   reused.  V4 uses a separate model version, feature version, environment
   switch, output path, workflow root, and neural Q-probe statistics.
2. A checkpoint contains online/target network state, Adam state, epsilon,
   episode/update counters, NumPy RNG state, Torch CPU/CUDA RNG state, and all
   feature/reward/traffic versions.  Checkpoints are only canonical at flushed
   episode boundaries, so no partial n-step queue can be silently resumed.
3. Standalone training may use CUDA.  Webots inference and physical fine tuning
   default to CPU because the network benchmark is below one millisecond and
   concurrent Webots processes should not contend for the GPU.  Checkpoints are
   loaded through `map_location` and remain device neutral.
4. Evaluation sets epsilon to zero, performs no optimizer step, writes no model,
   and rejects any model-state change.  Invalid checkpoints fail closed and do
   not fall back to another scheduler.
5. Count and Utility V2 keep independent models.  Training and fine tuning use
   their registered partitions; final evaluation uses exactly seeds
   61000--61005 and scenarios A/B/C.
6. The latest cumulative checkpoint advances to the next phase.  There is no
   validation/non-regression model-selection gate and no automatic rollback.
   Convergence diagnostics may stop training after the registered minimum but
   do not compare against the previous algorithm.
7. Every implementation step is followed by three reviews over the same final
   file hashes: algorithm/semantic review, runtime/resume review, and
   tests/final-diff review.  Any fix invalidates those reviews and restarts the
   three-review sequence for that step.

## Accepted architecture

- Context: 385 fields -> 128 -> LayerNorm/SiLU -> 64.
- Edge action: 72 fields -> 64/SiLU.
- Dispatch advantage: concatenated context/action embedding -> 64/SiLU -> 1.
- WAIT advantage: context embedding plus WAIT fields -> 32/SiLU -> 1.
- State value: context embedding -> 32/SiLU -> 1.
- Approximately 75k trainable parameters, with legal-action dueling centering.
- Five-step Expected SARSA, Huber loss, Adam at 1e-4, gradient norm cap 5,
  target synchronization every 500 optimizer steps.
- Epsilon 0.20 -> 0.02 in standalone; physical cap 0.01; evaluation 0.
