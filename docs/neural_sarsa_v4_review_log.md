# Neural SARSA v4 review log

## Step 1 — baseline preservation and design

- Review A (scope): passed. PR #6 preserves the 18 local source/workflow files
  at commit `91220a3`; Python caches, IDE state, generated results, and generated
  manifests were excluded.
- Review B (algorithm): passed after excluding generic replay and defining
  masked legal-only dueling centering, stored behavior probabilities, terminal
  n-step flushing, and a separate WAIT head.
- Review C (integration): passed after separating the v4 checkpoint/environment
  contract from v3, pinning CPU Webots inference, and fixing evaluation seeds to
  61000--61005 without a non-regression/rollback gate.

The two requested detailed proposal-review rounds are recorded in
`docs/neural_sarsa_v4_design_review.md`.

## Step 2 — shared traffic contract and v4 features

Initial Review A required changes: the parity assertion compared clipped v3
features with unbounded raw traffic features.  The comparison was moved into
the identical normalized and clipped feature space and all reviews restarted.

- Review A (algorithm/semantics): passed.  Both GraphImitation and SARSA call
  `evaluate_traffic_edge`; the same frozen snapshot produces the nine traffic
  fields, expert penalty, adjusted unfloored baseline cost, and endpoint loads.
  The penalty is observation-only for SARSA.
- Review B (runtime/safety): passed.  The helper is read-only, rejects stale
  types/non-finite costs, preserves GraphImitation's nonnegative adjusted-cost
  floor, and keeps the v3 feature/checkpoint ABI unchanged.
- Review C (tests/final diff): passed.  58 focused tests passed, the three files
  compiled from source, the 161x457/385+72 v4 feature smoke test passed, and
  `git diff --check` passed.

Reviewed source SHA-256:

```text
3C6FF71348BE8542AD774B6FDAC0F16CBD1BBCCA2F793562917E9172DA638ED9  controllers/factory_supervisor/traffic_features.py
BB9DDDAC63E3F83E20113E705C0DDCD655AA00FC889042E59E8EF7CE68C3D6AC  controllers/factory_supervisor/learning_scheduler.py
12231F487989458E4C04988E25811961B359D3BFC97B48B1BAEC00C8345F2ADC  controllers/factory_supervisor/sarsa_v4_features.py
```

## Step 3 — masked dueling Expected SARSA agent

- Review A (algorithm/semantics): passed.  Legal-only dueling centering, exact
  epsilon-greedy tie probabilities, five-step return maturation, terminal
  shorter-return flushing, and online-policy/target-value separation were read
  independently.  A constructed two-action case produced the exact Expected
  SARSA target `0.1*1 + 0.9*3 = 2.8`.
- Review B (device/persistence): passed.  CPU train/save/load/RNG continuation,
  CUDA inference, and CUDA backward/optimizer execution passed.  The checkpoint
  rejects incompatible feature/objective metadata and cannot be written with a
  partial trajectory or minibatch.
- Review C (tests/final diff): passed.  AST compilation, 41-transition n-step
  maturation, masked selection, state-hash round trip, legacy v3 agent tests,
  finite gradient clipping, and `git diff --check` passed.  The final network
  has 77,443 trainable parameters.

Reviewed source SHA-256:

```text
1B04F9B8F790E911EF4CCAC582EEE94AEAFF2639D28644567721ED153A3D9A11  controllers/factory_supervisor/sarsa_v4_agent.py
```

## Step 4 — commit-aware runtime and Supervisor integration

- Review A (algorithm/semantics): passed.  Transitions close only after an
  actual commit, rejection, or policy WAIT.  Event rewards remain conserved,
  rejection/defer penalties attach to the action that caused them, and terminal
  finalization flushes all pending n-step samples before the checkpoint.
- Review B (runtime/resume): passed.  V4 has an explicit environment switch and
  output path, simultaneous v3/v4 switches fail closed, incompatible V4 models
  never fall back to another scheduler, output is confined to the run root, and
  frozen evaluation rejects any model-state change.
- Review C (tests/final diff): passed.  Commit/reject/terminal and frozen-eval
  smoke tests passed; 55 legacy SARSA/advanced scheduler tests passed; factory
  opt-in/fail-closed, AST compilation, and `git diff --check` passed.

Reviewed source SHA-256:

```text
3194EE0AC411FFE25A783EBE6FE6DB7CF9F8E74CDB6B18510B9C657011846952  controllers/factory_supervisor/sarsa_v4_runtime.py
24183DEC831AB49CF6FD83DF573A8D9267BF89835DB3B9CB6372F1EA263D5F77  controllers/factory_supervisor/schedulers.py
904676A605FCF9313E7EFB4C17124619BCF74F280B0B25C8F37812B20A41A4D9  controllers/factory_supervisor/factory_supervisor.py
```

## Step 5 — resumable training and evaluation workflow

Initial Review B required changes: standalone initialization was merely
loadable rather than compared with the registered deterministic initial state,
and fine-tune initialization was not bound to its standalone parent hash.  Both
identity checks were added and the three reviews restarted.

- Review A (algorithm/partitions): passed.  Count and Utility V2 remain
  independent cumulative policies; standalone uses 21000--21059, Webots
  fine-tune uses 41000--41009, and frozen evaluation uses 61000--61005.  There
  is no validation/non-regression comparison, rollback, or evaluation-based
  selection.
- Review B (resume/safety): passed.  Source/configuration, manifest, input
  checkpoint, and initial-parent hashes are pinned.  Canonical records are
  written only after complete episodes; partial attempts remain isolated; a
  changed source or checkpoint blocks stale resume; every next phase requires
  three reviews of the previous phase evidence.
- Review C (execution/final diff): passed.  Review-gate tests and 16 existing
  workflow-protection tests passed.  A real CUDA-backed scenario A/seed 21000
  1800-second standalone episode completed 51 tasks, produced neural updates,
  wrote a loadable checkpoint, and took 15.1 wall
  seconds.  AST compilation, CLI help, and `git diff --check` passed.

Reviewed source SHA-256:

```text
7F8FC010E210AAE6C9EB636949D853DC3DA08C626EAE980C8463F698FB7EF80E  scripts/neural_sarsa_workflow_contract.py
C09F3CEDC818BAF609F74130F6BD2F537C140517A5251ABB869AF3C104C900E9  scripts/run_sarsa_v4_workflow.py
```

## Step 6 — tests and dependency contract

- Review A (focused behavior): passed.  Six neural v4 tests cover exact shared
  GraphImitation conflict penalties, 385+72 feature partitioning, illegal-action
  masks, exact epsilon-greedy expectation, terminal n-step flushing, checkpoint
  boundaries, commit/rejection attribution, frozen evaluation, versioned
  factory opt-in, and three-review hash binding.  CUDA backward passed.
- Review B (project regression): passed.  The full suite completed 461 tests in
  118.237 seconds with no failure.  Existing v3, GraphImitation,
  LearnedHungarian, runtime, workflow, reward, purity, and Webots launcher tests
  remain green.
- Review C (final diff/dependencies): passed.  All changed Python sources
  compiled from text, final `git diff --check` passed, and the installed
  `torch==2.14.1+cu126` matched the new CUDA-specific requirement.  `pip check`
  still reports the pre-existing `opencv-python 4.12.0.88` versus NumPy 2.4.3
  constraint; neither neural SARSA nor the project core imports OpenCV, so this
  review records it without changing NumPy before the formal run.

Reviewed source SHA-256:

```text
6E4FF199336E8D102914F4DB3343FA91A8089E54AD5C931ECD284F4B52F99C6C  requirements.txt
E18180822DACB558C2853CE77B11BD6C2E2955036806876C6DC1F363482C563A  requirements-torch-cu126.txt
8CD71B2496565364093D099E6D609CC945C12DFD0178ACAD5B0D1D18B39CF872  docs/ml_scheduler_1800_evaluation.md
AFA862E21CA144BF25D1E7DB09C299224D2F332B5F5655778CCF7727EC842D73  tests/test_sarsa_v4_neural.py
```
