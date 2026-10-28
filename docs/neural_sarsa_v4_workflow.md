# Neural SARSA v4 implementation and execution workflow

The baseline before this work is commit `91220a3` in pull request #6.  Generated
manifests, results, Python caches, and IDE configuration are not source changes.

## Review rule

Each step is complete only after three passing reviews cover identical final
file hashes:

1. algorithm, feature, and reward semantics;
2. runtime behavior, persistence, resume, and failure safety;
3. tests, static checks, and an independent final diff read.

If a review requires a change, all three reviews for that step restart.  Review
receipts are stored below the isolated run root and are checked by the workflow
before the next step starts.

## Steps

1. Preserve the baseline PR and record both design-review rounds.
2. Add a shared public traffic-edge contract and SARSA v4 feature adapter.
3. Add the PyTorch masked dueling Expected SARSA agent and checkpoint contract.
4. Add commit-aware runtime integration and fail-closed Supervisor selection.
5. Add the resumable standalone -> Webots fine-tune -> six-seed evaluation
   workflow without a non-regression/rollback phase.
6. Add focused tests, dependency installation instructions, and full-suite
   validation.
7. Run standalone training for Count and Utility V2.
8. Run Webots physical fine tuning for both objectives.
9. Freeze both models and evaluate A/B/C on seeds 61000--61005; aggregate paired
   results without using them to update or select the models.

## Execution guarantees

- Training, fine-tune, and evaluation manifests are disjoint and hashed.
- Completed episodes resume only when source, configuration, input checkpoint,
  and manifest hashes match.
- Evaluation never learns and never enables a fallback scheduler.
- Training proceeds directly to the next registered phase; there is no
  intermediate degradation test.
- An interrupted command resumes from its last complete episode/checkpoint.

