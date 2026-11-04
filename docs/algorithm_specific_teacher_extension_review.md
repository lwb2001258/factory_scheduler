# Algorithm-specific own-teacher evaluation extension

## Scope

The main algorithm-specific coefficient workflow must finish without source
changes.  Its current source contract hashes every Python file below
`controllers/` and `scripts/`.  The extension therefore lives in
`post_workflows/` and starts only after step 8 is complete, all three step-8
reviews pass, and the source workflow lock is released.

The extension evaluates four immutable analytical policies:

- `GraphImitationExpert_count`
- `GraphImitationExpert_utility_v2`
- `LearnedHungarianExpert_count`
- `LearnedHungarianExpert_utility_v2`

Each expert descriptor uses the coefficient bundle selected for its own
student policy.  There is no additional CMA-ES search, standalone training,
Webots fine-tuning, online update, or fallback.

## Review round 1: algorithm semantics

The missing evidence is not another student model.  It is the direct teacher
bound to the same coefficients as each student.  `GraphImitationExpert`
directly executes the Hungarian matching whose selected edges became binary
labels.  `LearnedHungarianExpert` directly executes the continuous cost matrix
that became Ridge regression targets.  Keeping these descriptors separate
answers how much each learned approximation loses relative to its own teacher.

An independently optimized `TrafficHungarian` remains a distinct reference.
It must not be substituted for either own-teacher comparison because its
coefficient hashes differ.

## Review round 2: evaluation and statistics

The evaluation reuses scenarios A/B/C and seeds 61000--61005, producing
`4 × 3 × 6 = 72` frozen Webots episodes.  Every manifest hash must match the
corresponding completed student evaluation.  Each result must have zero
learning, fallback, invalid scheduler output, timeout, and physical collision.

Comparisons are paired at `(scenario, seed)` before aggregation.  Confidence
intervals treat the six seeds as clusters and average the three scenarios
inside each seed.  The report contains both completed-task and utility deltas;
positive deltas always mean the own teacher performed better.  Each extension
step records three reviews over identical artifact hashes.
