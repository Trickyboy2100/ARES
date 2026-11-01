# Component A evidence snapshot

CPU aggregation only: no new simulator interactions. Historical smoke versions remain separate; failed smoke costs are retained.

Finalized recorded cost: 1258 episodes, 251600 physical steps, 11520 optimizer updates.
Unfinished excluded from final totals: []; their indexed physical-step lower bound is 0. Unrecorded current episodes and updates are unknown.

| Group | Family | Success | Distinct initial scenes |
|---|---|---:|---:|
| controls_fast_full::search_unique_fast | ALL | 30/36 | 6 |
| controls_fast_full::search_unique_fast | Y | 18/18 | 6 |
| controls_fast_full::search_unique_fast | YZ | 12/18 | 6 |
| controls_fast_task::search_unique_fast | ALL | 30/36 | 6 |
| controls_fast_task::search_unique_fast | Y | 18/18 | 6 |
| controls_fast_task::search_unique_fast | YZ | 12/18 | 6 |
| controls_fast_y_only::search_unique_fast | ALL | 25/36 | 6 |
| controls_fast_y_only::search_unique_fast | Y | 18/18 | 6 |
| controls_fast_y_only::search_unique_fast | YZ | 7/18 | 6 |
| controls_full::geometry | ALL | 11/36 | 6 |
| controls_full::geometry | Y | 7/18 | 6 |
| controls_full::geometry | YZ | 4/18 | 6 |
| controls_full::search | ALL | 30/36 | 6 |
| controls_full::search | Y | 16/18 | 6 |
| controls_full::search | YZ | 14/18 | 6 |
| controls_legacy::geometry | ALL | 4/12 | 2 |
| controls_legacy::geometry | Y | 2/6 | 2 |
| controls_legacy::geometry | YZ | 2/6 | 2 |
| controls_legacy::rls | ALL | 6/12 | 2 |
| controls_legacy::rls | Y | 2/6 | 2 |
| controls_legacy::rls | YZ | 4/6 | 2 |
| controls_legacy::search | ALL | 11/12 | 2 |
| controls_legacy::search | Y | 5/6 | 2 |
| controls_legacy::search | YZ | 6/6 | 2 |
| controls_rls::rls | ALL | 8/12 | 2 |
| controls_rls::rls | Y | 3/6 | 2 |
| controls_rls::rls | YZ | 5/6 | 2 |
| controls_task::search | ALL | 32/36 | 6 |
| controls_task::search | Y | 18/18 | 6 |
| controls_task::search | YZ | 14/18 | 6 |
| controls_y_only::search | ALL | 24/36 | 6 |
| controls_y_only::search | Y | 18/18 | 6 |
| controls_y_only::search | YZ | 6/18 | 6 |
| eval_full::model | ALL | 31/36 | 6 |
| eval_full::model | Y | 16/18 | 6 |
| eval_full::model | YZ | 15/18 | 6 |
| eval_full_2048::model | ALL | 32/36 | 6 |
| eval_full_2048::model | Y | 17/18 | 6 |
| eval_full_2048::model | YZ | 15/18 | 6 |
| eval_full_legacy::model | ALL | 10/12 | 2 |
| eval_full_legacy::model | Y | 5/6 | 2 |
| eval_full_legacy::model | YZ | 5/6 | 2 |
| eval_task::model | ALL | 35/36 | 6 |
| eval_task::model | Y | 18/18 | 6 |
| eval_task::model | YZ | 17/18 | 6 |
| eval_task_2048::model | ALL | 31/36 | 6 |
| eval_task_2048::model | Y | 17/18 | 6 |
| eval_task_2048::model | YZ | 14/18 | 6 |
| eval_y_only::model | ALL | 33/36 | 6 |
| eval_y_only::model | Y | 18/18 | 6 |
| eval_y_only::model | YZ | 15/18 | 6 |
| eval_y_only_2048::model | ALL | 26/36 | 6 |
| eval_y_only_2048::model | Y | 16/18 | 6 |
| eval_y_only_2048::model | YZ | 10/18 | 6 |

Training windows describe collected exploratory training episodes only (SAC optimization is off-policy); they do not show validation convergence. Same scene with multiple biases is not an independent scene draw. Pair tables retain initial-SHA and physical-prefix checks. PAIRS.json adds exploratory paired scene-cluster bootstrap intervals (20,000 resamples; fixed seed 20261101): all six bias/family conditions move together. Only six independent scenes exist, one used for stop engineering; a separate five-scene sensitivity excludes it. These intervals are not final statistical proof. No novelty or generalization claim follows from this component comparison.

Warnings requiring review: 0. Exact source/trace/checkpoint provenance is in RAW_SHA_INDEX.json.
