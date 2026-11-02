# Three strict STAP posterior regressions

Exact saved prefix equality rechecked, not just shared reset seed. JSON contains full81-action Q sweeps and candidate planning costs. These are development cases from prior experiments.

| case | initial sequence predicted success | posterior Q(retain → selected) | actual consequence |
|---|---:|---:|---|
|743020, mass.75, target−80mm|.999738|.000117→.000129|−60.17→−55mm; remaining61, proposed minimum predicted184 steps. HOOK consumes last33 steps; no PULL. Prior executed sequence succeeds.|
|743020, mass.75, target+80mm|.999829|.000963→.001765|−71.58→−55mm; candidate minimum51 versus remaining61. Actual HOOK/PULL take extra5 steps; feedback budget shrinks12→8. Target error29.98mm, but native region fails.|
|743021, mass1.25, target−80mm|.999959|.000056→.000140|−65.28→−55mm; candidate minimum45 versus remaining63. Native succeeds; target error43.31mm versus prior38.87mm. Not a simple timeout.|

Predicted vs actual tool XYZ errors:172.05,171.97,84.08mm; predicted cube errors140.57,142.75,16.35mm. Measured post-grasp states already include attachment, velocities, remaining budget, mass and target. No evidence that recurrent history is required. The skill critic's posterior ranking is poor even though it expresses near-zero absolute success. CEM still must return an argmax; clipping at−55mm is not itself the cause, and an expanded bound is not justified by this evidence.

Temperature calibration is monotone within a state. It cannot fix a wrong ranking mathematically. A low-confidence retain rule would select the already-executed prior sequence on these three known cases; that is an ex-post diagnostic, NOT three new successful physics runs. Prior budget guard rescued only one. Original STAP already has SCOD uncertainty filtering and risk-aware alternatives; posterior information, uncertainty or budget control alone are not novel contributions.

The new numerical regression also distinguishes probability saturation from scientific effects: float32 sigmoid values can all become1.0; logits and log-success retain ordering. `NUMERICS.json` checks three controlled cases (zero simulation evidence). Initial112 policy runs using probability ranking are retained as diagnostics; only separately executed corrected runs support the main comparison. New terminal contract auxiliaries are measured outcome labels, not certified geometric/IK reachability proofs.
