# DATA_DERIVED_PLOT / MEASURED_POSE_REPLAY

OBSERVED: Seed743051, mass scale.75, goal y+80mm; ordinary and relative+contract methods share exactly the same recorded first-skill prefix and90.78125mm grasp. Ordinary selects−65.625mm hook and finishes43.88mm from goal; candidate selects−62.5mm and finishes33.74mm away. Both retain the original native task criterion. The candidate rescues this complete task under the same100-step horizon.

INTERPRETATION: This is a real successor-action effect after an actual SkillResult. Selection uses stable logit ranking and frozen trained weights.

NOT ESTABLISHED: Generic auxiliary, contract-only and relative-only controls also rescue this case. This video does not show an independent contract/relative-method advantage. See the full paired results, including failures. Initial probability-ranked experiments are separate diagnostic runs.

Full101 frames: initial0 + actual100 steps at20Hz. No new physics steps. Colored actor boxes and target circles are projected geometry, not measured contact patches. Frame0 force-buffer readback is unvalidated and excluded.
