# ARES self-built contact-composition pilot, 2026-10-16

**All data here are synthetic research-task simulation results, NOT October field point clouds, robot readbacks, site geometry or official benchmark results.** Field records were not used in these new task labels or figures. Retained historical WideCabinet/field results are separate and unchanged.

Start with `../../reports/CONTACT_COMPOSITION_RESULTS_20261016.md`, the method/reproduction notes and paper working draft. `RESULTS.json` contains every exploratory and formal run, including failures. Formal v5 uses three paired seeds, 32768 actual online interactions each, and seven trained checkpoints including the one-seed all-phase credit ablation.

`contact_v5_*` holds training/evaluation episodes, logs, exact executed source snapshots and checkpoints. `heldout_*` holds the same frozen unseen-geometry cases for all policies; `causal_*` are post-hoc stage interventions on the first training seed, not independent multi-seed method confirmation. Development cases were used while debugging. `contact_single_execution_demo` is a four-case policy execution demo (two successes and two failures), not another training result.

`media/actual_16env_training_and_evaluation.gif` samples genuine online training followed by evaluation; per-frame physics steps/state provenance are retained. `actual_single_execution_2success_2failure.gif` uses a render-only color override with identical nonappearance USD attributes, checked in DISPLAY_ASSET_INVARIANCE.json. Neither is a fixed trajectory loop. No contact grasp is claimed. Figures label raw and independently confirmed results separately.

**ARTIFACT_MANIFEST.json hashes the full local archive, not just this Git subset.** The full ~250MB archive (including 120Hz physical state traces and generated robot assets) remains in the research workspace `runs/skill_composition_20261016_r0/`. RELEASE_MANIFEST.json hashes exactly the released subset. Checkpoint SHA values can be matched to the full manifest. Do not interpret omitted large files as nonexistent experiments.

Results are not yet a publishable method advantage: high seed variance; boundary-vs-all-phase seed31 policies are exactly equal; sampled terminal velocity audit disagreements retained conservatively. Independent checks are limited to sampled tool/workpiece/table/rail geometry, FK and soft limits, not full robot swept collision or field safety certification. Production ARES-R Beam/contact search is NOT_TESTED on these new tasks.

Loading a checkpoint uses the same 33D observation and two-dimensional bounded Cartesian action interface from the captured source. Rebuilding the existing JAKA asset and deploying Isaac Sim5.1/Lab2.3.0 is required; weights alone are not a portable real-robot controller. No driver changes, reboots or real-robot execution were performed.
