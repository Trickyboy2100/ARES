# Component A independent scientific review

CPU-only source/raw-data review; no additional physics or optimizer updates.

**Actual completed full-authority baseline:**4096 decisions,3840 updates,245 complete200-step episodes/49000 physical steps, including one forced training-tail episode. Final physical development evaluation is **31/36**, Y16/18 and YZ15/18. Its five failures never met native success, never entered completion hold and used zero settling-probe steps; final distance16–20mm indicates entrance/progress failure. These failures do not support further threshold tuning as the primary remedy.

## What is and is not fair

- All three policies receive identical92-dimensional information, including the declared Y/YZ family and actual applied-action history; no hidden bias value or future native-success input was found. Ground-truth terminal reward is ordinary simulator supervision, distinct from forbidden actor/stop oracle access.
- `full`, `y_only`, and `task` are **three-dimensional SAC actors with environment action projection**. Projected Z remains in SB3's requested-action replay, critic and entropy calculation, even when physically ignored. This is a valid common-network projection experiment, but **not a strict two-dimensional baseline with matched effective-action entropy**. Any gain must first be attributed to this interface package, not solely to physics-aware action dimension or a new contract method.
-4096 decisions do not imply identical physical steps, episode count, planning budget or scene exposure. Different completion times change those costs and coverage. Report each counter rather than declaring equal total data cost. gamma.99 is per component decision, not duration-correct SMDP discounting.
- Training samples YZ from six signed offset combinations; evaluation uses three of them. It is a declared finite development subset, not broad physical generalization. Known Y/YZ context is a shared calibration-family prior, not learned hidden-parameter identification.
- Seed732000 was already used for stopping-contract smoke development. Therefore the whole36-condition result is **development evaluation**, not six wholly unseen scenes. Preserve31/36 as the primary total and separately report the five-scene sensitivity excluding732000; never delete the reused scene to improve an apparent test score.
- Current observable hold uses shared biased geometry and frame-rate motion; no true native success is used to trigger it. In these36 full-policy trials, no hold entry occurred before native success. This empirical fact is not a proof of safe stopping under all bias/velocity conditions. Frame-rate stability can alias high-frequency physical oscillation.

## Remaining evidence needed

The y_only/task results and fixed2048-checkpoint evaluations are pending at this review snapshot. Compare actual physical outcomes on identical seed/bias/family and initialSHA, report recorded prefix equality separately, retain failures, and avoid selecting a checkpoint using these inspected scores. The frozen2048 and4096 CPU actor/Q reports are model diagnostics only; historical training success is not a checkpoint test.

A mature baseline may be substantially better than the earlier1024-decision pilot; that is baseline repair/training evidence, not an ARES method contribution. The multi-skill paper claim still requires real predecessor parameter effects and successor decisions beyond this component task.

## Additional deterministic strong comparator (declared before its runs)

The original pattern search fixes positive advance at 2 mm although the shared actor may request 4 mm. Full4096 failure diagnostics show frequent upper-bound advance. To avoid crediting RL for a simple untested nominal setting, `search_unique_fast` combines authority-aware deduplication with 4 mm positive advance, preserving any negative/backoff command, the existing alignment/IK/joint-increment gates, native200 budget and shared completion. It will be evaluated on the same36 conditions for all three authorities. Full authority has the same nine search points, so its comparison isolates advance; Y-only/task also change point deduplication and are composite strong baselines, not a clean single-factor ablation. This deterministic comparator is not a new method. Existing search results remain preserved. Pure dedup-only has CPU logic checks but no separate physical score unless explicitly recorded later.
