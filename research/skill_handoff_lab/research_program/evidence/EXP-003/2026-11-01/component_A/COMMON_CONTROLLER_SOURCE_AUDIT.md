# Common training/controller source audit

**PASS** — compared actual saved snapshots, not current working files. All manifest source hashes verified.

`make_episode`, `BaselineEnv`, `CompletionGate`, complete `Episode` including `step`, `SkillEnv` including reward, the entire training branch, and SAC construction are AST-identical across all three runs. Every other common source file is byte-identical. Runtime arguments differ only in declared authority and output path.

Full and Y-only source packages are identical. Task adds exactly the AuthoritySearchControl import, helper snapshot listing and `search_unique` evaluation/control branch. Removing only those three explicit changes makes the complete entry-file AST identical. The later `search_unique_fast` branch is **absent** from these training snapshots. No common training/physical/reward/stop implementation drift was found.

Shared SAC: critic LayerNorm, [128,128], learning rate 0.0003, buffer 50000, warmup 256, batch 128, one update per decision, alpha 0.02, gamma 0.99, tau 0.005, seed 31101, 4096 decisions. Gamma remains per decision, not duration-correct SMDP.

This static audit does not establish identical floating-point rollouts. Three-dimensional projected SAC remains a known baseline with the previously disclosed entropy/action-alias limitation, not a novel algorithm. Exact block/file/manifest hashes and source paths are in the companion JSON.
