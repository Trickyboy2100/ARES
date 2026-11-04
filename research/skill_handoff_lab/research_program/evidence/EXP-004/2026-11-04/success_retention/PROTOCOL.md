# Successor retention: declared comparison

Base a2fa4a6. Same PegInsertionSide-v1 native100, v4 persistent feedback controller,8 successor actions and64 inputs. No physical handoff resets. Fixed predecessor grasp1 isolates successor learning; this does not establish joint task-conditioned improvement. Development753200–215 is already seen. New native training initial states756000 onward are training, not held-out tests.

All branches start from the same ordinary96-episode checkpoint (31501), with its real replay, optimizer and sampler states. Seed31601 controls new behavior sampling, not independent neural initialization. Forty updates/full native episode, successor-only batches. Online CQL remains0. All original head updates are inherited for compatibility; no head changes or head-based deployment. The first factor is successful-trajectory Actor cross-entropy weight0 versus0.2. No new algorithm claim. This regularizes the actual recorded successful action, not a fabricated zero-correction target.

Second factor: same epsilon0.1 uniform exploration, with deterministic instead of stochastic Actor for the remaining0.9. Deployment remains deterministic in all branches; all8 actions retain support during training. This simple mature exploration control is NOT an exact Policy Decorator reproduction.

Action0 continues the active queue, including a previously modified queue. It cannot undo a replan. A3mm target shift can replace the entire remaining joint path; it is not a3mm bound on physical deviation. Action7 ends a refinement after one step, or holds up to4 in insertion. No deployment safety mask or nominal-path reset is introduced.

Primary outcomes: complete terminal native success; rescue on nominal failures; regression on nominal successes; all actual decisions/physical steps/planner calls. Same measured prefix is checked against the fresh nominal reference. Earlier targeted5 regressions are not an overall estimate.

## Primary literature and code examined

- [Policy Decorator paper](https://arxiv.org/html/2412.13630v1), §4.2; [official implementation](https://github.com/tongzhoumu/policy_decorator/blob/main/online/pi_dec_diffusion_maniskill2.py), lines466–479: progressive Bernoulli activation and bounded additive residual. Our queue replacement has different semantics, so its residual-magnitude guarantee does not transfer.
- [DAWN paper](https://arxiv.org/html/2602.10539v1), §3 and AppendixF.2; [official source](https://github.com/Guozheng-Ma/DAWN). Base-policy data anchoring and critic normalization are mature controls. DAWN reports that progressive exploration can become unnecessary with effective value learning; protection is a competing hypothesis to test, not an assumed solution. Existing local source snapshot `peg_value_20261103/prior_code/DAWN_online_dawn_pi_dec_diffusion_maniskill2_dawn.py` inspected around warmup, anchor/probe masks and updates.

Claims remain UNVERIFIED pending actual results. No ARES-R edits, hardware or sealed tests.

## Within-session revisions, after the first real results

1. At32 new native episodes, neither removal nor persistent BC changed9/16. Add4000 identical-count off-policy updates to each own real replay; separate these from physical training cost.
2. Removal then8/16; persistent9/16 with one rescue and one regression. Controlled sampling then10/16 with two rescues and one regression. Complete the fourth factorial cell (persistent BC + controlled sampling) without changing weights, actions or execution.
3. Successful-replay BC raises replan probability because success labels also attach to exploratory actions. Test constraint-source selection using **exactly the same removed/FINAL model and real replay**:64 extra successful anchor rows/update, weight0.2, all-success versus whole-episode all-zero-success anchors. Critic replay retains all positive/negative experiences. No new physical training data or hypothetical outcomes. This is mature data anchoring/BC, not originality.
4. The observed late-handoff rescues motivate posthoc same-authority rule comparisons: action4 in REFINE3 with≤15steps once; or action4 throughout refinement with≤28steps. Thresholds derive from seen development and cannot count as independent validation. No change to shared planner, action bounds, native100 or success criterion.
5. Both common-replay anchoring comparisons return nominal9/16 without rescue, while all-success anchoring remains regressive. One bounded coefficient revision lowers only nominal-anchor weight0.2→0.05 on the controlled-sampling replay. This is a tuning experiment, not a new method or a preregistered test.

All sampler comparisons use one inherited neural initialization31501 and behavior seed31601. Replacing stochastic Actor sampling also changes consumption of its shared NumPy generator; minibatch draws across behavior branches are not claimed identical. The later anchoring studies use identical parent model/replay and matched draw counts. Native measured-prefix comparisons are separate and are checked numerically.
