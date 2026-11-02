# Mechanism scope and prior art — 2026-11-02

The tested candidate uses a measured finite-difference terminal-effect map to orient the existing two-dimensional parameter box, then ordinary finite-horizon SMDP SAC. It adds no physical authority, no future-state policy observation and no different reward. It is an experimental representation hypothesis, **not established originality**.

- [LASER (ICRA2021)](https://arxiv.org/abs/2103.15793): already learns action representations using reconstruction and dynamic consistency for contact-rich RL. Merely learning a useful action space is not new.
- [ILA (2026 preprint)](https://arxiv.org/abs/2610.18855): jointly represents motion/contact/force evolution as action units for flow policies and offline improvement. A physical-effect representation alone is not a distinct paper claim.
- [PA-RL (2026 preprint)](https://arxiv.org/abs/2609.21609): uses parameterized potential fields with impedance execution, compared under the same RL algorithm. Our fixed planner interface differs, but that difference alone is not novelty.
- [RAPS (NeurIPS2021)](https://proceedings.neurips.cc/paper/2021/hash/b6846b0186a035fcc76b1b1d26fd42fa-Abstract.html): parameterized primitive interfaces are inherited background, not a contribution.
- STAP: the existing pinned log-Q/CEM implementation is retained. This run fits new boundary-success and transition models on exactly the shared physical dataset; this is a functional same-interface comparator, not an official benchmark reproduction or a converged released STAP policy.

The physical matrix distinguishes three claims: a post-contact parameter changes outcomes (observed); conditioning predicts effect magnitude/direction (bounded leave-scene-out evidence); conditioning improves whole-task learning beyond strong predecessor choices and ordinary RL (must be tested separately). No one of these implies the next. The new panel's joint grasp/contact oracle does not exceed its best nominal-grasp oracle.

The measured target is a finite terminal displacement under native100 including planner failure and budget consumption, not an instantaneous contact Jacobian. The same-state mapping uses measured tool-frame cube geometry, attachment and budget; these features are already available to ordinary RL.

The action-map Jacobian check covers the invertible parameter representation before the common no-op deadzone. Executed commands include a no-op atom; no claim of exact differential entropy over physical command atoms is made. See ENTROPY_SCOPE_ADDENDUM.json.
