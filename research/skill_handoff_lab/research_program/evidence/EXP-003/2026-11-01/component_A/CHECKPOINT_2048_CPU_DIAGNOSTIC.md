# Checkpoint2048 CPU actor/critic diagnostic — NOT_PHYSICS_OUTPUT

The actual saved full-authority checkpoint was loaded on CPU. Four raw21-dimensional observations, six zero feature slots and two declared-family bits reconstruct the92-dimensional input; actual hidden bias and future outcomes are not inputs. The checkpoint and all source trajectories have SHA256 in the JSON. No new physical interaction or optimizer update occurred.

First10 and recent10 **complete training episodes before the2048-decision boundary** were inspected at first suffix and, when present, first measured contact. The current frozen checkpoint predicts actions on those old states; it is not replayed as a new physical policy.

| Recorded state group | Historical exploratory behavior success | Current checkpoint mean first action(Y,Z,advance) | min-Q prefers actor over zero | min-Q prefers actor over zero-Z |
|---|---:|---|---:|---:|
| First10 training episodes | 0/10 | (-0.6646,+0.4906,+0.7874) | 9/10 | 7/10 |
| Recent10 beforecheckpoint | 10/10 | (-0.5785,+0.6014,+0.8208) | 10/10 | 8/10 |

The historical training improvement is useful evidence of learning activity, **not held-out convergence or a10/10 checkpoint score**. Initial distributions differ between windows, and actual historical actions were exploratory. Relative to earlier tiny pilots, the actor now selects positive Z on average, but this does not by itself establish suitable task-dependent Z behavior.

The JSON also contains Q1/Q2 over the9 candidate Y/Z offsets[-.8,0,.8] with nominal advance, the actor and zero-Z actions, and actor-action gradients. These are predicted rankings, not physical counterfactual outcomes. Soft Q contains reward/entropy effects and is not a success probability. Use the forthcoming independent evaluation and failure classification to decide whether these learned preferences are useful.
