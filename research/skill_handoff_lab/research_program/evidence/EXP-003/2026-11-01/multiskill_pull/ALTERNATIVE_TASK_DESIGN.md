# CPU source review: alternatives if PullTool has no conditional complementarity

No alternative physics or RL result is claimed here. The native PullCubeTool experiments must establish nontrivial parameter ranking before training.

## StackPyramid: reuse verified physical boundaries, change the question

`tools/handoff_physics.py::HandoffExecutor` already performs A grasp/lift/place/open/retreat and requires three consecutive valid release/stability observations. It then observes the actual A/B state, grasps C, and places C on their actual midpoint. The native horizon is 250 and is enforced by `NativeRecorder.step`. No reset separates these skills.

The old parameter pair is first A–B gap ±6 mm and then C center offset ±8 mm. Repeating that saturated grid is not a new opportunity test. A physically distinct action is the **placement axis of A relative to B**, which changes the support layout and the approach geometry required of C. Legal initial candidates could be 0, ±90, 180 degrees; a strong geometric baseline must screen table reach, collision geometry, C initial obstruction and complete successor planning cost, not blindly retain the old +X layout. At the second actual released/stable boundary, C center and approach can be recomputed from measured support poses. Only actual completed rollouts establish whether residual contact effects remain after this strong geometry.

The source-native objects are fixed 40 mm cubes. The native evaluator checks A/B proximity (about61.568 mm), top height, object static state and release. Varying cube size without revising the native evaluator would silently change the physical meaning of success and is rejected. A custom geometry task would have to be declared separately, so native initial-state layout/axis variation is the cleaner first experiment.

Useful falsification: if complete geometric axis search consistently chooses one successful layout and top midpoint always suffices, do not describe this as a learned cross-stage adaptation opportunity. Actual support-edge displacement/rotation after release and C contact outcome must provide information beyond complete current state, or the method must use that state directly without claiming extra posterior information.

## PokeCube: not preferred merely because it has a tool

Pinned ManiSkill `mani_skill/envs/tasks/tabletop/poke_cube.py` registers `PokeCube-v1` with only50 steps. It randomizes peg/cube placement and cube yaw, but the goal is always the cube's initial position+[0.1,0] with radius0.05. Native success requires cube inside that circle and robot static. Peg grasp/alignment are returned diagnostics, not extra native success requirements. Thus it is another candidate grasp→tool-contact chain but risks a horizon/geometry dominated benchmark; do not migrate solely to obtain two named phases.

## Decision before additional task engineering

Finish the bounded native PullTool comparison of fixed250/290 mm versus observed-geometry goal-distance selection on three additional scene seeds. If longest/goal rule dominates without conditional rank changes, preserve that negative result and use a new source-backed Pyramid placement-axis scout, retaining the existing strong release/stability boundaries and native success. No new network is warranted by parameter sensitivity alone.
