# DATA_DERIVED_PLOT / MEASURED_POSE_REPLAY

OBSERVED: Same native initial state, seed743041, mass scale1.25, goal y+80mm. Fixed100mm grasp /−65mm hook fails (native0, target error68.3mm);90mm grasp /same hook succeeds (native1,6.2mm). Both pass the real stable-grasp boundary; measured attachment x differs by about9.4mm. The yaw search also selects a different angle, as permitted by the same controller.

INTERPRETATION: The front-skill parameter affects attachment and subsequent geometric execution. This is physical controllability, not an RL gain or proof of target-dependent ranking reversal.

NOT ESTABLISHED: A single pair does not establish generalization or independent method novelty. The96-episode matrix is the quantitative context.

Full101 frames: initial0 + actual100 steps at20Hz. No new physics steps. Colored actor boxes and target circles are projected geometry, not measured contact patches. Frame0 force-buffer readback is unvalidated and excluded.
