# Full two-skill physical evidence

OBSERVED: Scene743020, density0.75, target y=-0.08. The first skill parameter and every measured prefix state are exactly equal. Keeping the planned hook succeeds; measured-state value replanning chooses another legal hook and reaches the native horizon before completing the pull.

INTERPRETATION: Posterior action-value misranking and geometric budget can outweigh the benefit of actual-state information. Deterministic budget guard is separately tested as a strong control.

NOT ESTABLISHED: This is neither proof against feedback generally nor a new RL method. Only one learned-model seed and development scenes. Selection occurred after outcomes were inspected.

MEASURED_POSE_REPLAY: saved actual physics, zero new physics. RAW is unannotated measured-pose rendering, not online capture. Frame0=initial; frame k=actual control step k at k/20 seconds. Yellow ring: projected declared40mm tolerance (occlusion-unaware geometric overlay). Colored box: actual cube geometry bounds; neither is a measured contact patch. Contact force is a simulator proxy; frame0 excluded.
