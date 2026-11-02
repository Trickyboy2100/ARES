# Same-state controller mechanism, not an RL improvement

OBSERVED: seed744000, support-gap correction0, desired top yaw45 degrees, top offset0. The complete250-step native trajectories share identical initial state and all recorded support-skill transitions. Nearest equivalent wrist orientation fails at C_PREPLACE; bounded feasible-equivalent selection completes the original native task and the additional yaw contract. Both movies contain251 frames at20fps, frame0 initial state, frame k physical step k. Six clickable curves share this frame index. Source is MEASURED_POSE_REPLAY, zero new physics; measured state reconstruction error1.19209e-7.

INTERPRETATION: checking equivalent cube orientations removes a kinematic planning failure. This supports retaining a deterministic controller primitive. It does not establish new RL gain or general contact adaptation. All terminal outcomes shown are offline annotations, not policy observations.

NOT ESTABLISHED: measured contact forces, real robot performance, dynamic safety certification, unseen-scene generalization or an independent learning mechanism. The inherited stride3 path executor is retained and disclosed. The requested yaw is modulo90 because this is an unmarked square cube; this experiment is an adaptation variant, not the unchanged benchmark leaderboard.
