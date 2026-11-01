# Endpoint-height intervention: six actual native100 episodes

Resource context: **SHARED_GPU_CONTENTION** with a separately authorized A training process on the same GPU0. Wall/planning durations are not exclusive-performance measurements. All six first-skill physical prefixes match their original .08m controls exactly, including actions and all recorded state arrays.

All600 control steps and3000 actual100Hz substeps are retained. Success2/6, rescued0, degraded0 relative to matched .08m cases. This panel has only two native seeds, not six independent scenes.

| Scene | Grasp x | Transfer endpoint z | Native success | Final base distance | TRANSFER tool/cube peak | Error |
|---|---:|---:|---|---:|---:|---|
|740000|0.070|0.10|False|0.867200m|0.000N|Horizon: NATIVE_HORIZON|
|740000|0.070|0.12|False|0.864728m|0.000N|Horizon: NATIVE_HORIZON|
|740001|0.035|0.10|False|0.994809m|5.326N|Horizon: NATIVE_HORIZON|
|740001|0.035|0.12|False|0.979294m|3.485N|Horizon: NATIVE_HORIZON|
|740001|0.070|0.10|True|0.579728m|0.000N|None|
|740001|0.070|0.12|True|0.572387m|0.000N|Horizon: NATIVE_HORIZON|

The intervention changes the endpoint height in both shared geometric search and execution. It does not constrain the entire swept path. Failed g035 still contacts the cube during TRANSFER at higher endpoints; this supports auditing path clearance and grasp compliance, not concluding that physical adaptation has no opportunity. The inherited controller is not fully collision-aware: cube and held-tool swept collision bodies are absent from its planning world.

The two successful g070 episodes remain successes. One ends with a horizon diagnostic during motion while native terminal distance satisfies its actual task criterion; both facts are retained rather than silently rewriting the failure log.

No new RL method, value predictor, or learned conditional advantage was tested. A staged vertical-first transfer would be a deterministic controller test, shared by all later policies. Broader grasp dominance (including g105) and state-conditioned second-action ranking remain unresolved.
