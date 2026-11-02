# Matched physical pair: frozen attachment versus one-shot transfer

**OBSERVED:** seed745000, load0.75, legal grasp35mm. Both actual runs share the exact first45 control-step records and verified PICK_TOOL boundary. Native terminal cube/base distances are835.20mm (fail) and597.42mm (pass), with unchanged600mm native threshold and100-step horizon. Transfer lasts21 versus30steps. Raw video preserves the measured geometry; boxes are projected cube bounds and may be occluded, not measured contact patches.

**INTERPRETATION:** segmented replanning changes the executed approach and later contact; the result does not require new attachment information. Read-only contact recordings and endpoint FK audits support this. The large relative-tool rotation begins after tool/cube contact, not before it.

**NOT ESTABLISHED:** a new RL mechanism, hidden-state necessity, universal stabilization, independent-seed gain, or a full G2A wrench evaluation. Fixed settling9 is a negative control. Finer screw integration rescues/regresses depending on grasp; low endpoint error alone does not guarantee task success.

**Provenance:** MEASURED_POSE_REPLAY; zero newphysics;101frames at20fps, frame k corresponds to stored physical step k. Original physics at100Hz, control20Hz. RAW and ANNOTATED share source states and camera; no artificial target region. Source and output SHA in MANIFEST.json. DATA curves exclude frame0 unvalidated contact buffer. Controller mode is a predeclared intervention, not a learned boundary decision in these two videos.
