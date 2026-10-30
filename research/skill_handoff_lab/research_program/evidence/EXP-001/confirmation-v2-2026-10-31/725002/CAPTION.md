# Seed 725002: contact-mode diagnostic (inspected development case)

**OBSERVED.** Identical recorded ALIGN boundary, same native 200-step task. Original servo / depth-preserving servo final successes: 0 / 1. Mode switches: 26 / 3; measured backward axial travel during servo: 41.37 / 0.00 mm. This is accumulated negative displacement, not net insertion loss. RAW and ANNOTATED show exactly the same two images at physical step 110, selected by largest old-servo one-step retreat. Rings project measured actor origins using the actual camera matrices; they do not assert a visible contact point. TRACE uses post-step physical states; servo mode is the action selected from the preceding observation. The goal-triggering action may lack a servo annotation because GoalReached interrupts logging after physical stepping.

**INTERPRETATION.** Returning to the initial standoff when alignment is temporarily lost can undo insertion progress. Holding current depth removes that commanded retreat. Both controllers retain integration and contact dynamics, so residual backward motion can remain.

**NOT ESTABLISHED.** No limit-cycle/stability proof, calibrated force sensing, new RL gain, or independent confirmation from these already-inspected cases. Simulation contact force is a proxy. Privileged object poses are used. Camera replay adds zero physics steps and is not real hardware. Native success requires position <=5 mm AND angle <=0.2 rad at step 200.

Sources: RAW/ANNOTATED = MEASURED_POSE_REPLAY; TRACE = DATA_DERIVED_PLOT. NO NEW RL CLAIM.
