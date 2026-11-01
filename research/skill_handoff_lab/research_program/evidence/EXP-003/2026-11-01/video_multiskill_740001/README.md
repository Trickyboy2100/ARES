# Full two-skill evidence — open one HTML

Open [SYNC_VIEWER.html](SYNC_VIEWER.html). Choose RAW/ANNOTATED, click any curve, or click either branch's first/second skill-decision button. Playback, parameter cards, measured SkillResult and curves share the same physical step. This is the strong geometry baseline versus a35mm legal grasp intervention; **no new RL gain is claimed**.

- [RAW.mp4](RAW.mp4), [ANNOTATED.mp4](ANNOTATED.mp4): all101 frames, complete native100-step task.
- [TRACE.png](TRACE.png), [TRACE.svg](TRACE.svg): full trajectory; dotted lines indicate each branch's second decision.
- [ANNOTATED_000.png](ANNOTATED_000.png), [ANNOTATED_050.png](ANNOTATED_050.png), [ANNOTATED_100.png](ANNOTATED_100.png), with corresponding RAW snapshots.
- [LEFT_SOURCE.json.gz](LEFT_SOURCE.json.gz), [RIGHT_SOURCE.json.gz](RIGHT_SOURCE.json.gz): complete byte-exact physical records, including both decisions and SkillResult.
- [DATA.json](DATA.json), [MANIFEST.json](MANIFEST.json), [QA.json](QA.json): signals, source/model/code SHA, camera matrices, encoded-frame and browser checks.
- [CAPTION.md](CAPTION.md): observed facts, interpretation, limits and invalid initial contact-buffer handling.
- [QA_BROWSER.html](QA_BROWSER.html): executable browser interface test; actual decoded media time and event seeks.

Reproduce using the network-isolated runtime and `research/skill_handoff_lab/tools/program_multiskill_video.py --left LEFT_SOURCE.json.gz --right RIGHT_SOURCE.json.gz --left-label '35 mm grasp intervention + strong suffix search' --right-label 'Strong geometry lookahead + suffix search' --out NEW_DIRECTORY`. Run `program_multiskill_video_qa.py --out NEW_DIRECTORY` with the available CPU Python for source/frame/browser validation. The generator never calls env.step. Video display geometry and source traces remain unchanged by annotation.

QA:101 frames each, allPTS=k/20; max state readback error1.1920929e−7; same initial hash; four source-backed decision buttons; no future SkillResult atstep0. Chrome seeks the2.05s event to2.049999s (1µs media-time quantization, far below the50ms frame period), recorded explicitly. All cube and tool bound corners remain in frame.

**Offline-analysis visibility:** the full-episode curves include future samples relative to the playback cursor, and “Full-task final” always denotes the recorded terminal outcome, not success at the current frame. These are reviewer-facing offline analysis; future samples and terminal labels were not policy observations. Parameter and measured SkillResult cards are separately restricted to records whose actual decision/completion boundary has already occurred.

**Matched-repeat100Hz contact addendum.** Original20Hz contact samples remain unchanged and separately plotted. A distinct instrumented physical repeat records5 actual100Hz substeps per control step. All100 original action/q/qvel/TCP/tool/cube/attachment samples match exactly, with the same initial hash; the full zero-difference audit, original/probe SHA and portable probe traces are in MANIFEST.json. The extra curve is the within-frame maximum over ((k−1)/20,k/20], not an instantaneous20Hz sample and not originally captured in the video episode. Failure-side first tool contact above0.5N is frame58/substep2, TRANSFER,4.011N (2.87s); successful-side first is frame69/substep3, HOOK,4.083N (3.43s), with overall peak16.614N. All recorded robot-link/cube forces in those1000 probe substeps are zero. The original sparse samples missed the failure's transfer collision; they cannot justify a no-tool-contact inference.

[TRACE_CONTACT_PHASE.png](TRACE_CONTACT_PHASE.png) compares original20Hz and matched100Hz peak signals with actual phases. [LEFT_CONTACT_REPEAT.json.gz](LEFT_CONTACT_REPEAT.json.gz) and [RIGHT_CONTACT_REPEAT.json.gz](RIGHT_CONTACT_REPEAT.json.gz) retain complete secondary-source records. The physical repeat budget belongs to the contact-probe experiment and is not counted again as video generation. The old full video package is retained locally at /home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-003/video_multiskill_740001_pre_contact_package.
