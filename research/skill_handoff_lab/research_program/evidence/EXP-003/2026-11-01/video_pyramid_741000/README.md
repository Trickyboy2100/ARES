# Pyramid full-task shared-controller repair

Open [SYNC_VIEWER.html](SYNC_VIEWER.html). The same-initial branches contain the complete250-step tasks, including support placement, open/retreat/settle verification and the actually permitted successor skill. Left has one decision because support verification fails; right has decisions at0 and78 and reaches native success. **SHARED_CONTROLLER_REPAIR_NOT_RL_GAIN**.

- [RAW.mp4](RAW.mp4), [ANNOTATED.mp4](ANNOTATED.mp4): full251 frames; switch while retaining time.
- [TRACE.png](TRACE.png), [TRACE.svg](TRACE.svg): A/B geometry, existing controller margin, gripper opening, C height and cube speed.
- [CAPTION.md](CAPTION.md): observed facts and causal/algorithmic limits.
- [LEFT_SOURCE.json.gz](LEFT_SOURCE.json.gz), [RIGHT_SOURCE.json.gz](RIGHT_SOURCE.json.gz): exact original250-step full-state traces and real boundary decisions.
- [DATA.json](DATA.json), [MANIFEST.json](MANIFEST.json), [QA.json](QA.json): source/model/code hashes, source-derived series, full-state/frame/browser verification.
- [ANNOTATED_100.png](ANNOTATED_100.png), [ANNOTATED_150.png](ANNOTATED_150.png), [ANNOTATED_250.png](ANNOTATED_250.png): individually inspectable same-time keyframes, with matching RAW files.

Regenerate using the network-isolated runtime and `research/skill_handoff_lab/tools/program_pyramid_video.py --left LEFT_SOURCE.json.gz --right RIGHT_SOURCE.json.gz --out NEW_DIRECTORY`; CPU validation uses `program_pyramid_video_qa.py --out NEW_DIRECTORY`. Zero env.step calls in the renderer. Future curve samples and “OFFLINE final result” are reviewer-facing analysis, not current-time success or policy inputs. Source-backed decision and SkillResult cards remain gated by physical step.


Contact visibility update: ANNOTATED.mp4 includes a synchronized **2x actual-RAW-pixel inset at steps 40–90**, projected around measured A/B positions. [CONTACT_CLOSEUP.png](CONTACT_CLOSEUP.png) is a 3x same-frame comparison at step68 (3.40s), when both branches are A_OPEN; it makes the recorded finger orientations visible. The HTML includes an expandable detail panel. Nearest-neighbor enlargement adds no geometry or resolution; no contact force is inferred from pixels. RAW.mp4 is unchanged. CONTACT_CROPS.json records every source pixel rectangle; contact_zoom_source.py reproduces the CPU operation. The original full video remains locally at `/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-003/video_pyramid_741000/`.
