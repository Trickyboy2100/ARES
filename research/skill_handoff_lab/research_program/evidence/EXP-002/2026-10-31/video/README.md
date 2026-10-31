# Full-episode synchronized physical evidence

Open [SYNC_VIEWER.html](SYNC_VIEWER.html) locally. Click any trace to seek; switch RAW/ANNOTATED without losing position; Previous/Next event jumps to measured contact onsets and contact-servo rollback events. Four linked panels show axial position, lateral and angular error, and force proxy. Current phase/mode appears above the video.

- [RAW.mp4](RAW.mp4): unannotated paired native-model pose replay, full 201 frames.
- [ANNOTATED.mp4](ANNOTATED.mp4): matching frames, native geometric guides, measured signals and mode strips (blue ALIGN, amber ADVANCE, green HOLD, grey prefix).
- [RAW.png](RAW.png), [ANNOTATED.png](ANNOTATED.png): same step150.
- [TRACE.png](TRACE.png) / [TRACE.svg](TRACE.svg): source-derived five-panel full trajectory.
- [CAPTION.md](CAPTION.md): observed evidence, interpretation and limitations.
- [DATA.json](DATA.json): 201 synchronized samples per controller, exact trace paths and SHA.
- [MANIFEST.json](MANIFEST.json): source execution manifests, native-model/source SHA, camera projection matrices and output hashes.
- [QA.json](QA.json): actual headless Chrome click/seek/switch checks, ffprobe frame timestamps and source-state readback.
- [QA_BROWSER.html](QA_BROWSER.html): executable browser test harness (temporary result appended after 2.5 seconds).

Actual RTX3080 rendering completed 2026-10-31 in network-isolated runtime; no hardware, no new physics steps, no RL updates. Both videos: 201 frames at20Hz; largest state readback difference 1.1920929e-7. All presentation timestamps exactly k/20. Historical results remain 19.19mm failure vs3.42mm success, not a new independent evaluation.

Reproduce from repository root with the existing isolated runtime:

```bash
bash research/skill_handoff_lab/tools/handoff_offline.sh \
  /home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python \
  research/skill_handoff_lab/tools/program_charger_video_sync.py \
  --root /home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-001 \
  --out /home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-002/new_video_output \
  --seed 725002
```

The exact render source is preserved as render_source.py; subsequent viewer-only improvements add angular traces and phase labels without rerendering or changing the MP4s. The manifest distinguishes render-source SHA from viewer-generator SHA. Initial frame now uses initial measurements, not step1; both plot cursors use 0..200; media-time flooring prevents half-frame advancement. Rollback events are restricted to actual CONTACT_SERVO, excluding unrelated approach motion.
