# New physical paired rescue: inspect one HTML

Open [SYNC_VIEWER.html](SYNC_VIEWER.html). It contains full task RAW/ANNOTATED switching, frame-aligned five-signal curves with full-episode/contact-detail zoom, click-to-seek, phase/mode labels and source-derived events. Both videos preserve the complete201-frame sequence. [CAPTION.md](CAPTION.md) distinguishes measured observations from offline simulator truth and limits the claim to a deterministic-control diagnostic.

- [RAW.mp4](RAW.mp4), [ANNOTATED.mp4](ANNOTATED.mp4): actual recorded-state replay, zero new physics.
- [TRACE_CONTACT.png](TRACE_CONTACT.png): expanded contact-stage view (steps80–200), preserving actual signals.
- [TRACE.png](TRACE.png), [TRACE.svg](TRACE.svg): full201 samples, six panels including controller mode.
- [LEFT_SOURCE.json.gz](LEFT_SOURCE.json.gz), [RIGHT_SOURCE.json.gz](RIGHT_SOURCE.json.gz): byte-exact original complete physical records, portable with this package.
- [DATA.json](DATA.json), [MANIFEST.json](MANIFEST.json), [QA.json](QA.json): measurements, source/code/model hashes, camera matrices and executable verification results.
- [QA_BROWSER.html](QA_BROWSER.html): browser QA harness; reports seek/switch/time mapping checks.
- [package_and_qa.py](package_and_qa.py): package, verify source observations, audit each encoded frame timestamp and run headless browser checks.

Reproduce with the isolated runtime and `tools/program_charger_video_sync.py --left LEFT_SOURCE.json.gz --right RIGHT_SOURCE.json.gz --left-label 'Geometry feedback' --right-label 'Progress search' --camera-side goal-positive-y --out NEW_OUTPUT_DIRECTORY` (paths relative to this package; invoke script using its actual repository path). The renderer infers and validates the common seed. Run `python package_and_qa.py NEW_OUTPUT_DIRECTORY` to regenerate the package and QA. Original execution manifests are embedded in MANIFEST.json; copied gzip source summaries additionally preserve physical configuration and readback. This is ARES-AdaptivePlug-v0, not an official native benchmark or live robot demonstration.

Selected camera uses the measured socket task frame: eye[-40,+300,+25]mm, target[-12,0,0]mm. The prior camera obscured the gap with the wrist; its complete local package and hashes are preserved in [ALTERNATE_CAMERA_INDEX.json](ALTERNATE_CAMERA_INDEX.json). Camera selection changes neither physics nor geometry.
