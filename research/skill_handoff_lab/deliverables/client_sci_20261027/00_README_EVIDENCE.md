# ARES 甲方 SCI 图文与真实实验交付入口

2026-10-27｜当前增量分支 `codex/ares-sci-autonomous-research-visual-20261027`，起点001bed5；前轮为 `codex/client-sci-paper-method-pilot-20261027`。运行以各任务源码快照/哈希为权威，不用最终HEAD追认旧源码。

**先看本轮：** `08_ITERATIVE_RESEARCH/index.html` 为实际成对视频/状态轨迹入口，`MENTOR_VISUAL_BRIEF_CN.pdf` 为精简图文册，`metrics/COMPLETED_RUNS_INDEX.json` 逐项记录真实新增预算。前轮v7整包在非Git `runs/client_sci_document_history_20261027/v7_001bed5/` 保全；精确历史PDF哈希见 `08_ITERATIVE_RESEARCH/references/V7_DOCUMENT_PRESERVATION.json`。下列v6/v7统计不能充作本轮新增。

前轮保全记录：当时ARES status为空，没有需stash的用户修改；旧研究分支79448a4保持，独立分支从98237503e71a1e3298c64c9d4ce1644d036b3c47继续。本轮在001bed5基础上先保全整包旧PDF/MD/LaTeX后增量更新，未覆盖唯一旧稿。运行大文件继续非Git保留。

## 一句话结论

已完成三轮真实闭环研究和414,720次实际在线交互：393,216正式＋5,120诊断＋12,288真实训练录制＋4,096失败录制，15完成checkpoint已加载验证。R3三seed普通22/25/20、候选24/24/18，强反馈26/32，配对+2/−1/−2不支持稳定方法增益。固定保持单seed同25成功、候选快0.208s的真实局部成本信号保留。实际Actor梯度机制成立，不等于SCI新算法性能已成立；不回到“RL必须不可替代”论证。

## 文件导航与证据等级

| 入口 | 内容 | 状态 |
|---|---|---|
| 01_CLIENT_PROPOSAL_CN.pdf / .md | 10页中文方案；第2/10页当前摘要/结论，第3–9页明确历史 | 实际结果与未证假设分开 |
| 02_PAPER_SEED/main.pdf / main.tex | 英文两栏八节工作稿、文献/图/表/构建脚本 | 可编译初稿，非已录用/可投稿声明 |
| 03_FIGURES/ | 系统/方法矢量图、实际Isaac截图、真实曲线、Fig6新实验 | 各图provenance逐项区分 |
| 04_TABLES/ | 从实际JSON/CSV生成的Booktabs表 | 历史EXACT；新run按完成记录 |
| 05_METHOD_AND_EXPERIMENT/ | 预注册、创新与风险、训练决策、存储清理 | PROTOTYPE/PROPOSED明确分开 |
| 06_RUNS_MANIFEST/ | checkpoint/源码/scene/曲线/视频索引及哈希 | 无敏感派生索引；大文件非Git |
| 07_MENTOR_BRIEF_CN.md / .pdf | 两页中文导师简报 | 与方案/主表一致 |
| 08_ITERATIVE_RESEARCH/ | 三轮研究、4页可视化简报、8对实际影片＋真实16环境在线训练、曲线/模型索引/源码文献分析 | ACTUAL_PHYSX，逐run独立验收 |

## 必须准确讲述的已有结果

历史v6b Geometry每组32例：阻尼反馈8；普通残差seed31/41/51为13/18/23，学习保持15/15/18。P2单独每16例：反馈5，普通残差6/6/7。不能把P1贡献包装成P2深槽优势。ID/Dynamics接近控制；新保持/等待未证实独立收益。总901,120历史交互含诊断和视频，不是本轮新增，也不是16个正式种子。

EXACT表示可追溯原始执行记录；REPRODUCED表示由冻结数据/脚本重建图表；PROPOSED表示计划/假设。Actor使用56维精确仿真状态，非已部署视觉或多模态Actor。真正的物理成功来自post-step状态与独立验收，不来自漂亮截图或reward。

## 可复现与安全

先读03_FIGURES/provenance/FIGURE_PROVENANCE.json、04_TABLES数据及06_RUNS_MANIFEST的实际运行映射。英文稿用 `bash 02_PAPER_SEED/build.sh`；中文PDF用 `python build_client_documents.py`（ReportLab/Pillow/pypdf）。生成器只读明确的合成实验数据，不访问现场目录。旧最终测试已读，不可用来调新方法。

当前交付：中文方案10页、英文两栏工作稿18页、导师简报2页、可视化简报4页。新增方法/曲线采用打印紧凑源图，宽图跨双栏；11条参考文献、完整LaTeX/表/构建脚本。`DOCUMENT_BUILD_MANIFEST.json`记录4份当前PDF哈希，`DOCUMENT_QA.json`核对页数/文本/生成源副本，`DOCUMENT_QA.md`记录当前逐页渲染目检。旧v7哈希索引是历史范围，不能校验已更新PDF。

公开包仅自建合成任务，不包含受限10月现场点云、生产配置、机器人读回或派生现场图。ARES-R只读、不操作实机、不改驱动/重启。渲染只展示真实PhysX身份；外观改善不改变碰撞、质量、初态或物理结果。

图表由本研究日志/代码或实际仿真输出生成，未复制相关论文示意图或网上机器人照片。原机器人资产沿用既有ARES项目；本包不增加第三方CAD/模型的再许可声明。相关工作论文/代码/资产的许可风险单列于NOVELTY_AND_RISK.md。

当前SCI核心判断：同样本/同权重的真实Actor梯度差、三seed/fixed消融已经完成，但没有稳定P2成功或强基线优势；恢复接口不等于学会恢复。无导轨零训练探针反馈/普通32、候选29，未证新方法泛化。先验证有用接触权限与固定保持成本信号，再决定是否扩种子/封存测试；不能只因稿件章节齐全称可投稿。

本轮实际清理：仅SHA相同的已完成文件硬链接合并，释放186,458,112字节（177.82MiB），删除独有证据/原路径/checkpoint均为0；见08/CLEANUP_APPLIED_FINAL.json。不要与前轮196.88MiB混算；历史硬链接证据需修改时另存新版本，不原地写。

## 当前完成范围与可移植包

66个完成run包含重复物理评价/相机/探针，不是66个独立训练种子；1174次评价也不等于1174独立场景。总在线成本使用metrics索引的charged字段，含已知失败4096；失败含reset的总physics tick未知，只给已知下界。训练主记录60Hz；通常120Hz trace为评价独立验收，额外展示训练另有120Hz/44字段sidecar。新增展示run真实3次rollout、288个优化小批量，不当正式性能模型或免费额外训练。

本轮可移植包以主代理生成的 `CLOSED_LOOP_BUNDLE_INDEX` 和包内 `CONTENT_SHA256` 为权威，根 `OPEN_ME.html` 通过相对videos链接播放；原工作区08/index.html是本地服务器入口，不是公共托管链接。旧包、旧DELIVERY_BUNDLE_INDEX及PUBLIC_FILE_HASHES仍保留为v7历史，不覆盖，不宣称它们校验当前整包。新9个MP4只选comparisons_final媒体版本；现场受限资料不进入任何公开包。

## 前轮v7完整交付与最终核验（历史，不代表当前文件哈希）

`06_RUNS_MANIFEST/GLOBAL_DELIVERY_QA.json`实际核验3份PDF、13份论文输入副本、111项图表输入/54项输出、27次正式物理运行和12个模型CPU加载及有限输出；53项单元测试通过。哈希/加载检查不替代物理或硬件安全认证。原始非Gittrace继续保留，公开Git只有无敏感派生证据。

完整可下载包位于研究根目录非Git `runs/client_sci_delivery_20261027/ARES_CLIENT_SCI_PACKAGE_20261027.tar.gz`，含本目录可编辑源文件、图表/PDF、当前实验工具和12模型归档；精确大小、SHA和成员哈希见 `06_RUNS_MANIFEST/DELIVERY_BUNDLE_INDEX.json`。归档不包括临时编译文件、受限现场资料或大型原始NPZ（另有逐run索引）。单独模型包为 `runs/client_sci_pilot_20261027/MODELS_AND_CONFIGS.tar.gz`。包索引不放入自己的归档，避免循环哈希。

| 本轮验收 | 结果 |
|---|---|
| 图文方案、LaTeX工作稿、两页简报、真实图表/截图 | PASS |
| 1→4→16环境真实训练、模型和独立物理验收 | PASS |
| 同预算三种子及真实输入/动作消融 | PASS |
| B3相对B2的P2独立方法收益 | FAIL（无成功集合增益，保留负结果） |
| 新最终测试/跨任务共享策略/实机部署 | NOT_TESTED |
| SCI新方法核心结果与可投稿性 | 尚不足；具体下一实验见TRAINING_DECISION.md |
