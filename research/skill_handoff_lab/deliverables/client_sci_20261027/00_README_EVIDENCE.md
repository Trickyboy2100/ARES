# ARES 甲方 SCI 图文与真实实验交付入口

2026-10-27｜工作分支 `codex/client-sci-paper-method-pilot-20261027`。历史研究基线 `79448a47ffda5c8c1887f53e33a5dbb90d3eb490`；运行以各任务源码快照/哈希为权威，不用最终HEAD追认旧源码。

保全核对：开始时ARES `status --porcelain --untracked-files=all`为空，没有需stash的未提交用户修改；原研究分支保持79448a4，独立工作分支从已取回工作单提交98237503e71a1e3298c64c9d4ce1644d036b3c47继续。没有声称备份了并不存在的修改。非Git历史数据保留；授权清理仅依SHA清单处理可恢复重复文件，详见存储报告。

## 一句话结论

已建立真实16环境PhysX接触RL，普通残差有历史几何适应初证；本轮196,608次新在线交互完成真实P2 pilot，反馈30/32、BODY残差30/29/30、任务坐标/结果历史均30/30/30。新机制进入Actor并改变物理轨迹，但未增加成功集合，G3不通过，不能称已形成SCI新算法。图文方案、编译论文、模型与负结果完整交付，不回到“RL必须不可替代”论证。

## 文件导航与证据等级

| 入口 | 内容 | 状态 |
|---|---|---|
| 01_CLIENT_PROPOSAL_CN.pdf / .md | 9页中文图文研究方案，可编辑正文与生成器 | PROPOSED方案 + EXACT历史 |
| 02_PAPER_SEED/main.pdf / main.tex | 英文两栏八节工作稿、文献/图/表/构建脚本 | 可编译初稿，非已录用/可投稿声明 |
| 03_FIGURES/ | 系统/方法矢量图、实际Isaac截图、真实曲线、Fig6新实验 | 各图provenance逐项区分 |
| 04_TABLES/ | 从实际JSON/CSV生成的Booktabs表 | 历史EXACT；新run按完成记录 |
| 05_METHOD_AND_EXPERIMENT/ | 预注册、创新与风险、训练决策、存储清理 | PROTOTYPE/PROPOSED明确分开 |
| 06_RUNS_MANIFEST/ | checkpoint/源码/scene/曲线/视频索引及哈希 | 无敏感派生索引；大文件非Git |
| 07_MENTOR_BRIEF_CN.md / .pdf | 两页中文导师简报 | 与方案/主表一致 |

## 必须准确讲述的已有结果

历史v6b Geometry每组32例：阻尼反馈8；普通残差seed31/41/51为13/18/23，学习保持15/15/18。P2单独每16例：反馈5，普通残差6/6/7。不能把P1贡献包装成P2深槽优势。ID/Dynamics接近控制；新保持/等待未证实独立收益。总901,120历史交互含诊断和视频，不是本轮新增，也不是16个正式种子。

EXACT表示可追溯原始执行记录；REPRODUCED表示由冻结数据/脚本重建图表；PROPOSED表示计划/假设。Actor使用56维精确仿真状态，非已部署视觉或多模态Actor。真正的物理成功来自post-step状态与独立验收，不来自漂亮截图或reward。

## 可复现与安全

先读03_FIGURES/provenance/FIGURE_PROVENANCE.json、04_TABLES数据及06_RUNS_MANIFEST的实际运行映射。英文稿用 `bash 02_PAPER_SEED/build.sh`；中文PDF用 `python build_client_documents.py`（ReportLab/Pillow/pypdf）。生成器只读明确的合成实验数据，不访问现场目录。旧最终测试已读，不可用来调新方法。

实际交付页数：中文图文方案9页、英文两栏工作稿10页、导师简报2页。八节英文源码、10条参考文献、独立图/表输入与构建脚本齐全；最终编译无未定义引用、Warning或Overfull box。两张学习曲线在英文稿均跨双栏，提高打印可读性。`DOCUMENT_BUILD_MANIFEST.json`记录3份PDF哈希；`DOCUMENT_QA.json`验证页数、文本及13项图/表精确副本，`DOCUMENT_QA.md`记录渲染目检。

公开包仅自建合成任务，不包含受限10月现场点云、生产配置、机器人读回或派生现场图。ARES-R只读、不操作实机、不改驱动/重启。渲染只展示真实PhysX身份；外观改善不改变碰撞、质量、初态或物理结果。

图表由本研究日志/代码或实际仿真输出生成，未复制相关论文示意图或网上机器人照片。原机器人资产沿用既有ARES项目；本包不增加第三方CAD/模型的再许可声明。相关工作论文/代码/资产的许可风险单列于NOVELTY_AND_RISK.md。

尚缺的SCI核心证据：相对充分调试普通残差的独立方法收益、关键输入/动作因果消融、更多种子与新封最终测试。无论新pilot正负，均保留真实结果并完成图文交付。

实际清理已完成：SHA相同副本硬链接合并160个，释放206,446,592字节（196.88MiB / 0.1923GiB），删除独有内容/原路径为0；三正式历史seed模型均实际加载通过。需要修改历史硬链接文件时必须原子替换。详见存储报告及逐文件清单。

## 完整交付与最终核验

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
