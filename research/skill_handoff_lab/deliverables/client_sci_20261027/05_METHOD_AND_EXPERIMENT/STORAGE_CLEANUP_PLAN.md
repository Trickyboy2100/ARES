# 历史研究存储清理计划（2026-10-27，执行前冻结）

范围限于本机非 Git 的已完成合成研究目录：`runs/contact_handoff_20261016_r1`、`skill_composition_20261016_r0`、`r1_20261014`、`multicontract_20261014`、`causal_20261014`、`industrial_20261014`、`mentor_package`。不处理受限 real_aligned、当前 client pilot、共享 cache、Git 文件/历史或190正在运行的任务。

采用内容保全的 **SHA256 完全一致文件硬链接合并**，而不是删除独有 checkpoint。逐文件先比较大小、设备号、权限及完整 SHA256；只有同设备同权限的重复文件参与。计划识别146组、160个冗余副本，逻辑重复内容206,085,233字节。准确物理释放量以去重前后唯一 inode 的 `st_blocks*512` 为准，不能用逻辑大小假称磁盘收益。

保留每个原始路径、原始字节、正式 final/必要 mid checkpoint、配置/源码快照、独立评估、失败 trace、负结果和旧图。最大项是旧 multicontract/grid_complete 的重复 action_audit/process_states，和导师包内对正式视频/模型的重复副本。每项 canonical、原副本、大小、SHA、引用及可重建性见 `06_RUNS_MANIFEST/STORAGE_DEDUP_PLAN.json`。

执行使用 `03_FIGURES/source/deduplicate_historical_evidence.py --execute`：先复核双端SHA，再建临时硬链接并原子替换重复副本；逐项验证SHA和inode。所有历史证据应视为不可变，未来修改必须原子替换，不可对硬链接原位写入。该过程不移除任何可寻址证据路径，也不改变历史SHA；若需要恢复成独立副本，可从同一canonical字节重新复制。

执行后抽样加载三个正式种子的模型参数，另验证所有合并路径字节；记录真实分配字节变化。未确认引用/只有唯一副本的大文件不清理。原始受控归档绝不清理。
