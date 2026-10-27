# 已执行的存储清理

2026-10-27，执行范围和算法与预先记录的 STORAGE_CLEANUP_PLAN.md 一致。

| 项目 | 实测值 |
|---|---:|
| 完整SHA相同组 | 146 |
| 合并冗余副本 | 160 |
| 所扫描文件去重前分配字节 | 2,961,850,368 |
| 去重后分配字节 | 2,755,403,776 |
| 扫描集合去重前 / 后 GiB | 2.7584 / 2.5662 |
| **实际释放** | **206,446,592 bytes = 196.88 MiB = 0.1923 GiB** |
| 删除独有内容 / 原路径 | **0 / 0** |

已用完全相同内容的硬链接替代冗余副本。该数值是唯一 inode 的块分配差，不是文件逻辑大小之和；不把压缩或新增训练文件的变化冒充清理收益。所有原路径和字节仍可核验，原实验SHA不变，正式负结果没有丢弃。

执行清单及SHA：`06_RUNS_MANIFEST/STORAGE_DEDUP_PLAN.json`、`STORAGE_DEDUP_RESULT.json`。正式 residual seeds31/41/51 的最终checkpoint均使用SB3 `PPO.load(..., device='cpu')` 实际加载通过，参数各15,879，记录步数各65,536；模型路径/SHA在 `STORAGE_MODEL_LOAD_CHECK.json`。

没有清理受控现场数据、唯一trace、正式mid/final checkpoint、Git文件、共享cache或当前client pilot。190当前运行任务未受影响。旧研究引用目录中的硬链接文件应继续保持不可变；若要修改，使用原子替换以保持其他引用的历史字节。
