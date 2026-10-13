# CASH方法状态与实际训练边界

普通MaskablePPO入口train_cash.py ≠ CASH。cash/model.py是后续约束条件化的候选集合Actor-Critic，cash/trainer.py提供PPO+value+successor-effect联合loss；目前仅人工张量单测更新过，未在真实数据上训练。

新增置换测试验证actor logits和effect预测随候选排列等变，value不变；动作ID未进入该网络。但尚未做真实训练后的置换评测，不能宣称已排除全部捷径。

真实effect目标应来自经验证的下一状态及可行性/代价，不从启发式分数或未来障碍伪造。现有fixture8维头没有被认作实测标签。完整三阶段CASH rollout、GAE、联合辅助头正式训练与冻结测试评测仍需接通，不能用SB3基线结果替代。

对照A：ARES-R启发式/等预算搜索；B：Ranker/普通PPO；C：真正联合effect loss的CASH。现在均未形成正式比较。方法优先研究后续准备动作和有限规划预算的价值传播；若固定测试仍无候选差异，应报告并转向Bridge Skill，不调指标制造优势。
