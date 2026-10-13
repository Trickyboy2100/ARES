# 真实学习数据缺口

正式合格handoff标签=0。诊断成功不能复制到Cabinet或其他候选；模型代理冲突、软件错误均UNKNOWN，未知边不能记失败。

现有缓存需要每个(history,stage,action,scene,attachment)的实际next_state、终止证据及源文件哈希。D0缺真实抓取结果分支；D1缺给定抓取后的带物center执行/规划转移；D2缺换站后新观测与preplace完整路径。两候选一场景不能形成按场景隔离的train/validation/test集合。

18单测PASS，含16步人工fixture PPO与候选顺序置换，fixture排除正式数据。真实Ranker/PPO/CASH训练NOT_TESTED。下一步先收集冻结scene级candidate evaluation、保留unknown、统计预算success@K；有多个合格场景后才做场景级划分。不得提前把PLACE新障碍送入PICK策略。
