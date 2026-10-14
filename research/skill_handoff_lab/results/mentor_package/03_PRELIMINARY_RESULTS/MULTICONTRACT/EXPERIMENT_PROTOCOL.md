# 冻结多合同实验协议与来源

- 基础commit：3c20dcd；190个人研究目录；两张空闲RTX4090各一个16环境Worker，其余无研究计算。无驱动修改、重启、实机调用、ARES-R修改。
- 训练程序v19 SHA256：04c87b75af09ef247e9537124ffa69b358d727ff3c051466bb61f45f0e598b1c。全链程序v20为独立版本，source副本保留，不混淆来源。
- suite_v5 SHA256：197ed381fb21d9314a4c3e9add7281440bb6e04721d05f2206bbc33bb6500a98；研究碰撞USD：1857a1c59c7693845c1f73d62fc94d61859e7668df07f2d87b52b14d1ebdb760。
- reward_v2在训练前冻结；每方法种子31、47、59固定预算16,384，验证/测试64案例seed1731。所有最终checkpoint固定预算，不选最佳，不重试失败job。
- 命令与残差裁剪前后、实际关节与全部刚体位姿、物理时钟、目标/合同、每episode终止类别、模型/日志/场景保存。每模型真实1024次optimizer更新，16,384次控制交互、32,768个env physics steps；验证模型更新应为0。
- 训练视频为真实物理训练间隔采样time-lapse，不是实时录像、预制轨迹或循环任务。记录每帧phase、交互量及每env关节，GIF转MP4只编码格式转换。

v19原manifest中`scope`及`limitations`的“single wide cabinet”是未更新的历史说明字段；以`task_groups`、`task_suite_sha256`、`task_split`、实际scene_specs和USD边界为准，本包明确更正其解释，原JSON不改写。frozen_evaluation旧`scope`字符串也不应被解释为同分布测试：val/test为冻结的不同几何组。action_audit的policy_observation为更新命令后的状态观测，不是该行raw_action的精确输入；离线输入敏感性报告明确注明这一点。

范围：固定夹爪＋可靠焊接托盘代理、独立固定基座右臂；未验证接触抓取、全部底盘双臂、部署传感器误差、载荷/摩擦随机化。研究USD明确应用CollisionAPI到实际Mesh Gprim并使用convexHull；不是原始凹CAD完整验收。过程FCL为120Hz离散采样，非连续扫掠碰撞。

独立网格复核435746个物理状态：1281个原报告成功的重复episode均通过，0成功待复核。不能把这些重复episode当1281个独立场景。标准PPO失败有约15–16mm实测工作空间越界，v21因此补充1mm明确跟踪容差的实测状态硬终止。冻结seed31 PPO在同64验证案例修复后仍0/64，其中32个workspace终止、3个姿态终止，平均episode2.177083s；其余为超时。新失败时间不回填v19表。名义与合同v21各64/64，0实际workspace失败。

审计程序两个哈希c03eb/f09da的完整源码在raw_compact保存且校验SHA一致；差别仅为未保存刚体位姿时的pose_source说明文字，实际网格均使用OBSERVED_PHYSX_BODY_POSES，所有碰撞/约束数学不变。
