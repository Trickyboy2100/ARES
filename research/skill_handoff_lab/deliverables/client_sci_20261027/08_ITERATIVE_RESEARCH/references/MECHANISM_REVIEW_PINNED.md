# 三个最接近开源机制：动作设计与复现边界

2026-10-27；**只读源码/论文，不运行对方环境，不复制结果、资产或未授权代码。** 本轮限HACMan++、Fabrica、RecoveryChaining三项。精确源码文件及SHA256见`PINNED_PUBLIC_CODE_MANIFEST.json`，缓存为非Git `runs/public_reference_code_20261027/`。19个选定文件共260,793字节；没有下载预训练模型、场景资产或依赖。普通Git clone TLS失败，改用系统curl/官方contents API取固定commit原始字节；没有关闭证书校验。

## 1. HACMan++：把作用位置、朝向、接触后运动分开

- 来源：[RSS2024官方论文](https://sgmp-rss2024.github.io/static/hacman%2B%2B_camera_ready.pdf)，§IV，尤其primitive参数及Actor/Critic maps；仓库`JiangBowen0008/HACManPP`，pin **e1c00f23b0db209d1963d2b6f33fd452a60196ac**。
- 真正进入决策的代码：`hacman/algos/location_policy.py:94–190`，对点/基元Q施加合法grounding mask，训练采样、评估argmax；`hacman/algos/multi_td3.py:80–101`按primitive参数维数屏蔽无效输出。`hacman_bin/primitives.py:193–225,284–344`将pre-contact、接触朝向与后接触推进分离。这比“移动阶段目标几毫米”更明确地改变作用方式。
- 对ARES的直接启发：在同一合法控制接口中独立检验接触点/接近方向/工具yaw/接触后推进，记录实际接触与物体yaw响应。强反馈和Residual基线必须享有同样参数权限；有用的实际物理参数成立后再学，而不是把新增参数本身称创新。
- 禁止整体移植：该文件`move_to`存在Virtual `set_qpos`，Poke前后可调用`robots.reset`。这不适用于ARES持续接触/不瞬移合同，只能借动作语义、用本项目DLS/servo重新实现连续执行。固定tree未找到主代码顶层LICENSE（只有third_party许可），**不能因公开GitHub就假定MIT或复制实质代码**；本轮只读/独立设计，不导入代码。

## 2. Fabrica：路径规范化与残差已有；持件执行不等于自由工件

- 来源：[CoRL2025正式论文入口](https://proceedings.mlr.press/v305/tian25a.html)与本轮实读源码；仓库`yunshengtian/Fabrica`，pin **215a30fe51e59299588a5b5e417a9cb934fb393e**。本轮官方PDF下载/解析失败，故以下逐行实施判断来自固定源码，论文贡献描述沿用前轮已核对记录；不宣称本轮重新完整精读了其PDF。
- 核心`learning/isaacgymenvs/tasks/fabrica/fabrica_task_assemble.py:193–225,427–475`同时对观察和动作做path-frame/尺度变换，叠加名义方向，并通过ctrl target执行。故“局部坐标+残差”已是直接先例，不能独立作ARES新颖性。`pre_physics_step`和`post_physics_step`明确物理刷新顺序，可借鉴post-step记录规范。
- 关键不适配处：同文件`ATTACH_PLUG_TO_GRIPPER=True`，`post_physics_step:181–182`及`_attach_plug_to_gripper:393–413`同步plug pose/root state。它表达持件假设，**不可复制为ARES自由方块物理动作或声称真实抓取**。本轮继续让自由工件由接触运动；允许的工具yaw还需用旋转后的真实工具包络验收，不能只旋转渲染。
- 许可已实读：根`LICENSE`为MIT（2025 Yunsheng Tian），`learning/LICENSE.txt`为NVIDIA BSD-3-Clause（2018–2023），资产另有`assets/licenses`。若将来复制兼容片段，须保留对应notice并单独核资产；本轮不复制旧IsaacGym代码、不装旧环境，只复用比较思想。

## 3. RecoveryChaining：真正执行名义后缀，不把预测值当物理成功

- 来源：[IROS2025论文](https://merl.com/publications/docs/TR2025-152.pdf)，§V-A/B及Lazy变体；仓库`shivamvats/recovery-chaining`，pin **4894acc3768678de4dee6358df14c461a411162e**。
- `scripts/learn_recoveries.py:339–386`从名义执行采失败和完整轨迹，`:464–517`建立恢复训练，`:633–690`评估在真实失败之后连续进入recovery（该路径`reset_failure=False`）。`belief_srs/envs/discrete_action_wrapper.py:62–183`把primitive与terminal nominal-chain动作并列，记录真实`timesteps`；其执行后缀结果才可作为后续完成证据。
- ARES可借：将“纠偏后重新交回名义技能”作为可计价动作，保留实际后缀任务结果和总时间；短期不需要同时加入分类器。论文Lazy变体使用预测奖励并非每次都执行后缀，**不能把该预测标为本项目ACTUAL_PHYSX成功**。若ARES没有可救回状态，先改合法作用方式，不通过软件异常伪造恢复。
- 复现/许可边界：固定tree未见LICENSE；README仍写under review，不能作为最终发表状态权威，发表信息以MERL/IROS为准。`nominal_q_value.py`及`primitive_skills.py`在此commit为0字节占位，不能把它们报成已读完整模块；实际wrapper/value文件另取并记录SHA。没有装其Python3.10/robosuite环境或复制源码，许可未解决前只借机制。

## 已反馈给方法实现的两条有界研究切口

| 切口 | 先验假设，不是测得结果 | 必要同权限对照 | 淘汰/转向依据 |
|---|---|---|---|
| 工具yaw/接触位置的实际控制权 | 改接触法向/作用点可纠正物体偏转而不是仅改参考 | 同参数强反馈、规则/有限真实搜索、普通参数Residual | 真实工具/物体无有益差异时不扩大网络；统一旋转包络和约束 |
| 持久参数的有效执行策略梯度 | 当前未刷新参数不参与新物理动作，去掉其无用score可能改善有限样本训练 | 相同stateful执行器的完整Gaussian PPO与边缘化版本 | 是否有独立跨seed物理收益；不以数学正确/梯度非零代替结果 |

第二项属于已有参数化动作/统计边缘化思想，详见`ACTIVE_EXECUTION_LIKELIHOOD_NOTE.md`。v6有效似然56/57的负结果继续保留；R3若改变执行器，必须与享有同持久参数接口的PPO比较，不能把执行器改善归给likelihood。

**已发给method agent，不等待论文写完再指导动作实现。** R1真实结果为两32k模型同25/32、同7失败，仅说明该新增历史未改善当前开发集；不与v7不同split的30/32相减。R2/R3结果以新的运行清单为准，本文件不预填优势。
