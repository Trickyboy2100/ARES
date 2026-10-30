# VISUAL EVIDENCE PROTOCOL｜原始结果 + 英文学术 Highlight

本规范贯穿单篇 SCI。优先选真正能够解释**物理控制/学习机制**的少量
代表性救回、退化、共同失败案例。不要每轮大量生成PDF或视频。
所有圆圈、箭头、数字、英文解释必须来自**真实测量/来源可追溯的轨迹**。

## 四层证据图

1. **RAW**：原始模拟执行中的图像/原始测量曲线，无圈注、来源不覆盖。
2. **ANNOTATED**：同一个物理时间与视角，圈出charger/socket接触ROI、
   工具实际位置/姿态偏差、真实接触、物体运动和动作变更时刻，
   用英文短语如 `Observed Misalignment`, `Repeated Rollback`,
   `Insertion Progress`, `Depth-Preserving Alignment`。
3. **TRACE**：同步画实际任务坐标下轴向深度、横向/姿态误差、
   真实接触力代理、ALIGN/ADVANCE/HOLD模式、时间步和预算。
   若未实际记录某项，不允许编造曲线或矢量箭头。
4. **CAPTION**：英语论文图注依次标记 `OBSERVED`（真实发生）、
   `INTERPRETATION`（可能原因）和 `NOT ESTABLISHED`（尚未验证）。
   各图一并给相同case的原生终态成功与失败，不靠单个视频推成功率。

来源标签四选一：`ACTUAL_SIMULATION_RENDER`（在线物理拍摄）、
`MEASURED_POSE_REPLAY`（从已保存姿态重渲染，新增physics步数0）、
`DATA_DERIVED_PLOT`（实测CSV/JSON计算）、`SCHEMATIC`（理论概念图）。
不把姿态重播称新的物理试验或实机。真实物体位姿、对齐/接触数据用
原始episode验证，视觉圈注由实际相机投影或可验证ROI生成；不靠肉眼
臆测毫米测量。

## EXP-001首选案例

- `721000`：原积分+goal hold终态误差34.57mm、失败；
  有界servo+goal hold误差4.01mm、原生成功；现有本机
  `runs/research_program/EXP-001/visuals/controller_pair.mp4`
  属于MEASURED_POSE_REPLAY，完整物理因果需回看源trace。
- `725002`：原积分4.79mm成功、第一版servo19.19mm失败、
  事后current-depth hold3.42mm成功。
- `725005`：原积分23.65mm失败、第一版servo23.54mm失败、
  事后current-depth hold3.92mm成功。
  **两例只用于已看诊断，不是新n=8确认结果**。
- 冻结8例servo旧版成功5、积分成功1、五救回一退化、
  精确双侧p=0.21875；下一轮需在未看新场景独立确认最新控制器。

## 机理理论边界

源码判据：横向偏差<0.4mm、角度<0.03rad时允许一次2mm轴向推进；
旧版未对齐时回退到较早接近深度；修正版保留当前实测轴向深度、
继续修正横向与姿态。可能对应**接触阶段切换与撤回造成进展损失**，
应通过真实时序验证切换次数和轴向位移，而非称已经证明极限环/稳定性。
经典Hybrid Position/Force Control可作理论背景，但当前代码是
位姿伺服+仿真接触代理，并非已认证力/位混合控制或新增RL算法。

## 输出结构与真实性

每组 `RAW.{png,mp4}`, `ANNOTATED.{png,mp4}`, `TRACE.png`,
`CAPTION.md`, `visual_manifest.json`。清晰英语英文标注，论文图片300dpi
或矢量版本；保留time/step、seed、policy、raw/annotated SHA、
执行源码/配置/模型SHA、任务原生判据/时限、真实raw source路径及
标注生成脚本。原始/圈注必须可在同一时间点并列比较；视频与图像原始版
不能被标注版覆盖。无独立RL增量时一律注明 `NO NEW RL CLAIM`。
