# 10月Preland失败审计：原档缺失，结论暂定

**不是请求级复核完成。** 来源为用户说明、已跟踪SITE_20261012_REFERENCE及ARES-R smoke README；原tar/索引/交接说明/request/planning/native拒绝日志未取得。均标UNVERIFIED_SECONDARY，不能制造原日志哈希或精确时间。

## 已知叙事顺序（精确日期/时间待原档）

|阶段|二级记录|当前可下结论|不可下结论|
|---|---|---|---|
|2026-10-10至12观测|space2/object4/camera1，目标[.604866,-.184327,1.152342]|与9月object3配置不同；待原识别结果核验|实际柜口/物体实例和当前状态已确认|
|Center→Transit|SUCCESS，181 knots，约48.28mm模型净距|曾报告规划成功，非任务/执行成功|与后续同world/TCP/attached/限位|
|Transit→Preland|FAILURE，0 knots|没有可接受轨迹；原因待原log|目标被硬碰撞堵死、几何不可能或可恢复|
|直接→Z+70mm Preland|SUCCESS，301 knots，约22.3668mm模型净距/.5255°水平误差；原生封装拒绝|规划与执行资格应分开；J3软限位拒绝需保留|真实执行成功或与Transit版本公平比较|
|重跑/重绑定|冻scene重跑不可靠|世界更新/参数差异应独立归因|一次SUCCESS保证可重复，所有失败归于碰撞|

目标Preland[.604866,-.184327,1.222342]m、yaw=.044793rad、LEVEL_YAW_TARGET、clearance10mm、virtual central exclusion=false均为二级报告。关闭虚拟规则不关闭真实机器人/世界碰撞、水平、原生关节限位。

历史motion配置J3物理区间±2.268928rad，margin.15→±2.118928rad（约121.406°）。这不是10月当前阈值证明。必须取得直接301轨迹的J3 extrema及同次packager下/上界，逐节点/插值/速度/加速度复算，不能引用9月阈值冒充实际拒绝记录。

## 公平请求指纹（所有项齐全才分组）

start/left_hold q与状态时间；target/TCP语义及SE3；object4识别原结果/置信度；Robot/工具/碰撞球/附着T与包络；完整world cuboid/mesh及scene/observation/base/calibration revisions；central exclusion与clearance；orientation mode/yaw/path姿态容差；IK分支/goal q；原生软限位/速度/加速度；cuRobo版本/config/seed/timeout。对规范化内容及原字节分别计算SHA。

同指纹重复seed不是独立场景。不同排除规则、附着物、目标姿态或world的SUCCESS不能直接当算法优于FAILURE。planner失败、IK失败、J3 native reject、scene mismatch、timeout/OOM/software error分别分类；原log不足标UNKNOWN，不能作失败训练标签。

## 下一步有界回放

导入原档→逐请求manifest/差异表→在无硬件的190固定配置，先重放1个Center→Transit、1个失败Transit→Preland、1个J3拒绝直达。只各一次+预声明重复，保留原失败与参数。新轨迹只有原生限位/附着/姿态/碰撞/场景重绑均验收后才有规划级反馈，不推断物理放置。
