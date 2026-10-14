# 10月任务Real-to-Sim审计入口

2026-10-14；指定基准3c20dcd，实际研究HEAD f5e9eb0；未回退。ARES-R只读HEAD9de17d2。

## 目前能交付什么

完成独立双臂模型FK、旧USD安装和物体数字几何核查；输出历史空间代理叠加图，并在190用Isaac Sim5.1完成静态参照渲染。**不是10月现场重建，不是物理放置成功；没有增加PPO训练。** 原16环境、WideCabinet代码和正负结果均保留。

![任务对照](TASK_COMPARISON.png)

左图为9月冻结BODY代理及历史机器人状态；右图对照旧工业PPO研究设置与二级材料报告的10月目标。两种目标的TCP/物体中心语义尚未统一，不能由图中坐标差推断真实操作行程。

## 物体与空间来源

|对象|实际取得的来源|证据等级|不能推出的结论|
|---|---|---|---|
|旧金属Tray|main.usd原网格，1217顶点，可见包络99.5×179.5×30mm|CAD（旧数字模型）|与10月实物是同一实例、真实夹持点已正确|
|工业PPO物体|同包络焊接盒，研究质量/附着参数|研究代理，现场UNVERIFIED|耳片、夹爪接触及实物载荷已验证|
|10月附着物|只有二级任务摘录，原包络/T_link6_object未取得|UNVERIFIED|Link6 AABB就是料盘外尺寸|
|9月工位|9/21 clear_final SceneSnapshot及cuboids|HISTORICAL / SCAN_PROXY|柜口CAD、扫描空腔或自由空间已重建|
|10月工位|目标/Preland报告坐标；原点云、场景manifest缺失|UNVERIFIED|旧障碍世界就是本次现场世界|

![几何来源对照](GEOMETRY_SOURCES.png)

关键证据：两臂分别推导的旧WORLD↔BODY映射相差120°，不能用单一变换；28组历史关节姿态的独立USD关节链与URDF差异≤3.92e-8m（仅模型内部验证）。历史左tool3、右tool2；不能共用右TCP。旧耳片GraspFrames比网格中心高约144mm，旧抓取别名仅相距2.944mm；先核对坐标意图，再用于接触。旧研究速度上限1.57rad/s与历史现场0.1rad/s不同，研究耗时不能当实机执行耗时。

## 两个可执行RL方案

|路线|实际任务相关性|当前可运行基础|先补什么|判断|
|---|---|---|---|---|
|A 技能级RL|直接对应Transit/Preland、yaw、软限位、重绑定与准备决策|现有cuRobo Oracle/技能缓存/Gym；可复用规划级训练|原请求约束指纹、同一起点可执行准备转移、独立完整轨迹验收|更接近目前报告的现场失败；原始数据不足，暂不声称建立了可信10月训练集|
|B 局部残差RL|对应真实Preland后的对齐、进入、接触及恢复|现有16env PhysX/PPO，工程复用最强|实物/工位几何、实际进入方向与TCP语义、接触和完整过程验收|更适合现有并行物理平台；WideCabinet结果不能外推为现场任务成功|

不是重新论证是否应采用RL，而是确定学习对象。A/B可共用Skill Contract与日志；先完成一条真实条件一致的Preland到操作链，依据数据覆盖选择正式论文主实验。

## 下一轮最小工作单

1. 导入原档，冻结request/results/native拒绝/识别/q/工具/配置和点云manifest；区分object4与旧object3。缺项保持UNKNOWN。
2. 单env：每臂完整BODY安装、独立TCP、左HOLD_CURRENT、右实际持物；校正Tray root/网格/耳片/附着链。旧外观可复用，但未测底盘代理不能称真实CAD。
3. 真实Preland起点：回放一条约束一致且独立验收的轨迹，再用Articulation控制物理步进；检查姿态、软限位、速度、过程碰撞/接触与目标误差。已附着范围不包含真实接触抓取验证。
4. 同应用1→4→16 cloning，独立状态/场景/目标/reset；保留扫描表面、实体代理、未知和证据支持的自由空间分层。通过单env后再训练，不重刷WideCabinet。

## 最少需补充的原始材料

请给`ARES-R_2026-10-12_Preland_Planning_Records`完整归档或实际路径，以及对应点云/SceneSnapshot/运行配置导出。实际Tray CAD若无，可先给长宽高、耳片/夹持点尺寸、抓取方向与带尺/夹持照片。只需已有材料，不要求新扫描或连接现场。

详细文件：REAL_TO_SIM_AUDIT.md、FRAME_ALIGNMENT.json、TRAY_AND_GRIPPER_GEOMETRY.md、FIELD_SCENE_GEOMETRY.md、REAL_SCENE_OVERLAY.png、OCTOBER_PLANNING_FAILURE_ANALYSIS.md、RL_METHOD_REVIEW.md、SIM_RECONSTRUCTION_PLAN.md。Isaac截图及独立进程日志见isaac51_static_v2；physics_steps=0，任务完成=false。
