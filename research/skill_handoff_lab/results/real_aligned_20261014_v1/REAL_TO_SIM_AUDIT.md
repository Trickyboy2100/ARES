# 10月任务Real-to-Sim有界审计

## 结论与范围

**10月原始归档未取得，不能声称完成10月请求级核验或真实柜口重建。** 本轮在指定3c20dcd工业PPO基准上审计，当前工作树f5e9eb0后续成果保留，未回退、未扩大训练、未覆盖16env代码。ARES-R本地HEAD9de17d2虽于10月12日提交，其说明明确未包含10月未提交现场工作。记录的现场HEAD a9ec30ee…来自二级摘录，不是本机有效现场版本。

搜索范围：本研究workspace与robot-research内相关归档、ARES-R含ignored文件、已授权190个人研究目录。`data/site_exports/`为空；未找到Preland tar或原始PLY/NPZ/depth。没有连接32现场设备、Epic/机器人/AMR，没有重新扫描，没有生产配置更改。已向用户索取原档路径。

## 对照与证据等级

|项|3c20dcd工业RL|现场参照/新目标|来源与可信度|
|---|---|---|---|
|机器人|固定基座右臂；无左臂/底盘物理任务|右臂持物、左臂HOLD_CURRENT、换站后重新观测|现场10月说明UNVERIFIED；9月Snapshot HISTORICAL|
|安装|9月BODY修正的右臂USD|双臂完整安装与工具须分别复用|robot_world+9/21 SDK校准 HISTORICAL；USD/URDF REPLAY_VERIFIED|
|当前状态|研究起点与微扰|实测right center不等于named center|10月q/读回缺失UNVERIFIED；9/28 named配置HISTORICAL|
|对象|焊接99.5×179.5×30mm盒，0.3kg近似载荷|实际托盘耳片抓取及10月attached envelope|旧网格CAD；PPO研究代理；现场实例UNVERIFIED|
|目标|held-center[.45475,-.48,1.45]；+X进入260×180柜口|检测[.604866,-.184327,1.152342]；Preland Z+70mm|10月用户说明+SITE_REFERENCE二级摘录UNVERIFIED|
|姿态|研究水平姿态、yaw=0|LEVEL_YAW_TARGET约.044793rad，明确前向|Oct二级摘录UNVERIFIED；历史+Z为TCP前向约定HISTORICAL|
|场景|4个研究墙体，无现场点云/epoch|新PLACE观测、SceneSnapshot和目标/障碍重新绑定|ARES-R协议代码REPLAY_VERIFIED；Oct原场景UNVERIFIED|
|速度|执行器上限1.57rad/s、独立研究控制器|历史现场0.1rad/s、0.2rad/s²|3c20dcd代码/9月motion config HISTORICAL；10月有效值UNVERIFIED|
|保护|研究姿态/限位/净力与代理碰撞|原生封装软限位、附着物、完整场景和约束版本|不能把planning SUCCESS当执行成功；原始Oct封装日志缺失|

held-center与Epic/TCP目标语义尚未统一，因此图中的位置差只是坐标数值对照，不是已验证的TCP行程。

## 可直接复用

原始JAKA/EG2网格与固定URDF的哈希、关节顺序和几何链；原始金属托盘可用于数字几何参照；SceneSnapshot/场景失效协议；16env cloning、PPO/残差、日志与独立过程检查。复用不等于现场物理已标定。原main双臂必须重建各自安装，不能只整体平移场景，也不能沿用左抓→右交接演示。

## 主要审计发现

1. 原main左右完整变换分别推导的WORLD↔BODY候选相差120°/1.498m，单一全局刚体变换不一致。需保留原件、独立research层按每臂安装重新构建。
2. 独立USD关节局部坐标链与URDF对28个历史双臂姿态一致，最大位置差3.92e-8m；这仅是模型内部验证，不是10月SDK读回验收。
3. 历史左tool3、右tool2的SE3不同，不能共用right TCP。当前Oct工具/夹爪开度未知。
4. 原金属Mesh中心Z=.996111m，而Ear GraspFrames和碰撞代理中心Z≈1.140335m，约144mm错位；两个旧抓取别名相距仅2.944mm。详见几何审计，不能未经核验用于接触任务。
5. 原始点云缺失，只有9/21冻结cuboids。不得把AABB空腔填实、未覆盖区域当自由空间，或把历史画面称10月工位。

## 最小补充材料

①10月10–12原始Preland档案（requests/results/轨迹/原生封装拒绝/现场读回/识别结果）及当时dirty配置patch；②对应SceneSnapshot/点云capture与BODY处理manifest、PLY/NPZ/depth；③实际托盘CAD或量尺照片及夹持点/T_link6_object；④同次左右q、active tool SE3/开度、底盘站位、有效robot_world与motion配置。只需现存导出，不需新硬件扫描。

其余产物是明确标注的HISTORICAL/PROXY_ONLY参照，不能作为正式real-aligned学习数据。
