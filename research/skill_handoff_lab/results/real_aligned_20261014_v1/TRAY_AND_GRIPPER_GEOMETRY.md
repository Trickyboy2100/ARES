# 托盘、夹爪、TCP与附着几何

## 三者不能直接等同

|几何|当前证据|尺寸/坐标含义|可用范围|
|---|---|---|---|
|原/World/Tray/Mesh|CAD：旧数字网格1217顶点；实物同一性UNVERIFIED|旧WORLD轴向bbox99.5×179.5×30mm；Tray根局部bbox179.5×30×99.5mm，轴旋转可交换尺寸|静态形状、耳片形状检查；不是10月计量|
|PPO WeldedTray|REPLAY_VERIFIED研究代理|采用上述WORLD bbox尺寸再按研究T_link6_object挂载；没有耳片/金属空腔；近似0.3kg惯量|可靠已持物仿真，不是接触抓取或真实托盘实例证明|
|10月attached envelope|UNVERIFIED，原请求缺失|Link6局部保守AABB可能含夹爪/物体、偏置、膨胀；不能当实际外尺寸|取得原request和SE3后才能比较|

## 定量问题（旧CAD，非现场新测量）

原Mesh世界中心[.072776981,.319279401,.996111300]m；Tray根[.113305242,.286956560,1.444067842]m。根坐标不在几何中心，局部bbox中心偏移[.032322841,-.447956541,-.040528261]m。附着若把根直接当物体中心会严重错位。

CollisionProxies Body/Ears位于Z≈1.140335m，GraspFrames YMinus/YPlus也在该高度，与Mesh中心差约144.22mm（物体仅30mm高）。这些代理/帧无法未经核验当作当前网格耳片。Root默认bbox只覆盖可见Mesh，不会揭露purpose/visibility不同的隐藏碰撞代理错位。TRAY_COMPONENTS逐组件记录，不只看根bbox。

两个legacy别名tray_grasp_point/tray_grasp_init_point相距2.944mm，均靠近同一Y+侧；不是可靠的左右耳双点定义。旧demo在缺点时甚至以同一耳点fallback，并用FixedJoint锁物，不能继承为真实接触抓取结果。

## 坐标链

T_BODY_TCP(side,q)=T_BODY_controllerBase(side)×T_controller_model(side)×FK_URDF(q)×T_Link6_TCP(side)。物体应另有T_TCP_grasp×T_grasp_object，不能由包络外尺寸倒推出。PPO现有T_Link6_object只是研究定义。

9/21 SDK历史读回：左active tool3，平移[1.949,-33.966,80.200]mm；右tool2，[27.946,-2.365,81.402]mm；两者完整姿态分别在FRAME_ALIGNMENT。原URDF 4C2_Link5与SDK active TCP不是同一个标记。左右真实工具、开度、指垫安装和10月夹持读回均需当次导出。

## 必须补充的最低量测

真实托盘外L/W/H、底厚/翻边/空腔、两耳片宽高厚与中心距、夹持面和物体质量/重心；带尺顶/侧/底视图、夹持状态近照及机器人/柜体同框照片；左右指垫宽厚与开闭读回、抓取方向、T_TCP_grasp与T_Link6_object。最好CAD/STL含单位、坐标原点与版本。没有这些数据，不声明三种几何是同一实际物体。

已有网格不删除；旧碰撞代理也保留为审计对象。新物理环境先修正数字几何对应关系，再进行接触验收。
