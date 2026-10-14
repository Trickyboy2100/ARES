# 现场几何：PROXY_ONLY

## 已取得与未取得

现有9/21 clear_final SceneSnapshot是HISTORICAL，包含BODY观测ID、左右实测q、校准revision、13个residual AABB、support table和robot/chassis代理。原BODY点云仅有ID及sha256，PLY/NPZ/depth/有效点/去机器人点均不在本机/190研究目录。10月原SceneSnapshot、Epic object4识别和点云缺失，因此本轮不能生成真正的raw/filtered/self-filtered点云叠加、实际柜口截面或净距。

REAL_SCENE_OVERLAY显示历史wireframe cuboids和二级报告的10月目标，明确PROXY_ONLY。没有虚构扫描点或烘干机CAD，没有用旧AABB生成“真实自由空间”。旧block_final另有synthetic_block_goal，不能当真实现场障碍；参照采用clear_final，不混用三种云/世界。

## 相机权威值按运行版本判断

system.epic_pointcloud记录T_BODY_camera revision ccb214ea…、state COMMISSIONED，9/20双臂独立crosscheck报告支持：两链平移差.944mm/旋转差.722°，生产选right H1，left只校验。旧pointcloud_pipeline.site顶层仍UNCOMMISSIONED/T=null，其self_filter原因又提及已commission。更早worklog/generated/body_collision_world属于旧失败注册结果。

故9月冻结snapshot优先按其manifest中的ccb214ea与23e44f93校验版本，而不是按文件名覆盖。10月权威矩阵必须由当次运行配置、dirty版本、capture manifest和校验日志确定，不能把9月已标定推断成10月仍有效。

## 分层几何规则

1. 实体CAD/可确认支撑面：独立几何与版本，不让柜口空腔成为整块实心box。
2. 扫描表面：原点/网格surface，保留噪声、遮挡、机器人去除mask；SCAN_PROXY不等于体积实心。
3. 未知区域：未观察/遮挡显式unknown；盒外空白不是已证自由。
4. 证据自由空间：只有传感ray/free-space carve或可验证CAD开口才能标记known free。
5. robot自滤波代理与硬环境碰撞需分开；历史chassis过滤代理.56×.70×1.15m不是已测AGV CAD。

取得数据后，以BODY原/有效/去机器人云和两臂/底盘占用分层，叠加实测TCP/物体/目标；沿实际入口法向及目标高度切片，使用物体/夹爪OBB、姿态误差和不确定性计算几何余量。现在不能提供真实开口宽度、深度、接触面、摩擦或窄区净距。

PICK→AMR_MOVE使原scene/trajectory失效；PLACE新观测重建后，必须重绑scene_snapshot_id、observation_id、base_pose_revision、calibration、attachment和active TCP。历史BODY云不能当换站后的实时世界。
