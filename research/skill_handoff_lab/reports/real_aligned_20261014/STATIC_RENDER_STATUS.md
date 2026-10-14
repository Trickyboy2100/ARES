# Isaac5.1静态参照验收

190 GPU0单进程、独立进程组、180秒超时上限。初版曝光偏高但正常退出，原图和日志保留在isaac51_static；第二版dome intensity=250提升可读性，保留在isaac51_static_v2。两个版本都不是物理任务：physics_steps=0、physics_task_completed=false、october_scene_reconstructed=false。

双臂安装采用历史BODY模型，每臂分别装配；关节为9月Snapshot，不冒充10月读回。夹爪为固定CAD外观，实际10月开度未知。展示的是重新居中的旧金属料盘，没有伪造右手附着；下方底盘外观仅旧USD位置拟合，不能当真实现场安装CAD。黄色/紫色球为二级摘录的目标/Preland；线框为9月冻结障碍代理，不代表完整柜口或自由空间。

已查看1280×960实际渲染图，非空且双臂/料盘/目标标记可见。精确Isaac版本、退出状态、耗时、场景与渲染脚本SHA见process_manifest.json/capture_manifest.json，原始日志见isaac.log。任务和接触均NOT_TESTED。停止后190四卡均2MiB、0%利用率，没有残留审计GPU任务。

对应STATIC USD压缩保存用于复现。解压后内容SHA必须等于capture_manifest的scene_sha256。它是静态数字参照，禁止直接启用动力学后称为已经验收的物理训练环境。
