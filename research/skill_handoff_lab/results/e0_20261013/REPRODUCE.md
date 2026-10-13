# E0 真实带物规划与成本pilot复现

基线ARES `4dc7b64`；ARES-R `9de17d2`只读；cuRobo `8e734f3`。原URDF/碰撞盒/软限位/工具审计见各request中collision_model与完整SE3。190依赖和启动方式沿用`tools/server190_oracle.sh`；每批单GPU0、总限时240秒、最多2 attempts。原始请求中的绝对路径是190个人目录，不得将它们直接用于实机。

目录角色：

- `requests_v2/endpoint_preflight.json`：完整1.15m底盘代理的16个失败端点，GPU请求0。`requests/`是首次CPU导出遇到subdivisions参数错误留下的准备文件，不是规划结果。
- `requests_lower` / `execution_lower`：8个固定跨IK组合，全部保留原始成功/姿态拒绝/无路径状态。
- `requests_continuation` / `execution_continuation`：4个固定同分支IK延拓。
- `requests_variations` / `execution_variations`：6组预冻结目标×2滚转，预先指定split；12套原始完整请求/轨迹/日志/独立验收。
- `cache.json`：只有完整模型内已验收动作组的PREPLACE一步候选评估缓存。本轮6组无排除，无未知转移被转换成失败。
- `learning_seed17_v2`、`learning_seed23_v2`：正式本轮pilot，含源码SHA；`learning_seed17_no_aux_v2`为effect=0消融。无v2后缀的较早同数据运行也保留，但导师图采用v2。
- `runtime_source`：实际executor、两版独立验证源码与训练源码；验证SHA逐项匹配。v1验证程序后来增加planner SUCCESS条件成v2，所有正样本生产状态本来均为SUCCESS；没有把原失败改成成功。
- `isaac51_*`：真正执行的安装/兼容检查/捕获日志，不是物理成功标签。
- `data_manifest.json`：所有数据SHA、源commit及已核验24套request/result/validation计数。

在工作区根目录（ARES与ARES-R分别放`repos/`，已有科研Python环境）可执行：

```bash
PYTHONPATH=repos/ARES:repos/ARES-R/src envs/cash/bin/python -m research.skill_handoff_lab.tools.audit_e0_artifacts --run runs/e0_20261013
PYTHONPATH=repos/ARES:repos/ARES-R/src envs/cash/bin/python -m research.skill_handoff_lab.tools.train_e0_cost --dataset runs/e0_20261013/cache.json --output runs/e0_reproduction_new --seed 17 --steps 2048
```

重新规划请向190新输出目录运行冻结batch，禁止覆盖现有execution目录。新机器须显式绑定robot_yaml/robot_yaml_urdf位置并记录新request SHA，不能只改机器名仍引用旧SHA。E0生成命令使用`prepare_e0 --chassis-policy full`作负对照、`--chassis-policy historical_lower`作声明模型对照，再加`--continuous-goal`生成连续IK。托盘来源是现有USD `/World/Tray`包围体；源USD SHA和尺寸在每个request/frozen_design里记录，未当成测量CAD。

本轮不支持“从一个固定持物状态直接选另一抓取分支并执行”的在线策略复现；准备边缺失，不假造。所报告成功仅为研究模型内有效规划路径，不是Isaac物理或实际工业任务成功。
