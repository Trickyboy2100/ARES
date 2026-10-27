# Actual PhysX same-case two-policy录像协议（录制前冻结）

本轮所有录像为**重新执行的真实闭环 PhysX**，不是旧视频改标题、轨迹插值或状态重播。正式版本Isaac Sim5.1 / Isaac Lab2.3.0；每侧独立冷启动进程，保持同scene/config、caseID、eval seed与机器人/工件初态，初始q/qd、工件pose/线角速度和推板位置逐项校验。隐藏接触求解器cache不声称可恢复或位相完全一致。

## R0预声明选择

仅作历史描述性再评价，不用于调新算法。以历史v6b Test-Geometry三seed独立成功数13/18/23的**中位seed41**，不是表现最好seed，比较当时真实阻尼反馈与普通ResidualPPO。保持当时精确source snapshot、scene、附着/软限位/预算；不混新v7场景。

从已冻结独立case表自动选：

- case0：最小ID的P1 baseline失败而residual成功；
- case1：最小ID的P2 baseline失败而residual成功；
- case9：最小ID的P2共同成功；
- case3：最小ID的P2共同失败。

同一冷启动完整16env/32case批次可录多个case，不为某个案例重新挑seed/初态。新的真实再执行结果若不匹配历史，保留并报告差异，不裁成期望结论。全体32例配对结果保留；选片不替代总体成功率。

## R1/R2/R3选择

每轮算法/参数训练与完整验证完成后，按完成的独立验收表，优先最小ID的P2真实救回；不存在则最小ID共同失败，再不存在则共同成功。优先每轮至少一组同case基线/候选真实闭环。可以追加其他类别，但不能只选最漂亮视频或称个例为总体优势；源/模型按该轮实际引擎及checkpoint SHA，不用HEAD替代旧运行。

R3录像训练种子预指定为31，不依据最终三种子表现选最好模型；其片段属于个例，不作为多种子代表值。左右使用同87D/5D、同event refresh及24s执行合同的ordinary/qualified模型，不用不同接口的强反馈替代算法配对。

## 相机与时间

- 原始相机1920×1080 RGB，固定任务局部俯斜视角，仅加材质/灯光/相机；外观修改前后验证全部物理/几何属性签名一致。
- 在真实post-step、reset前录制；60Hz控制每2步采帧（标称30Hz），并补录实际阶段切换/终止事件，同时保留原生120Hz trace。原始MP4为固定30fps编码，但插入事件帧后其帧号不是严格物理时钟；逐帧JSON中的真实时间戳权威，成对成片按这些物理时间重新采样，不以原始编码时长推断任务时长。
- 渲染只调用`sim.render()`，不调用任何额外`sim.step()`，不写机器人/自由工件状态。默认旧GIF采集关闭，另保留真实逐帧tick、物理时钟、stage、动作与接触代理。
- 并排H264 MP4/GIF/filmstrip由原始独立相机帧组装；使用同物理时间轴，较早结束侧明确标END/HOLD，保留另一侧超时。开场/结束静止时间注明是展示保持，不循环动作、不冒充物理步。
- PASS/FAIL来自整段独立验收，而非图片推断；distance/yaw/lateral/contact均来自对应原始物理记录。接触为净力代理，非标定实机力觉。

## R3额外16环境总览（录制前声明）

在同一闭环录制应用内增加总览相机，不启动16个应用、不修改任何物理状态。相机6Hz记录最初12个物理秒，另保存0/3/6/10s原1920×1080 RGB截图，以及每帧全部16环境的case、q/qd、物体pose/速度、阶段和目标。该镜头明确属于**live PhysX evaluation，非网络梯度更新期间的training画面**。独立状态CSV与物理tick保留，不能将静态模型复制当成并行交互。

R3若采用更长任务时间及新重接触合同，必须与R3自身的同合同控制/RL比较；不将R1/R2的12s结果和R3结果直接作为算法因果差异。电影时间轴仍以各run真实物理时间为准。

## 存储及来源

原始120Hz NPZ、完整相机MP4、逐帧JSON和checkpoint留非Git：`runs/client_sci_closed_loop_20261027/paired_videos/`。Git只放小GIF/filmstrip/矢量轨迹图、CSV/HTML索引、SHA与重现脚本，不放现场数据或大原档。每项绑定actual task/helper/config/scene/model/action/trace SHA、初态检查、相机、physics ticks、fps、退出状态及独立验收。

所有任务为SYNTHETIC ARES自建实验；不是实机、金隅现场、真实料盘抓取或官方benchmark。不会修改ARES-R、连接机器人控制器或修改服务器驱动。
