# R0：历史控制器的真实闭环重执行

这是历史 v6b Test-Geometry 的描述性重新执行，不是新方法实验，也不用于调参。采用原三训练种子成功数13/18/23的中位seed41，未选择最好seed51。反馈和残差各自独立冷启动同一16环境、32案例、eval seed5901。保持原始task snapshot、场景、动作/工具/软限位合同及模型。摄像只改变外观和相机，物理属性签名前后相等；源方法AST证明及原始脚本随run保留。

|真实独立验收|调试后的历史反馈|历史普通残差PPO seed41|
|---|---:|---:|
|P1自由工件推送|3/16|12/16|
|P2导槽送入|5/16|6/16|
|合计|8/32|18/32|

32例有11个残差救回、1个残差退化、7个共同成功、13个共同失败。这个单历史checkpoint不等于多种子新方法优势；P2仅有一个案例净增益，不能由总计提升推断P2问题已解决。全部32例见`tables/R0_all_paired_cases.csv`，自动选片规则及来源SHA见`R0_VIDEO_SELECTION.json`。

四组预声明片段：case0=P1救回，case1=P2救回，case3=P2共同失败，case9=P2共同成功。每组均有1920×1080、H264、30fps成对MP4、GIF、filmstrip及真实轨迹图。MP4在非Git `runs/client_sci_closed_loop_20261027/paired_videos/comparisons/R0_case_*/paired_actual_physx.mp4`；本目录`index.html`为本地浏览入口，不伪造公网视频地址。

精确初态q/qd、自由工件pose/线角速度、推板位置逐项匹配≤1e−7，不声称PhysX隐藏接触缓存完全相同。每帧均来自真实post-step、reset前的相机，渲染额外physics steps=0。较早结束侧明确保持其真实终态，开场/结束显示停留不计交互。原生120Hz trace、实际动作、逐帧物理时钟和post-step终态可按`VIDEO_PROVENANCE.json`追溯。轨迹图中contact是30Hz净力代理，不是标定力觉。

验收范围为采样OBB、FK、现场来源研究软限位、工件姿态/速度与稳定保持。不是连续碰撞检测、完整手臂网格安全认证、实机、现场柜体或真实料盘抓取。旧checkpoint、旧结果和负例不覆盖。
