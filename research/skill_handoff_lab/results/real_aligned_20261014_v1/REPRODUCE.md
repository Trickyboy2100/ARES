# 只读审计复现

工作区根目录`/home/andyee/robot-research/ARES-Skilllab`。不需要连接机器人。保留原资产与旧实验；输出路径必须是新目录。

```bash
PYTHONPATH=repos/ARES:repos/ARES-R/src envs/kinematics/bin/python repos/ARES/research/skill_handoff_lab/tools/real_to_sim_audit.py --ares repos/ARES --ares-r repos/ARES-R --output runs/real_aligned_audit_replay
MPLCONFIGDIR=/tmp/ares-audit-mpl PYTHONPATH=repos/ARES:repos/ARES-R/src:envs/kinematics/lib/python3.13/site-packages envs/cash/bin/python repos/ARES/research/skill_handoff_lab/tools/plot_real_alignment.py --ares repos/ARES --ares-r repos/ARES-R --audit runs/real_aligned_audit_replay
PYTHONPATH=repos/ARES:repos/ARES-R/src envs/kinematics/bin/python repos/ARES/research/skill_handoff_lab/tools/build_site_reference.py --ares repos/ARES --ares-r repos/ARES-R --audit runs/real_aligned_audit_replay --asset runs/multicontract_20261014/asset_mesh_colliders_v5
envs/cash/bin/python -m unittest discover -s repos/ARES/research/skill_handoff_lab/tests -p test_real_alignment_artifacts.py -v
```

脚本将独立使用USD的joint局部frames/轴计算各link FK，而非给USD赋同一URDF FK再宣称对齐。离线测试仅校验冻结证据内部一致及未错误宣称物理成功，不能证明实物安全。

静态渲染不需要重建：解压结果目录的SITE_REFERENCE_STATIC.usda.gz，校验解压内容SHA与STATIC_SCENE_MANIFEST/capture_manifest一致；在190既有Isaac5.1个人环境运行capture_site_reference.py，或复用run_isaac51_capture.py的独立进程/超时管理。渲染timeline保持stop、rep.step(delta_time=0)，因此0物理步。capture.py对应截图v2，初版截图曝光较高仅保留作为调试历史。

source_manifest记录原文件SHA；artifact_manifest记录完整交付件与补充模型/脚本SHA。引用的September配置与二级October摘录不能替代缺失的October原档。新机器需按manifest版本复核，不将本地目录路径视为原始现场路径。
