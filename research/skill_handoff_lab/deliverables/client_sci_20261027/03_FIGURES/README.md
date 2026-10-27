# 可重建图表与证据边界

运行（工作区根目录）：

```bash
MPLCONFIGDIR=/tmp/ares_mpl envs/cash/bin/python \
  repos/ARES/research/skill_handoff_lab/deliverables/client_sci_20261027/03_FIGURES/source/build_evidence_figures.py
```

Fig1/2为原创矢量架构图，**不是**数据驱动效果图：已实现/历史实测和方法独立增益未证实的部分用实线/虚线及文字状态区分。Fig2是实际v7接口，不画不存在的successor-Q头或learned阶段切换；接触目标命令不等于实际执行位移。Fig4/5完全从已公开、已冻结的v6b实际episode与逐case独立验收重建，不从报告手填成功数字。PDF/SVG保留文字，PNG仅作预览。英文图注见`caption_en.md`，所有原始输入SHA、输出图和CSV的SHA见`provenance/FIGURE_PROVENANCE.json`。

历史证据版本`79448a47ffda5c8c1887f53e33a5dbb90d3eb490`。旧Test-*已被阅读，不能作为新方法未触碰的最终测试。Fig4左侧是探索训练已结束episode的8192交互分箱；没有episode的分箱保留缺口，不插值成0成功。粗线为3seed均值，阴影为观察范围，**不是置信区间**。右侧32k/64k验证严格只取同32案例；DirectPPO缺少32k验证，不补造曲线。

Fig5分别显示P1/P2，不用易任务掩盖P2。三seed针对同一批案例配对，并非3倍独立工位样本。`historical_seed_summary.csv`给出seed SD/range；`historical_paired_differences.csv`给出逐seed配对救回/丢失集合计数及固定案例重采样区间，**不是跨训练seed或总体泛化置信区间**。所有raw和independent成功分别保留。

新pilot只有实际完成且独立验收后才绘制Fig6：

```bash
.../build_evidence_figures.py --pilot-csv <实际运行汇总CSV>
```

接口见`../04_TABLES/PILOT_INGESTION.md`。模型/方法缺失seed不做补值；未验收结果显示Not tested，不画虚构成功柱。Fig6只消费`split=validation`，不混训练结束8例检查；推理遮蔽使用同checkpoint，不能当额外训练种子。FigS2为新pilot真正在线episode曲线，空分箱不补成功，短pilot不能宣称收敛。Fig3由独立Isaac渲染脚本产生，原图、物理身份和appearance-only diff另存其provenance，global manifest核对raw SHA并引用完整记录；它是历史模型LIVE再评估，不是新v7训练画面。不要将该状态自动继承到其他图的实验。

全部当前图表为SYNTHETIC自建研究任务；非官方benchmark、非金隅现场、非实机抓取，未包含受限点云、生产配置、真实机器人读数或现场派生图。
