# EXP-002 有限执行效果机制：训练前CPU科学审查

日期：2026-10-31。范围：只读源码审查和纯NumPy接口检查；**本审查没有物理步、GPU运行或RL训练**。训练成绩以本轮正式运行结果为准，不在这里预判。

## 审查版本

| 源文件 | SHA256 |
|---|---|
| program_contract_effect.py | `8bfc1de65cb2ebaa0bc4c93f0f145e1a415f2b4b8db96c5998e50516a1f66206` |
| program_adaptive_rl.py | `339990fa35ee56a25a2c636ded542c839e02a4d9f897c3acb9bac57c5a470f14` |
| program_adaptive_contact.py | `0dab2da19d2d4519974616476e340c587988400b52c40baffa0ef15945b235b2` |

若后续源码有变动，以上结论对应这些SHA，不能替代新版本审查。

## 已修问题与接口公平性

- 成功奖励现在在保持到原生第200步后计算：终态成功+5、否则−1；不再奖励“瞬时到位后丢失”的净+4。
- 控制停止使用**有偏测量坐标**下的位置≤4mm及角误差≤0.03rad；真实`evaluate()`仅用于终态奖励和证据，不再直接决定在线停止。评估奖励可以使用真值，但不代表有部署传感器实现。
- 两策略均获得相同四帧，每帧21维，合计84维；普通SAC尾部补6个零，候选追加6个确定性派生特征，网络输入均90维。CPU测试逐元素确认原84维一致。
- 新增双方共享的推进允许比例、单步关节增量裁剪比例。效果模块排除推进允许<0.5、裁剪>0.5的窗口观测。CPU构造检查中两种排除场景的`used`均为0，允许推进且未裁剪的停滞样本才更新分数。
- 未发现隐藏`bias_mm`、种子或真实孔位直接进入策略输入。普通策略同样能看到动作、结果位移、力和权限信号。环境内部真值用于构造有偏观测是仿真实现，不等于网络获得真值。

这些检查只证明代码接口及限定行为，**不证明传感器噪声鲁棒性、真实接触机理准确性或训练有效性**。

## 候选机制的准确表述

这是“同历史Residual SAC附加有限窗口的几何执行兼容特征”原型，仍为 **UNVERIFIED**。候选网格表示真实孔轴在有偏测量坐标中的位置，即负传感偏差；`hypotheses-yz`方向正确。不能将输出直接命名为已校准的偏差后验。

分数使用固定0.5mm净空、方向误差门控、轴向进展与接触力判据，没有从数据估计观测噪声或验证似然校准。四个相邻物理窗口相关，也不满足独立观测假设。推进允许和关节增量裁剪是改善归因的代理，仍不等同于实际TCP轴向作用或接触力方向；积分控制、姿态误差、接触模式切换也可能造成停滞。当前“posterior”是代码变量名，应在论文中称**启发式归一化兼容分数**。

额外6维没有增加隐藏状态信息，但提供手工物理先验和计算；若候选优于普通SAC，首先只能归因于这种表示偏置，不能直接归因于新RL算法或独立科学创新。与SCOUT、在线RLS、几何搜索的关系见同目录`PRIOR_ART_REVIEW.md`。

## 强基线事实及不可直接拼接的协议

已只读检查以下真实结果：

- `runs/research_program/EXP-002/search_v3_1/RESULTS.json`
- `runs/research_program/EXP-002/search_v3_-1/RESULTS.json`

两初态727000、727002 × 两个偏差±1.2mm：搜索**4/4**完整200步成功；RLS **2/4**。这是**4个设计条件、2个独立初态**，不是4个独立泛化场景。搜索已在这些设计条件达到上限，不能略去它而仅对比较弱RLS或普通SAC。

重要协议差别：上述保存的`program_adaptive_contact.py`仍以真实原生成功信号停止。现在的新接口已改成有偏测量停止。因此旧搜索4/4需保留为真实历史证据，但**不能直接当作当前SAC同协议成绩**；最终配对需以新接口重新运行搜索/几何/RLS，并同时记录源码SHA、初态和实际prefix状态。

## 如何解释两组1024决策试验

1024是预设RL决策预算，不是物理步数或成熟收敛证明。prefix和终态保持也消耗实际物理步；不同策略耗时会让同决策预算覆盖不同episode数和不同初态比例。必须同时披露决策数、物理步、完整episode、prefix成本、优化器更新、训练种子和评估初态。不能在未收敛小预算下用普通SAC低成功率证明机制优势。

评估必须使用同一新版环境、相同未见初态×偏差组合；规划器随机调用可能被历史episode影响，宜冷过程或核对实际prefix边界。应区分成功率、配对救回/退化和成功时决策/物理耗时。若搜索与候选都成功，则只能检验成本差异；不得事后将成本换成原先未声明的主要贡献。

若候选没有超过搜索，保留负结果并分析：特征是否经常`used=0`、与真实孔轴偏差是否一致、错误置信是否使退化增加、策略是否依赖额外分数而忽略真实接触历史。下一有信息量实验是这些机理消融和新物理条件，而不是简单延长同一低信息训练。

## 研究判断

可以执行这两组有界真实训练以验证接口和学习信号，且不能将其命名为新算法成果。发表级增量仍需同历史/同预算强基线、排除手工先验或动作权限差异、独立场景与机制消融。原生任务和自建感知偏差任务必须分开汇报，ARES-R与实机部署无新增证据。

### 评估协议落实（结果尚未读取）

主研究执行者已预定：729000–729003 × 偏差[-1.2,0,1.2]mm，共12个完整任务开发条件；geometry/search/rls和两个1024决策训练模型均按当前observed-stop源码执行。两训练结束冻结后才读取评估，不据此挑模型。该组为development evaluation，不是封存测试，也不是12个独立初态。本条记录协议，不填入尚不存在的结果。

## 追加：critic-only LayerNorm的SB3 2.7.0接口核对

本节为已知强基线修正建议，不是新方法。只读核对本地固定`external/sb3-2.7.0`中的`sac/policies.py`、`common/policies.py`、`sac/sac.py`，未复制DAWN实现。CPU实例化以下独立接口原型，检查通过：actor含0个LN、双critic含4个LN、双target含4个LN；target初始state_dict与online逐项相等；critic optimizer包含全部参数；对动作求梯度有限。未运行优化器更新或物理环境。

```python
from torch import nn
from stable_baselines3.sac.policies import SACPolicy

class CriticLNPolicy(SACPolicy):
    def make_critic(self, features_extractor=None):
        critic = super().make_critic(features_extractor)
        rebuilt = []
        for i, q in enumerate(critic.q_networks):
            old, parts = list(q.children()), []
            for j, layer in enumerate(old):
                parts.append(layer)
                if isinstance(layer, nn.Linear) and j < len(old) - 1:
                    parts.append(nn.LayerNorm(layer.out_features))
            net = nn.Sequential(*parts).to(self.device)
            critic.add_module(f"qf{i}", net)
            rebuilt.append(net)
        critic.q_networks = rebuilt
        return critic
```

适用当前ReLU `[64,64]` MLP结构。`SACPolicy._build`分别创建online和target critic，再复制参数并建立optimizer，因此必须在`make_critic`中修改；只覆盖Python列表而不覆盖注册的`qf0/qf1`会留下错误参数，训练之后替换则会使optimizer漏掉新参数。actor构造不改。保存/重载时应把该类放在可导入的稳定研究模块并做完整`SAC.save/load`验证；CPU原型未做磁盘checkpoint往返。

### 已有critic能诊断什么

对实际训练replay/真实评估轨迹中的同一90维状态，固定比较：零残差、actor确定性输出、actor的Y-only投影、Y正负修正、Z正负修正、有限回撤。记录两Q、min-Q排序、两Q分歧、动作梯度、与训练动作支持的距离；不要只查看actor输出。SB3实际actor优化的是`alpha*log_prob - min(Q1,Q2)`，确定性均值动作不是完整随机策略目标。

可以计算Q对动作扰动的有限差分敏感性，以及隐藏激活矩阵的奇异值/有效秩，但“低秩”或“动作Q差小”只是诊断，不能单独证明LN会救回或是失败根因。Q是带未来熵项的折扣soft return，不是校准成功概率；与一次确定性终态0/1直接比较会混淆价值定义。最有因果信息的是同真实边界独立物理配对动作的方向效果与critic排序是否相反。

如果Q偏爱负Z而Y-only真实救回，应首先记录为学到有害动作维度/价值排序错误。Y-only属于**部署动作投影消融**，动作权限改变，不能把投影checkpoint得分当原三维SAC成绩。应同时给强控制和其他策略同样投影，或单独命名为动作空间诊断。

训练LN应从新初始化按相同预算运行，其他条件固定。若同时改学习率、熵系数、动作维度或预热方式，不能将变化归因于LN。既有失败checkpoint及其完整评估保留，不以新版本覆盖。1024决策不足以声称成熟普通SAC被超越，即使LN短期救回也属于强基线修复。
