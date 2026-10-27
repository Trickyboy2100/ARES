# 训练完成清单、失败保全与预算

根目录（非Git）`runs/client_sci_closed_loop_20261027/`。本清单不替代逐文件SHA总索引；实际执行快照、所有失败和额外评估均保留，sealed final未启封。

## 已完成正式12模型：每个32768在线交互，共393216

```text
r1/param_ppo_seed31
r1/effect_param_ppo_seed31
r2/yaw_param_ppo_seed31
r2/yaw_effect_param_ppo_seed31
r3/event_ordinary_seed31
r3/event_ordinary_seed41
r3/event_ordinary_seed51
r3/event_qualified_seed31
r3/event_qualified_seed41
r3/event_qualified_seed51
r3/fixed_ordinary_seed31
r3/fixed_qualified_seed31
```

每项对应同名`_development`独立冷启动32例评估。每模型训练后的8例train场景检查不是额外训练，不混进32例开发成功率，也不作为独立测试场景数量。R1/R2正式源分别在`r1/engine`、`r2/engine_train`；R3权威源`r3/engine_training`。不要拿当前HEAD代替实际`task_source.py`。

## 已完成真实optimizer/物理smokes：共5120

`r3/smoke4_qualified_optimizer`：4096交互，84D历史诊断源engine_corrected，真实optimizer步骤，不纳入正式方法比较。

`r3/smoke4_real_gradient_audit`：1024交互，最终87D源，512条真实rollout审计，无额外虚构数据；不纳入正式模型统计。

因此正式+smoke已完成398336，而非把预计录像预算预先算作已完成。

## 必须保留的软件失败

`r3/visual_training_qualified_seed31_12288/exception.txt`：真实第一个完整16×256 rollout后，视频observer的instance-level train closure在SB3保存初始模型时捕获Env/TorchScript，引发`PickleError: ScriptFunction cannot be pickled`。失败发生在原PPO `super.train()`之前，0 optimizer更新；`ppo/progress.csv`明确记录`time/total_timesteps=4096`，必须计入4096真实交互，不能因无最终manifest记0。保留exception、实际源、梯度样本、进度与已捕获影像。

这不是正式算法训练失败；12个正式worker均完成。但不能只打包成功run而删掉该失败目录。录像重试必须新目录、单列实际交互；若重试12288成功，总数应414720，而不是忽略失败后写410624。

因此在该失败后的可核验已消耗量为398336+4096=402432；任何进行中的重试按其实际counter继续累加，不能因为尚无最终manifest把它漏记。

## 物理负结果，不是软件错误

`r3/probe4_recontact` 0/8、`r3/probe4_recontact_corrected` 3/8、`r3/probe4_exact_recontact` 3/8及其single/gain variants全部保留。它们反映对正常控制前缀的工程回归/保守包络约束，不包装成算法成败，也不被后来的8/8 smoke覆盖。

`r3/engine_train`等未被任何run引用的staging源不是一次训练，不填虚假0-step模型或耗时；有实际进度但未完成的worker则按已执行量计。

## 额外评估与重复案例

R1的`feedback32_gpu0`、`feedback32_gpu1`及后来补做`feedback32_zero_steering`都是实际完成的32例反馈评估；全部物理成本计入总索引，但共享32案例，不能当96独立任务。首次查询只看目录尾部造成遗漏而补做，不能隐去重复成本。

所有物理probe、各model train-check与development、真实录像重执行都按实际physics/control步另计。不要把它们加成在线训练交互，也不要把它们从计算预算隐藏。

## 探索性transfer已完成：0训练

预指定event seed31、冻结`transfer_push.json`（SHA `58f68b0247f91e413ad2d35d1d05de265cbdc16adebd35bbd19af735eaf75a84`），物理family=push、无导轨、未见质量/摩擦/方向组合，case seed102951。

```text
r3/transfer_push_feedback32           raw32/32
r3/transfer_push_ordinary_seed31      raw32/32
r3/transfer_push_qualified_seed31     raw29/32
```

独立验收另查总索引。仅探索性零样本transfer，不是正式final或跨工位已验证泛化；没有candidate优势。

## 进程退出

GPU0正式队列与transfer、GPU1正式队列与transfer均已正常结束。2026-10-27 10:04 UTC只读`ps`检查未见这些`run_contact_v8.py` worker，仅有查询自身；不杀其他进程、不以Kit对其他卡的轻量context误报为仍有训练。GPU2录像由独立录制worker管理，其失败/重试/最终退出单独记录。

完整复现入口见`REPRODUCE_FROZEN_R3.md`。相关完整pytest结果为74 passed（包含函数式masked-policy tests），不是用unittest72替代。
