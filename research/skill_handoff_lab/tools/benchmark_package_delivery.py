"""Copy only this public benchmark iteration's key evidence; never field data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser()
p.add_argument('--runs',type=Path,required=True)
p.add_argument('--figures',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
lab=Path(__file__).resolve().parents[1]
for directory,pattern in [('tools','benchmark*.py'),('configs','benchmark*.json'),('configs','maniskill_scout_freeze*.txt'),('tests','test_benchmark_residual.py')]:
    target=a.output/'code'/directory
    target.mkdir(parents=True,exist_ok=True)
    for source in (lab/directory).glob(pattern):shutil.copy2(source,target/source.name)
evidence={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in a.figures.iterdir()
          if x.is_file() and x.name!='EVIDENCE_SHA256.json' and x.suffix not in ['.aux','.out','.log']}
(a.figures/'EVIDENCE_SHA256.json').write_text(json.dumps(evidence,indent=2))
for source in a.figures.iterdir():
    if source.is_file() and source.suffix not in ['.aux','.out','.log']:
        shutil.copy2(source,a.output/source.name)
models=[]
for run in sorted((a.runs/'runs').iterdir()):
    final=run/'final_ckpt.pt'
    if not final.exists(): continue
    target=a.output/'models'/run.name
    target.mkdir(parents=True,exist_ok=True)
    for name in ['final_ckpt.pt','upstream_ppo_source.py','controller_source.py','pilot_protocol.json','launch.json','training.log']:
        if (run/name).exists(): shutil.copy2(run/name,target/name)
    models.append({'run':run.name,'final_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
                   'bytes':final.stat().st_size})
index=[]
for path in sorted(a.runs.rglob('*')):
    if not path.is_file() or a.output.resolve() in path.resolve().parents: continue
    relative=path.relative_to(a.runs)
    if any(part in ['maniskill_assets','converted','import_meshes','__pycache__'] for part in relative.parts):continue
    index.append({'path':str(relative),'bytes':path.stat().st_size,
                  'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(a.output/'RAW_DATA_INDEX.json').write_text(json.dumps({'root':str(a.runs.resolve()),
    'scope':'PUBLIC_SYNTHETIC_BENCHMARK_ONLY_NO_FIELD_DATA','files':index},indent=2))
(a.output/'MODELS.json').write_text(json.dumps(models,indent=2))
results=json.loads((a.output/'RESULTS.json').read_text())
table=''.join(f"<tr><td>{r['run']}</td><td>{r['final_success']}/{r['n']}</td><td>{r['scope']}</td></tr>" for r in results if 'eval64' in r['run'] or 'validation64' in r['run'])
(a.output/'index.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>ARES | Benchmark Pivot</title>
<style>body{margin:0;background:#edf2f5;color:#193447;font:16px/1.7 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:40px}h1{font-size:34px;margin-bottom:4px}h2{color:#087b83}section{background:white;padding:25px;margin:22px 0;border-radius:14px}video,img{width:100%;border-radius:8px}a{color:#087b83}table{width:100%;border-collapse:collapse;font-size:13px}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}.note{padding:12px;background:#fff3d9;border-left:4px solid #c2943b}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:700px){.grid{grid-template-columns:1fr}main{padding:18px}}</style>
<main><h1>ARES：公开基准与 JAKA 双轨研究</h1><p>2026-10-27 · 真实物理执行 · 完整保留正负结果</p>
<p><a href="BENCHMARK_BRIEF_CN.pdf">中文图文简报 PDF</a>　|　<a href="PAPER_PIVOT_DRAFT.pdf">论文补充稿 PDF</a>　|　<a href="PAPER_PIVOT_DRAFT.tex">LaTeX</a>　|　<a href="MODELS.json">模型清单</a></p>
<div class="note">这里是普通反馈 / PPO / 残差 PPO 的真实基线与因果验证，不是已证明的新算法。所有 64 案例属于开发验证集，不能当作密封测试或 3×64 个独立场景。Panda 成绩不是 JAKA 成绩。</div>
<section><h2>同案例：同时展示改善与退化</h2><div class="grid"><div><h3>改善案例</h3><video controls preload="metadata" poster="paired_improvement.png" src="paired_improvement.mp4"></video></div><div><h3>退化案例</h3><video controls preload="metadata" poster="paired_regression.png" src="paired_regression.mp4"></video></div></div><p>左：反馈；右：seed31 普通残差 PPO。ManiSkill / SAPIEN，半速呈现。两侧使用逐项核对的同一初始物理状态；筛选案例仅用于解释，完整成功率使用全部案例。</p></section>
<section><h2>全部开发验证结果</h2><table><tr><th>运行</th><th>最终成功</th><th>范围</th></tr>'''+table+'''</table><p><a href="RESULTS.json">完整含失败数值</a> | <a href="PAIRED_CAUSAL_AUDIT.json">残差置零与逐案例因果审计</a></p><img src="native_validation.png"></section>
<section><h2>真实训练与物理轨迹</h2><img src="learning_curves.png"><img src="feedback_physics.png"><p>训练曲线来自实际日志；未平滑掩盖退化。每颗种子的 checkpoint 都保留。普通残差收益不等于论文方法创新。</p></section>
<section><h2>Isaac 应用轨道：已运行，但范围有限</h2><img src="isaac_drawer.png"><p><a href="application_readiness/isaac_drawer4_front_view/actual_execution.mp4">官方 Franka 抽屉真实执行</a> | <a href="application_readiness/jaka_eg2_float_init/actual_actuation.mp4">JAKA/EG2 物理开合 smoke</a></p><p class="note">JAKA 已物理步进 720 步，但关节限位与 mimic 独立检查失败。不是合格抓取、任务迁移或工业安全证据。<a href="JAKA_INDEPENDENT_AUDIT.json">查看失败审计</a></p></section>
<section><h2>证据、许可与下一步</h2><p><a href="RAW_DATA_INDEX.json">完整原始数据哈希索引</a> | <a href="DELIVERY_SHA256.json">本包哈希</a> | <a href="BENCHMARK_SCOUT_DECISION.md">研究选择与边界</a></p><p>主平台：ManiSkill3；Stack 快速闭环，Peg 为不同接触条件候选；JAKA 保留 Isaac Lab。下一轮应围绕执行结果驱动的合法参数/恢复决策做同接口多种子消融，而不是继续扩大旧 P2。</p><p>仅公开合成实验，不含现场点云或生产配置。ManiSkill 资产 CC BY-NC 提示仍需商业使用复核；本包不分发仿真资产。</p></section></main></html>''')
for name in ['jaka_eg2_float_init','isaac_drawer4_front_view']:
    source=a.runs/name
    if not (source/'manifest.json').exists():continue
    target=a.output/'application_readiness'/name
    target.mkdir(parents=True,exist_ok=True)
    for filename in ['manifest.json','actual_actuation.mp4','actual_execution.mp4','physical_joint_trace.npz','source.py']:
        if (source/filename).exists():shutil.copy2(source/filename,target/filename)
hashes={str(x.relative_to(a.output)):hashlib.sha256(x.read_bytes()).hexdigest()
        for x in a.output.rglob('*') if x.is_file() and x.name!='DELIVERY_SHA256.json'}
(a.output/'DELIVERY_SHA256.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps({'models':len(models),'raw_files':len(index),'delivery_files':len(hashes)}))
