"""Compact review generated from measured summaries; links to full evidence."""
import argparse,json,html,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);s=json.loads((a.root/'SUMMARY.json').read_text());arms=s['arms']
for f in a.root.glob('*.json'):shutil.copy2(f,a.out/f.name)
shutil.copytree(a.root/'figures',a.out/'figures',dirs_exist_ok=True)
shutil.copytree(a.root/'video',a.out/'video',dirs_exist_ok=True)
shutil.copy2(a.root/'FROZEN_RESULTS.tex',a.out/'FROZEN_RESULTS.tex')
def table(names,cp):
 rows=[]
 for n in names:
  v=arms[n];e=v['evaluation'][str(cp)];rows.append(f'<tr><td>{html.escape(n)}</td><td>{e["success"]}/{e["n"]}</td><td>{len(e["rescue"])}</td><td>{len(e["regression"])}</td><td>{v["training_n"]}</td><td>{v["updates"]}</td></tr>')
 return '<table><tr><th>Arm</th><th>Success</th><th>Rescue</th><th>Regression</th><th>Training episodes</th><th>Updates</th></tr>'+''.join(rows)+'</table>'
initial=table(['online_sac','online_success_bc','relative_sac64','relative_bc64'],64)
fresh=[]
for n,v in arms.items():
 if n.startswith('confirm_'):fresh.append(f'<tr><td>{html.escape(n)}</td><td>{v["success"]}/{v["n"]}</td><td>{len(v["rescue"])}</td><td>{len(v["regression"])}</td></tr>')
fresh='<table><tr><th>Frozen policy/control</th><th>Success</th><th>Rescue</th><th>Regression</th></tr>'+''.join(fresh)+'</table>'
pooled=json.loads((a.root/'CONFIRMATION_SUMMARY.json').read_text())['pooled']
pooled_table='<table><tr><th>Frozen method</th><th>Success/32</th><th>Rescue</th><th>Regression</th><th>Plan queries</th></tr>'+''.join(f'<tr><td>{n}</td><td>{v["success"]}</td><td>{len(v["rescue"])}</td><td>{len(v["regression"])}</td><td>{v["planning_queries"]}</td></tr>'for n,v in pooled.items())+'</table>'
clock=json.loads((a.root/'clock_probe/COMPARISONS.json').read_text())if(a.root/'clock_probe/COMPARISONS.json').exists()else []
c=s['new_cost'];body=f'''<!doctype html><meta charset="utf-8"><title>ARES — execution reference study</title><style>body{{font:17px system-ui;color:#183248;background:#f4f7fa;max-width:1120px;margin:35px auto;padding:20px}}section{{background:white;padding:25px;margin:20px 0;border-radius:12px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}img{{width:100%}}a{{color:#087a8e}}small{{color:#526779}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}</style>
<h1>Execution reference, learning and plan progress</h1><p>Native Peg100 · actual grasp and physical continuation · no hardware · no sealed tests</p>
<section><h2>1. Matched64-episode factorial</h2><p>Historical measured endpoints are unchanged. New-interface arms collected their own experience. All16 states are seen development data.</p>{initial}<img src="figures/factorial_learning_curve.png"><img src="figures/final64_rescue_regression.png"><img src="figures/online_training_curve.png"></section>
<section><h2>2. Equal-budget continuation and stochastic repeat</h2><p>Same inherited neural initialization; independent replay/exploration seeds. No claim of independent neural initialization. Fixed endpoints and declared selected checkpoints are distinct.</p><img src="figures/matched_extension_repeat.png"><p><a href="FROZEN_SELECTION_31601.json">Seed31601 selection</a> · <a href="FROZEN_SELECTION_31602.json">Seed31602 selection</a></p></section>
<section><h2>3. Frozen new natural initial states</h2><p>760200–760215 were evaluated after all four policies were frozen;760216–760231 were added adaptively without changing models or rules. Rescue/regression is paired against common nominal continuation; these 32 states are not a sealed publication test. Nominal19/32 remains strongest; the observed learned-interface benefit is fewer regressions, not new rescues.</p>{pooled_table}<img src="figures/frozen_confirmation32.png"><details><summary>Separate cohort results</summary>{fresh}</details></section>
<section><h2>4. Physical mechanism and counterexample</h2><p>Two complete trajectories share 94 actual physical steps. A 3 mm target correction restarts the remaining plan; target magnitude alone does not describe execution cost. Geometry and schedule explanations are separated by zero-offset replanning probes.</p><img src="figures/schedule_restart_mechanism.png"><p><a href="PREDECESSOR_DIAGNOSTIC.json">All shared-grasp pairs</a> · <a href="SCHEDULE_MECHANISM.json">Measured queue/source evidence</a> · <a href="LATE_REPLAN_VALUES.json">Frozen Actor and Q</a></p><pre>{html.escape(json.dumps(clock,indent=2))}</pre></section>
<section><h2>5. Evidence, scope and costs</h2><p>New complete native episodes: {c['episodes']}; actual physical steps: {c['steps']}; planner queries: {c['planning_queries']}; summed planner wall time: {c['planning_wall_s']:.2f}s. Historical data cost is separate. Summed time is not elapsed wall time under parallel runs.</p><p><a href="README.md">Research conclusions</a> · <a href="PRIOR_ART.md">Nearest papers and code</a> · <a href="SUMMARY.json">Full learning/cost data</a> · <a href="RAW_MANIFEST.json">Archive SHA</a> · <a href="video/SYNC_VIEWER.html">Full measured-state video and synchronized curves</a></p><p>Reference separation is not a new SAC algorithm. Smaller immediate command disturbance does not establish improved online learning. Remaining claims are stated individually in the manuscript and claim matrix.</p></section>'''
(a.out/'REVIEW.html').write_text(body)
