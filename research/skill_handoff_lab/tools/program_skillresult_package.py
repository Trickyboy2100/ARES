"""Package actual evidence, checkpoints, costs and review; verify archive hashes."""
import argparse,hashlib,json,shutil,tarfile,subprocess
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();summary=json.loads((a.out/'SUMMARY.json').read_text())
 for name in ['models31110','video_grasp','video_successor']:
  if (a.root/name).exists():shutil.copytree(a.root/name,a.out/name,dirs_exist_ok=True)
 for group,names in [('training_physics',['train96']),('policy_physics',['eval112','corrected112'])]:
  archive=a.out/(group+'.tar.gz');index=[]
  with tarfile.open(archive,'w:gz') as tar:
   for name in names:
    for path in sorted((a.root/name).rglob('*')):
     if path.is_file():tar.add(path,arcname=str(path.relative_to(a.root)),recursive=False);index.append(dict(path=str(path.relative_to(a.root)),sha256=sha(path),bytes=path.stat().st_size))
  with tarfile.open(archive) as tar:
   for row in index:assert hashlib.sha256(tar.extractfile(row['path']).read()).hexdigest()==row['sha256']
  (a.out/(group+'_FILES.json')).write_text(json.dumps(index,indent=2));assert archive.stat().st_size<95_000_000
 for name in ['TRAIN.log','EVAL.log','CORRECTED.log']:
  if (a.root/name).exists():shutil.copy2(a.root/name,a.out/name)
 videos=[]
 for path in a.out.glob('video_*/*.mp4'):
  ff=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,r_frame_rate,width,height','-of','json',str(path)]));assert int(ff['streams'][0]['nb_read_frames'])==101;videos.append(dict(path=str(path.relative_to(a.out)),sha256=sha(path),probe=ff))
 rows=[]
 for name,v in summary['methods'].items():
  pair=summary['paired_vs_ordinary'][name];rows.append(f"<tr><td>{name}</td><td>{v['success']}/{v['n']}</td><td>{v['native']}/{v['n']}</td><td>{pair['rescue']} / {pair['regression']}</td><td>{v['mean_wall_s']:.2f}</td></tr>")
 links=''.join(f'<p><a href="{n}/SYNC_VIEWER.html">{n}: synchronized RAW / English annotated full episode</a></p><video controls preload="metadata" src="{n}/ANNOTATED.mp4"></video>' for n in ['video_grasp','video_successor'] if (a.out/n).exists())
 html='''<!doctype html><meta charset="utf-8"><title>ARES SCI | SkillResult evidence</title><style>body{background:#102133;color:#e7edf3;font:17px system-ui;max-width:1450px;margin:32px auto;padding:20px}a{color:#70cbec}img,video{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #415363;text-align:left}.badge{color:#ffc56b}p{line-height:1.6}</style><h1>SkillResult: physical effect ≠ independent RL gain</h1><p class="badge">UNVERIFIED independent method gain · actual offline simulation · no physical robot motion</p><p>96 parameter interventions across8 native initial scenes; bounded frozen-model comparisons across2 further scenes ×2 masses ×2 declared targets. Native100-step success retained AND40mm declared goal. Initial probability-ranking runs are diagnostic only; primary results use separately executed log-ranking correction. No final-test or multisample-seed claim.</p><img src="KEY_FIGURE.png"><h2>Actual corrected physical results</h2><table><tr><th>Method</th><th>Full-task</th><th>Native only</th><th>Rescue / regression vs ordinary</th><th>Mean episode wall seconds</th></tr>'''+''.join(rows)+'''</table><p>Rescues share initial state; terminal-head ablations additionally have exactly equal executed prefixes. Ordinary terminal Q is also the supervised-score baseline. UQ uses3 bootstrap critics, not original SCOD. Foresight is a fixed-suffix state-value analogue. Calibrated probabilities may still fail on new scenes; see metrics.</p><h2>Physical mechanisms, full100-step videos</h2>'''+links+'''<h2>Audit and reproduction</h2><p><a href="README.md">Research decision</a> · <a href="REGRESSION_MECHANISM.md">Three regressions</a> · <a href="SUMMARY.json">Exact metrics</a> · <a href="EPISODES.csv">Every episode</a> · <a href="BUDGET.json">Planning budget vs physics</a> · <a href="PROTOCOL.md">Predeclared design and limitations</a> · <a href="REPRODUCE.md">Reproduction</a></p>'''
 (a.out/'REVIEW.html').write_text(html)
 (a.out/'PACKAGE.json').write_text(json.dumps(dict(physical_episodes=summary['episodes'],physics_control_steps=summary['physics_steps'],optimizer_updates=13600,training_seed=31110,archives={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in a.out.glob('*.tar.gz')},videos=videos,checkpoint_sha256={p.name:sha(p) for p in (a.root/'models31110').glob('*.pt')},evaluator_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('program_skillresult_*.py')},scope='Public simulator assets and actual measured states only; zero hardware operations. All original outcomes including diagnostic numerical failures retained.'),indent=2))
 print('VERIFIED PACKAGE',summary['episodes'],len(videos),'videos')
if __name__=='__main__':main()
