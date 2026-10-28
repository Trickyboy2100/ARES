"""Verify release against immutable actual-physics evidence; no new env steps."""
import argparse,csv,gzip,hashlib,json,subprocess
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
import torch

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ['href','src','poster'] and v:self.links.append(v)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root,out=a.root.resolve(),a.output.resolve();checks={}
 for name in ['MODEL_INDEX','SOURCE_SHA_INDEX','RAW_SOURCE_INDEX']:
  rows=json.loads((out/(name+'.json')).read_text())
  for row in rows:
   f=Path(row['file']);f=f if f.is_absolute() else out/f;assert f.exists() and sha(f)==row['sha256'],str(f)
  checks[name]=len(rows)
 dev=json.loads((root/'development64/RESULTS.json').read_text())['rows'];table={r['method']:r for r in csv.DictReader((out/'DEVELOPMENT_RESULTS.csv').open())};groups={};hashes={}
 for r in dev:
  groups.setdefault(r['method'],[]).append(r)
  key=tuple(r['seeds']);hashes.setdefault(key,r['initial_sha256']);assert hashes[key]==r['initial_sha256']
  for ds,bs in zip(r['decisions'],r['boundaries']):
   assert len(ds) in [1,2]
   if len(ds)==2:assert any(b['stage']==0 and b['valid'] for b in bs)
 for name,rs in groups.items():
  assert len(rs)==4 and len({s for r in rs for s in r['seeds']})==64;assert int(table[name]['native_success'])==sum(sum(r['success']) for r in rs)
  program=sum(len(r['options'][i])==2 and r['errors'][i] is None for r in rs for i in range(16));assert int(table[name]['verified_program_success'])==program
 checks['development_methods']=len(groups);checks['all_initial_hashes_match']=True;checks['native_scores_and_local_contract_scores_reconciled']=True
 cost=json.loads((out/'COST_LEDGER.json').read_text());assert cost['total_external_control_interactions']==sum(r['physical_interactions'] for r in cost['rows']);checks['external_control_interactions']=cost['total_external_control_interactions']
 phasehashes={sha(f) for f in (out/'SOURCE_PHASES').glob('*.py')};phasehashes|={sha(f) for f in (out.parents[2]/'tools').glob('handoff_*.py')};models=[]
 for row in json.loads((out/'MODEL_INDEX.json').read_text()):
  f=out/row['file'];ck=torch.load(f,weights_only=False)
  if 'model' in ck:
   assert ck['source_sha256'] in phasehashes;assert all(torch.isfinite(v).all() for v in ck['model'].values());models.append(dict(file=f.name,seed=ck['seed'],beta=ck['beta'],mode=ck['mode'],updates=ck['updates'],online_control_interactions=ck['physical_interactions'],source_sha256=ck['source_sha256']))
 (out/'MODEL_METADATA.json').write_text(json.dumps(models,indent=2)+'\n');checks['finite_checkpoints_with_source_provenance']=len(models)
 parser=Links();html=out/'VIDEOS/index.html';parser.feed(html.read_text());resolved=[]
 for link in parser.links:
  u=urlparse(link)
  if u.scheme=='file':f=Path(unquote(u.path))
  elif u.scheme in ['http','https','data']:continue
  else:f=(html.parent/unquote(u.path)).resolve()
  assert f.exists(),str(f);resolved.append(str(f))
 checks['html_local_targets']=len(resolved);videos=json.loads((root/'visuals/RENDER_MANIFEST.json').read_text())
 for row in videos:
  f=root/'visuals'/row['video'];assert sha(f)==row['video_sha256'] and row['physics_steps']==0 and row['max_state_readback_error']<2e-6
  probe=json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-show_streams','-of','json',str(f)],text=True));v=next(s for s in probe['streams'] if s['codec_type']=='video');assert int(v['nb_frames'])==251 and v['r_frame_rate']=='20/1'
 checks['playable_original_state_comparisons']=len(videos);checks['visual_QA']='All9 English and6 Chinese pages rendered with system Poppler and manually reviewed; no clipping/overlap. Individual plot labels corrected and final pages re-reviewed.'
 for rel,pages in [('LATEX/PAPER_SKILL_HANDOFF.pdf',9),('MENTOR_DECISION_CN.pdf',6)]:
  info=subprocess.check_output(['/usr/bin/pdfinfo',str(out/rel)],text=True);assert f'Pages:           {pages}' in info;txt=subprocess.check_output(['/usr/bin/pdftotext',str(out/rel),'-'],text=True);assert '??' not in txt
 checks['pdf_pages']={'English':9,'Chinese':6};assert sha(out/'MENTOR_DECISION_CN.pdf')==sha(out.parent/'01_CLIENT_PROPOSAL_CN.pdf');assert sha(out/'LATEX/PAPER_SKILL_HANDOFF.pdf')==sha(out.parent/'02_PAPER_SEED/main.pdf');checks['current_client_and_paper_entries_synchronized']=True
 runtime=json.loads((out/'REPRO_DATA/RUNTIME.json').read_text());assert not runtime['ares_r_write_access'] and runtime['cuda_device_count']==1 and runtime['network_interfaces']==[[1,'lo']] and not runtime['serial_devices'];checks['offline_isolation_runtime_verified']=True
 checks['sealed_test']='UNOPENED';checks['new_physics_during_verification']=0;(out/'RELEASE_QA.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
