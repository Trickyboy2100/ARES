#!/usr/bin/env python3
"""Cache selected pinned public code for read-only mechanism review, not import.

No assets, dependencies or external checkpoints are fetched. Cache is non-Git.
Missing root LICENSE is recorded rather than interpreted as open permission.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import base64
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[8]
CACHE=ROOT/'runs/public_reference_code_20261027'
OUT=Path(__file__).resolve().parent
SOURCES={
 'HACManPP':('JiangBowen0008/HACManPP','e1c00f23b0db209d1963d2b6f33fd452a60196ac',[
   'README.md','hacman/algos/location_policy.py','hacman/algos/multi_td3.py',
   'hacman/utils/primitive_utils.py','hacman_bin/primitives.py',
   'hacman_suite/env_wrappers/primitives.py']),
 'Fabrica':('yunshengtian/Fabrica','215a30fe51e59299588a5b5e417a9cb934fb393e',[
   'README.md','LICENSE','learning/LICENSE.txt',
   'learning/isaacgymenvs/tasks/fabrica/fabrica_task_assemble.py',
   'learning/isaacgymenvs/tasks/fabrica/fabrica_fixplug_task_assemble.py']),
 'recovery-chaining':('shivamvats/recovery-chaining','4894acc3768678de4dee6358df14c461a411162e',[
   'README.md','scripts/learn_recoveries.py','belief_srs/skills/nominal_skills.py',
   'belief_srs/skills/primitive_skills.py','belief_srs/skills/recovery_skill.py',
   'belief_srs/skills/nominal_q_value.py','belief_srs/envs/discrete_action_wrapper.py',
   'belief_srs/skills/value_fn.py']),
}

def fetch(task):
 name,repo,commit,rel=task
 url=f'https://raw.githubusercontent.com/{repo}/{commit}/{rel}'
 target=CACHE/name/rel
 if not target.is_file():
  # curl uses the system TLS backend; Python/OpenSSL and shallow Git both
  # encounter intermittent EOFs on this host. Never disable certificate checks.
  try:
   data=subprocess.run(['curl','-fsSL','--retry','1','--connect-timeout','8',url],
                       check=True,capture_output=True,timeout=45).stdout
  except subprocess.CalledProcessError:
   api=f'https://api.github.com/repos/{repo}/contents/{rel}?ref={commit}'
   payload=subprocess.run(['curl','-fsSL','--connect-timeout','8',api],
                          check=True,capture_output=True,timeout=30).stdout
   obj=json.loads(payload)
   if obj.get('encoding')!='base64':raise ValueError('Unexpected public API content encoding')
   data=base64.b64decode(obj['content'])
  target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(data)
 else:data=target.read_bytes()
 return {'repository':repo,'commit':commit,'path':rel,'source_url':url,
         'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
         'cache_path':str(target),'status':'FETCHED_READ_ONLY_NOT_EXECUTED'}

def main():
 tasks=[(name,repo,commit,rel) for name,(repo,commit,paths) in SOURCES.items() for rel in paths]
 def bounded_fetch(task):
  try:return fetch(task)
  except Exception as exc:
   name,repo,commit,rel=task
   return {'repository':repo,'commit':commit,'path':rel,'status':'FETCH_FAILED_NOT_READ',
           'error':f'{type(exc).__name__}: {str(exc)[:180]}'}
 with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(bounded_fetch,tasks))
 result={'scope':'Selected public source only; no install, imported code, weights or assets',
         'date':'2026-10-27','files':rows,
         'root_license_tree_audit':{
          'JiangBowen0008/HACManPP':'No top-level LICENSE found in recursive pinned tree; third-party notices do not license own code',
          'shivamvats/recovery-chaining':'No LICENSE file found in recursive pinned tree; permission unresolved',
          'yunshengtian/Fabrica':'Root MIT and separate learning/LICENSE.txt fetched; review both before reuse'}}
 (OUT/'PINNED_PUBLIC_CODE_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'files':len(rows),'fetched':sum(x['status']=='FETCHED_READ_ONLY_NOT_EXECUTED' for x in rows),
                   'bytes':sum(x.get('bytes',0) for x in rows),'cache':str(CACHE),
                   'failed':[x['path'] for x in rows if x['status']=='FETCH_FAILED_NOT_READ']},indent=2))

if __name__=='__main__':main()
