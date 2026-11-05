"""Recompute terminal physics, pair effects and own-policy learning; no score selection."""
import argparse, collections, gzip, hashlib, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
groups, index, training = {}, {}, {}
contract_checks=[]
def check_contract(b,c,entry):
    if entry is None:return
    def extract(row):
        q=next(v for v in row['decisions'] if v['stage'] and v['step']==entry['step'] and v['obs']==entry['s'])
        return {k:q[k] for k in ['obs','queue_before','active_target_before']}
    left,right=extract(b),extract(c)
    assert left==right,('execution contract mismatch',b['seed'],entry['step'])
    contract_checks.append(dict(seed=b['seed'],step=entry['step'],queue_and_reference_sha=hashlib.sha256(json.dumps(left,sort_keys=True).encode()).hexdigest()))
for f in sorted(a.root.rglob('*.json.gz')):
    d = json.load(gzip.open(f, 'rt'))
    if 'steps' not in d or 'summary' not in d:
        continue
    s = d['summary']; steps = d['steps']; assert len(steps) == 100, f
    last = steps[-1]
    success = bool(last['evaluation']['success'][0]) if 'evaluation' in last else bool(last['native_success'])
    assert success == bool(s['success']), f
    name = f.relative_to(a.root).parts[0]
    g = groups.setdefault(name, dict(episodes=0, physical_steps=0, successes=0, planner_queries=0, planner_wall_s=0., episode_wall_s=0.))
    g['episodes'] += 1; g['physical_steps'] += len(steps); g['successes'] += success
    g['planner_queries'] += s.get('planner_queries', len(s.get('queries', [])))
    g['planner_wall_s'] += s.get('query_wall_s', sum(q.get('wall_s', 0) for q in s.get('queries', [])))
    g['episode_wall_s'] += s['wall_s']
    index[str(f.relative_to(a.root))] = dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(), seed=s['seed'], success=success, steps=len(steps))
basepath = a.root/'dev_rule/RESULTS.json'
baseline = {r['seed']:r['success'] for r in json.loads(basepath.read_text())} if basepath.exists() else {}
def compare(rows, reference=None):
    reference=baseline if reference is None else reference
    return dict(n=len(rows), success=sum(r['success'] for r in rows),
                rescue=[r['seed'] for r in rows if r['seed'] in reference and r['success'] and not reference[r['seed']]],
                regression=[r['seed'] for r in rows if r['seed'] in reference and not r['success'] and reference[r['seed']]])
for f in sorted(a.root.glob('*/HISTORY.json')):
    h=json.loads(f.read_text()); tr=[v for v in h if v['kind']=='train']; curves={}
    for v in tr:check_contract(v['baseline'],v['row'],v['option']['entry'])
    for c in sorted({v['checkpoint'] for v in h if v['kind']=='eval'}):
        z=[v for v in h if v['kind']=='eval' and v['checkpoint']==c]
        curves[c]=compare([v['row'] for v in z])
        curves[c]['adopted_episodes']=sum(v['option']['entry'] is not None for v in z)
    training[f.parent.name]=dict(own_episodes=len(tr), own_success=sum(v['row']['success'] for v in tr),
        rescue=[v['row']['seed'] for v in tr if v['delta']>0], regression=[v['row']['seed'] for v in tr if v['delta']<0],
        own_gate_decisions=sum(len(v['gate_decisions']) for v in tr), own_adoptions=sum(v['option']['entry'] is not None for v in tr),
        optimizer_updates=sum(v['optimizer_updates'] for v in tr), warm_updates=json.loads((f.parent/'MANIFEST.json').read_text()).get('warm_updates',500), curve=curves)
pairs=json.loads((a.root/'pairs/PAIRS.json').read_text())
for v in pairs:check_contract(v['baseline'],v['candidate'],v['option']['entry'])
(a.root/'EXECUTION_PREFIX_CHECKS.json').write_text(json.dumps(dict(checked=len(contract_checks),checks=contract_checks,scope='Complete saved joint queue, cursor, previous action, nominal failure flag, hold state, semantic phase, active target and observed state at adoption; full native physical prefix separately checked during collection. No PhysX state restore.'),indent=2))
mechanisms=[]
for v in pairs:
    if not v['delta']:continue
    m={k:v[k] for k in ['seed','delta','a_b','a_c','front_grasp','option_duration','budget']};m['branches']={}
    for key in ['baseline','candidate']:
        d=json.load(gzip.open(v[key]['path'],'rt'));ph={}
        for z in d['steps']:ph.setdefault(z['phase'],[z['step'],z['step']])[1]=z['step']
        x=d['steps'][-1];m['branches'][key]=dict(phase_spans=ph,final_head_in_hole=x['evaluation']['peg_head_pos_at_hole'][0],grasped=x['grasped'],success=x['evaluation']['success'][0],queries=len(d['summary']['queries']))
    mechanisms.append(m)
(a.root/'MECHANISM_CASES.json').write_text(json.dumps(mechanisms,indent=2))
controls={f.parent.name:compare(json.loads(f.read_text())) for f in sorted(a.root.glob('dev_*/RESULTS.json'))}
cf=a.root/'confirm_rule/RESULTS.json';cb={v['seed']:v['success']for v in json.loads(cf.read_text())}if cf.exists()else{}
confirmation={f.parent.name:compare(json.loads(f.read_text()),cb)for f in sorted(a.root.glob('confirm_*/RESULTS.json'))}
tool={}
for name in ['tool_scout','tool_early']:
    f=a.root/name/'RESULTS.json'
    if not f.exists():continue
    rows=json.loads(f.read_text());by={(r['seed'],r['grasp']):r['success']for r in rows if r['mode']=='none'}
    tool[name]={m:dict(n=len(z),success=sum(r['success']for r in z),rescue=[(r['seed'],r['grasp'])for r in z if r['success']and not by[(r['seed'],r['grasp'])]],regression=[(r['seed'],r['grasp'])for r in z if not r['success']and by[(r['seed'],r['grasp'])]],errors=dict(collections.Counter(str(r['error'])for r in z)))for m in sorted({r['mode']for r in rows})for z in [[r for r in rows if r['mode']==m]]}
summary=dict(groups=groups, total={k:sum(g[k] for g in groups.values()) for k in ['episodes','physical_steps','planner_queries','planner_wall_s','episode_wall_s']},
    training=training, controls=controls, confirmation=confirmation,tool=tool,pairs=dict(n=len(pairs),baseline_success=sum(v['G_b'] for v in pairs),candidate_success=sum(v['G_candidate'] for v in pairs),effects=dict(collections.Counter(v['delta'] for v in pairs))),
    scope='Complete native100 trajectories, actual terminal predicate. Unique executed cost; shared pair and baseline costs must also be charged per comparison arm. Summed episode time is not elapsed wall clock.')
(a.root/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
(a.root/'RAW_INDEX.json').write_text(json.dumps(index,indent=2))
print(json.dumps(summary,indent=2))
