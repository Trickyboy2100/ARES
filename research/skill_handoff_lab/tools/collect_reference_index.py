"""Pin public upstream code read for mechanism analysis; never run upstream code."""
import argparse,hashlib,json,subprocess
from pathlib import Path
PROJECTS={
 'STAP':('agiachris/STAP',['README.md','LICENSE','stap/planners/cem.py','stap/planners/utils.py','stap/agents/sac.py']),
 'SeqDex':('sequential-dexterity/SeqDex',['README.md','LICENSE','dexteroushandenvs/policy_sequencing/transition_value_trainer.py','dexteroushandenvs/scripts/bi_optimization.py']),
 'ViSkill':('med-air/ViSkill',['README.md','LICENSE','viskill/agents/sc_sac.py','viskill/agents/sc_ddpg.py']),
 'raps':('mihdalal/raps',['README.md','LICENSE','rlkit/rlkit/envs/primitives_wrappers.py']),
 'recovery-chaining':('shivamvats/recovery-chaining',['README.md','belief_srs/skills/skill_chain.py','belief_srs/skills/recovery_skill.py','belief_srs/envs/subtask_wrapper.py','belief_srs/skills/skill_chaining.py']),
 'skimo':('clvrai/skimo',['README.md','skimo_agent.py','skimo_rollout.py','pretrained_models.md']),
 'skill-composition-benchmark':('han20192019/skill-composition-benchmark',['README.md','LICENSE','mani_skill/envs/tasks/tabletop_skill_compose/cube_out_and_in_compose.py','mani_skill/envs/tasks/tabletop_skill_compose/tools_usage_compose.py','mani_skill/envs/tasks/tabletop_skill_compose/obstacle_avoidance_compose_1.py','mani_skill/examples/motionplanning/panda/solutions/cube_out_and_in_compose.py']),
 'ManiSkill':('mani-skill/ManiSkill',['README.md','LICENSE','mani_skill/envs/tasks/tabletop/push_cube.py','examples/baselines/ppo/ppo.py']),
 'robocasa':('robocasa/robocasa',['README.md','LICENSE','robocasa/utils/subtask_labeling/base.py','robocasa/environments/kitchen/kitchen.py'])}

def main(a):
    rows=[]
    for name,(repo,files) in PROJECTS.items():
        root=a.refs/name;sha=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();entries=[]
        for file in files:
            content=subprocess.check_output(['git','-C',str(root),'show',f'HEAD:{file}'])
            entries.append({'path':file,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),'url':f'https://github.com/{repo}/blob/{sha}/{file}'})
        rows.append({'project':name,'upstream':f'https://github.com/{repo}','commit':sha,'source_files':entries})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'date':'2026-10-16','upstream_executed':False,'projects':rows},indent=2))
    print([(r['project'],r['commit'][:10]) for r in rows])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--refs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
