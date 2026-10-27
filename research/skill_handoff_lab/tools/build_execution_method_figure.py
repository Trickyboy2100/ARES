"""Vector diagram grounded in one completed R3 training source/config, not a claim of benefit."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    m=json.loads((a.run/'manifest.json').read_text());c=json.loads((a.run/'run_config.json').read_text())
    assert (a.run/'RUN_COMPLETE.json').exists() and c['recontact_enabled'] and m['action_dim']==5
    assert c['policy_mode']=='qualified_gaussian' and m['optimizer_updates']>0 and m['parameters_changed']
    helper=a.run/'contact_masked_policy_v8.py';assert helper.exists()
    text=helper.read_text();assert 'self.distribution.log_prob(actions)*self.mask' in text
    if 'dependency_sha256' in m:assert sha(helper)==m['dependency_sha256']['contact_masked_policy_v8.py']
    fig,ax=plt.subplots(figsize=(7,5.4));ax.set(xlim=(0,7),ylim=(0,5.4));ax.axis('off')
    ink='#203544';teal='#177e91';gold='#c98a45';muted='#5d6e7d'
    def box(x,y,w,h,title,body,color=teal):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05,rounding_size=.08',fc='#f1f6f8',ec=color,lw=1.3))
        ax.text(x+.10,y+h-.15,title,ha='left',va='top',fontsize=10.5,fontweight='bold',color=color)
        ax.text(x+.10,y+h-.48,body,ha='left',va='top',fontsize=9.3,color=ink,linespacing=1.35)
    def arrow(x,y,u,v,label=None,color=teal):
        ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=11,lw=1.2,color=color))
        if label:ax.text((x+u)/2,(y+v)/2+.08,label,ha='center',fontsize=8,color=color)
    ax.text(.1,5.17,'Execution-aware PPO for stateful contact control',fontsize=13.5,fontweight='bold',color=ink)
    ax.text(.1,4.91,'R3 source-grounded implementation; performance advantage is not implied.',fontsize=9,color=muted)
    box(.15,2.8,3.2,1.75,'Observed state + Actor–Critic',
        f'{m["observation_dim"]}D: pose, velocity, goals and geometry;\nresponse history + controller memory.\n5D: XY residual, contact, yaw, approach.\nMLP 64 × 64; standard PPO + GAE.')
    box(3.75,2.8,3.1,1.75,'Shared stateful executor',
        'Nominal feedback + DLS servo.\nHold / refresh legal parameters.\nController-owned geometric recontact.\nNo learned high-level stage switch.')
    box(.15,.75,3.2,1.7,'Execution-active likelihood',
        r'$\ell_t=\sum_jm_{tj}\log\mathcal{N}(a_{tj};\mu_{\theta j},\sigma^2_{\theta j})$'+'\nPre-action mask is in the observation.\nInactive coordinates integrated out.\nMasked log-probability enters ratio.\nEntropy coefficient = 0.',color=gold)
    box(3.75,.75,3.1,1.7,'PhysX + independent checks',
        'One 16-env PhysX app, 120 Hz.\nFree workpiece; no pose overwrite.\nObserved response feeds next state.\nSampled geometry / tool checks,\nnot hardware safety or field transfer.')
    arrow(3.42,3.8,3.68,3.8);arrow(5.3,2.73,5.3,2.52)
    arrow(3.68,1.6,3.42,1.6);arrow(1.75,2.52,1.75,2.73,color=gold)
    ax.plot([6.93,6.93,1.05,1.05],[1.8,4.72,4.72,4.62],c=teal,lw=1.1);arrow(1.05,4.72,1.05,4.62)
    ax.text(4.5,4.76,'next observed state; no future-success input',ha='center',fontsize=8,color=teal)
    ax.text(.15,.41,'Implemented mechanism only; success gain and seed robustness require independent evidence.',fontsize=8.3,color=muted)
    ax.text(.15,.16,f'{a.run.name} | source {m["task_source_sha256"][:12]} | {m["online_training_interactions"]:,} interactions\nNot a new PPO theorem. Not field / hardware transfer. Contract and execution traces retained.',fontsize=8.1,color=muted)
    fig.tight_layout(pad=.3);a.output.mkdir(parents=True,exist_ok=True);outputs=[]
    for ext in ['pdf','svg','png']:
        p=a.output/f'FigR_method.{ext}';fig.savefig(p,dpi=180,bbox_inches='tight');outputs.append(p)
    plt.close(fig)
    (a.output.parent/'METHOD_FIGURE_PROVENANCE.json').write_text(json.dumps({'run':str(a.run.resolve()),'observation_dim':m['observation_dim'],'action_dim':m['action_dim'],'policy_mode':c['policy_mode'],'nominal_controller_not_learned_high_level_switch':True,'source_script_sha256':sha(Path(__file__)),'inputs':{str(p.resolve()):sha(p) for p in [a.run/'manifest.json',a.run/'run_config.json',a.run/'task_source.py',helper]},'outputs':{p.name:sha(p) for p in outputs}},indent=2)+'\n')
    print(json.dumps({'figure':str(outputs[0]),'state_dim':m['observation_dim'],'action_dim':m['action_dim']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
