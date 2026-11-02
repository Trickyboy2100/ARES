"""Source-linked trace figures and restrained captions for two physical pairs."""
import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 captions={'video_grasp':'''OBSERVED: Same native initial state, seed743041, mass scale1.25, goal y+80mm. Fixed100mm grasp /−65mm hook fails (native0, target error68.3mm);90mm grasp /same hook succeeds (native1,6.2mm). Both pass the real stable-grasp boundary; measured attachment x differs by about9.4mm. The yaw search also selects a different angle, as permitted by the same controller.

INTERPRETATION: The front-skill parameter affects attachment and subsequent geometric execution. This is physical controllability, not an RL gain or proof of target-dependent ranking reversal.

NOT ESTABLISHED: A single pair does not establish generalization or independent method novelty. The96-episode matrix is the quantitative context.''',
 'video_successor':'''OBSERVED: Seed743051, mass scale.75, goal y+80mm; ordinary and relative+contract methods share exactly the same recorded first-skill prefix and90.78125mm grasp. Ordinary selects−65.625mm hook and finishes43.88mm from goal; candidate selects−62.5mm and finishes33.74mm away. Both retain the original native task criterion. The candidate rescues this complete task under the same100-step horizon.

INTERPRETATION: This is a real successor-action effect after an actual SkillResult. Selection uses stable logit ranking and frozen trained weights.

NOT ESTABLISHED: Generic auxiliary, contract-only and relative-only controls also rescue this case. This video does not show an independent contract/relative-method advantage. See the full paired results, including failures. Initial probability-ranked experiments are separate diagnostic runs.'''}
 for name,caption in captions.items():
  out=a.root/name;d=json.loads((out/'DATA.json').read_text());fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained');keys=list(d['series'][0]['signals'])
  for ax,key in zip(axes.flat,keys):
   for s,col in zip(d['series'],['#c37920','#128a97']):
    ax.plot(range(101),s['signals'][key],color=col,label=s['label'])
    for event in s['decisions']:ax.axvline(event['step'],color=col,ls=':',alpha=.5)
   ax.set(title=key,xlabel='Recorded physical control step');ax.grid(alpha=.2)
   if key=='Cube/base distance (mm)':ax.axhline(600,color='black',ls='--',label='Native threshold')
   if key=='Declared target distance (mm)':ax.axhline(40,color='black',ls='--',label='Target threshold')
  axes[0,0].legend(fontsize=8);fig.savefig(out/'TRACE.png',dpi=300);fig.savefig(out/'TRACE.pdf');plt.close(fig)
  (out/'CAPTION.md').write_text('# DATA_DERIVED_PLOT / MEASURED_POSE_REPLAY\n\n'+caption+'\n\nFull101 frames: initial0 + actual100 steps at20Hz. No new physics steps. Colored actor boxes and target circles are projected geometry, not measured contact patches. Frame0 force-buffer readback is unvalidated and excluded.\n')
  m=json.loads((out/'MANIFEST.json').read_text());m['outputs']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='MANIFEST.json'};(out/'MANIFEST.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
