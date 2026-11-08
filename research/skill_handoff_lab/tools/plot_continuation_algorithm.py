"""Scientific schematic of the implemented, unvalidated candidate."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
f,ax=plt.subplots(figsize=(12,5));ax.set(xlim=(0,12),ylim=(0,5));ax.axis('off')
def box(x,y,w,h,text,c='#edf5f4'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc=c,ec='#306966',lw=1.3));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(x,y,u,v):ax.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=14,lw=1.4,color='#344054'))
box(.2,3.1,2.1,1.1,'Native reset\nReal grasp → SkillResult\nNo state restoration')
box(2.8,3.1,2.1,1.1,'Recorded prefix equality\nQueue + reference + budget\nActor still sees 83 features')
box(5.5,3.8,2.3,.8,'Reference choice\nthen frozen continuation')
box(5.5,2.5,2.3,.8,'Different legal Option\nthen SAME continuation')
box(8.5,3.1,3.1,1.1,'Actual terminal outcome contrast\nPolicy SHA + duration + full cost\nKeep regressions and ties')
box(8.5,.5,3.1,1.3,'Own-policy SMDP updates\nSAC / supervised / retention\nSame legal Options + native 100')
box(4.7,.5,3.1,1.3,'Executed witness choices\nSoft support vs constrained update\nCurrent-reference paired evidence','#fff4e5')
box(.4,.5,3.3,1.3,'Fixed checkpoints 0 / 64 / 128\nRefresh pairs; accumulate witnesses\nNo runtime action override')
for q in [(2.3,3.65,2.8,3.65),(4.9,3.65,5.5,4.2),(4.9,3.65,5.5,2.9),(7.8,4.2,8.5,3.8),(7.8,2.9,8.5,3.4),(10,3.1,10,1.8),(8.5,1.15,7.8,1.15),(4.7,1.15,3.7,1.15),(1.4,1.8,1.4,3.1)]:arrow(*q)
f.suptitle('Continuation-bound evidence for executable skill policy improvement',fontsize=15)
f.text(.5,.01,'SAC, imitation and constrained optimization are established. Finite witness feasibility is not a task-performance guarantee.',ha='center',fontsize=9)
f.tight_layout(rect=(0,.03,1,.95))
for ext in ['png','pdf']:f.savefig(a.out/f'ALGORITHM.{ext}',dpi=180)
