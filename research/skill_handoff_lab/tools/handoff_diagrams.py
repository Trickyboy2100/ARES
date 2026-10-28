"""Paper diagrams: evidence status is explicitly distinct from industrial context."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
 def box(ax,x,y,w,h,label,color='#e6f2ef',dash=False):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',lw=1.5,edgecolor='#6e8a94',facecolor=color,linestyle='--' if dash else '-'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=12)
 def arrow(ax,x,y,xx,yy,dash=False):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',lw=1.8,color='#40697a',linestyle='--' if dash else '-'))
 def save(fig,name):
  fig.savefig(a.output/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(a.output/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,6.8));ax.set(xlim=(0,12),ylim=(0,7));ax.axis('off')
 ax.text(.15,6.7,'One paper: executable parameter adaptation at verified boundaries',fontsize=19,weight='bold',color='#173a4a')
 box(ax,.3,4.9,3.2,1.05,'ARES-R perception\nmultimodal scene', '#f1f2f4',True);box(ax,4.3,4.9,3.2,1.05,'Skill Contract / Runtime\nindustrial motivation','#f1f2f4',True);box(ax,8.3,4.9,3.2,1.05,'JAKA / AMR / cuRobo\nNOT evaluated here','#f1f2f4',True);arrow(ax,3.62,5.43,4.18,5.43,True);arrow(ax,7.62,5.43,8.18,5.43,True)
 ax.text(.3,4.38,'GRAY / DASHED: context only; no visual policy, hardware transfer or production result',fontsize=11,color='#66717c')
 box(ax,.3,2.65,3.2,1.1,'Public native physics\nPanda + StackPyramid');box(ax,4.3,2.65,3.2,1.1,'Two-boundary policy\nlegal three-choice parameters');box(ax,8.3,2.65,3.2,1.1,'MPLib + joint servo\nactual grasp / release');arrow(ax,3.62,3.2,4.18,3.2);arrow(ax,7.62,3.2,8.18,3.2)
 box(ax,4.3,.6,3.2,1.1,'Measured SkillResult\ncontact / pose / stability');arrow(ax,9.9,2.52,7.62,1.15);arrow(ax,5.9,1.82,5.9,2.52)
 ax.text(.3,.1,'GREEN: implemented native-physics experiment | same executor, observations and action permissions for RL baselines',fontsize=11,color='#285e58');save(fig,'Fig1_scope_and_system')
 fig,ax=plt.subplots(figsize=(12,7.1));ax.set(xlim=(0,12),ylim=(0,7.5));ax.axis('off');ax.text(.2,7.08,'Two physical decisions, not a one-shot outcome classifier',fontsize=20,weight='bold',color='#173a4a')
 box(ax,.3,5.1,2.4,1.2,'Initial state s0\nnext C geometry');box(ax,3.3,5.1,2.4,1.2,'Actor chooses A\nplacement direction');box(ax,6.3,5.1,2.4,1.2,'Pick A / place A\nOPEN + retreat');box(ax,9.3,5.1,2.4,1.2,'Verify contact\nrelease + stability')
 for x in [2.85,5.85,8.85]:arrow(ax,x,5.7,x+.32,5.7)
 box(ax,9.3,2.8,2.4,1.2,'Actual posterior s1\nno reset / teleport');arrow(ax,10.5,4.95,10.5,4.15)
 box(ax,6.3,2.8,2.4,1.2,'Actor chooses C\nphysical grasp yaw');box(ax,3.3,2.8,2.4,1.2,'Pick C / place C\nOPEN + verify');box(ax,.3,2.8,2.4,1.2,'Original native\nterminal success Y')
 for x in [9.15,6.15,3.15]:arrow(ax,x,3.4,x-.32,3.4)
 box(ax,.4,.45,5.2,1.25,'Shared PPO: terminal Y to both visited decisions\ngamma=1, actual option durations logged', '#e9eff7');box(ax,6.4,.45,5.2,1.25,'Candidate: PPO + beta E[risk(s,a)]\nRisk fitted ONLY on paired training rollouts','#faeee6')
 arrow(ax,1.5,2.65,2.7,1.85,True);arrow(ax,9,1.85,7.6,2.65,True);ax.text(.4,2.04,'No future outcome labels enter the deployed actor',fontsize=11,color='#526a78');ax.text(.4,.05,'Conventional candidate under test; parameterization, downstream value and paired subtraction are NOT novelty claims.',fontsize=10.5,color='#78563b');save(fig,'Fig2_two_boundary_method')
if __name__=='__main__':main()
