"""Plot original20Hz and independent matched-repeat100Hz evidence; no physics."""
import argparse,json,hashlib,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    a=argparse.ArgumentParser();a.add_argument('package',type=Path);p=a.parse_args().package
    data=json.loads((p/'DATA.json').read_text());ss=data['series'];cols=['#ab6200','#00839c'];fig,axs=plt.subplots(3,1,figsize=(15,11),sharex=True,gridspec_kw={'height_ratios':[1,1.25,.7]})
    for s,c in zip(ss,cols):
        sig=s['signals'];axs[0].plot(sig['Cube/base distance (mm)'],color=c,label=s['label']);axs[1].plot(sig['Tool/cube contact proxy (N)'],color=c,ls=':',lw=1.8);axs[1].plot(sig['Matched 100Hz peak (N)'],color=c,lw=2.5)
        phase={'TRANSFER':0,'HOOK':1,'PULL':2,'HOLD':3};axs[2].step(range(101),[phase.get(x,float('nan')) for x in s['phase']],where='post',color=c)
    axs[0].legend(loc='upper left',fontsize=10);axs[0].axhline(600,color='gray',ls='--',lw=1);axs[0].set_ylabel('Cube/base distance (mm)')
    axs[1].set_ylabel('Tool/cube force (N)');axs[1].set_ylim(-.5,21);axs[1].text(.02,.94,'Dotted: original 20 Hz sample\nSolid: matched-repeat 100 Hz maximum per control frame',transform=axs[1].transAxes,va='top',fontsize=10,bbox={'facecolor':'white','edgecolor':'#ccc'})
    for s,c,xytext in zip(ss,cols,[(51,9),(80,15)]):
        e=s['contact_secondary_source']['first_tool_contact_over_half_N'];k=e['control_step'];v=s['signals']['Matched 100Hz peak (N)'][k]
        axs[1].annotate(f"First >0.5 N: {k}/{e['substep']} {e['phase']}\nsubstep force {e['force_N']:.2f} N",xy=(k,v),xytext=xytext,color=c,fontsize=11,arrowprops={'arrowstyle':'->','color':c})
    axs[2].set_yticks(range(4),['TRANSFER','HOOK','PULL','HOLD']);axs[2].set_ylabel('Actual source phase');axs[2].set_xlabel('20 Hz physical control frame; solid force = maximum of five actual 100 Hz substeps')
    for ax in axs:ax.set_xlim(40,100);ax.grid(alpha=.2)
    fig.suptitle('Native PullCubeTool: temporal aliasing audit / exactly matched control-state repeats\nDATA_DERIVED PLOT / NO NEW RL CLAIM',fontsize=15);fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(p/f'TRACE_CONTACT_PHASE.{ext}',dpi=180)
    shutil.copy2(__file__,p/'contact_plot_source.py');m=json.loads((p/'MANIFEST.json').read_text());m['outputs']={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p.iterdir() if x.is_file() and x.name!='MANIFEST.json'};(p/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
if __name__=='__main__':main()
