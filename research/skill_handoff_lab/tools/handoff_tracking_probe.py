"""Single-variable native250 physical intervention on previously observed failures."""
import argparse,gzip,json
from pathlib import Path
from handoff_vector import VectorHandoff
from handoff_physics import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);ex=VectorHandoff(trace=True);rows=[]
    try:
        for batch in [0,3]:
            digest=None
            for name,strides in [('original',{}),('full_C_approach_samples',{'C_REACH':1,'C_DESCEND':1})]:
                ex.path_stride_by_label=strides
                r=ex.run(list(range(420000+16*batch,420016+16*batch)),lambda o,i,st:(1,dict(kind='nominal')))
                if digest is None:digest=r['initial_sha256']
                assert digest==r['initial_sha256']
                f=a.out/f'{name}_{batch}.trace.json.gz'
                with gzip.open(f,'wt') as h:json.dump(dict(initial=ex.initial,transitions=ex.rows,plans=ex.plans),h,separators=(',',':'))
                r.update(method=name,batch=batch,trajectory_file=f.name,trajectory_sha256=sha(f));rows.append(r)
                (a.out/'RESULTS.json').write_text(json.dumps(dict(rows=rows,scope='KNOWN_FAILURE_DIAGNOSTIC_NOT_GENERALIZATION',intervention='Only C_REACH/C_DESCEND position stride 3->1; targets, gains, six tracking holds, native250 unchanged',physical_interactions=sum(r['physical_interactions'] for r in rows)),separators=(',',':')))
                print(json.dumps(dict(method=name,batch=batch,success=sum(r['success']),errors=r['errors'],wall_s=r['wall_s'])),flush=True)
    finally:ex.close()
if __name__=='__main__':main()
