"""Content-preserving exact-duplicate coalescing, never broad deletion.

Only explicit historical synthetic-research roots may be supplied. Emits a
reviewable plan first; --execute requires that byte-stable plan. Every original
path remains available. Never point this at Git, field data or active workers.
"""
import argparse, collections, hashlib, json, os, stat, time
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def blocks(paths):
    seen=set();total=0
    for p in paths:
        s=p.stat();i=(s.st_dev,s.st_ino)
        if i not in seen:total+=s.st_blocks*512;seen.add(i)
    return total
def plan(roots,out):
    sizes=collections.defaultdict(list);paths=[]
    for root in roots:
        assert root.is_dir() and 'real_aligned' not in str(root) and '.git' not in str(root)
        for p in root.rglob('*'):
            if p.is_symlink() or not p.is_file() or '.git' in p.parts:continue
            s=p.stat()
            if s.st_size>=128*1024:
                paths.append(p);sizes[(s.st_size,s.st_dev,stat.S_IMODE(s.st_mode))].append(p)
    groups=[]
    for z in sizes.values():
        if len(z)<2:continue
        hashes=collections.defaultdict(list)
        for p in z:hashes[sha(p)].append(p)
        for h,ps in hashes.items():
            if len({(p.stat().st_dev,p.stat().st_ino) for p in ps})<2:continue
            ps=sorted(ps,key=lambda p:(len(str(p)),str(p)));canonical=ps[0]
            groups.append({'sha256':h,'bytes':canonical.stat().st_size,'canonical':str(canonical),'duplicates':[str(p) for p in ps[1:] if p.stat().st_ino!=canonical.stat().st_ino],'evidence_reference':'All original filenames retained; original manifests remain valid byte-for-byte','reconstruction':'Hard links; no recomputation required'})
    data={'operation':'SHA256_VERIFIED_HARDLINK_COALESCING','created_utc_unix':time.time(),'roots':[str(r) for r in roots],'minimum_bytes':128*1024,'files_scanned':len(paths),'allocated_bytes_before':blocks(paths),'groups':groups,'paths_scanned':[str(p) for p in paths],'exclusions':['field data','Git tracked files/history','all current client pilot runs','shared caches','unique traces/models/logs']}
    out.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({'groups':len(groups),'duplicates':sum(len(g['duplicates']) for g in groups),'logical_duplicate_bytes':sum(g['bytes']*len(g['duplicates']) for g in groups),'plan':str(out)}))
def execute(p):
    data=json.loads(p.read_text());applied=[]
    for g in data['groups']:
        c=Path(g['canonical']);assert sha(c)==g['sha256']
        for name in g['duplicates']:
            d=Path(name);assert d.stat().st_dev==c.stat().st_dev and sha(d)==g['sha256']
            if d.stat().st_ino==c.stat().st_ino:continue
            tmp=d.with_name(d.name+'.dedup-link-in-progress');assert not tmp.exists()
            os.link(c,tmp);os.replace(tmp,d)
            assert sha(d)==g['sha256'] and d.stat().st_ino==c.stat().st_ino
            applied.append({'path':str(d),'canonical':str(c),'sha256':g['sha256'],'bytes':g['bytes']})
    after=blocks([Path(x) for x in data['paths_scanned']]);result={'operation':data['operation'],'allocated_bytes_before':data['allocated_bytes_before'],'allocated_bytes_after':after,'allocated_bytes_released':data['allocated_bytes_before']-after,'coalesced':applied,'content_deleted':False,'warning':'Hard-linked historical evidence is immutable: future edits must use atomic replacement, never in-place writes.'}
    q=p.with_name(p.stem+'_RESULT.json');q.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='coalesced'}))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--roots',type=Path,nargs='+');a.add_argument('--plan',type=Path,required=True);a.add_argument('--execute',action='store_true');x=a.parse_args()
    if x.execute:execute(x.plan)
    else:plan([r.resolve() for r in x.roots],x.plan)
