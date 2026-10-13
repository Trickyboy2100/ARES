#!/usr/bin/env python3
import argparse,csv,json,collections,hashlib
from pathlib import Path

def read(path): return [json.loads(x) for x in path.read_text().splitlines()]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--run",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    c={r["candidate_id"]:r for r in read(a.run/"candidates.jsonl")}
    t={r["candidate_id"]:r for r in read(a.run/"transitions.jsonl")}
    o={r["candidate_id"]:r for r in read(a.run/"outcomes.jsonl")}
    fields=["scenario_id","candidate_id","grasp_symmetry","center_yaw_rad","ik_branch_index",
      "candidate_legal","geometric_start_set_member","cpu_linear_proxy_clear",
      "center_limit_margin_rad","center_singularity","baseline_predecessor_to_center_cost",
      "full_chain_proxy_cost","failure_codes"]
    with (a.out/"P2_1_CANDIDATE_LEDGER.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for cid,row in c.items():
            joined={**row,**o[cid]}; joined["failure_codes"]="|".join(o[cid]["failure_codes"])
            w.writerow({k:joined.get(k) for k in fields})
    scene_fields=["scenario_id","candidate_slots","legal","start_members","legal_successes",
                  "legal_failures","within_legal_endpoint_sensitive","minimum_full_cost",
                  "maximum_full_cost","heuristic_regret"]
    scene_rows=[]
    for sid in sorted({r["scenario_id"] for r in c.values()}):
        ids=[i for i,r in c.items() if r["scenario_id"]==sid]; legal=[i for i in ids if c[i]["candidate_legal"]]
        valid=[i for i in legal if o[i]["geometric_start_set_member"]]
        costs=[o[i]["full_chain_proxy_cost"] for i in valid if o[i]["full_chain_proxy_cost"] is not None]
        regret=None
        if valid:
            heuristic=min(valid,key=lambda i:o[i]["baseline_predecessor_to_center_cost"])
            regret=o[heuristic]["full_chain_proxy_cost"]-min(costs)
        scene_rows.append({"scenario_id":sid,"candidate_slots":len(ids),"legal":len(legal),
          "start_members":sum(o[i]["geometric_start_set_member"] for i in ids),
          "legal_successes":len(valid),"legal_failures":len(legal)-len(valid),
          "within_legal_endpoint_sensitive":len({o[i]["geometric_start_set_member"] for i in legal})>1,
          "minimum_full_cost":min(costs) if costs else None,"maximum_full_cost":max(costs) if costs else None,
          "heuristic_regret":regret})
    with (a.out/"P2_1_SCENE_SUMMARY.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=scene_fields); w.writeheader(); w.writerows(scene_rows)
    counts=collections.Counter(code for row in o.values() for code in row["failure_codes"])
    with (a.out/"P2_1_FAILURE_COUNTS.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["failure_code","count"]); w.writerows(counts.most_common())
    try:
        import matplotlib.pyplot as plt
        names=[r["scenario_id"].replace("-p21","") for r in scene_rows]
        legal=[r["legal"] for r in scene_rows]; members=[r["start_members"] for r in scene_rows]
        fig,ax=plt.subplots(figsize=(11,4.5)); x=range(len(names)); ax.bar([i-.18 for i in x],legal,.36,label="legal center terminal")
        ax.bar([i+.18 for i in x],members,.36,label="downstream endpoint member")
        ax.set_xticks(list(x),names,rotation=35,ha="right"); ax.set_ylabel("candidate count / 12 fixed slots")
        ax.legend(); fig.tight_layout(); fig.savefig(a.out/"p2_1_candidate_sensitivity.png",dpi=180); plt.close(fig)
    except Exception as exc: (a.out/"PLOT_NOT_GENERATED.txt").write_text(repr(exc)+"\n")
    hashes=[]
    for path in sorted(a.run.iterdir()):
        if path.is_file(): hashes.append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    (a.out/"SOURCE_HASHES.json").write_text(json.dumps(hashes,indent=2)+"\n")
if __name__=="__main__": main()
