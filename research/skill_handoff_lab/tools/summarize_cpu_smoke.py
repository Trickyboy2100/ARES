#!/usr/bin/env python3
"""Create derived tables/plot/hash manifest from an immutable smoke directory."""
import argparse,csv,hashlib,json
from pathlib import Path

def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
def main():
    p=argparse.ArgumentParser(); p.add_argument("--run",type=Path,required=True); a=p.parse_args(); root=a.run
    cs=rows(root/"candidates.jsonl"); oo=rows(root/"outcomes.jsonl")
    byc={x["candidate_id"]:x for x in cs}; scenes=sorted({x["scenario_id"] for x in oo})
    with (root/"baseline_comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["scenario_id","legal_candidates","heuristic_selected_candidate","heuristic_edge_cost","heuristic_geometry_success","equal_budget_oracle_success","feasible_candidates"]); w.writeheader()
        for sid in scenes:
            out=[x for x in oo if x["scenario_id"]==sid]; legal=[x for x in out if byc[x["candidate_id"]]["candidate_legal"]]
            pick=min(legal,key=lambda x:x["edge_cost_baseline_score"]) if legal else None
            w.writerow({"scenario_id":sid,"legal_candidates":len(legal),"heuristic_selected_candidate":pick["candidate_id"] if pick else "","heuristic_edge_cost":pick["edge_cost_baseline_score"] if pick else "","heuristic_geometry_success":int(pick["geometry_successor_start_feasible"]) if pick else 0,"equal_budget_oracle_success":int(any(x["geometry_successor_start_feasible"] for x in legal)),"feasible_candidates":sum(x["geometry_successor_start_feasible"] for x in legal)})
    width,height=900,390; bars=[]
    for i,sid in enumerate(scenes):
        out=[x for x in oo if x["scenario_id"]==sid]; good=sum(x["geometry_successor_start_feasible"] for x in out)
        x=75+i*135; h=55*good; bars.append(f'<rect x="{x}" y="{300-h}" width="70" height="{h}" fill="#0072B2"/><text x="{x+35}" y="325" text-anchor="middle" font-size="16">{sid}</text><text x="{x+35}" y="{285-h}" text-anchor="middle" font-size="18">{good}/4</text>')
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><text x="450" y="35" text-anchor="middle" font-family="sans-serif" font-size="22">CPU geometric successor-start feasibility</text><line x1="45" y1="300" x2="870" y2="300" stroke="#333"/><line x1="45" y1="80" x2="45" y2="300" stroke="#333"/><text x="20" y="195" transform="rotate(-90 20 195)" text-anchor="middle" font-family="sans-serif" font-size="16">feasible candidates (of 4)</text>{''.join(bars)}<text x="450" y="370" text-anchor="middle" font-family="sans-serif" font-size="14">CPU geometry only; cuRobo / contact physics / real execution not tested</text></svg>'''
    (root/"candidate_sensitivity.svg").write_text(svg)
    sources=[x for x in root.iterdir() if x.is_file() and x.name!="SHA256SUMS"]
    (root/"SHA256SUMS").write_text("".join(hashlib.sha256(x.read_bytes()).hexdigest()+"  "+x.name+"\n" for x in sorted(sources)))
if __name__=="__main__": main()
