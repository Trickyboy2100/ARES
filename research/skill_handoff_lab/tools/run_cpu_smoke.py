#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from research.skill_handoff_lab.bench.generate import run
def main():
    p=argparse.ArgumentParser(); p.add_argument("--ares",type=Path,required=True); p.add_argument("--ares-r",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); print(json.dumps(run(a.ares.resolve(),a.ares_r.resolve(),a.output.resolve()),indent=2))
if __name__=="__main__": main()
