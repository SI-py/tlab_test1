"""External weekly replication on full sampled-user Yambda histories.

Run 'design' first and inspect results/yambda_design.json before develop/final.
The design stage reads only user IDs, configuration and prior-cohort metadata.
"""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from recsys.yambda import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['design','develop','final','all'],default='design')
    p.add_argument('--config',type=Path,default=ROOT/'configs/yambda.json')
    p.add_argument('--data',type=Path,default=ROOT/'data/raw')
    p.add_argument('--task2-manifest',type=Path,default=ROOT.parent/'artifacts/task2/yambda/data_manifest.json')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/yambda')
    p.add_argument('--results',type=Path,default=ROOT/'artifacts/yambda/results.json')
    p.add_argument('--design',type=Path,default=ROOT/'results/yambda_design.json')
    run(p.parse_args())
