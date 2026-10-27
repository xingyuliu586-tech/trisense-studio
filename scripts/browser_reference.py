"""Refresh numeric fixtures for the browser inference parity check."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from trisense.engine import Engine
from trisense.features import read_features


def main():
    engine=Engine();sample=read_features(ROOT/'examples/synthetic_sample.npz')
    records={}
    for predictor,mode,rate in [('q3','none',0),('q3','vision',1),('ensemble','none',0),('ensemble','audio+vision',.3)]:
        r=engine.analyze(sample,predictor,mode,rate,'middle')
        records[f'{predictor}:{mode}:{rate}']={key:r[key] for key in ('prediction','attribution','intervention')}
    (ROOT/'tests/browser_reference.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Refreshed 4 Python inference references.')


if __name__=='__main__':main()
