"""Compare the software with the original functions without importing its dependencies."""
import argparse
import ast
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from trisense.engine import Engine
from trisense.features import read_features, mask_segment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--research-root',type=Path,required=True)
    args=parser.parse_args()
    source=(args.research_root/'q2_pipeline.py').read_text(encoding='utf-8-sig')
    tree=ast.parse(source)
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('normalize','batch_arrays','MaskedFusionNet')]
    namespace={'np':np,'torch':torch,'nn':nn,'F':F}
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<original-research-functions>','exec'),namespace)
    engine=Engine();sample=read_features(ROOT/'examples/synthetic_sample.npz')
    norm={name:{key:np.asarray(value,np.float32) for key,value in values.items()} for name,values in engine.normalization.items()}
    diffs=[]
    with torch.inference_mode():
        for mode,rate in [('none',0),('audio',.3),('vision',1),('audio+vision',1)]:
            altered,_=mask_segment(sample,mode,rate,'middle')
            part={k:v[None] for k,v in altered.arrays.items()}
            old_inputs=namespace['batch_arrays'](part,np.array([0]),norm)
            new_inputs=engine.tensors(altered)
            assert all(torch.equal(a,b) for a,b in zip(old_inputs,new_inputs))
            for name in ('q2_baseline','q2_robust','q3_predictor'):
                ref=namespace['MaskedFusionNet']().eval()
                ref.load_state_dict(torch.load(ROOT/'models'/f'{name}.pt',weights_only=True,map_location='cpu'))
                old=ref(*old_inputs);new=engine._model(name)(*new_inputs)
                delta=max(float((a-b).abs().max()) for a,b in zip(old,new))
                assert delta < 1e-6
                diffs.append({'checkpoint':name,'mode':mode,'max_absolute_difference':delta})
    report={'scope':'Synthetic compatibility checks only; not a re-evaluation of dataset performance.', 'cases':diffs,'max_absolute_difference':max(x['max_absolute_difference'] for x in diffs)}
    (ROOT/'docs').mkdir(exist_ok=True)
    (ROOT/'docs/parity_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
