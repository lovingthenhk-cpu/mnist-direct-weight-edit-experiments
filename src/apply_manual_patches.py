#!/usr/bin/env python3
"""Explicit per-element manual weight modifications; no optimizer."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
src=ROOT/'experiments/exp-004-targeted-bias-and-confusion/parameters.json'
out=ROOT/'experiments/exp-005-pixelwise-weight-edits'
p=json.loads(src.read_text())
p['experiment']='exp-005-pixelwise-weight-edits'
p['authoring']='Explicit list of 30 manually chosen replacement weight values. No training, fitting, gradient, grid search, or optimizer.'
# (class, row, column, NEW ABSOLUTE weight) - ALL CHOSEN BY LLM after inspection
# of printed class-mean difference maps. Zero values are not computed from data.
PATCHES=[
 # 4: horizontal center and right-side stem vs loop 9, and absent lower loop 6.
 (4,3,7,-0.70),(4,4,7,-0.70),(4,3,8,-0.55),(4,4,6,-0.55),
 (4,4,9,1.40),(4,5,9,1.45),(4,7,4,1.40),(4,8,4,1.00),
 (4,7,9,1.40),(4,8,9,1.10),
 (4,9,5,-0.80),(4,10,5,-0.80),(4,10,6,-0.75),
 # 9: strong top loop, weak lower left (vs 8), weak left crossbar (vs 4).
 (9,3,7,1.45),(9,4,7,1.60),(9,3,8,1.35),(9,4,6,1.45),
 (9,7,4,-0.55),(9,8,4,-0.65),
 (9,9,5,-0.75),(9,10,4,-0.65),(9,10,5,-0.75),(9,11,5,-0.75),
 # 5: left-middle upstroke and right-middle empty pocket separate it from 3.
 (5,5,4,1.10),(5,5,5,1.50),(5,6,4,1.25),
 (5,5,8,-0.65),(5,6,7,-0.55),(5,6,8,-0.60),
 # 7: short top bar, no middle left loop; stronger lower diagonal.
 (7,7,5,-0.55),(7,7,6,-0.55),(7,11,4,0.70),(7,12,4,0.65)
]
assert len(PATCHES)==len(set((d,r,c) for d,r,c,_ in PATCHES))
log=[]
for d,r,c,new in PATCHES:
 old=p['weights'][d][r*14+c]
 p['weights'][d][r*14+c]=new
 log.append({'digit':d,'row_0based':r,'col_0based':c,'old':old,'new':new,'manual_rationale':'Contrastive stroke edit as described in decision_before_eval.md'})
(out/'parameters.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n')
(out/'literal_edits.json').write_text(json.dumps(log,indent=2,ensure_ascii=False)+'\n')
print('Manually changed',len(PATCHES),'distinct weights, all biases unchanged')
