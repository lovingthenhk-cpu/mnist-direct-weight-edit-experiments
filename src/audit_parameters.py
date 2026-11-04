#!/usr/bin/env python3
"""Deterministic read-only parameter diff audit. Never calculates weight updates."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
paths=sorted((ROOT/'experiments').glob('exp-*/parameters.json'))
rows=[]
for i,p in enumerate(paths):
 d=json.loads(p.read_text())
 w=d['weights'];b=d['biases']
 assert len(w)==10 and all(len(x)==196 for x in w) and len(b)==10
 now=sum(x*x for row in w for x in row)
 rec={'experiment':d['experiment'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
      'weight_count':sum(map(len,w)),'bias_count':len(b),'weight_l2_squared':round(now,5)}
 if i:
  old=json.loads(paths[i-1].read_text());rec['weight_entries_changed_from_previous']=sum(x!=y for r1,r2 in zip(old['weights'],w) for x,y in zip(r1,r2));rec['bias_entries_changed_from_previous']=sum(x!=y for x,y in zip(old['biases'],b))
 else:rec['weight_entries_changed_from_previous']=None;rec['bias_entries_changed_from_previous']=None
 rows.append(rec)
 print(rec['experiment'],'W changed:',rec['weight_entries_changed_from_previous'],'bias changed:',rec['bias_entries_changed_from_previous'])
(ROOT/'validation/parameter_audit.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
