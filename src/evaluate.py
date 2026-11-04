#!/usr/bin/env python3
"""Inference/evaluation only: no gradients, no parameter updates."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
import cupy as cp
ROOT=Path(__file__).resolve().parent.parent

def load(split):
 def _read(kind):
  stem=('train' if split=='train' else 't10k')+'-'+kind
  with gzip.open(ROOT/'data'/(stem+'.gz'),'rb') as f:data=f.read()
  if 'images' in kind: return np.frombuffer(data,dtype=np.uint8,offset=16).reshape(-1,28,28)
  return np.frombuffer(data,dtype=np.uint8,offset=8)
 return _read('images-idx3-ubyte'),_read('labels-idx1-ubyte')

def prep(x):
 # Fixed non-parametric preprocessing; never modified by any label.
 return x.reshape(-1,14,2,14,2).mean((2,4)).reshape(-1,196).astype(np.float32)/255.

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('params')
 ap.add_argument('--split',choices=['dev','test'],default='dev')
 ap.add_argument('--out',default=None)
 args=ap.parse_args()
 p=Path(args.params); data=json.loads(p.read_text())
 W=np.array(data['weights'],dtype=np.float32); B=np.array(data['biases'],dtype=np.float32)
 assert W.shape==(10,196) and B.shape==(10,)
 x,y=load('train' if args.split=='dev' else 'test')
 if args.split=='dev':
  idx=np.random.RandomState(20261104).permutation(len(y))[:5000]
  x=x[idx]; y=y[idx]
 # CUDA inference, no autograd or differentiable optimization is involved.
 X=cp.asarray(prep(x)); wg=cp.asarray(W); bg=cp.asarray(B)
 scores=cp.asnumpy(X@wg.T+bg)
 pred=np.argmax(scores,axis=1)
 cm=np.zeros((10,10),dtype=np.int32)
 for true,got in zip(y,pred):cm[true,got]+=1
 wrong=np.where(pred!=y)[0]
 margin=np.sort(scores,axis=1)[:,-1]-np.sort(scores,axis=1)[:,-2]
 # Select low-margin and high-confidence errors, not derivatives.
 picks=sorted(wrong,key=lambda i: (-margin[i],i))[:12]
 results={'experiment':data['experiment'],'split':args.split,'samples':len(y),'correct':int(np.sum(pred==y)),
 'accuracy':float(np.mean(pred==y)),'per_digit_accuracy':{str(j):round(float(cm[j,j]/max(1,sum(cm[j]))),4) for j in range(10)},
 'confusion_matrix':cm.tolist(), 'wrong_by_predicted':{str(j):int(sum((pred==j)&(y!=j))) for j in range(10)},
 'wrong_examples':[{'sample_index':int(i),'truth':int(y[i]),'prediction':int(pred[i]),
 'scores':[round(float(v),3) for v in scores[i]],'margin':round(float(margin[i]),3),
 'pixels_14x14':[''.join('█' if z>.5 else '+' if z>.15 else ' ' for z in row) for row in prep(x[i:i+1]).reshape(14,14)]} for i in picks],
 'gpu_name':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), 'weight_min':float(W.min()),'weight_max':float(W.max()),
 'biases':B.tolist()}
 out=Path(args.out) if args.out else p.parent/f'results_{args.split}.json'
 out.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
 print('split:',args.split, 'ACC:',round(results['accuracy'],5),'correct:',results['correct'],'/',len(y))
 print('per-digit:',results['per_digit_accuracy'])
 print('confusion rows=true cols=pred:')
 for c,row in enumerate(cm):print(c,' '.join(f'{n:4d}' for n in row))
 print('high-confidence mistakes:')
 for w in results['wrong_examples'][:5]: print('actual',w['truth'],'got',w['prediction'],'score diff',w['margin'])
 print('wrote:',out)
if __name__=='__main__':main()
