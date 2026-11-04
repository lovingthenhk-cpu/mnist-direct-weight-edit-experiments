import json,argparse,time
from pathlib import Path
import numpy as np
import cupy as cp
from nearest_manual import split_data,features
def grad(a):
 x=cp.zeros_like(a);y=cp.zeros_like(a)
 x[...,1:-1]=(a[...,2:]-a[...,:-2])*0.5
 y[...,1:-1,:]=(a[...,2:,:]-a[...,:-2,:])*0.5
 return x,y
def evaluate(path,split='dev',batch=32):
 t=time.time();p=json.loads(Path(path).read_text())
 refs,labels,query,actual,refidx,qidx=split_data(split)
 ref=cp.asarray(features(refs,p).astype('f'))
 q=features(query,p).astype('f')
 norm=(ref*ref).sum(axis=1)[None,:]
 s=p['shortlist_literal'];k=p['number_of_neighbors_literal']
 pred=[];errs=[]
 for st in range(0,len(q),batch):
  a=cp.asarray(q[st:st+batch]);n=len(a)
  d=cp.sum(a*a,axis=1)[:,None]+norm-2*a@ref.T
  idx=cp.argpartition(d,s-1,axis=1)[:,:s];d0=cp.take_along_axis(d,idx,axis=1)
  r=ref[idx].reshape(n,s,28,28);x=a.reshape(n,1,28,28)
  rx,ry=grad(r);xx,xy=grad(x)
  rp=cp.pad(r,((0,0),(0,0),(1,1),(1,1)))
  rpx=cp.pad(rx,((0,0),(0,0),(1,1),(1,1)))
  rpy=cp.pad(ry,((0,0),(0,0),(1,1),(1,1)))
  best=cp.full(r.shape,cp.inf,dtype='f')
  for dy in (-1,0,1):
   for dx in (-1,0,1):
    v=rp[:,:,1+dy:29+dy,1+dx:29+dx]
    vx=rpx[:,:,1+dy:29+dy,1+dx:29+dx]
    vy=rpy[:,:,1+dy:29+dy,1+dx:29+dx]
    dd=(x-v)**2
    dd+=p['image_gradient_x_weight_literal']*(xx-vx)**2
    dd+=p['image_gradient_y_weight_literal']*(xy-vy)**2
    dd+=p['shift_penalty_literal']*(dx*dx+dy*dy)
    best=cp.minimum(best,dd)
  dist=p['global_distance_weight_literal']*d0+p['local_distance_weight_literal']*best.sum((2,3))
  sel=cp.argpartition(dist,k-1,axis=1)[:,:k]
  ds=cp.asnumpy(cp.take_along_axis(dist,sel,axis=1))
  ids=cp.asnumpy(cp.take_along_axis(idx,sel,axis=1))
  ord=np.argsort(ds,axis=1);ds=np.take_along_axis(ds,ord,axis=1);ids=np.take_along_axis(ids,ord,axis=1)
  votes=np.tile(np.array(p['class_bias_literals'],float)[None,:],(n,1))
  for j in range(k):votes[np.arange(n),labels[ids[:,j]]]+=1/(np.sqrt(np.maximum(0,ds[:,j]))+p['distance_epsilon_literal'])
  yy=np.argmax(votes,axis=1)
  pred.extend(yy.tolist())
  for i in range(n):
   if yy[i]!=actual[st+i]:
    errs.append({'query_index':int(qidx[st+i]),'true':int(actual[st+i]),'pred':int(yy[i]),'nearest_ref_indices':refidx[ids[i,:3]].tolist(),'scores':votes[i].round(5).tolist()})
  if st%800==0:print('done',st,'/',len(q),'elapsed',round(time.time()-t,2),flush=True)
 pred=np.array(pred)
 cm=np.zeros((10,10),int)
 for a,b in zip(actual,pred):cm[a,b]+=1
 out={'experiment':p['experiment'],'split':split,'n_reference':len(refs),'samples':len(query),'correct':int(np.sum(pred==actual)),'accuracy':float(np.mean(pred==actual)),'confusion':cm.tolist(),'class_recall':{str(i):float(cm[i,i]/cm[i].sum()) for i in range(10)},'errors':errs[:50],'parameters':p,'time_s':round(time.time()-t,2)}
 (Path(path).parent/('results_'+split+'.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print('ACCURACY',out['accuracy'],'CORRECT',out['correct'],'/',len(query),'SEC',out['time_s'],flush=True)
 print('RECALL',out['class_recall'],flush=True)
 return out
if __name__=='__main__':
 z=argparse.ArgumentParser();z.add_argument('param');z.add_argument('--split',choices=['dev','test'],default='dev');z.add_argument('--batch',type=int,default=32);a=z.parse_args()
 evaluate(a.param,a.split,a.batch)
