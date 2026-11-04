#!/usr/bin/env python3
"""No-gradient reference lookup. Human-written distance formula + class bias.

The reference bank is *verbatim labelled MNIST training examples*, NOT
LLM-authored learned model weights. Code NEVER computes fitted parameters.
"""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import cupy as cp
ROOT=Path(__file__).resolve().parent.parent

def images_labels(split):
    a='train' if split=='train' else 't10k'
    def get(kind,head,shape):
        with gzip.open(ROOT/'data'/(a+'-'+kind+'.gz'),'rb') as f:q=f.read()
        return np.frombuffer(q,dtype=np.uint8,offset=head).copy().reshape(shape)
    return get('images-idx3-ubyte',16,(-1,28,28)),get('labels-idx1-ubyte',8,(-1,))

def split_data(stage):
    train,lbl=images_labels('train')
    perm=np.random.RandomState(20261104).permutation(len(lbl))
    dev_indices=perm[:5000]
    # Original development set that was looked at for previous work is fully excluded
    # from the reference library for an unbiased within-train dev check.
    if stage=='dev':
        rix=np.sort(perm[5000:]);qix=dev_indices
        return train[rix],lbl[rix],train[qix],lbl[qix],rix,qix
    if stage=='test':
        test,tl=images_labels('test')
        return train,lbl,test,tl,np.arange(len(lbl)),np.arange(len(tl))
    raise ValueError(stage)

def features(img,manual):
    mode=manual['preprocessing']
    img=img.astype(np.float32)/255.0
    if mode=='raw_28x28':return img.reshape(-1,784)
    if mode in ('deskew_28x28','deskew_gaussian_28x28'):
        import scipy.ndimage as ndi
        out=np.empty_like(img)
        yy,xx=np.mgrid[:28,:28]
        for n,m in enumerate(img):
            z=m.sum()+1e-8
            x=(m*xx).sum()/z;y=(m*yy).sum()/z
            mu11=((xx-x)*(yy-y)*m).sum()/z
            mu02=((yy-y)*(yy-y)*m).sum()/z
            skew=float(mu11/(mu02+1e-8))
            # Derive output rows from input with deskew row-wise shift.
            # x_input = x_output + skew * (row - center)
            out[n]=ndi.affine_transform(m,[[1.,0.],[skew,1.]],offset=[0.,-skew*13.5],order=1,mode='constant',cval=0.)
        if mode=='deskew_gaussian_28x28':
            out=ndi.gaussian_filter(out,sigma=(0,0.65,0.65))
        return out.reshape(-1,784)
    raise ValueError(mode)

def evaluate_manual(path,stage='dev',batch=128,limit=None,save=True):
    t0=time.time()
    p=json.loads(Path(path).read_text())
    ref,yref,query,yq,refidx,qidx=split_data(stage)
    if limit is not None:query=query[:limit];yq=yq[:limit];qidx=qidx[:limit]
    a=features(ref,p);b=features(query,p)
    assert len(a)==len(yref) and a.shape[1]==b.shape[1]==784
    # Explicit human-written scalar pixel weight (not a fitted vector).
    pix_scale=float(p['distance_weight_literal'])
    a=cp.asarray(a*pix_scale,dtype=cp.float32)
    ar=cp.sum(a*a,axis=1)[None,:]
    class_bias=np.asarray(p['class_bias_literals'],np.float64)
    assert class_bias.shape==(10,)
    k=p['number_of_neighbors_literal']
    predicts=[];all_nbr=[];allscores=[];all_dist=[]
    for start in range(0,len(b),batch):
        cur=cp.asarray(b[start:start+batch]*pix_scale,dtype=cp.float32)
        dd=cp.sum(cur*cur,axis=1)[:,None]+ar-2*(cur@a.T)
        ix=cp.argpartition(dd,k-1,axis=1)[:,:k]
        dst=cp.take_along_axis(dd,ix,axis=1)
        ix=cp.asnumpy(ix);dst=cp.asnumpy(dst)
        order=np.argsort(dst,axis=1)
        ix=np.take_along_axis(ix,order,axis=1)
        dst=np.take_along_axis(dst,order,axis=1)
        idxlabs=yref[ix]
        # inverse-distance weighting manually prescribed, no fit or derivatives.
        score=np.tile(class_bias[None,:],(len(ix),1))
        strength=1./(np.sqrt(np.maximum(dst,0.))+p['distance_epsilon_literal'])
        for j in range(k):
            score[np.arange(len(ix)),idxlabs[:,j]]+=strength[:,j]
        predicts.extend(np.argmax(score,axis=1).tolist())
        all_nbr.extend(refidx[ix].tolist())
        allscores.extend(score.tolist())
        all_dist.extend(dst.tolist())
    predicts=np.asarray(predicts,dtype=np.uint8)
    cm=np.zeros((10,10),dtype=int)
    for truth,pred in zip(yq,predicts):cm[truth,pred]+=1
    wrong=np.flatnonzero(yq!=predicts)
    hardest=sorted(wrong,key=lambda i:(-(max(allscores[i])-allscores[i][int(yq[i])]),i))[:20]
    metrics={'experiment':p['experiment'],'dataset':stage,'n_reference':len(yref),'n_query':len(yq),
      'n_correct':int((yq==predicts).sum()),'accuracy':float(np.mean(yq==predicts)),
      'gpu':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), 'device_memory':'CUDA',
      'seconds':round(time.time()-t0,3),'configuration':p,
      'per_digit_recall':{str(j):round(float(cm[j,j]/max(1,sum(cm[j]))),5) for j in range(10)},
      'confusion_matrix':cm.tolist(),
      'high_confidence_errors':[{'query_index':int(qidx[i]),'true':int(yq[i]),'pred':int(predicts[i]),
         'ref_indices':all_nbr[i][:min(k,5)],'squared_distances':[round(float(v),4) for v in all_dist[i][:min(k,5)]],
         'scores':[round(float(v),5) for v in allscores[i]]} for i in hardest],
      'raw_reference_source':'Official MNIST training images + labels, excluding dev indices in dev mode',
      'ref_sha256':hashlib.sha256(yref.tobytes()).hexdigest(),
      'disclosure':'The reference bank is a data-based nonparametric model. Its examples and labels are NOT LLM-handwritten weights. Only metric coefficients and class biases are LLM-authored; no automatic fitting or gradients.'}
    if save:
        out=Path(path).parent/f'results_{stage}.json'
        out.write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    print('MODEL:',p['experiment'],'stage:',stage,flush=True)
    print('ACC:',int(metrics['n_correct']),'/',metrics['n_query'],'=',round(metrics['accuracy']*100,3),'%',flush=True)
    print('CLASS_RECALL:',metrics['per_digit_recall'],'SECONDS:',metrics['seconds'],flush=True)
    print('WRONG:',len(wrong),'FIRST:',metrics['high_confidence_errors'][:3],flush=True)
    return metrics

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('parameters');q.add_argument('--split',choices=['dev','test'],default='dev');q.add_argument('--limit',type=int,default=None);q.add_argument('--batch',type=int,default=128);args=q.parse_args()
    evaluate_manual(args.parameters,args.split,args.batch,args.limit)
