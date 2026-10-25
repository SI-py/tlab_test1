"""Full-denominator validation and LightGBM learner helpers."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
from .v2_experiment import read_snapshot

class FullRankingMetric:
    """Top-k metrics retain missed positives; equal weight for validation weeks."""
    def __init__(self, frame, truths, cutoffs, k=10):
        npos,folds={},{}
        for fold,(truth,cutoff) in enumerate(zip(truths,cutoffs)):
            day=int(pd.Timestamp(cutoff).value//86400000000000)
            for u,n in truth.groupby('user_id').size().items():
                q=int(u)+day*2_000_000
                npos[q]=n;folds[q]=fold
        if not npos:
            raise ValueError('No positive validation queries')
        qids=np.asarray(sorted(npos),dtype=np.int64)
        inv=pd.Index(qids).get_indexer(frame.query_id)
        if np.any(inv<0):
            raise ValueError('Candidate query absent from declared truth')
        self.inv = inv
        self.sizes = np.bincount(inv,minlength=len(qids))
        first = np.r_[0,np.cumsum(self.sizes)[:-1]]
        assert np.all(np.diff(inv)>=0), 'Queries must be contiguous and sorted'
        self.position = np.arange(len(frame))-np.repeat(first,self.sizes)
        self.shape=(len(qids),max(k,int(self.sizes.max())))
        self.labels=np.zeros(self.shape,np.float64)
        self.labels[inv,self.position]=frame.label.to_numpy()
        self.discount=1/np.log2(np.arange(1,k+1)+1)
        self.k=k
        self.npos=np.asarray([npos[int(q)] for q in qids])
        self.folds=np.asarray([folds[int(q)] for q in qids])
        cumulative=np.cumsum(self.discount)
        self.idcg=cumulative[np.minimum(self.npos,k)-1]

    def per_query(self,scores):
        matrix=np.full(self.shape,-np.inf,np.float64)
        matrix[self.inv,self.position]=scores
        # Stable sort preserves item-ID order even at a tied top-k boundary.
        top=np.argsort(-matrix,axis=1,kind='stable')[:,:self.k]
        hits=np.take_along_axis(self.labels,top,axis=1)
        ndcg=(hits*self.discount).sum(axis=1)/self.idcg
        recall=hits.sum(axis=1)/self.npos
        return ndcg,recall

    def metrics(self,scores):
        ndcg,recall=self.per_query(scores)
        weeks=[]
        for fold in sorted(set(self.folds)):
            mask=self.folds==fold
            weeks.append({'ndcg@10':float(ndcg[mask].mean()),'recall@10':float(recall[mask].mean()),'users':int(mask.sum())})
        return {'ndcg@10':float(np.mean([w['ndcg@10'] for w in weeks])),
                'recall@10':float(np.mean([w['recall@10'] for w in weeks])),'weeks':weeks}

    def lgb_callback(self):
        def callback(y_true,y_pred):
            return 'full_ndcg10',self.metrics(y_pred)['ndcg@10'],True
        return callback


def encode_categories(train,val,cats):
    levels={}
    for col in cats:
        train[col]=train[col].astype('category')
        levels[col]=train[col].cat.categories.tolist()
        if val is not None:
            val[col]=pd.Categorical(val[col].where(val[col].isin(levels[col])),categories=levels[col])
    return levels


def fit_ranker(train,columns,cats,cfg,params,seed,val=None,metric=None,trees=None):
    t=time.perf_counter()
    model=lgb.LGBMRanker(objective='lambdarank',metric='None',
        n_estimators=int(trees or cfg['max_trees']),learning_rate=cfg['learning_rate'],
        num_leaves=params['num_leaves'],reg_lambda=params['reg_lambda'],
        min_child_samples=params['min_child_samples'],reg_alpha=.1,
        colsample_bytree=.85,subsample=.85,subsample_freq=1,
        lambdarank_truncation_level=15,random_state=seed,n_jobs=cfg['num_threads'],
        verbosity=-1,deterministic=True,force_col_wise=True)
    group=train.groupby('query_id',sort=False).size().to_numpy()
    kwargs={}
    if val is not None:
        kwargs={'eval_set':[(val[columns],val.label)],'eval_group':[val.groupby('query_id',sort=False).size().to_numpy()],
                'eval_metric':metric.lgb_callback(),'callbacks':[lgb.early_stopping(100,first_metric_only=True,verbose=False)]}
    model.fit(train[columns],train.label,group=group,categorical_feature=[c for c in cats if c in columns],**kwargs)
    return model,time.perf_counter()-t


def _development(args,cfg):
    trains=[];vals=[];truths=[];stats=None
    for cutoff in cfg['protocol']['train_cutoffs']:
        f,_,stats=read_snapshot(args.output/'snapshots'/f'train_{cutoff}')
        trains.append(f)
    train=pd.concat(trains,ignore_index=True).sort_values(['query_id','item_id']).reset_index(drop=True)
    del trains
    for cutoff in cfg['protocol']['validation_cutoffs']:
        f,t,stats=read_snapshot(args.output/'snapshots'/f'validation_{cutoff}')
        vals.append(f);truths.append(t)
    val=pd.concat(vals,ignore_index=True).sort_values(['query_id','item_id']).reset_index(drop=True)
    del vals
    return train,val,truths,stats

