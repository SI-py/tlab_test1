"""Bounded manual/importance/random feature replication on relative-time Yambda.

No KION duration rule, fabricated metadata or retrospective pair deduplication.
Raw listening histories remain complete; only collaborative/catalog matrices
restrict items. Future relevance is unique recommendation-driven listened items.
"""
from __future__ import annotations
import gc
import json
import os
from pathlib import Path
import time
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy import sparse
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

DAY=86400

def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def design(args,cfg):
    if args.design.exists():
        plan=read(args.design)
        if plan['config']!=cfg:raise ValueError('Existing design has another configuration; use a fresh design/output')
        return plan
    ids=pq.read_table(args.data/'yambda50m_multi.parquet',columns=['uid']).column(0).to_numpy()
    prior=set(read(args.task2_manifest)['cohort'])
    eligible=np.array(sorted(set(map(int,ids))-prior),dtype=np.int64)
    wanted=sum(cfg['users'].values())
    if len(eligible)<wanted:raise ValueError('Insufficient users outside Task2 cohort')
    chosen=np.random.default_rng(cfg['seed']).permutation(eligible)[:wanted]
    cohorts={};start=0
    for role,count in cfg['users'].items():
        cohorts[role]=sorted(map(int,chosen[start:start+count]));start+=count
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'config':cfg,'cohorts':cohorts,
      'excluded_task2_users':sorted(prior),'eligible_users':len(eligible),
      'population':'Uniform users without future-activity conditioning; all their listen histories retained.',
      'time':'Relative seconds; day=t/86400. These are not Unix/calendar dates.',
      'target':'Unique future listened tracks with is_organic==0; repeated previously listened tracks eligible.',
      'target_scope':'All recorded recommendation-driven listen events, including zero played ratio. Observed interaction, not satisfaction; not the official Yambda Listen+ threshold (>50 percent) or official benchmark.',
      'history':'All past listens, including organic and outside-catalog tracks. Artist=smallest associated artist ID; missing has no affinity.',
      'catalog':'Top10000 tracks by all-sampled-user listen counts strictly before first train cutoff, ties raw item ID; frozen thereafter.',
      'candidate_policy':'Common pop500/BM25-ItemKNN200/ALS100 union; no target injection; no seen exclusion.',
      'collaborative_representation':'Binary past user-track matrix inside the frozen catalog; repeated listening counts remain in manual/automatic features. Popularity counts all listens in the past30days. ALS32/10iterations/alpha10/reg0.5/seed17, CF and BLAS one thread.',
      'selection':'Shared manual/all-window fits; train-fitted all-window importance order; four prefixes versus four independent uniform random subsets of sizes1/2/3/5. Fixed d7/L2=15/seed17/2000/staged50.',
      'final':'Three fixed arms, three seeds each; no final model/subset selection. Three final weeks evaluated once after develop.',
      'primary_contrast':'Importance-selected MINUS manual; all future-positive users with common cold fallback; equal-three-week user-macro NDCG@10, three seed models per arm. Random comparison, Recall and individual weeks secondary/descriptive. No post-hoc procedure switch; no non-inferiority margin.',
      'update_policy':'Global CF and prefix features update before each final week using only its past; GBDT feature subsets, tree counts and learned weights remain fixed across all three final weeks.',
      'training_sampling':'Per query uniformly keep at most20 retrieved positives, then min(3*n_kept_positive,25,available) uniform true-zero-label negatives, without replacement. Other true positives are excluded from fit, never relabeled negative. Same pool for all arms.',
      'bootstrap':'Primary point estimate is equal mean of three week-level user-macro metrics. Resample the union of users active in at least one final week once, apply multiplicities to every week, compute weighted user macro within each week, then equal mean of three weeks. All-active main; warm-active union separately; no-positive queries undefined, never artificial zero. Users without positives in all three weeks are outside the conditional ranking estimand.',
      'limitations':['Yambda is an external conceptual replication, not exact feature transfer from KION.',
        'Final weeks precede Task2 evaluation periods, and Task2 users are excluded; the raw dataset was nevertheless known before this experiment.',
        'Eligible population is conditioned on exclusion of the earlier Task2 cohort, whose own eligibility was based on later-prefix activity; uniform sampling refers only to the remaining8000 users.',
        'Static artist mapping historical availability is unknown; fixed-catalog and no-positive conditioning explicitly limit inference.',
        'One random-search RNG seed and two development weeks do not estimate search/training/period uncertainty. Bootstrap intervals descriptive, no multiplicity correction.']}
    save(args.design,plan);return plan

class Histories:
    def __init__(self,data,plan):
        self.cfg=plan['config'];self.users=np.array(sorted(sum(plan['cohorts'].values(),[])),np.int64)
        lookup={int(u):j for j,u in enumerate(self.users)}
        chunks={key:[] for key in ['u','i','t','organic']}
        columns=['uid','timestamp','item_id','event_type','is_organic']
        for batch in pq.ParquetFile(data/'yambda50m_multi.parquet').iter_batches(
                batch_size=self.cfg['parquet_batch_users'],columns=columns):
            for row,uid in enumerate(batch.column(0).to_numpy()):
                if int(uid) not in lookup:continue
                arrays={name:batch.column(j)[row].values for j,name in enumerate(columns[1:],1)}
                keep=pc.equal(arrays['event_type'].cast(pa.string()),'listen').to_numpy(zero_copy_only=False)
                chunks['u'].append(np.full(int(keep.sum()),lookup[int(uid)],np.int16))
                for name,source,dtype in [('i','item_id',np.int32),('t','timestamp',np.uint32),('organic','is_organic',np.uint8)]:
                    chunks[name].append(arrays[source].to_numpy(zero_copy_only=False)[keep].astype(dtype,copy=False))
        for key,values in chunks.items():setattr(self,key,np.concatenate(values))
        del chunks;gc.collect()
        if np.any(self.u[1:]<self.u[:-1]):raise ValueError('Official sequential source must remain ordered by uid')
        self.indptr=np.r_[0,np.cumsum(np.bincount(self.u,minlength=len(self.users)))]
        # Compact canonical artist lookup; do not create a millions-entry Python dict.
        mapping=pq.read_table(data/'yambda_artist_item_mapping.parquet',columns=['item_id','artist_id'])
        item=mapping.column('item_id').to_numpy();artist=mapping.column('artist_id').to_numpy()
        order=np.lexsort((artist,item));si=item[order];first=np.r_[True,si[1:]!=si[:-1]]
        self.mapping_items=si[first].copy();self.mapping_artists=artist[order[first]].copy()
        del mapping,item,artist,order,si,first;gc.collect()
        self.a=self.artist(self.i)
        self.artist_ids=np.unique(self.a[self.a>=0]);self.apos=np.searchsorted(self.artist_ids,self.a)
        prefix=self.t<int(min(self.cfg['train_days'])*DAY)
        tracks,counts=np.unique(self.i[prefix],return_counts=True)
        chosen=np.lexsort((tracks,-counts))[:self.cfg['catalog_size']]
        self.items=np.sort(tracks[chosen]);self.ipos=pd.Index(self.items).get_indexer(self.i).astype(np.int32)
        self.user_index=pd.Index(self.users);self.item_index=pd.Index(self.items)
    def artist(self,items):
        items=np.asarray(items);positions=np.searchsorted(self.mapping_items,items)
        valid=(positions<len(self.mapping_items))
        safe=np.minimum(positions,len(self.mapping_items)-1)
        valid &= self.mapping_items[safe]==items
        return np.where(valid,self.mapping_artists[safe].astype(np.int32),-1).astype(np.int32)
    def labels(self,role,day,plan):
        positions=self.user_index.get_indexer(plan['cohorts'][role])
        mask=(self.t>=day*DAY)&(self.t<(day+self.cfg['horizon_days'])*DAY)&(self.organic==0)&np.isin(self.u,positions)
        pairs=pd.DataFrame({'user_id':self.users[self.u[mask]],'item_id':self.i[mask]}).drop_duplicates()
        return pairs.sort_values(['user_id','item_id']).reset_index(drop=True)

class Metric:
    def __init__(self,frame,truths,days):
        self.frame=frame;self.records=[]
        for truth,day in zip(truths,days):
            for uid,group in truth.groupby('user_id',sort=True):
                self.records.append((int(day*2000000+uid),set(map(int,group.item_id)),int(uid),day))
        self.groups={int(q):np.asarray(ix) for q,ix in frame.groupby('query_id',sort=False).indices.items()}
    def rows(self,scores):
        output=[];items=self.frame.item_id.to_numpy();scores=np.asarray(scores).copy()
        if 'user_cold' in self.frame:
            cold=self.frame.user_cold.to_numpy()>0
            scores[cold]=-self.frame.loc[cold,'pop_rank'].to_numpy()
        for q,truth,uid,day in self.records:
            ix=self.groups.get(q,np.empty(0,int));order=np.lexsort((items[ix],-scores[ix]))[:10];selected=items[ix[order]]
            hits=np.array([int(i) in truth for i in selected],float)
            discount=1/np.log2(np.arange(1,len(hits)+1)+1)
            idcg=(1/np.log2(np.arange(1,min(10,len(truth))+1)+1)).sum()
            captured=sum(int(i) in truth for i in items[ix])
            output.append({'user_id':uid,'day':day,'ndcg':float((hits*discount).sum()/idcg),
              'recall':float(hits.sum()/len(truth)),'candidate_recall':captured/len(truth),'positives':len(truth)})
        return pd.DataFrame(output)
    def metrics(self,scores):
        rows=self.rows(scores);weeks=[]
        for day,part in rows.groupby('day',sort=True):
            weeks.append({'day':int(day),'users':len(part),'ndcg@10':float(part.ndcg.mean()),'recall@10':float(part.recall.mean())})
        return {'ndcg@10':float(np.mean([w['ndcg@10'] for w in weeks])),
                'recall@10':float(np.mean([w['recall@10'] for w in weeks])),'weeks':weeks}

def training_sample(frame,seed,cfg):
    rng=np.random.default_rng(seed);selected=[]
    for _,group in frame.groupby('query_id',sort=True):
        positives=group.index[group.label.eq(1)].to_numpy()
        zeros=group.index[group.label.eq(0)].to_numpy()
        if not len(positives):continue
        keep=rng.choice(positives,min(len(positives),cfg['training_positive_cap']),replace=False)
        negative=rng.choice(zeros,min(3*len(keep),cfg['training_negative_cap'],len(zeros)),replace=False)
        selected.extend(keep);selected.extend(negative)
    return frame.loc[selected].sort_values(['query_id','item_id']).reset_index(drop=True)

class Prefix:
    def __init__(self,h,day,cfg):
        from implicit.nearest_neighbours import BM25Recommender
        from implicit.als import AlternatingLeastSquares
        started=time.perf_counter();self.h=h;self.day=day;self.cfg=cfg
        self.past=h.t<day*DAY;inside=self.past&(h.ipos>=0)
        self.x=sparse.csr_matrix((np.ones(int(inside.sum()),np.float32),(h.u[inside],h.ipos[inside])),shape=(len(h.users),len(h.items)))
        self.x.data[:]=1;self.x.sort_indices()
        recent=inside&(h.t>=(day-30)*DAY)
        self.pop=np.bincount(h.ipos[recent],minlength=len(h.items)).astype(np.float32)
        with threadpool_limits(limits=1):
            self.bm25=BM25Recommender(K=200,K1=1.2,B=.8,num_threads=1)
            self.bm25.fit(self.x,show_progress=False)
            self.als=AlternatingLeastSquares(factors=cfg['als_factors'],regularization=cfg['als_regularization'],
              alpha=cfg['als_alpha'],iterations=cfg['als_iterations'],num_threads=1,random_state=17)
            self.als.fit(self.x,show_progress=False)
        self.fit_seconds=time.perf_counter()-started
    def candidates(self,users):
        h=self.h;rows=[];batch=self.cfg['candidate_batch_users']
        for start in range(0,len(users),batch):
            query=np.asarray(users[start:start+batch]);up=h.user_index.get_indexer(query)
            with threadpool_limits(limits=1):
                scores={'pop':np.broadcast_to(self.pop,(len(up),len(h.items))),
                  'bm25':(self.x[up]@self.bm25.similarity).toarray(),
                  'als':self.als.user_factors[up]@self.als.item_factors.T}
            orders={name:np.argsort(-score,axis=1,kind='stable') for name,score in scores.items()}
            ranks={}
            for name,order in orders.items():
                rank=np.empty_like(order);np.put_along_axis(rank,order,np.arange(len(h.items))[None,:]+1,axis=1);ranks[name]=rank
            for j,uid in enumerate(query):
                chosen=np.unique(np.concatenate([orders[name][j,:budget] for name,budget in self.cfg['budgets'].items()]))
                part=pd.DataFrame({'user_id':int(uid),'item_id':h.items[chosen]})
                for name,score in scores.items():
                    part[name+'_score']=score[j,chosen].astype(np.float32);part[name+'_rank']=ranks[name][j,chosen].astype(np.float32)
                rows.append(part)
        return pd.concat(rows,ignore_index=True).sort_values(['user_id','item_id']).reset_index(drop=True)
    def features(self,pairs):
        h=self.h;u=h.user_index.get_indexer(pairs.user_id);i=h.item_index.get_indexer(pairs.item_id)
        artist=h.artist(pairs.item_id);ai=np.searchsorted(h.artist_ids,artist)
        frame=pairs.copy();numeric={};groups={'manual':[c for c in pairs if c.endswith('_score') or c.endswith('_rank')]}
        frame['item_identity']=pairs.item_id.astype(str);frame['artist_identity']=np.where(artist>=0,artist.astype(str),'missing')
        cats=['item_identity','artist_identity'];groups['manual']+=cats
        alluser=np.bincount(h.u[self.past],minlength=len(h.users)).astype(np.float32)
        allitem=np.bincount(h.ipos[self.past&(h.ipos>=0)],minlength=len(h.items)).astype(np.float32)
        last=np.zeros(len(h.users),np.float64);np.maximum.at(last,h.u[self.past],h.t[self.past])
        first=np.full(len(h.users),np.inf);np.minimum.at(first,h.u[self.past],h.t[self.past])
        base={'user_recency':np.where(alluser[u]>0,self.day-last[u]/DAY,999),
          'user_history_span':np.where(alluser[u]>0,(last[u]-first[u])/DAY,0),'user_cold':(alluser[u]==0).astype(np.float32)}
        inside=self.past&(h.ipos>=0);ilast=np.zeros(len(h.items),np.float64)
        np.maximum.at(ilast,h.ipos[inside],h.t[inside]);base['item_recency']=np.where(allitem[i]>0,self.day-ilast[i]/DAY,999)
        pairgap=np.full(len(pairs),999,np.float32);lasttrack=np.full(len(pairs),'missing',object)
        distinct=np.zeros(len(pairs),np.float32)
        for position,ix in pd.DataFrame({'u':u}).groupby('u',sort=False).indices.items():
            begin,end=h.indptr[position:position+2];times=h.t[begin:end];before=times<self.day*DAY
            tracks=h.i[begin:end][before];times=times[before];local=h.ipos[begin:end][before]
            if not len(times):continue
            distinct[ix]=len(np.unique(tracks));lasttrack[ix]=str(int(tracks[times==times.max()].min()))
            latest=np.full(len(h.items),-np.inf);valid=local>=0;np.maximum.at(latest,local[valid],times[valid])
            stamps=latest[i[ix]];pairgap[ix]=np.where(np.isfinite(stamps),self.day-stamps/DAY,999)
        base['pair_recency']=pairgap;base['user_distinct_tracks']=distinct
        frame['user_last_item']=lasttrack;cats.append('user_last_item');groups['manual'].append('user_last_item')
        for name,values in base.items():numeric[name]=np.asarray(values,np.float32);groups['manual'].append(name)
        allpair=None
        for window in [0]+self.cfg['manual_windows'][:-1]+self.cfg['automatic_windows']:
            mask=self.past if not window else self.past&(h.t>=(self.day-window)*DAY)
            inside=mask&(h.ipos>=0)
            counts=sparse.csr_matrix((np.ones(int(inside.sum()),np.float32),(h.u[inside],h.ipos[inside])),shape=self.x.shape)
            uv=np.bincount(h.u[mask],minlength=len(h.users)).astype(np.float32)
            iv=np.bincount(h.ipos[inside],minlength=len(h.items)).astype(np.float32)
            pv=np.asarray(counts[u,i]).ravel();allpair=pv if window==0 else allpair
            valid=mask&(h.a>=0)
            profile=sparse.csr_matrix((np.ones(int(valid.sum()),np.float32),(h.u[valid],h.apos[valid])),shape=(len(h.users),len(h.artist_ids)))
            affinity=np.zeros(len(pairs),np.float32);known=artist>=0
            affinity[known]=np.asarray(profile[u[known],ai[known]]).ravel()
            values={'user_count':uv[u],'item_count':iv[i],'pair_count':pv,
              'user_share':uv[u]/np.maximum(alluser[u],1),'item_share':iv[i]/np.maximum(allitem[i],1),
              'pair_share':pv/np.maximum(allpair,1),'artist_affinity':affinity,
              'artist_share':affinity/np.maximum(uv[u],1)}
            group='manual' if window in self.cfg['manual_windows'] else 'window_'+str(window)
            for name,values in values.items():
                column=name+'_'+str(window);numeric[column]=np.asarray(values,np.float32);groups.setdefault(group,[]).append(column)
            del counts,profile
        return pd.concat([frame,pd.DataFrame(numeric,index=frame.index)],axis=1),cats,groups

def snapshot(h,plan,role,day,output):
    path=output/f'{role}_{day}';file=path/'features.parquet'
    if file.exists():
        stats=read(path/'stats.json')
        if stats.get('config')!=plan['config']:raise ValueError('Cached snapshot configuration changed; use a fresh output')
        return pd.read_parquet(file),pd.read_parquet(path/'truth.parquet'),stats
    cfg=plan['config'];started=time.perf_counter();prefix=Prefix(h,day,cfg)
    truth=h.labels(role,day,plan);users=sorted(truth.user_id.unique())
    if not users:raise ValueError(f'Empty positive query interval {role} day{day}; stop without adjusting from quality')
    pairs=prefix.candidates(users);positives=pd.MultiIndex.from_frame(truth[['user_id','item_id']])
    pairs['label']=pd.MultiIndex.from_frame(pairs[['user_id','item_id']]).isin(positives).astype(np.int8)
    pairs['query_id']=pairs.user_id.to_numpy(np.int64)+day*2000000
    if role=='train':
        pairs=training_sample(pairs,cfg['seed']+day,cfg)
    frame,cats,groups=prefix.features(pairs)
    rawcounts=np.bincount(h.u[prefix.past],minlength=len(h.users))
    activepos=h.user_index.get_indexer(users);cold=[int(uid) for uid,pos in zip(users,activepos) if rawcounts[pos]==0]
    catalogcounts=np.asarray(prefix.x.sum(axis=1)).ravel()
    stats={'day':day,'role':role,'config':cfg,'groups':groups,'cats':cats,'rows':len(frame),'query_users':len(users),
      'no_positive_users':len(plan['cohorts'][role])-len(users),'cold_active_users':len(cold),'cold_user_ids':cold,
      'warm_active_users':len(users)-len(cold),
      'warm_active_without_catalog_history':int(((rawcounts[activepos]>0)&(catalogcounts[activepos]==0)).sum()),
      'outside_catalog_positive_pairs':int((~truth.item_id.isin(h.items)).sum()),
      'cf_fit_seconds':prefix.fit_seconds,'snapshot_seconds':time.perf_counter()-started,
      'full_history_listens':int(prefix.past.sum()),'catalog_history_listens':int((prefix.past&(h.ipos>=0)).sum())}
    path.mkdir(parents=True,exist_ok=True);frame.to_parquet(file,index=False);truth.to_parquet(path/'truth.parquet',index=False);save(path/'stats.json',stats)
    del prefix;gc.collect();return frame,truth,stats

def columns(stats,groups):
    # Membership differs across procedures, feature order does not.
    selected=set(groups)
    return list(dict.fromkeys(stats['groups']['manual']+[c for g,cols in stats['groups'].items() if g in selected for c in cols]))

def development(h,plan,args):
    cfg=plan['config'];train=[];val=[];truth=[];snapstats=[]
    for role,days,dest in [('train',cfg['train_days'],train),('validation',cfg['validation_days'],val)]:
        for day in days:
            frame,t,stats=snapshot(h,plan,role,day,args.output);dest.append(frame);snapstats.append(stats)
            if role=='validation':truth.append(t)
    return pd.concat(train,ignore_index=True),pd.concat(val,ignore_index=True),truth,stats,snapstats

def develop(h,plan,args):
    from .v3_catboost import fit_classifier
    if args.results.exists() and read(args.results).get('status') in ['developed','complete']:return
    cfg=plan['config'];train,val,truths,stats,snapstats=development(h,plan,args)
    metric=Metric(val,truths,cfg['validation_days']);universe=['window_'+str(w) for w in cfg['automatic_windows']]
    public_stats=[{k:v for k,v in s.items() if k not in ['config','groups','cats','cold_user_ids']} for s in snapstats]
    results=read(args.results) if args.results.exists() else {'scope':'External Yambda replication; fixed design before final weekly quality.',
      'config':cfg,'status':'developing','snapshot_costs':public_stats,'trials':[]}
    if results['config']!=cfg:raise ValueError('Existing development results have different settings')
    if results['status'] in ['developed','complete']:return
    def fit(name,groups):
        cached=next((trial for trial in results['trials'] if trial['name']==name),None)
        if cached is not None:return None,cached
        cols=columns(stats,groups)
        model,record=fit_classifier(train,cols,stats['cats'],cfg,17,iterations=cfg['max_trees'],
          val=val,metric=metric,staged_period=cfg['staged_period'])
        trial={'name':name,'groups':groups,'columns':cols,'features':len(cols),'trees':int(model.tree_count_),
          'fit_seconds':record['fit_seconds'],'metrics':record['metrics']}
        if name=='all_windows':trial['feature_importances']=dict(zip(cols,map(float,model.feature_importances_)))
        results['trials'].append(trial);save(args.results,results)
        print('YAMBDA_DEVELOP',json.dumps({k:v for k,v in trial.items() if k!='columns'}),flush=True)
        return model,trial
    model,manual=fit('manual',[]);del model;gc.collect()
    model,allwindows=fit('all_windows',universe)
    importances=allwindows['feature_importances']
    order=sorted(universe,key=lambda g:(-sum(importances.get(c,0) for c in stats['groups'][g]),cfg['automatic_windows'].index(int(g.split('_')[1]))))
    del model;gc.collect();rng=np.random.default_rng(cfg['random_subset_seed']);winners={'manual':manual}
    # Plan all random subsets before their first fit, never expand after outcomes.
    random=[list(rng.choice(universe,k,replace=False)) for k in cfg['subset_cardinalities']]
    results['importance_order']=order;results['random_subsets']=random;save(args.results,results)
    for arm in ['importance','random']:
        trials=[]
        for j,k in enumerate(cfg['subset_cardinalities']):
            chosen=order[:k] if arm=='importance' else [g for g in universe if g in random[j]]
            model,trial=fit(arm+'_'+str(k),chosen);trials.append(trial);del model;gc.collect()
        winners[arm]=max(trials,key=lambda r:r['metrics']['ndcg@10'])
    results.update(status='developed',selected=winners,manual_columns=stats['groups']['manual'],categoricals=stats['cats'],
      shared_fit_seconds=manual['fit_seconds']+allwindows['fit_seconds'],catalog_items=len(h.items),history_listens=len(h.t))
    save(args.results,results)

def bootstrap(a,b,cfg):
    joined=a.set_index(['user_id','day'])[['ndcg','recall']].join(b.set_index(['user_id','day'])[['ndcg','recall']],lsuffix='_a',rsuffix='_b',how='inner')
    rng=np.random.default_rng(cfg['bootstrap_seed']);out={}
    for key in ['ndcg','recall']:
        wide=(joined[key+'_b']-joined[key+'_a']).unstack('day').reindex(columns=cfg['test_days'])
        observed=wide.notna().to_numpy();d=wide.fillna(0).to_numpy();n=len(wide)
        point=float(np.mean(d.sum(axis=0)/observed.sum(axis=0)));means=[]
        for _ in range(cfg['bootstrap_draws']):
            weights=np.bincount(rng.integers(0,n,n),minlength=n)
            denominator=(weights[:,None]*observed).sum(axis=0)
            if (denominator==0).any():raise ValueError('Bootstrap week has no weighted positive queries')
            means.append(float(np.mean((weights[:,None]*d).sum(axis=0)/denominator)))
        out[key]={'delta':point,'ci95':np.quantile(means,[.025,.975]).tolist(),'users':n,'queries':len(joined)}
    return out

def final(h,plan,args):
    from .v3_catboost import fit_classifier,catboost_categories
    from catboost import CatBoostClassifier
    cfg=plan['config'];result=read(args.results)
    if result['status']!='developed':raise ValueError('Developed frozen choices required; final evaluation is not a model-selection stage')
    train,val,_,stats,snapstats=development(h,plan,args)
    addition=training_sample(val,cfg['seed']+433,cfg)
    pool=pd.concat([train,addition],ignore_index=True).sort_values(['query_id','item_id']).reset_index(drop=True)
    del train,val,addition;gc.collect();modelstats={}
    for arm,choice in result['selected'].items():
        files=[];seconds=0
        for seed in cfg['seeds']:
            model,record=fit_classifier(pool,choice['columns'],stats['cats'],cfg,seed,iterations=choice['trees'])
            file=args.output/f'final_{arm}_{seed}.cbm';model.save_model(str(file));files.append(file.name);seconds+=record['fit_seconds']
            del model;gc.collect()
        modelstats[arm]={'files':files,'fit_seconds_all_seeds':seconds,
          'model_bytes_all_seeds':sum((args.output/f).stat().st_size for f in files)}
    del pool;gc.collect()
    # Only after every fixed refit completes are final weekly labels sliced.
    perarm={arm:[] for arm in result['selected']};weekresults=[]
    for day in cfg['test_days']:
        frame,truth,st=snapshot(h,plan,'test',day,args.output);metric=Metric(frame,[truth],[day]);week={}
        cold=frame.user_id.isin(st['cold_user_ids']).to_numpy()
        for arm,choice in result['selected'].items():
            features=frame[choice['columns']].copy(deep=False);catboost_categories(features,stats['cats']);score=np.zeros(len(frame))
            for file in modelstats[arm]['files']:
                model=CatBoostClassifier();model.load_model(str(args.output/file))
                score+=model.predict(features,prediction_type='RawFormulaVal',thread_count=cfg['num_threads']);del model;gc.collect()
            score/=len(cfg['seeds']);score[cold]=-frame.loc[cold,'pop_rank'].to_numpy()
            rows=metric.rows(score);rows['cold']=rows.user_id.isin(st['cold_user_ids'])
            perarm[arm].append(rows)
            week[arm]={'warm':{key:float(rows.loc[~rows.cold,key].mean()) for key in ['ndcg','recall','candidate_recall']},
              'all_active':{key:float(rows[key].mean()) for key in ['ndcg','recall','candidate_recall']}}
            del features,score
        weekresults.append({'day':day,'systems':week,'population':{k:st[k] for k in ['query_users','no_positive_users','warm_active_users','cold_active_users','warm_active_without_catalog_history','outside_catalog_positive_pairs']},
          'candidate_rows':len(frame),'cf_fit_seconds':st['cf_fit_seconds'],'snapshot_seconds':st['snapshot_seconds']})
        del frame,truth,metric;gc.collect()
    pooled={arm:pd.concat(parts,ignore_index=True) for arm,parts in perarm.items()}
    summaries={};paired={}
    for scope in ['warm','all_active']:
        local={arm:rows[~rows.cold] if scope=='warm' else rows for arm,rows in pooled.items()}
        summaries[scope]={arm:{key:float(rows.groupby('day')[key].mean().mean()) for key in ['ndcg','recall','candidate_recall']} for arm,rows in local.items()}
        paired[scope]={b+'_minus_'+a:bootstrap(local[a],local[b],cfg) for a,b in [('manual','importance'),('manual','random'),('random','importance')]}
    result.update(status='complete',test_weeks=weekresults,pooled_equal_week=summaries,paired_user_bootstrap=paired,
      final_models=modelstats,pooled_estimand='Equal mean of three week-level user macro metrics. Bootstrap user multiplicities shared across all weeks; all-active main, warm separate.',
      bootstrap_population='Resample the union of users with positive targets in at least one final week; warm uses the warm-active union. No-positive-in-all-three-weeks users are outside this conditional ranking estimand. Intervals condition on these three fixed periods and fitted seed ensembles; they do not estimate calendar-period or retraining variability.',
      final_selection_rule='No final winner or subset is selected; compare all three frozen arms.')
    result['primary_result']={'reference':'manual','challenger':'importance','population':'all_future_positive_with_common_cold_fallback',
      'metric':'equal_three_week_macro_ndcg@10','reference_value':summaries['all_active']['manual']['ndcg'],
      'challenger_value':summaries['all_active']['importance']['ndcg'],
      'bootstrap':paired['all_active']['importance_minus_manual']['ndcg']}
    save(args.results,result)

def run(args):
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
    cfg=read(args.config);plan=design(args,cfg)
    if args.stage=='design':
        print(json.dumps({k:v for k,v in plan.items() if k not in ['cohorts','excluded_task2_users']},ensure_ascii=False,indent=2));return
    if args.results.exists() and read(args.results).get('status')=='complete':
        print('YAMBDA_ALREADY_COMPLETE',str(args.results),flush=True);return
    # Fresh-clone reproduction needs no external Task2 checkout: the published
    # design contains the already fixed cohorts, and official sources are public.
    import urllib.request
    import shutil
    args.data.mkdir(parents=True,exist_ok=True)
    for filename,url in [('yambda50m_multi.parquet',cfg['source_url']),('yambda_artist_item_mapping.parquet',cfg['artist_url'])]:
        destination=args.data/filename
        if destination.exists():continue
        temporary=destination.with_suffix('.part')
        with urllib.request.urlopen(url) as response,temporary.open('wb') as stream:
            shutil.copyfileobj(response,stream,length=1024*1024)
        temporary.replace(destination)
    args.output.mkdir(parents=True,exist_ok=True)
    h=Histories(args.data,plan)
    print('YAMBDA_HISTORY',len(h.users),len(h.t),'full listens; fixedcatalog',len(h.items),flush=True)
    if args.stage in ['develop','all']:develop(h,plan,args)
    if args.stage in ['final','all']:final(h,plan,args)
