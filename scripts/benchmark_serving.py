"""Measure selective features and complete warm-state recommendation requests.

Loads existing prefix/model caches, never fits CF or GBDT models. Each request
contains the same 128 prefix-observed users. Startup I/O and prefix aggregate
construction are reported separately from steady-state request latency.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from recsys.v2_experiment import full_data
from recsys.v2_candidates import MultiSourceCandidates
from recsys.v4_quality_features import QualityWinnerFeatureStore
from recsys.v5_features import AnchorWinnerFeatureStore
from recsys.v4_predict import load_models, score_frame, reciprocal_ranks


def cache_file(cache, name, cutoff):
    plain=cache/f'{name}_{cutoff}.pkl'
    if plain.exists():return plain
    matches=list(cache.glob(f'{name}_{cutoff}_*.pkl'))
    if len(matches)!=1:raise ValueError(f'Expected one existing {name} cache, found {len(matches)}')
    return matches[0]


def summarize(values):
    return {'runs_seconds':values,'mean_seconds':float(np.mean(values)),
            'min_seconds':float(np.min(values)),'max_seconds':float(np.max(values))}


def rank_top10(candidates,raw,weights):
    if len(weights)==1:
        values=raw[next(iter(weights))]
    else:
        values=np.zeros(len(candidates),np.float64)
        for name,weight in weights.items():values+=weight*reciprocal_ranks(candidates,raw[name])
    result=candidates[['user_id','item_id']].assign(score=values)
    return result.sort_values(['user_id','score','item_id'],ascending=[True,False,True]).groupby('user_id',sort=False).head(10)


def main(args):
    if args.batch_size!=128:raise ValueError('This comparison fixes batch size at128')
    selection=json.loads((args.run/'selection.json').read_text());cfg=selection['config']
    metadata=json.loads((args.run/'final_models.json').read_text())
    levels=json.loads((args.run/'final_categories.json').read_text())
    cutoff=cfg['protocol']['test_cutoff']
    startup={};started=time.perf_counter()
    with cache_file(args.cache,'index',cutoff).open('rb') as stream:index=pickle.load(stream)
    startup['index_load_seconds']=time.perf_counter()-started
    started=time.perf_counter();extras={}
    for name in cfg['extra_models']:
        with cache_file(args.cache,name,cutoff).open('rb') as stream:extras[name]=pickle.load(stream)
    startup['extra_generator_load_seconds']=time.perf_counter()-started
    started=time.perf_counter();events,items,user_metadata=full_data(args.data)
    events=events.loc[events.timestamp<pd.Timestamp(cutoff)].copy()
    startup['raw_read_and_prefix_seconds']=time.perf_counter()-started
    settings=cfg.get('candidate_generator',{})
    generator=MultiSourceCandidates(index,user_metadata,conditional=settings.get('conditional',True),
        prior=settings.get('conditional_prior',20),extra_models=extras)
    users=np.sort(np.random.default_rng(args.seed).choice(index.users,args.batch_size,replace=False))
    started=time.perf_counter();candidates=generator.candidates(users,cfg['budgets'])
    startup['initial_candidate_generation_seconds']=time.perf_counter()-started
    candidates['query_id']=candidates.user_id
    representation=extras[cfg.get('representation_model','weighted_hybrid_als')]
    report={'scope':'Existing prefix fits; warm-state request benchmark, no CF/GBDT training',
        'cutoff':cutoff,'user_sampling':'Uniform sorted prefix users, without future-activity filtering','batch_size':args.batch_size,
        'candidate_rows':len(candidates),'seed':args.seed,'gbdt_threads':4,'blas_threads':1,
        'warmup_runs':1,'timed_runs':args.runs,'shared_startup':startup,'cases':{},
        'timing_boundary':'Full request = generate candidates + selective store.build + saved model scoring + top10. Prefix fits, cache/model loading and aggregate-store constructor excluded and reported separately.',
        'cold_definition':'Observed first load in this process; OS filesystem cache is not flushed, so this is not a physical cold-disk claim.'}
    # Ordinary numerical replay: full materialization versus selective columns,
    # using identical candidates, prefix caches and already fitted factors.
    started=time.perf_counter()
    reference_store=AnchorWinnerFeatureStore(events,items,user_metadata,index,
        representation_model=representation,anchor_item_ids=cfg['anchor_item_ids'])
    reference,_,_=reference_store.build(candidates)
    report['reference_full_constructor_plus_build_seconds']=time.perf_counter()-started
    del reference_store;gc.collect()
    cases={'manual':{'manual':1.},'auto_selected':{'auto_selected':1.},
           'primary':selection['primary_mixture']['weights']}
    for case,weights in cases.items():
        if 'rank_fusion' in weights:raise ValueError('This measured primary is restricted to model mixtures')
        specs={name:metadata[name] for name in weights}
        required=list(dict.fromkeys(c for name in specs for c in specs[name]['columns']))
        started=time.perf_counter();models=load_models(args.run,specs)
        cold_load=time.perf_counter()-started
        del models;gc.collect()
        started=time.perf_counter();models=load_models(args.run,specs)
        repeat_load=time.perf_counter()-started
        started=time.perf_counter()
        store=AnchorWinnerFeatureStore(events,items,user_metadata,index,required_columns=required,
            representation_model=representation,anchor_item_ids=cfg['anchor_item_ids'])
        constructor=time.perf_counter()-started
        frame,_,_=store.build(candidates)
        difference=[]
        for col in required:
            a=reference[col].to_numpy();b=frame[col].to_numpy()
            equal=np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fci' else np.array_equal(a,b)
            if not equal:
                difference.append(col)
        if difference:raise ValueError(f'Selective feature arithmetic changed: {difference}')
        frame['query_id']=candidates.query_id.to_numpy();frame['item_id']=candidates.item_id.to_numpy()
        def feature_only():
            return store.build(candidates)[0]
        def request():
            local=generator.candidates(users,cfg['budgets']);local['query_id']=local.user_id
            features,_,_=store.build(local)
            features['query_id']=local.query_id.to_numpy();features['item_id']=local.item_id.to_numpy()
            raw=score_frame(features,models,specs,levels,4)
            return rank_top10(local,raw,weights)
        feature_only();request()
        feature_times=[];request_times=[]
        for _ in range(args.runs):
            started=time.perf_counter();feature_only();feature_times.append(time.perf_counter()-started)
            started=time.perf_counter();request();request_times.append(time.perf_counter()-started)
        report['cases'][case]={'models':list(specs),'seed_models':sum(len(m['files']) for m in specs.values()),
            'required_feature_columns':len(required),'constructed_auto_windows':[w for w in store.windows if w not in (0,1,7,30)],
            'latent_or_anchor_columns':sum(c.startswith(('learned_','latent_product_')) for c in required),
            'feature_parity_exact':True,'first_ensemble_load_seconds':cold_load,'repeat_ensemble_load_seconds':repeat_load,
            'prefix_aggregate_store_constructor_seconds':constructor,
            'feature_only':summarize(feature_times),'full_warm_request':summarize(request_times)}
        print(case,json.dumps(report['cases'][case]),flush=True)
        del store,models,frame;gc.collect()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('artifacts/v5'))
    parser.add_argument('--data',type=Path,default=Path('data/raw'))
    parser.add_argument('--cache',type=Path,default=Path('artifacts/cache'))
    parser.add_argument('--output',type=Path,default=Path('results/serving_benchmark.json'))
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--runs',type=int,default=3)
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):main(args)
