"""Full-data, user-disjoint temporal benchmark: prepare -> search -> final.

Run modules with BLAS threads=1. The final phase is the only test-label access.
"""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import json
import pickle
import os
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .collaborative import CollaborativeIndex
from .v2_features import TemporalFeatureStore
from .v2_protocol import V2Config, create_manifest, load_manifest, labels_for_split
from .metrics import evaluate, paired_bootstrap


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n')


def full_data(root):
    wanted = {'user_id','item_id','last_watch_dt','total_dur','watched_pct'}
    events = pd.read_csv(root / 'interactions.csv', usecols=lambda col: col in wanted,
                         dtype={'user_id':'int32', 'item_id':'int32'}, parse_dates=['last_watch_dt'])
    events = events.rename(columns={'last_watch_dt':'timestamp'})
    return events, pd.read_csv(root/'items.csv').set_index('item_id'), pd.read_csv(root/'users.csv').set_index('user_id')




def index_at(events, cutoff, cfg, cache):
    path = cache / f'index_{pd.Timestamp(cutoff).date()}.pkl'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open('rb') as stream:
            index = pickle.load(stream)
        print('INDEX_CACHE', cutoff, index.stats, flush=True)
        return index
    with threadpool_limits(limits=1):
        index = CollaborativeIndex(events, cutoff, cfg['collaborative'])
    # Inference needs parameters, not WARP optimizer accumulators (~1GiB).
    if index.lightfm is not None:
        model = index.lightfm
        index.lightfm = SimpleNamespace(**{k:getattr(model,k) for k in
            ['user_embeddings','item_embeddings','user_biases','item_biases']})
        del model
    with path.open('wb') as stream:
        pickle.dump(index, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print('INDEX_FIT', cutoff, index.stats, 'cache_bytes', path.stat().st_size, flush=True)
    return index


def training_sample(pairs, seed, negatives=80, hard=20):
    """All retrieved positives; hard-ranked + uniform negatives, never inject positives."""
    rng = np.random.default_rng(seed)
    selected = []
    ranks = [c for c in pairs if c.endswith('_rank')]
    for _, frame in pairs.groupby('user_id', sort=True):
        positive = frame[frame.label.eq(1)]
        if positive.empty:
            continue  # all-negative groups have zero ranking gradient
        negative = frame[frame.label.eq(0)]
        quality = negative[ranks].min(axis=1)
        hard_ids = quality.sort_values(kind='stable').index[:hard]
        rest = negative.index.difference(hard_ids, sort=False).to_numpy()
        random_ids = rng.choice(rest, min(negatives, len(rest)), replace=False)
        selected.extend(positive.index.tolist() + hard_ids.tolist() + random_ids.tolist())
    return pairs.loc[selected].sort_values(['user_id', 'item_id']).reset_index(drop=True)


def source_training_sample(pairs, seed):
    """KION classifier recipe on the FULL union: every positive + <=3x negatives.

    Uniform negatives without replacement, capped at 25/query; queries with no
    retrieved positive are omitted from supervised training only.
    """
    rng=np.random.default_rng(seed)
    selected=[]
    for _,frame in pairs.groupby('user_id',sort=True):
        positive=frame.index[frame.label.eq(1)].to_numpy()
        negative=frame.index[frame.label.eq(0)].to_numpy()
        if not len(positive):continue
        selected.extend(positive.tolist())
        selected.extend(rng.choice(negative,min(3*len(positive),25,len(negative)),replace=False).tolist())
    return pairs.loc[selected].sort_values(['user_id','item_id']).reset_index(drop=True)


def build_snapshot(events, items, users, cfg, manifest, role, cutoff, out, cache, truth_override=None):
    cutoff = pd.Timestamp(cutoff)
    stem = f'{role}_{cutoff.date()}'
    dest = out / 'snapshots' / stem
    if (dest/'stats.json').exists():
        return dest
    t = time.perf_counter()
    truth_all = labels_for_split(events, manifest, role, cutoff) if truth_override is None else truth_override
    index = index_at(events, cutoff, cfg, cache)
    extras={}
    generator=index
    if cfg.get('candidate_generator'):
        from .v3_models import extra_models
        from .v2_candidates import MultiSourceCandidates
        extras=extra_models(index,users,items,cfg,cache,events=events)
        settings=cfg['candidate_generator']
        generator=MultiSourceCandidates(index,users,conditional=settings.get('conditional',True),prior=settings.get('conditional_prior',20),extra_models=extras)
    truth = truth_all[truth_all.user_id.isin(index.users)].reset_index(drop=True)
    # This benchmark averages over warm users with future positives. Conditioning
    # only schedules queries; it never changes fit, item eligibility or candidates.
    query_users = np.sort(truth.user_id.unique())
    truth_index = pd.MultiIndex.from_frame(truth[['user_id','item_id']])
    dest.mkdir(parents=True, exist_ok=True)
    truth.to_parquet(dest/'truth.parquet', index=False)
    parts = []
    original_rows, original_positives = 0, 0
    for start in range(0,len(query_users),128):
        candidates = generator.candidates(query_users[start:start+128], cfg['budgets'])
        candidates['label'] = pd.MultiIndex.from_frame(candidates[['user_id','item_id']]).isin(truth_index).astype('int8')
        original_rows += len(candidates)
        original_positives += int(candidates.label.sum())
        if role == 'train':
            sample_seed=cfg['protocol']['seed']+int(cutoff.day)+start
            if cfg.get('training_sampling')=='source_classifier':
                candidates=source_training_sample(candidates,sample_seed)
            else:
                candidates = training_sample(candidates,sample_seed,
                                             cfg['train_negatives'],cfg['train_hard_negatives'])
        parts.append(candidates)
    pairs = pd.concat(parts,ignore_index=True)
    del parts
    pairs.to_parquet(dest/'pairs.parquet',index=False)
    if cfg.get('feature_store')=='anchor_v5':
        from .v5_features import AnchorWinnerFeatureStore
        store=AnchorWinnerFeatureStore(events,items,users,index,
            anchor_item_ids=cfg['anchor_item_ids'],
            representation_model=extras.get(cfg.get('representation_model','hybrid_als')))
    elif cfg.get('feature_store')=='quality_v4':
        from .v4_quality_features import QualityWinnerFeatureStore
        store=QualityWinnerFeatureStore(events,items,users,index,
            representation_model=extras.get(cfg.get('representation_model','hybrid_als')))
    elif cfg.get('feature_store')=='winner_v3':
        from .v3_features import WinnerFeatureStore
        store=WinnerFeatureStore(events,items,users,index,representation_model=extras.get('hybrid_als'))
    else:
        store = TemporalFeatureStore(events,items,users,index)
    groups = None
    feature_parts = []
    # Each query-independent batch writes immediately, keeping working memory bounded.
    for part_no, start in enumerate(range(0,len(pairs),60000)):
        batch = pairs.iloc[start:start+60000].reset_index(drop=True)
        frame,cats,groups = store.build(batch)
        frame['user_id'] = batch.user_id.to_numpy()
        frame['item_id'] = batch.item_id.to_numpy()
        frame['label'] = batch.label.to_numpy()
        frame['query_id'] = batch.user_id.to_numpy(np.int64) + int(cutoff.value//86400000000000)*2_000_000
        file = dest/f'features_{part_no:03d}.parquet'
        frame.to_parquet(file,index=False)
        feature_parts.append(file.name)
        del frame
    stats = {'role':role,'cutoff':str(cutoff),'query_users':len(query_users),
             'candidate_rows_before_sampling':original_rows,'positive_candidates_before_sampling':original_positives,
             'training_rows_after_sampling':len(pairs),'truth_pairs':len(truth),
             'cold_user_truth_pairs':len(truth_all)-len(truth),
             'cold_item_truth_pairs':int((~truth.item_id.isin(index.items)).sum()),
             'feature_parts':feature_parts,'cats':cats,'groups':groups,'index_stats':index.stats,
             'extra_model_stats':{name:model.stats for name,model in extras.items()},
             'training_sampling':cfg.get('training_sampling','hard_plus_uniform'),
             'seconds':time.perf_counter()-t}
    save(dest/'stats.json',stats)
    print('SNAPSHOT',json.dumps(stats),flush=True)
    del store,index,generator,extras,pairs
    gc.collect()
    return dest


def read_snapshot(path):
    stats = json.loads((path/'stats.json').read_text())
    frames = [pd.read_parquet(path/name) for name in stats['feature_parts']]
    frame = pd.concat(frames,ignore_index=True)
    frame.attrs['cutoff'] = stats['cutoff']
    return frame,pd.read_parquet(path/'truth.parquet'),stats


def features_for(stats, variant):
    cats,groups = stats['cats'],stats['groups']
    base = cats + groups['retrieval'] + groups['manual'] + groups['history']
    auto = [c for g,cols in groups.items() if g.startswith('window_') for c in cols]
    latent = groups.get('latent',[])
    if variant == 'manual': cols = base
    elif variant == 'auto_windows': cols = base + auto
    elif variant == 'auto_full': cols = base + auto + latent
    elif variant == 'representations':
        cols = cats + [c for c in groups['retrieval'] if not (c.startswith('pop_') or '_pop_' in c)] + [c for c in groups['history'] if not c.endswith('_pop')] + latent
    else: raise ValueError(variant)
    return list(dict.fromkeys(cols))


def prepare(args,cfg,events,items,users,manifest):
    for role,key in [('train','train_cutoffs'),('validation','validation_cutoffs')]:
        for cutoff in cfg['protocol'][key]:
            build_snapshot(events,items,users,cfg,manifest,role,cutoff,args.output,args.cache)
    save(args.output/'prepared.json',{'config':cfg,'packages':{
        p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','implicit','lightfm','lightgbm','catboost']}})


def main():
    parser=argparse.ArgumentParser(description='Temporal KION recommendation experiment')
    parser.add_argument('phase', choices=['prepare','search','final'])
    parser.add_argument('--config', type=Path, default=Path('configs/v4.json'))
    parser.add_argument('--data', type=Path, default=Path('data/raw'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/v4'))
    parser.add_argument('--cache', type=Path, default=Path('artifacts/cache'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    args.cache.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(args.config.read_text())
    events,items,users=full_data(args.data)
    path=args.output/'protocol.json'
    if path.exists():
        manifest=load_manifest(path)
    else:
        from .exposure import exposure_from_recipe
        manifest=create_manifest(path,events.user_id.unique(),exposure_from_recipe(events.user_id.unique()),V2Config(**cfg['protocol']))
    if args.phase=='prepare':
        prepare(args,cfg,events,items,users,manifest)
    elif args.phase=='search':
        if cfg.get('search_engine')=='source_v5':
            from .v5_search import search
        else:
            from .v4_search import search
        del events,items,users
        gc.collect()
        search(args,cfg,manifest)
    else:
        from .v4_search import final
        final(args,cfg,manifest,events,items,users)


if __name__=='__main__':
    main()
