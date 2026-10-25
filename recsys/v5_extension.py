"""Append fixed training-prefix anchor features to the development snapshots."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import pickle
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .anchor_features import AnchorFeatures, select_anchors
from .v2_experiment import save


def load_prefix_representation(cutoff, cache):
    path = Path(cache)/f'weighted_hybrid_als_{pd.Timestamp(cutoff).date()}.pkl'
    with path.open('rb') as stream:
        return pickle.load(stream)


def augment_snapshot(parent, destination, model, anchor_ids):
    started=time.perf_counter()
    original=json.loads((parent/'stats.json').read_text())
    if (destination/'stats.json').exists():
        return json.loads((destination/'stats.json').read_text())
    destination.mkdir(parents=True,exist_ok=True)
    for filename in ('pairs.parquet','truth.parquet'):
        shutil.copyfile(parent/filename,destination/filename)
    builder=AnchorFeatures(model,anchor_ids)
    for filename in original['feature_parts']:
        frame=pd.read_parquet(parent/filename)
        extra=builder.build(frame[['user_id','item_id']])
        pd.concat([frame,extra],axis=1).to_parquet(destination/filename,index=False)
        del frame,extra
    stats=copy.deepcopy(original)
    stats['anchor_item_ids']=list(map(int,anchor_ids))
    stats['groups']['latent']=list(dict.fromkeys(stats['groups'].get('latent',[])+list(builder.columns)))
    stats['extension_seconds']=time.perf_counter()-started
    stats['base_seconds']=original['seconds']
    stats['seconds']=stats['base_seconds']+stats['extension_seconds']
    save(destination/'stats.json',stats)
    return stats


def run(args):
    parent,output=args.parent,args.output
    cfg=copy.deepcopy(json.loads((parent/'prepared.json').read_text())['config'])
    output.mkdir(parents=True,exist_ok=True)
    events=pd.read_csv(args.data/'interactions.csv',usecols=['user_id','item_id','last_watch_dt'],
        dtype={'user_id':'int32','item_id':'int32'},parse_dates=['last_watch_dt'])
    events=events.rename(columns={'last_watch_dt':'timestamp'})
    earliest=min(cfg['protocol']['train_cutoffs'],key=pd.Timestamp)
    model=load_prefix_representation(earliest,args.cache)
    prefix=events[events.timestamp<pd.Timestamp(earliest)]
    recent=prefix[prefix.timestamp>=pd.Timestamp(earliest)-pd.Timedelta(days=30)]
    pop=recent.item_id.value_counts().reindex(model.items,fill_value=0).to_numpy(np.float32)
    anchors=select_anchors(model,pop,max_anchors=args.max_anchors,pool_size=args.pool_size)
    del prefix,recent,pop,model,events
    gc.collect()
    shutil.copyfile(parent/'protocol.json',output/'protocol.json')
    for role,key in [('train','train_cutoffs'),('validation','validation_cutoffs')]:
        for cutoff in cfg['protocol'][key]:
            model=load_prefix_representation(cutoff,args.cache)
            name=f'{role}_{cutoff}'
            stats=augment_snapshot(parent/'snapshots'/name,output/'snapshots'/name,model,anchors)
            print('ANCHOR_FEATURES',name,stats.get('extension_seconds'),flush=True)
            del model
            gc.collect()
    cfg.update(feature_store='anchor_v5',search_engine='source_v5',
        anchor_item_ids=list(map(int,anchors)),parent_output=str(parent))
    save(args.config_output,cfg)
    save(output/'prepared.json',{'config':cfg})
    return cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=Path('artifacts/v4'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/v5'))
    parser.add_argument('--data',type=Path,default=Path('data/raw'))
    parser.add_argument('--cache',type=Path,default=Path('artifacts/cache'))
    parser.add_argument('--config-output',type=Path,default=Path('configs/v5.json'))
    parser.add_argument('--max-anchors',type=int,default=16)
    parser.add_argument('--pool-size',type=int,default=1000)
    run(parser.parse_args())


if __name__=='__main__':
    main()
