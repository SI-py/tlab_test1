"""Source-bound, prefix-aligned caches for the RecTools-style hybrid models."""
import gc
import json
import pickle
from pathlib import Path
import numpy as np
from .hybrid_als import HybridALS


def extra_models(index,users,items,cfg,cache,events=None):
    models={}
    for name,params in cfg.get('extra_models',{}).items():
        weighted=params.get('weighting')=='rectools_kion'
        if weighted:
            from .weighted_als import WeightedHybridALS
        file=cache/f'{name}_{index.cutoff.date()}.pkl'
        if file.exists():
            with file.open('rb') as stream:model=pickle.load(stream)
            if not np.array_equal(model.users,index.users) or not np.array_equal(model.items,index.items):
                raise ValueError('Hybrid cached-factor alignment mismatch')
            print('EXTRA_CACHE',name,str(index.cutoff),model.stats,flush=True)
        else:
            if weighted:
                if events is None: raise ValueError('Weighted source models require completed-pair attributes')
                model=WeightedHybridALS(params).fit(events,index=index,
                    user_metadata=users,item_metadata=items)
            else:
                model=HybridALS(params).fit(index.x,users=index.users,items=index.items,
                    user_metadata=users,item_metadata=items)
            with file.open('wb') as stream:pickle.dump(model,stream,pickle.HIGHEST_PROTOCOL)
            print('EXTRA_FIT',name,str(index.cutoff),model.stats,flush=True)
        models[name]=model
    return models
