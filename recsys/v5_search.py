"""Bounded learned-geometry extension of the exact same V4 development pairs.

V4 candidates, cohorts, baseline models and scores are reused verbatim. Only two
additional feature families (two CatBoost settings each) are evaluated before
final evaluation. Anchor selection never uses validation outcomes.
"""
from __future__ import annotations
import gc
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .v2_experiment import save,features_for
from .v2_search import FullRankingMetric
from .v2_protocol import load_manifest
from .v3_catboost import fit_classifier,catboost_categories
from .v4_search import _development,mixture_search,final
from .v4_predict import generator_rrf


def matched_catboost_comparisons(parent_trials,anchor_trials,cutoffs):
    """Describe saved development trials with the same family/params/seed.

    Parent per-variant winners may be LightGBM, so best-to-best comparisons do
    not isolate adding anchors. These records use the matching CatBoost trial,
    regardless of which family won parent selection. Each side may have its
    own validation-selected tree count; those counts remain explicit.
    This diagnostic never selects a model or feeds a metric back into search.
    """
    families={'anchor_representations':'representations','anchor_auto_full':'auto_full'}
    metric_names=('ndcg@10','recall@10')
    comparisons=[]
    for anchor_index,anchor in enumerate(anchor_trials):
        base=families.get(anchor.get('variant'))
        if base is None or anchor.get('family')!='catboost' or anchor.get('seed') is None:
            continue
        for parent_index,parent in enumerate(parent_trials):
            if (parent.get('variant')!=base or parent.get('family')!='catboost'
                    or parent.get('seed')!=anchor['seed'] or parent.get('params')!=anchor['params']):
                continue
            parent_metrics={key:float(parent['metrics'][key]) for key in metric_names}
            anchor_metrics={key:float(anchor['metrics'][key]) for key in metric_names}
            parent_weeks=parent['metrics'].get('weeks',[])
            anchor_weeks=anchor['metrics'].get('weeks',[])
            if len(parent_weeks)!=len(anchor_weeks) or len(parent_weeks)!=len(cutoffs):
                raise ValueError('Matched development trials have incompatible validation weeks')
            weeks=[]
            for cutoff,parent_week,anchor_week in zip(cutoffs,parent_weeks,anchor_weeks):
                if parent_week.get('users')!=anchor_week.get('users'):
                    raise ValueError('Matched development trials have different validation query counts')
                before={key:float(parent_week[key]) for key in metric_names}
                after={key:float(anchor_week[key]) for key in metric_names}
                weeks.append({'cutoff':cutoff,'query_users':parent_week.get('users'),
                    'parent_metrics':before,'anchor_metrics':after,
                    'delta_anchor_minus_parent':{key:after[key]-before[key] for key in metric_names}})
            comparisons.append({'parent_variant':base,'anchor_variant':anchor['variant'],
                'family':'catboost','params':anchor['params'],'seed':anchor['seed'],
                'parent_trial_index':parent_index,'extension_trial_index':anchor_index,
                'parent_trees':parent.get('trees'),'anchor_trees':anchor.get('trees'),
                'parent_metrics':parent_metrics,'anchor_metrics':anchor_metrics,
                'delta_anchor_minus_parent':{key:anchor_metrics[key]-parent_metrics[key] for key in metric_names},
                'weeks':weeks,
                'scope':'Matched CatBoost family/params/seed; each side independently selects its tree prefix on development data. Not a best-family comparison or fixed-tree causal effect.'})
    return comparisons


def search(args,cfg,manifest):
    started=time.perf_counter()
    parent=Path(cfg.get('parent_output','artifacts/v4'))
    previous=json.loads((parent/'selection.json').read_text())
    prediction_file=parent/'validation_predictions.parquet'
    train,val,truths,stats=_development(args,cfg)
    metric=FullRankingMetric(val,truths,cfg['protocol']['validation_cutoffs'])
    old=pd.read_parquet(prediction_file)
    columns=['user_id','item_id','query_id','label']
    if not val[columns].equals(old[columns]):
        raise ValueError('Extension validation pairs or labels differ from the baseline')
    best={key:dict(value) for key,value in previous['best'].items()}
    predictions={key:old[key].to_numpy().copy() for key in best}
    rrf=generator_rrf(val,previous['rank_fusion']['weights'])
    if not np.array_equal(rrf,old.rank_fusion.to_numpy()):
        raise ValueError('Frozen generator fusion replay changed')
    del old
    records=[]
    variant_columns={}
    for variant,base in [('anchor_representations','representations'),('anchor_auto_full','auto_full')]:
        cols=features_for(stats,base)
        old_columns=set(previous['best'][base]['columns'])
        additions=set(cols)-old_columns
        if not old_columns<set(cols) or any(not col.startswith('learned_anchor_') for col in additions):
            raise ValueError('Controlled geometry ablation must preserve every base feature and only add anchors')
        variant_columns[variant]=cols
    for variant,cols in variant_columns.items():
        cats=[col for col in stats['cats'] if col in cols]
        for params in cfg['catboost_grid']:
            model,record=fit_classifier(train,cols,cats,{**cfg,'catboost_params':params},
                cfg['seeds'][0],iterations=cfg['catboost_max_trees'],val=val,metric=metric,
                staged_period=cfg['catboost_staged_period'])
            frame=val[cols].copy(deep=False)
            catboost_categories(frame,cats)
            scores=model.predict(frame,prediction_type='RawFormulaVal',thread_count=cfg['num_threads'])
            trial={'variant':variant,'family':'catboost','params':params,'seed':cfg['seeds'][0],
                   'columns':cols,'categoricals':cats,'trees':int(model.tree_count_),
                   'features':len(cols),'fit_seconds':record['fit_seconds'],
                   'fit_record':record,'metrics':metric.metrics(scores)}
            records.append(trial)
            if variant not in best or trial['metrics']['ndcg@10']>best[variant]['metrics']['ndcg@10']:
                best[variant]=trial;predictions[variant]=scores
                model.save_model(str(args.output/f'dev_{variant}.cbm'))
            save(args.output/'search_partial.json',records)
            print('V5_VALIDATION',json.dumps({key:value for key,value in trial.items() if key not in ['columns','categoricals','fit_record']}),flush=True)
            del model,frame
            gc.collect()
    mixture,mixture_trials=mixture_search(val,predictions,metric,rrf)
    automatic=max([key for key in best if key!='manual'],key=lambda key:best[key]['metrics']['ndcg@10'])
    decisions={**previous,'config':cfg,'best':best,'best_automatic_variant':automatic,
               'primary_mixture':mixture,
               'geometry_extension':{'anchor_item_ids':cfg['anchor_item_ids'],
                   'selection_data':'first training cutoff prefix only; no validation outcomes',
                   'families':['anchor_representations','anchor_auto_full'],
                   'parameter_trials':len(records),'rotation_invariant':True},
               'extension_search_seconds':time.perf_counter()-started,
               'peak_resident_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))}
    parent_search_file=parent/'validation_search.json'
    parent_search=json.loads(parent_search_file.read_text())
    matched=matched_catboost_comparisons(parent_search['trials'],records,
        cfg['protocol']['validation_cutoffs'])
    save(args.output/'validation_search.json',{'parent_search':str(parent_search_file),
         'extension_trials':records,'mixtures':mixture_trials,
         'matched_catboost_comparisons':matched,
         'matched_comparison_scope':'Saved development metrics only; exact family/params/seed matches. Model-family winners and quality choices are unchanged.'})
    save(args.output/'feature_specification.json',{**previous['feature_specification'],
         'geometry_extension':decisions['geometry_extension']})
    val[columns].assign(**predictions,rank_fusion=rrf).to_parquet(args.output/'validation_predictions.parquet',index=False)
    save(args.output/'selection.json',decisions)
    print('V5_SELECTED',json.dumps({'automatic':automatic,'mixture':mixture,
          'seconds':decisions['extension_search_seconds']}),flush=True)
