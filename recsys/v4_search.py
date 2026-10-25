"""Source-derived classifiers, bounded feature grammar and final evaluation.

All choices use the declared two development weeks. The final cohort is opened
once, after the complete selection JSON and fixed-tree refits exist.
"""
from __future__ import annotations
import gc
import json
import time
import resource
import sys
import importlib.metadata
import os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from .v2_experiment import save, read_snapshot, features_for, build_snapshot
from .v2_search import FullRankingMetric, encode_categories, fit_ranker
from .v4_predict import reciprocal_ranks, generator_rrf_basis, weighted_rank_sum, combine_scores
from .v2_protocol import labels_for_split
from .v3_catboost import fit_classifier, source_negative_sample, catboost_categories
from .metrics import evaluate, paired_bootstrap


def _development(args,cfg):
    """Concatenate already ordered snapshots without another full-frame sort.

    Wide validation has millions of rows; an unnecessary sorting copy can add
    several GiB. Reject a broken ordering contract instead of hiding it.
    """
    trains=[];validations=[];truths=[];stats=None
    for role,key,destination in [('train','train_cutoffs',trains),('validation','validation_cutoffs',validations)]:
        for cutoff in cfg['protocol'][key]:
            frame,truth,stats=read_snapshot(args.output/'snapshots'/f'{role}_{cutoff}')
            q=frame.query_id.to_numpy();item=frame.item_id.to_numpy()
            if np.any(q[1:]<q[:-1]) or np.any((q[1:]==q[:-1])&(item[1:]<=item[:-1])):
                raise ValueError('Snapshot must be ordered by unique query and item IDs')
            destination.append(frame)
            if role=='validation':truths.append(truth)
    del frame,truth
    train=pd.concat(trains,ignore_index=True)
    del trains
    validation=pd.concat(validations,ignore_index=True)
    del validations
    gc.collect()
    return train,validation,truths,stats


def rank_fusion_search(val,metric,seed=84921,random_trials=64):
    scorers=[c[:-5] for c in val if c.endswith('_rank') and c!='cf_rank']
    basis=generator_rrf_basis(val,scorers)
    eye=np.eye(len(scorers))
    candidates=[*eye,np.ones(len(scorers))/len(scorers)]
    rng=np.random.default_rng(seed)
    candidates.extend(rng.dirichlet(np.full(len(scorers),.5),size=random_trials))
    best_score,best_weight,best_metrics=-np.inf,None,None
    trials=[]
    def consider(weight):
        nonlocal best_score,best_weight,best_metrics
        measured=metric.metrics(weighted_rank_sum(basis,weight))
        trials.append({'weights':weight.tolist(),'metrics':measured})
        if measured['ndcg@10']>best_score:
            best_score,best_weight,best_metrics=measured['ndcg@10'],weight.copy(),measured
    for weight in candidates:consider(weight)
    # Sparse forward refinement can eliminate unhelpful generators; a random
    # Dirichlet search alone tends to give every weak source a nonzero weight.
    for _ in range(2):
        start=best_weight.copy()
        for unit in eye:
            for amount in [.1,.25,.5,.75]:consider((1-amount)*start+amount*unit)
    return {'weights':dict(zip(scorers,best_weight.tolist())),
            'metrics':best_metrics,'trials':len(trials),'seed':seed},weighted_rank_sum(basis,best_weight),trials


def mixture_search(val,predictions,metric,rrf):
    names=list(predictions)+['rank_fusion']
    ranked={name:reciprocal_ranks(val,score) for name,score in predictions.items()}
    ranked['rank_fusion']=rrf
    trials=[]
    # A deliberately bounded grid: individual systems and all pairwise convex
    # rank blends. No test feedback can expand this search.
    for i,first in enumerate(names):
        trials.append({'weights':{first:1.},'metrics':metric.metrics(ranked[first])})
        for second in names[i+1:]:
            for weight in [.25,.5,.75]:
                score=weight*ranked[first]+(1-weight)*ranked[second]
                trials.append({'weights':{first:weight,second:1-weight},'metrics':metric.metrics(score)})
    return max(trials,key=lambda x:x['metrics']['ndcg@10']),trials


def search(args,cfg,manifest):
    started=time.perf_counter()
    train,val,truths,stats=_development(args,cfg)
    metric=FullRankingMetric(val,truths,cfg['protocol']['validation_cutoffs'])
    records=[];best={};predictions={};importance={}
    seed=cfg['seeds'][0]
    variants=['manual','auto_windows','auto_full','representations']
    def retain(variant,record,model,pred):
        records.append(record)
        save(args.output/'search_partial.json',records)
        print('V4_VALIDATION',json.dumps({k:v for k,v in record.items() if k not in ['columns','categoricals','fit_record']}),flush=True)
        if variant not in best or record['metrics']['ndcg@10']>best[variant]['metrics']['ndcg@10']:
            best[variant]=record
            predictions[variant]=pred
            if record['family']=='catboost':
                model.save_model(str(args.output/f'dev_{variant}.cbm'))
                importance[variant]=dict(zip(record['columns'],model.feature_importances_.tolist()))
            else:model.booster_.save_model(str(args.output/f'dev_{variant}.txt'))
    def classifier(variant,columns,params):
        local={**cfg,'catboost_params':params}
        model,fit_record=fit_classifier(train,columns,stats['cats'],local,seed,
            iterations=cfg.get('catboost_max_trees',2000),val=val,metric=metric,
            staged_period=cfg.get('catboost_staged_period',50))
        cats=[c for c in stats['cats'] if c in columns]
        frame=val[columns].copy(deep=False)
        catboost_categories(frame,cats)
        pred=model.predict(frame,prediction_type='RawFormulaVal',thread_count=cfg['num_threads'])
        record={'variant':variant,'family':'catboost','params':params,'seed':seed,
                'trees':int(model.tree_count_),'features':len(columns),'columns':columns,
                'categoricals':cats,'fit_seconds':fit_record['fit_seconds'],
                'fit_record':fit_record,'metrics':metric.metrics(pred)}
        retain(variant,record,model,pred)
        del model,frame
        gc.collect()
    for variant in variants:
        for params in cfg.get('catboost_grid',[{'depth':9,'l2_leaf_reg':27}]):
            classifier(variant,features_for(stats,variant),params)
    # A finite relational grammar is generated in the store. We select WINDOW
    # GROUPS rather than thousands of individual values: fitted-model feature
    # importance orders groups, complete temporal validation chooses the prefix.
    # Learned factors are held out here to isolate automatic aggregation.
    group_importance={group:float(sum(importance.get('auto_windows',{}).get(col,0.) for col in cols))
                      for group,cols in stats['groups'].items() if group.startswith('window_')}
    order=sorted(group_importance,key=lambda group:(-group_importance[group],group))
    manual=features_for(stats,'manual')
    selection_trials=[]
    params=best['auto_windows']['params']
    for count in sorted({min(n,len(order)) for n in [1,2,3,5]}):
        if not count:continue
        columns=list(dict.fromkeys(manual+[col for group in order[:count] for col in stats['groups'][group]]))
        previous=len(records)
        classifier('auto_selected',columns,params)
        records[-1]['selected_window_groups']=order[:count]
        selection_trials.append({'groups':order[:count],'metrics':records[-1]['metrics']})
    # Source #2 LambdaRank remains a separate family and is tuned on exactly
    # the same candidate lists and source-classifier training pool.
    levels=encode_categories(train,val,stats['cats'])
    save(args.output/'development_categories.json',levels)
    grid=cfg.get('lightgbm_grid',[{'num_leaves':15,'reg_lambda':.1,'min_child_samples':80},
                                {'num_leaves':31,'reg_lambda':5.,'min_child_samples':100}])
    for variant in variants:
        columns=features_for(stats,variant)
        for params in grid:
            model,seconds=fit_ranker(train,columns,stats['cats'],cfg,params,seed,val,metric)
            pred=model.predict(val[columns],num_threads=cfg['num_threads'])
            record={'variant':variant,'family':'lightgbm','params':params,'seed':seed,
                    'trees':int(model.best_iteration_),'features':len(columns),'columns':columns,
                    'categoricals':[c for c in stats['cats'] if c in columns],
                    'fit_seconds':seconds,'metrics':metric.metrics(pred)}
            retain(variant,record,model,pred)
            del model
            gc.collect()
    rrf,rrf_scores,rrf_trials=rank_fusion_search(val,metric,random_trials=cfg.get('rrf_trials',64))
    mixture,mixture_trials=mixture_search(val,predictions,metric,rrf_scores)
    automatic=max([name for name in best if name!='manual'],key=lambda name:best[name]['metrics']['ndcg@10'])
    feature_spec={'grammar':{'entities':['user','item'],'paths':['direct','genre','country','content_type'],
                    'aggregations':['count','share','qualified_count','qualified_share'],
                    'manual_windows':[1,7,30,'all'],'generated_windows':order},
                  'group_ordering':'train-fitted CatBoost importance sums; model and iteration selected on development weeks',
                  'group_importance':group_importance,'group_selection_trials':selection_trials,
                  'constraint':'manual features retained; selection over prefixes of ordered generated groups'}
    decisions={'config':cfg,'best':best,'best_automatic_variant':automatic,
        'rank_fusion':rrf,'primary_mixture':mixture,'categoricals':stats['cats'],
        'feature_specification':feature_spec,
        'candidate_policy':'one fixed raw-prefix union for every model; never inject a target',
        'selection_metric':'equal mean of two full-denominator temporal NDCG@10 values',
        'training_rows':len(train),'validation_rows':len(val),
        'search_seconds':time.perf_counter()-started}
    decisions['peak_resident_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
    save(args.output/'validation_search.json',{'trials':records,'rank_fusion_trials':rrf_trials,'mixtures':mixture_trials})
    save(args.output/'feature_specification.json',feature_spec)
    val[['user_id','item_id','query_id','label']].assign(**predictions,rank_fusion=rrf_scores).to_parquet(args.output/'validation_predictions.parquet',index=False)
    save(args.output/'selection.json',decisions)
    print('V4_SELECTED',json.dumps({'automatic':automatic,'mixture':mixture,'seconds':decisions['search_seconds']}),flush=True)






def _diagnostic_plan(available):
    pairs=[('representations_minus_manual','manual','representations'),
           ('anchors_minus_representations','representations','anchor_representations'),
           ('anchors_minus_auto_full','auto_full','anchor_auto_full')]
    return {'scope':'Fixed diagnostic comparisons declared before final access; descriptive, no multiplicity correction',
            'seed':2026,'draws':2000,
            'comparisons':{name:{'reference':reference,'challenger':challenger}
                           for name,reference,challenger in pairs if reference in available and challenger in available}}




def _refit_models(args,cfg,decisions):
    """Refit the selected recipes using the same train+sampled-validation pool."""
    train,val,truths,stats=_development(args,cfg)
    addition=source_negative_sample(val,cfg['protocol']['seed']+433)
    final_train=pd.concat([train,addition],ignore_index=True).sort_values(['query_id','item_id']).reset_index(drop=True)
    del train,val,truths,addition
    gc.collect()
    metadata={}
    for variant,choice in decisions['best'].items():
        if choice['family']!='catboost':continue
        paths=[];times=[]
        for seed in cfg['seeds']:
            model,detail=fit_classifier(final_train,choice['columns'],choice['categoricals'],
                {**cfg,'catboost_params':choice['params']},seed,iterations=choice['trees'])
            path=args.output/f'final_{variant}_{seed}.cbm'
            model.save_model(str(path));paths.append(path.name);times.append(detail['fit_seconds'])
            print('REFIT',variant,seed,detail['fit_seconds'],flush=True)
            if seed==cfg['seeds'][0]:
                pd.DataFrame({'feature':choice['columns'],'importance':model.feature_importances_}).sort_values('importance',ascending=False).to_csv(args.output/f'{variant}_importance.csv',index=False)
            del model
        metadata[variant]={'family':'catboost','files':paths,'columns':choice['columns'],
            'categoricals':choice['categoricals'],'trees':choice['trees'],'fit_seconds':sum(times),'seeds':cfg['seeds']}
    levels=encode_categories(final_train,None,stats['cats'])
    save(args.output/'final_categories.json',levels)
    for variant,choice in decisions['best'].items():
        if choice['family']!='lightgbm':continue
        paths=[];times=[]
        for seed in cfg['seeds']:
            model,seconds=fit_ranker(final_train,choice['columns'],stats['cats'],cfg,choice['params'],seed,trees=choice['trees'])
            path=args.output/f'final_{variant}_{seed}.txt';model.booster_.save_model(str(path))
            paths.append(path.name);times.append(seconds)
            print('REFIT',variant,seed,seconds,flush=True)
            del model
        metadata[variant]={'family':'lightgbm','files':paths,'columns':choice['columns'],
            'categoricals':choice['categoricals'],'trees':choice['trees'],'fit_seconds':sum(times),'seeds':cfg['seeds']}
    save(args.output/'final_models.json',metadata)
    rows=len(final_train);del final_train;gc.collect()
    return metadata,levels,rows




def _score_snapshot(output,dest,feature_parts,metadata,levels,num_threads):
    """Score one frozen variant ensemble at a time with column projection.

    Part boundaries and seed accumulation order match score_frame exactly.
    Returned timing covers feature conversion and prediction, excluding model
    loading, projected I/O and collection; the caller records total wall time.
    """
    from .v4_predict import load_models,score_frame
    predictions={};seconds=0.
    for variant,spec in metadata.items():
        local={variant:spec}
        models=load_models(output,local)
        parts=[]
        for file in feature_parts:
            frame=pd.read_parquet(dest/file,columns=spec['columns'])
            started=time.perf_counter()
            scores=score_frame(frame,models,local,levels,num_threads)[variant]
            seconds+=time.perf_counter()-started
            parts.append(scores)
            del frame,scores
        predictions[variant]=np.concatenate(parts)
        del models,parts
        gc.collect()
    return predictions,seconds


def final(args,cfg,manifest,events,items,users):
    decisions=json.loads((args.output/'selection.json').read_text())
    metadata,levels,training_rows=_refit_models(args,cfg,decisions)
    analysis_plan=_diagnostic_plan(decisions['best'])
    truth_all=labels_for_split(events,manifest,'test',cfg['protocol']['test_cutoff'])
    dest=build_snapshot(events,items,users,cfg,manifest,'test',cfg['protocol']['test_cutoff'],args.output,args.cache,truth_override=truth_all)
    gc.collect()
    st=json.loads((dest/'stats.json').read_text())
    pairs=pd.read_parquet(dest/'pairs.parquet');truth=pd.read_parquet(dest/'truth.parquet')
    score_started=time.perf_counter()
    predictions,predict_seconds=_score_snapshot(args.output,dest,st['feature_parts'],metadata,levels,cfg['num_threads'])
    prediction_wall_seconds=time.perf_counter()-score_started
    for col in pairs:
        if col.endswith('_score') and col!='cf_score':predictions['generator_'+col[:-6]]=pairs[col].to_numpy()
    predictions['popularity']=-pairs.pop_rank.to_numpy()
    combined=combine_scores(pairs.assign(query_id=pairs.user_id),
        {name:predictions[name] for name in metadata},decisions)
    predictions['rank_fusion']=combined['rank_fusion']
    predictions['primary']=combined['primary']
    metrics={};peruser={}
    for name,score in predictions.items():
        measured,byuser=evaluate(pairs,score,truth)
        measured.pop('sampled_logloss',None)
        metrics[name]=measured;peruser[name]=byuser
        byuser.to_csv(args.output/f'{name}_per_user.csv',index=False)
        print('V4_TEST',name,json.dumps(measured),flush=True)
    auto=decisions['best_automatic_variant']
    comparisons={name:paired_bootstrap(peruser['manual'],peruser[name]) for name in [auto,'primary']}
    comparisons['manual_vs_popularity']=paired_bootstrap(peruser['popularity'],peruser['manual'])
    diagnostics={name:{**spec,'bootstrap':paired_bootstrap(peruser[spec['reference']],peruser[spec['challenger']],
                  seed=analysis_plan['seed'],n_boot=analysis_plan['draws'])}
                 for name,spec in analysis_plan['comparisons'].items()}
    # An additional full-cohort metric includes users with no raw prefix. They
    # receive the same global recent-popularity fallback for every system.
    warm=np.sort(truth.user_id.unique())
    cold_truth=truth_all[~truth_all.user_id.isin(warm)]
    cold_metrics={};all_user_metrics={}
    if len(cold_truth):
        index=__import__('recsys.v2_experiment',fromlist=['index_at']).index_at(events,cfg['protocol']['test_cutoff'],cfg,args.cache)
        top=np.lexsort((index.items,-index.pop))[:10]
        cold_ids=np.sort(cold_truth.user_id.unique())
        cold_pairs=pd.DataFrame({'user_id':np.repeat(cold_ids,len(top)),
                                 'item_id':np.tile(index.items[top],len(cold_ids))})
        cold_score=np.tile(-np.arange(len(top)),len(cold_ids))
        cold_metrics,cold_by=evaluate(cold_pairs,cold_score,cold_truth)
        cold_metrics.pop('sampled_logloss',None)
        cold_by.to_csv(args.output/'cold_fallback_per_user.csv',index=False)
        for name,by in peruser.items():
            both=pd.concat([by,cold_by],ignore_index=True)
            all_user_metrics[name]={key+'@10':float(both[key].mean()) for key in ['ndcg','recall','map']}
            all_user_metrics[name]['users']=len(both)
        del index
    else:all_user_metrics=metrics
    # History-length segments are diagnostics only; they cannot change selection.
    prefix=events[events.timestamp<pd.Timestamp(cfg['protocol']['test_cutoff'])]
    lengths=prefix.groupby('user_id').size()
    segments={}
    for name in ['manual',auto,'primary','popularity']:
        by=peruser[name].copy();by['history']=by.user_id.map(lengths)
        by['segment']=pd.cut(by.history,[0,3,10,30,np.inf],labels=['1-3','4-10','11-30','31+'])
        segments[name]=by.groupby('segment',observed=True).agg(users=('user_id','size'),ndcg=('ndcg','mean'),recall=('recall','mean')).reset_index().to_dict('records')
    predictions_path=args.output/'test_predictions.parquet'
    temporary=predictions_path.with_suffix('.parquet.tmp')
    pairs[['user_id','item_id','label']].assign(**predictions).to_parquet(temporary,index=False)
    os.replace(temporary,predictions_path)
    save(args.output/'results.json',{'metrics':metrics,'all_future_active_user_metrics':all_user_metrics,
        'cold_user_fallback':cold_metrics,'paired_bootstrap':comparisons,'segments':segments,
        'diagnostic_paired_bootstrap':diagnostics,
        'costs':metadata,'prediction_seconds_all_models':predict_seconds,
        'prediction_wall_seconds_with_load_io':prediction_wall_seconds,
        'peak_resident_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)),
        'primary_selected_by_validation':decisions['primary_mixture'],'best_automatic_variant':auto,
        'test_snapshot':st,'final_training_rows':training_rows,
        'protocol':'User-disjoint confirmation on a previously studied calendar week; prefix-only transductive retrieval'})
    print('COMPLETE_V4',flush=True)
