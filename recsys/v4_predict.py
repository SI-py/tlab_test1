"""Apply a selected recommendation ensemble at its declared cutoff for arbitrary user IDs.

This module never reads cohort files, test truth, or label-selection helpers.
Raw unknown users receive the prefix's recent-popularity fallback. Predictions
are ranking scores, not calibrated click probabilities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from threadpoolctl import threadpool_limits

from .v2_experiment import full_data, index_at
from .v2_candidates import MultiSourceCandidates
from .v3_models import extra_models
from .v3_catboost import catboost_categories
from .v4_quality_features import QualityWinnerFeatureStore






def load_models(output, metadata):
    """Load all frozen seed models, validating their ordered feature schema."""
    output = Path(output)
    result = {}
    for variant, spec in metadata.items():
        files = spec.get('files', [])
        if not files:
            raise ValueError(f'No frozen model files for {variant}')
        family = spec['family']
        result[variant] = []
        for filename in files:
            if family == 'catboost':
                model = CatBoostClassifier()
                model.load_model(str(output/filename))
                actual = model.feature_names_
                expected_cats = sorted(spec['columns'].index(c) for c in spec.get('categoricals', []))
                if model.get_cat_feature_indices() != expected_cats:
                    raise ValueError(f'Categorical schema mismatch: {variant}/{filename}')
                trees = model.tree_count_
            elif family == 'lightgbm':
                model = lgb.Booster(model_file=str(output/filename))
                actual = model.feature_name()
                trees = model.num_trees()
            else:
                raise ValueError(f'Unknown model family: {family}')
            if list(actual) != list(spec['columns']):
                raise ValueError(f'Feature schema mismatch: {variant}/{filename}')
            if 'trees' in spec and int(spec['trees']) != trees:
                raise ValueError(f'Tree count mismatch: {variant}/{filename}')
            result[variant].append(model)
    return result


def score_frame(frame, models, metadata, levels, num_threads=1):
    """Return seed-averaged RAW scores per variant, preserving input row order.

    No ranks are computed here, so pair batches may be split freely. CatBoost
    receives explicit strings; LightGBM receives the saved training categories,
    with unseen values mapped to missing, exactly as during validation.
    """
    result = {}
    for variant, ensemble in models.items():
        if not ensemble:
            raise ValueError(f'Empty model ensemble: {variant}')
        spec = metadata[variant]
        absent = set(spec['columns'])-set(frame.columns)
        if absent:
            raise ValueError(f'Missing features for {variant}: {sorted(absent)}')
        if frame.empty:
            result[variant] = np.empty(0, dtype=float)
            continue
        features = frame.loc[:, spec['columns']].copy(deep=False)
        if spec['family'] == 'catboost':
            catboost_categories(features, spec.get('categoricals', []))
        elif spec['family'] == 'lightgbm':
            for column in spec.get('categoricals', []):
                if column not in levels:
                    raise ValueError(f'Missing frozen LightGBM vocabulary: {column}')
                features[column] = pd.Categorical(
                    features[column].where(features[column].isin(levels[column])),
                    categories=levels[column])
        else:
            raise ValueError(f"Unknown model family: {spec['family']}")
        scores = np.zeros(len(frame), dtype=np.float64)
        for model in ensemble:
            if spec['family'] == 'catboost':
                scores += model.predict(features, prediction_type='RawFormulaVal',
                                        thread_count=num_threads)
            else:
                scores += model.predict(features, raw_score=True, num_threads=num_threads)
        scores /= len(ensemble)
        if not np.isfinite(scores).all():
            raise ValueError(f'Non-finite prediction: {variant}')
        result[variant] = scores
    return result


def reciprocal_ranks(frame, scores):
    """Per-query reciprocal ranks with item-ID ties; do not split a query."""
    queries = frame.query_id.to_numpy() if 'query_id' in frame else frame.user_id.to_numpy()
    order = np.lexsort((frame.item_id.to_numpy(), -np.asarray(scores), queries))
    if not len(order):
        return np.empty(0, dtype=float)
    query = queries[order]
    boundary = np.r_[True, query[1:] != query[:-1]]
    first = np.maximum.accumulate(np.where(boundary, np.arange(len(query)), 0))
    ranks = np.empty(len(frame), np.int64)
    ranks[order] = np.arange(len(query))-first+1
    return 1.0/(60.0+ranks)


def generator_rrf_basis(frame, sources):
    """Shared float64 generator-rank basis for offline selection and inference."""
    ranks = frame[[source+'_rank' for source in sources]].to_numpy(dtype=np.float64)
    if not np.isfinite(ranks).all() or (ranks < 1).any():
        raise ValueError('Generator ranks must be finite and at least one')
    return 1.0/(60.0+ranks)


def weighted_rank_sum(basis, weights):
    """Fixed column-order float64 accumulation, independent of query batching.

    BLAS ``basis @ weights`` may choose different reduction kernels at different
    batch sizes, introducing tiny score differences that can break tied ranks.
    This explicit accumulation is shared by offline selection and inference.
    """
    basis = np.asarray(basis, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if basis.ndim != 2 or weights.shape != (basis.shape[1],):
        raise ValueError('Rank basis and weights have incompatible shapes')
    result = np.zeros(basis.shape[0], dtype=np.float64)
    for column, weight in enumerate(weights):
        result += basis[:, column]*weight
    return result


def generator_rrf(frame, weights):
    """Use identical ordered float64 accumulation online and offline."""
    sources = list(weights)
    values = np.asarray([weights[source] for source in sources], dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Fusion weights must be finite and non-negative')
    return weighted_rank_sum(generator_rrf_basis(frame, sources), values)


def combine_scores(frame, scores, selection):
    """Apply frozen generator RRF and primary model-rank mixture verbatim."""
    fused = generator_rrf(frame, selection['rank_fusion']['weights'])
    result = {**scores, 'rank_fusion': fused}
    primary = np.zeros(len(frame), dtype=float)
    for variant, weight in selection['primary_mixture']['weights'].items():
        if not np.isfinite(weight) or weight < 0:
            raise ValueError('Mixture weights must be finite and non-negative')
        if variant not in result:
            raise ValueError(f'Frozen mixture model unavailable: {variant}')
        values = result[variant] if variant == 'rank_fusion' else reciprocal_ranks(frame, result[variant])
        primary += weight*values
    result['primary'] = primary
    return result


def recent_popularity_fallback(index, users, k=10):
    order = np.lexsort((index.items, -index.pop))[:k]
    return pd.DataFrame([(uid, index.items[i], float(index.pop[i]))
                         for uid in sorted(set(users)) for i in order],
                        columns=['user_id', 'item_id', 'score'])


def run(args):
    selection = json.loads((args.model_dir/'selection.json').read_text())
    cfg = selection['config']
    metadata = json.loads((args.model_dir/'final_models.json').read_text())
    levels_path = args.model_dir/'final_categories.json'
    levels = json.loads(levels_path.read_text()) if levels_path.exists() else {}
    for variant, spec in metadata.items():
        frozen = selection['best'][variant]
        for key in ['family', 'columns', 'categoricals', 'trees']:
            if key in frozen and spec[key] != frozen[key]:
                raise ValueError(f'Final model differs from frozen selection: {variant}/{key}')
    needed={name for name in selection['primary_mixture']['weights'] if name!='rank_fusion'}
    absent=needed-set(metadata)
    if absent:raise ValueError(f'Frozen primary model metadata missing: {sorted(absent)}')
    primary_metadata={name:spec for name,spec in metadata.items() if name in needed}
    required_columns=list(dict.fromkeys(
        column for spec in primary_metadata.values() for column in spec['columns']))
    query = sorted({int(uid.strip()) for uid in args.users.split(',') if uid.strip()})
    if not query:
        raise ValueError('At least one user ID is required')
    cutoff = pd.Timestamp(cfg['protocol']['test_cutoff'])
    events, items, users = full_data(args.data)
    # Feature constructors only receive past rows; no label helper is called.
    events = events.loc[events.timestamp < cutoff].copy()
    args.cache.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        index = index_at(events, cutoff, cfg, args.cache)
    warm = [u for u in query if u in index.uindex]
    cold = [u for u in query if u not in index.uindex]
    outputs = []
    if warm:
        models = load_models(args.model_dir, primary_metadata)
        with threadpool_limits(limits=1):
            extras = extra_models(index, users, items, cfg, args.cache, events=events)
        settings = cfg.get('candidate_generator', {})
        generator = MultiSourceCandidates(index, users,
            conditional=settings.get('conditional', True),
            prior=settings.get('conditional_prior', 20), extra_models=extras)
        if cfg.get('feature_store')=='anchor_v5':
            from .v5_features import AnchorWinnerFeatureStore
            store=AnchorWinnerFeatureStore(events,items,users,index,
                anchor_item_ids=cfg['anchor_item_ids'],
                representation_model=extras.get(cfg.get('representation_model','hybrid_als')),
                required_columns=required_columns)
        else:
            store = QualityWinnerFeatureStore(events, items, users, index,
                representation_model=extras.get(cfg.get('representation_model', 'hybrid_als')),
                required_columns=required_columns)
        batch_size = int(cfg['collaborative'].get('max_batch', 128))
        for start in range(0, len(warm), batch_size):
            candidates = generator.candidates(warm[start:start+batch_size], cfg['budgets'])
            frame, _, _ = store.build(candidates)
            frame['user_id'] = candidates.user_id.to_numpy()
            frame['item_id'] = candidates.item_id.to_numpy()
            frame['query_id'] = frame.user_id
            predictions = score_frame(frame, models, primary_metadata, levels, cfg.get('num_threads', 4))
            combined = combine_scores(frame, predictions, selection)
            result = candidates[['user_id', 'item_id']].assign(score=combined['primary'])
            result = result.sort_values(['user_id', 'score', 'item_id'], ascending=[True, False, True])
            result = result.groupby('user_id', sort=False).head(10)
            result['recommendation_source'] = 'frozen_ensemble'
            outputs.append(result)
    if cold:
        result = recent_popularity_fallback(index, cold)
        result['recommendation_source'] = 'recent_popularity_cold_user'
        outputs.append(result)
    result = pd.concat(outputs, ignore_index=True).sort_values(
        ['user_id', 'score', 'item_id'], ascending=[True, False, True])
    result['rank'] = result.groupby('user_id').cumcount()+1
    if 'title' in items:
        result['title'] = result.item_id.map(items.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(json.dumps({'cutoff': str(cutoff), 'requested_users': len(query),
                      'cold_users': len(cold), 'recommendations': len(result),
                      'output': str(args.output)}, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/raw'))
    parser.add_argument('--cache', type=Path, default=Path('artifacts/cache'))
    parser.add_argument('--users', required=True, help='Comma-separated IDs; arbitrary IDs are permitted.')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
