"""Validation-only CatBoost replication of the fourth-place KION approach.

Model/sampling source (not a copied implementation):
https://github.com/blondered/ods_MTS_RecSys_Challenge_solution/blob/6d8832bda8bbc6d927c04bbcb134eb28f19246db/src/models/train_second_stage.py

Unlike the author's sampled random holdout, early stopping here uses complete
candidate lists on two declared temporal validation weeks, with every missed
positive retained in the ideal NDCG denominator. Final-test data is never read.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from .v2_experiment import features_for, save
from .v2_search import FullRankingMetric, _development


class CatBoostFullNDCG:
    """Full-denominator mean-of-weeks NDCG; train metric is intentionally ignored."""
    def __init__(self, metric, validation_rows):
        self.metric = metric
        self.validation_rows = int(validation_rows)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        if len(target) != self.validation_rows:
            return 0.0, 1.0
        return self.metric.metrics(np.asarray(approxes[0]))['ndcg@10'], 1.0

    def get_final_error(self, error, weight):
        return error / weight if weight else 0.0


def source_negative_sample(train, seed=17):
    """Keep every retrieved positive and min(3*npos,25) uniform negatives/query.

    The input is already the existing pipeline's training pool. This additional
    source-like sampling does not recover negatives removed at snapshot creation.
    It never samples validation pairs or injects missing target items.
    """
    rng = np.random.default_rng(seed)
    selected = []
    for _, group in train.groupby('query_id', sort=True):
        positives = group.index[group.label.eq(1)].to_numpy()
        negatives = group.index[group.label.eq(0)].to_numpy()
        if not len(positives):
            continue
        count = min(3 * len(positives), 25, len(negatives))
        sampled = rng.choice(negatives, count, replace=False)
        selected.extend(positives.tolist() + sampled.tolist())
    sampled = train.loc[selected].sort_values(['query_id', 'item_id']).reset_index(drop=True)
    assert int(sampled.label.sum()) == int(train.label.sum())
    return sampled


def catboost_categories(frame, categorical_columns):
    """Explicit strings handle Arrow/pandas StringDtype and missing categories."""
    for column in categorical_columns:
        if column in frame:
            frame[column] = frame[column].astype('string').fillna('__MISSING__').astype(str)


def fit_classifier(train, columns, cats, config=None, seed=17, iterations=2000,
                   val=None, metric=None, checkpoint_iterations=None,
                   patience_blocks=3, staged_period=None):
    """Fit the source classifier and return ``(best_model, fit_record)``.

    Sampling is the caller's responsibility; ``source_negative_sample`` is
    available separately. Without validation this fits exactly ``iterations``
    trees for a frozen final refit. With validation, default early stopping
    evaluates full NDCG every iteration. ``checkpoint_iterations=100`` instead
    continues boosting in blocks and evaluates FULL validation only once/block.
    That faster mode resets per-fit RNG/permutation state between blocks and is
    explicitly an algorithm adaptation, not bitwise single-fit reproduction.
    Returned checkpoint models are copies of the best validation checkpoint,
    so later non-improving blocks cannot alter the selected parameters.

    Recommended for large validation sets: ``staged_period=50`` trains one
    continuous model without evaluation, scans staged predictions every 50
    trees against full validation, then copies and shrinks to the best prefix.
    This preserves the continuous boosting RNG stream; only the selection grid
    differs from per-iteration early stopping. It never resets RNG between fits.
    """
    config = dict(config or {})
    iterations = int(iterations)
    if iterations < 1:
        raise ValueError('iterations must be positive')
    if (val is None) != (metric is None):
        raise ValueError('Provide both validation frame and full ranking metric, or neither')
    if checkpoint_iterations is not None and (int(checkpoint_iterations) < 1 or patience_blocks < 1):
        raise ValueError('Checkpoint iterations and patience_blocks must be positive')
    if staged_period is not None and (int(staged_period) < 1 or val is None):
        raise ValueError('staged_period requires validation and must be positive')
    if staged_period is not None and checkpoint_iterations is not None:
        raise ValueError('Choose either staged_period or init-model checkpoints')
    columns = list(columns)
    cats = [c for c in cats if c in columns]
    xtrain = train.loc[:, columns].copy(deep=False)
    catboost_categories(xtrain, cats)
    train_pool = Pool(xtrain, train.label, cat_features=cats)
    validation_pool = None
    if val is not None:
        xval = val.loc[:, columns].copy(deep=False)
        catboost_categories(xval, cats)
        validation_pool = Pool(xval, val.label, cat_features=cats)
    params = dict(depth=9, learning_rate=.03, l2_leaf_reg=27,
                  bootstrap_type='Bernoulli', subsample=.97, scale_pos_weight=3,
                  thread_count=int(config.get('num_threads', config.get('threads', 4))),
                  random_seed=int(seed), loss_function='Logloss',
                  allow_writing_files=False, task_type='CPU')
    params.update(config.get('catboost_params', {}))
    params.update(iterations=iterations, random_seed=int(seed))
    started = time.perf_counter()
    history = []
    if val is None:
        model = CatBoostClassifier(**params)
        model.fit(train_pool, verbose=False)
        mode = 'fixed_trees_no_validation'
        last_iteration = model.tree_count_
    elif staged_period is not None:
        mode = 'single_fit_staged_validation_selection'
        model = CatBoostClassifier(**params)
        model.fit(train_pool, verbose=False)
        period = int(staged_period)
        best_score, best_trees = -np.inf, None
        for step, predictions in enumerate(model.staged_predict(
                validation_pool, prediction_type='RawFormulaVal', eval_period=period,
                thread_count=params['thread_count']), start=1):
            trees = min(step * period, model.tree_count_)
            measured = metric.metrics(predictions)
            checkpoint = {'trees': trees, 'metrics': measured,
                          'elapsed_seconds': time.perf_counter()-started}
            history.append(checkpoint)
            print('CATBOOST_STAGED', json.dumps(checkpoint), flush=True)
            if measured['ndcg@10'] > best_score:
                best_score, best_trees = measured['ndcg@10'], trees
        last_iteration = model.tree_count_
        model = model.copy()
        model.shrink(best_trees)
    elif checkpoint_iterations is None:
        if len(train) == len(val):
            raise ValueError('Custom metric requires distinct train and validation row counts')
        model = CatBoostClassifier(**params, eval_metric=CatBoostFullNDCG(metric, len(val)))
        model.fit(train_pool, eval_set=validation_pool, early_stopping_rounds=200,
                  use_best_model=True, verbose=False)
        mode = 'full_ndcg_each_iteration'
        last_iteration = model.tree_count_
    else:
        mode = 'checkpointed_init_model_adaptation'
        current = best = None
        best_score, stale, trained = -np.inf, 0, 0
        while trained < iterations:
            block_size = min(int(checkpoint_iterations), iterations-trained)
            block_params = {**params, 'iterations': block_size}
            following = CatBoostClassifier(**block_params)
            following.fit(train_pool, init_model=current, verbose=False)
            current = following
            trained += block_size
            predictions = current.predict(validation_pool, prediction_type='RawFormulaVal',
                                          thread_count=params['thread_count'])
            measured = metric.metrics(predictions)
            checkpoint = {'trees': trained, 'metrics': measured,
                          'elapsed_seconds': time.perf_counter()-started}
            history.append(checkpoint)
            print('CATBOOST_CHECKPOINT', json.dumps(checkpoint), flush=True)
            score = measured['ndcg@10']
            if score > best_score:
                best_score, stale = score, 0
                best = current.copy()
            else:
                stale += 1
            if stale >= patience_blocks:
                break
        model = best
        last_iteration = trained
    record = {'mode': mode, 'seed': int(seed), 'params': params,
              'trees': int(model.tree_count_), 'trained_through_iteration': int(last_iteration),
              'fit_seconds': time.perf_counter()-started,
              'features': len(columns), 'checkpoint_history': history,
              'checkpoint_iterations': checkpoint_iterations,
              'staged_period': staged_period,
              'selection_adaptation': ('full NDCG on continuous-model tree-prefix grid'
                                       if staged_period is not None else None),
              'patience_blocks': patience_blocks if checkpoint_iterations is not None else None,
              'algorithm_adaptation': mode == 'checkpointed_init_model_adaptation'}
    if validation_pool is not None:
        predictions = model.predict(validation_pool, prediction_type='RawFormulaVal',
                                    thread_count=params['thread_count'])
        record['metrics'] = metric.metrics(predictions)
    return model, record

