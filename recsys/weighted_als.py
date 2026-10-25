"""RecTools' weighted KION iALS recipe on strictly retained prefix rows.

KION keeps the LAST watch of each pair and aggregates duration/percentage over
its watches (https://arxiv.org/html/2209.00325#S2.SS2). If that last date precedes
the cutoff, every contributing watch precedes it under the documented event-date
semantics. Those cumulative fields can therefore weight the retained prefix;
they cannot reconstruct a future-dated row or exact within-window watch time.

Recipe: last_watch_dt < cutoff, total_dur >= 300, weight=3 if watched_pct>10
else 1, positive confidence alpha*weight, missing confidence 1. Native fitting
uses only users/items with qualifying prefix interactions, as RecTools does.
https://rectools.readthedocs.io/en/latest/examples/5_benchmark_iALS_with_features.html

The returned factor matrices and score columns retain the RAW index alignment.
Its seen-item exclusions, catalog eligibility, cohorts, and labels must remain
unchanged outside this helper. Inactive raw users get a qualified-prefix count
popularity fallback in scores(); their zero factor rows do not encode that
fallback. Inactive raw item factor rows and scores remain zero for active users.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits


def source_categorical_features(ids, metadata, fields, *, multiple=()):
    """RecTools-style category counts; wholly absent metadata IDs have zero rows.

    Missing fields of a PRESENT object get an Unknown category, as in the KION
    notebook. Repeated multi-value tokens are counted, as SparseFeatures does.
    Vocabulary uses present objects among the qualifying prefix IDs only.
    """
    ids = np.asarray(ids)
    if metadata is None or not metadata.index.is_unique:
        raise ValueError('metadata must have a unique object-ID index')
    missing = set(fields)-set(metadata.columns)
    if missing:
        raise ValueError(f'missing metadata fields: {sorted(missing)}')
    present = pd.Index(ids).isin(metadata.index)
    present_positions = np.flatnonzero(present)
    frame = metadata.reindex(ids[present])
    chunks, names = [], []
    for field in fields:
        values = frame[field].fillna('Unknown').astype(str)
        if field in multiple:
            tokens = [[s.strip().lower() for s in value.split(',') if s.strip()]
                      or ['unknown'] for value in values]
        else:
            tokens = [[value] for value in values]
        vocabulary = sorted({token for row in tokens for token in row})
        lookup = {value: j for j, value in enumerate(vocabulary)}
        encoded = np.zeros((len(ids), len(vocabulary)), np.float32)
        rows = np.repeat(present_positions, [len(row) for row in tokens])
        cols = [lookup[value] for row in tokens for value in row]
        np.add.at(encoded, (rows, cols), 1)
        chunks.append(encoded)
        names.extend(f'{field}={value}' for value in vocabulary)
    return (np.concatenate(chunks, axis=1) if chunks else
            np.empty((len(ids), 0), np.float32)), tuple(names)


def weighted_prefix_matrix(events, index, *, min_duration=300, pct_threshold=10,
                           high_weight=3, base_weight=1):
    """Build an aligned CSR; inspect duration/pct only AFTER the strict date cut.

    Missing percentages receive base_weight, matching ``np.where(pct>10,3,1)``.
    Missing durations fail the explicit >=300 qualification. Duplicate retained
    pairs raise instead of silently summing already aggregated observations.
    """
    date_column = 'timestamp' if 'timestamp' in events.columns else 'last_watch_dt'
    required = {'user_id', 'item_id', date_column, 'total_dur', 'watched_pct'}
    if required - set(events.columns):
        raise ValueError(f'missing columns: {sorted(required-set(events.columns))}')
    cutoff = pd.Timestamp(index.cutoff)
    if pd.isna(cutoff) or cutoff.tz is not None:
        raise ValueError('index.cutoff must be a timezone-naive date')
    dates = pd.to_datetime(events[date_column])
    prefix = events.loc[dates < cutoff, ['user_id', 'item_id', 'total_dur', 'watched_pct']]
    prefix_rows = len(prefix)
    duration = pd.to_numeric(prefix.total_dur, errors='raise')
    qualified = prefix.loc[duration >= min_duration, ['user_id', 'item_id', 'watched_pct']]
    if qualified.duplicated(['user_id', 'item_id']).any():
        raise ValueError('KION input must have one aggregate row per retained user-item pair')
    pct = pd.to_numeric(qualified.watched_pct, errors='raise')
    weight = np.where(pct > pct_threshold, high_weight, base_weight).astype(np.float32)
    rows = pd.Index(index.users).get_indexer(qualified.user_id)
    cols = pd.Index(index.items).get_indexer(qualified.item_id)
    if (rows < 0).any() or (cols < 0).any():
        raise ValueError('qualified prefix IDs are absent from the raw index')
    if (not np.isfinite(weight).all() or (weight <= 0).any()
            or not np.isfinite(min_duration) or not np.isfinite(pct_threshold)):
        raise ValueError('qualification thresholds and positive weights must be finite')
    matrix = sparse.csr_matrix((weight, (rows, cols)),
                              shape=(len(index.users), len(index.items)), dtype=np.float32)
    matrix.sort_indices()
    stats = {'cutoff': cutoff.isoformat(), 'prefix_rows': prefix_rows,
             'qualified_rows': len(qualified), 'qualified_pairs': matrix.nnz,
             'base_weight_rows': int(np.count_nonzero(weight == base_weight)),
             'high_weight_rows': int(np.count_nonzero(weight == high_weight)),
             'sum_weights': float(weight.sum(dtype=np.float64))}
    return matrix, stats


class WeightedHybridALS:
    """Fit(events,index,user_metadata,item_metadata); scores(raw_user_positions).

    Config: factors=32, iterations=10, alpha=10, regularization=.5, seed=32,
    features='bias' ('none'/'bias'/'metadata'), fit_features_together=False,
    threads=1, use_cg=True, cg_steps=3, max_batch=1024. Qualification defaults
    min_duration=300, pct_threshold=10, high_weight=3, base_weight=1.

    Joint training reproduces RecTools' unconstrained solve then fixed-column
    restoration, not exact constrained-coordinate minimization. Separate mode
    fits latent ALS first, then independently regresses paired explicit features.
    Metadata vocabulary is learned only from qualifying prefix user/item IDs.
    """

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.user_factors = self.item_factors = None

    def fit(self, events, index, user_metadata=None, item_metadata=None, *, users_metadata=None):
        from implicit.cpu.als import AlternatingLeastSquares

        started = time.perf_counter()
        if users_metadata is not None:
            if user_metadata is not None and user_metadata is not users_metadata:
                raise ValueError('provide only one of user_metadata and users_metadata')
            user_metadata = users_metadata
        self.users, self.items = np.asarray(index.users).copy(), np.asarray(index.items).copy()
        if (self.users.ndim != 1 or self.items.ndim != 1
                or len(np.unique(self.users)) != len(self.users)
                or len(np.unique(self.items)) != len(self.items)):
            raise ValueError('raw index IDs must be unique vectors')
        self.cutoff = pd.Timestamp(index.cutoff)
        options = {key: self.config.get(key, value) for key, value in
                   [('min_duration', 300), ('pct_threshold', 10), ('high_weight', 3), ('base_weight', 1)]}
        weighted, weight_stats = weighted_prefix_matrix(events, index, **options)
        self.active_users = np.diff(weighted.indptr) > 0
        self.active_items = np.bincount(weighted.indices, minlength=len(self.items)) > 0
        if not weighted.nnz:
            raise ValueError('no duration-qualified interactions strictly before cutoff')
        active_user_ids, active_item_ids = self.users[self.active_users], self.items[self.active_items]
        x = weighted[self.active_users][:, self.active_items].tocsr()
        count = np.bincount(weighted.indices, minlength=len(self.items)).astype(np.float32)
        self.cold_user_prior = count / max(float(count.max()), 1.)
        del weighted
        n_users, n_items = x.shape
        factors = int(self.config.get('factors', 32))
        iterations = int(self.config.get('iterations', 10))
        alpha = float(self.config.get('alpha', 10))
        regularization = float(self.config.get('regularization', .5))
        cg_steps = int(self.config.get('cg_steps', 3))
        if (min(factors, iterations, cg_steps) < 1 or not np.isfinite(alpha) or alpha <= 0
                or not np.isfinite(regularization) or regularization <= 0):
            raise ValueError('factors, iterations, cg_steps, alpha and regularization must be positive')
        mode = self.config.get('features', 'bias')
        self.user_feature_names = self.item_feature_names = ()
        if mode == 'none':
            fu, fi = np.empty((n_users, 0), np.float32), np.empty((n_items, 0), np.float32)
        elif mode == 'bias':
            fu, fi = np.ones((n_users, 1), np.float32), np.ones((n_items, 1), np.float32)
            self.user_feature_names = self.item_feature_names = ('constant_bias',)
        elif mode == 'metadata':
            fu, self.user_feature_names = source_categorical_features(
                active_user_ids, user_metadata, ['sex', 'age', 'income'])
            fi, self.item_feature_names = source_categorical_features(
                active_item_ids, item_metadata, ['genres', 'content_type'], multiple=['genres'])
        else:
            raise ValueError("features must be 'none', 'bias' or 'metadata'")
        du, di = fu.shape[1], fi.shape[1]
        model = AlternatingLeastSquares(
            factors=factors, iterations=iterations, alpha=alpha, regularization=regularization,
            random_state=int(self.config.get('seed', 32)), dtype=np.float32,
            num_threads=int(self.config.get('threads', 1)),
            use_cg=bool(self.config.get('use_cg', True)))
        model.cg_steps = cg_steps
        together = bool(self.config.get('fit_features_together', False)) and du + di > 0
        with threadpool_limits(limits=1, user_api='blas'):
            if together:
                rng = np.random.default_rng(int(self.config.get('seed', 32)))
                uf = np.column_stack((fu, rng.random((n_users, factors))*.01,
                                      np.zeros((n_users, di)))).astype(np.float32)
                vf = np.column_stack((np.zeros((n_items, du)),
                                      rng.random((n_items, factors))*.01, fi)).astype(np.float32)
                confidence = alpha*x
                transposed = confidence.T.tocsr()
                for _ in range(iterations):
                    model.solver(confidence, uf, vf, regularization, model.num_threads)
                    if du:
                        uf[:, :du] = fu
                    model.solver(transposed, vf, uf, regularization, model.num_threads)
                    if di:
                        vf[:, -di:] = fi
            else:
                model.fit(x, show_progress=False)
                user_chunks, item_chunks = [model.user_factors], [model.item_factors]
                confidence = alpha*x
                # Source initializes paired-side CG with the native RNG. A
                # single all-one feature converges to the exact scalar solution.
                if du:
                    rng = np.random.default_rng(int(self.config.get('seed', 32)))
                    paired = rng.random((n_items, du), dtype=np.float32)*.01
                    model.solver(confidence.T.tocsr(), paired, fu, regularization, model.num_threads)
                    user_chunks.append(fu)
                    item_chunks.append(paired)
                if di:
                    rng = np.random.default_rng(int(self.config.get('seed', 32)))
                    paired = rng.random((n_users, di), dtype=np.float32)*.01
                    model.solver(confidence, paired, fi, regularization, model.num_threads)
                    user_chunks.append(paired)
                    item_chunks.append(fi)
                uf, vf = np.column_stack(user_chunks), np.column_stack(item_chunks)
        if not np.isfinite(uf).all() or not np.isfinite(vf).all():
            raise FloatingPointError('weighted ALS produced non-finite factors')
        self.user_factors = np.zeros((len(self.users), uf.shape[1]), np.float32)
        self.item_factors = np.zeros((len(self.items), vf.shape[1]), np.float32)
        self.user_factors[self.active_users] = uf
        self.item_factors[self.active_items] = vf
        self.n_user_features, self.n_item_features = du, di
        self.stats = {**weight_stats, 'fit_seconds': time.perf_counter()-started,
                      'raw_users': len(self.users), 'raw_items': len(self.items),
                      'active_users': n_users, 'active_items': n_items,
                      'inactive_raw_users': int((~self.active_users).sum()),
                      'inactive_raw_items': int((~self.active_items).sum()),
                      'latent_factors': factors, 'total_factors': uf.shape[1],
                      'user_features': du, 'item_features': di,
                      'metadata_encoding': ('RecTools category counts; absent metadata IDs zero; present missing fields Unknown'
                                            if mode == 'metadata' else 'none'),
                      'fit_features_together': bool(together),
                      'qualification': options, 'alpha': alpha,
                      'confidence': 'qualified positive=alpha*weight; missing=1',
                      'cold_user_fallback': 'qualified-prefix item user counts / maximum count',
                      'cold_item_policy': 'zero factor row, eligible raw catalog preserved',
                      'stored_matrix_bytes': int(sum(a.nbytes for a in [self.user_factors,
                          self.item_factors, self.active_users, self.active_items, self.cold_user_prior]))}
        return self

    def scores(self, user_positions):
        """RAW row positions; inactive users get prefix popularity, -1 gets zero."""
        if self.user_factors is None:
            raise RuntimeError('fit must precede scores')
        positions = np.asarray(user_positions)
        if positions.ndim != 1 or (positions.size and positions.dtype.kind not in 'iu'):
            raise ValueError('user_positions must be an integer vector')
        positions = positions.astype(np.int64)
        if (positions < -1).any() or (positions >= len(self.users)).any():
            raise IndexError('raw user position out of range; only -1 denotes unknown')
        if len(positions) > int(self.config.get('max_batch', 1024)):
            raise ValueError('scores batch exceeds max_batch')
        result = np.zeros((len(positions), len(self.items)), np.float32)
        known = positions >= 0
        warm_rows = np.flatnonzero(known)
        active = self.active_users[positions[known]]
        result[warm_rows[active]] = self.user_factors[positions[warm_rows[active]]] @ self.item_factors.T
        result[warm_rows[~active]] = self.cold_user_prior
        return result
