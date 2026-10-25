"""Vectorized candidate unions and KION-style conditional popularity.

ConditionalPopularityRecommender is an established KION source component,
not a new learned representation. This implementation aggregates the frozen
prefix user-item matrix by static user attributes. Metadata is assumed to have
been available at the cutoff; it is never inferred from future interactions.

Source config: https://github.com/asash/bert4rec_repro/blob/main/evaluation/configs/mts_kion_competition/mts_kion_config.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


ATTRIBUTES = {'sex_pop': 'sex', 'age_pop': 'age',
              'income_pop': 'income', 'kids_pop': 'kids_flg'}


class MultiSourceCandidates:
    """Union top budgets with deterministic item-ID ties and excluded seen items.

    ``prior`` is a Dirichlet pseudo-count over item interactions. For category
    g, score(i|g) = (count(g,i) + prior*p(i))/(sum_i count(g,i) + prior).
    Unknown/missing categories fall back to p(i), the normalized global prefix
    item frequency. Only ``index.x`` and static metadata enter these counts.
    No new collaborative model is trained. The index remains unchanged.
    """
    def __init__(self, index, users_metadata, conditional=True, prior=20.0,
                 extra_models=None):
        self.index = index
        self.conditional = bool(conditional)
        self.prior = float(prior)
        self.extra_models = dict(extra_models or {})
        reserved = {'bm25', 'rp3beta', 'pop', 'als', 'lightfm', 'ease', 'cf', *ATTRIBUTES}
        for name, model in self.extra_models.items():
            if not isinstance(name, str) or not name or name in reserved:
                raise ValueError(f'Extra model has invalid or reserved scorer name: {name!r}')
            if (not np.array_equal(getattr(model, 'users', None), index.users)
                    or not np.array_equal(getattr(model, 'items', None), index.items)):
                raise ValueError(f'Extra model {name!r} users/items are not aligned with index')
            if not callable(getattr(model, 'scores', None)):
                raise ValueError(f'Extra model {name!r} has no scores(user_positions) method')
        if not np.isfinite(self.prior) or self.prior < 0:
            raise ValueError('prior must be finite and non-negative')
        metadata = users_metadata.copy()
        if 'user_id' in metadata.columns:
            metadata = metadata.set_index('user_id')
        if not metadata.index.is_unique:
            raise ValueError('Duplicate user metadata IDs')
        self.metadata = metadata
        self.global_probability = np.asarray(index.x.sum(axis=0)).ravel().astype(np.float64)
        self.global_probability /= self.global_probability.sum()
        self.global_probability = self.global_probability.astype(np.float32)
        self.category_tables = {}
        self.category_indexes = {}
        if self.conditional:
            prefix_metadata = metadata.reindex(index.users)
            for source, column in ATTRIBUTES.items():
                if column not in prefix_metadata:
                    self.category_tables[source] = np.empty((0, len(index.items)), np.float32)
                    self.category_indexes[source] = {}
                    continue
                values = prefix_metadata[column]
                codes, categories = pd.factorize(values, sort=True, use_na_sentinel=True)
                known = np.flatnonzero(codes >= 0)
                membership = sparse.csr_matrix(
                    (np.ones(len(known), np.float32), (codes[known], known)),
                    shape=(len(categories), len(index.users)))
                counts = (membership @ index.x).toarray().astype(np.float64)
                denominator = counts.sum(axis=1, keepdims=True) + self.prior
                counts += self.prior * self.global_probability[None, :]
                # Prefix categories all have interactions; the max also makes
                # prior=0 safe for empty category tables.
                counts /= np.maximum(denominator, np.finfo(float).tiny)
                self.category_tables[source] = counts.astype(np.float32)
                self.category_indexes[source] = {value: i for i, value in enumerate(categories)}

    def __getattr__(self, name):
        index = self.__dict__.get('index')
        if index is None:
            raise AttributeError(name)
        return getattr(index, name)

    def scores(self, users):
        users = list(users)
        result = self.index.scores(users)
        if self.extra_models:
            positions = np.asarray([self.index.uindex.get(uid, -1) for uid in users], dtype=np.int64)
            for name, model in self.extra_models.items():
                values = np.asarray(model.scores(positions), dtype=np.float32)
                if values.shape != (len(users), len(self.items)):
                    raise ValueError(f'Extra model {name!r} returned incorrect score shape')
                if not np.isfinite(values).all():
                    raise ValueError(f'Extra model {name!r} returned non-finite scores')
                if (positions < 0).any():
                    values = values.copy()
                    values[positions < 0] = 0
                result[name] = values
        if self.conditional:
            query_metadata = self.metadata.reindex(users)
            for source, column in ATTRIBUTES.items():
                values = np.broadcast_to(self.global_probability,
                                         (len(users), len(self.items))).copy()
                if column in query_metadata:
                    category_index = self.category_indexes[source]
                    rows = np.asarray([category_index.get(v, -1)
                                       if pd.notna(v) else -1
                                       for v in query_metadata[column]], dtype=np.int64)
                    known = rows >= 0
                    values[known] = self.category_tables[source][rows[known]]
                result[source] = values
        return result

    score_users = scores

    def candidates(self, users, budgets=None):
        users = np.asarray(sorted(set(users)))
        batch_size = int(self.index.config.get('max_batch', 256))
        if batch_size < 1:
            raise ValueError('max_batch must be positive')
        if budgets is not None:
            budgets = dict(budgets)
            if any(not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 0
                   for n in budgets.values()):
                raise ValueError('Candidate budgets must be non-negative integers')
        parts = []
        output_columns = None
        for start in range(0, len(users), batch_size):
            batch = users[start:start + batch_size]
            scores = self.scores(batch)
            selected_budgets = {source: 60 for source in scores} if budgets is None else budgets
            unavailable = set(selected_budgets) - set(scores)
            if unavailable:
                raise ValueError(f'Scorers unavailable: {sorted(unavailable)}')
            profiles, positions = self.index._profiles(batch)
            seen = profiles.toarray() != 0
            selected = np.zeros(seen.shape, dtype=bool)
            ranks = {}
            for name, values in scores.items():
                valid = np.isfinite(values) & ~seen
                if name in {'bm25', 'rp3beta'}:
                    valid &= values > 0
                if name not in {'pop', *ATTRIBUTES}:
                    valid &= positions[:, None] >= 0
                if name == 'ease':
                    allowed = np.zeros(len(self.items), dtype=bool)
                    allowed[self.index.ease_columns] = True
                    valid &= allowed[None, :]
                # Items are sorted IDs. Stable descending order therefore uses
                # item ID to resolve ties, exactly matching core candidates().
                order = np.argsort(-np.where(valid, values, -np.inf), axis=1, kind='stable')
                rank = np.empty(values.shape, dtype=np.int32)
                np.put_along_axis(rank, order,
                                  np.broadcast_to(np.arange(1, len(self.items) + 1, dtype=np.int32),
                                                  values.shape), axis=1)
                rank[~valid] = len(self.items) + 1
                ranks[name] = rank
                if name in selected_budgets:
                    selected |= valid & (rank <= selected_budgets[name])
            rows, columns = np.where(selected)
            data = {'user_id': batch[rows], 'item_id': self.items[columns]}
            for name, values in scores.items():
                data[name + '_score'] = values[rows, columns].astype(float)
                data[name + '_rank'] = ranks[name][rows, columns].astype(np.int64)
            data['cf_score'] = data['bm25_score']
            data['cf_rank'] = data['bm25_rank']
            output_columns = list(data)
            parts.append(pd.DataFrame(data))
        if parts:
            return pd.concat(parts, ignore_index=True)
        # Keep an empty query result usable by downstream schema-aware code.
        if output_columns is None:
            score_names = list(self.scores([]))
            output_columns = ['user_id', 'item_id']
            for name in score_names:
                output_columns.extend([name + '_score', name + '_rank'])
            output_columns.extend(['cf_score', 'cf_rank'])
        return pd.DataFrame(columns=output_columns)
