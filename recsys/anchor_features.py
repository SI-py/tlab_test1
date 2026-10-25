"""Stable item-identity landmarks for rotation-invariant ALS geometry features.

Select IDs ONCE on the first declared training prefix and save them in the
experiment manifest. Later prefixes reuse these IDs with their current factors;
they never reselect, align using future data, or fit a coordinate transform.

For user u, candidate i and fixed item anchor a, expose u·a, cos(u,a),
cos(i,a), cos(u,a)*cos(i,a), plus anchor availability. All are invariant under
the SAME orthogonal transformation of user, item and anchor latent factors.
They need not stay numerically identical across independently fitted prefixes:
the underlying user/item preferences may genuinely change.

This is an inference-time landmark representation, not ProtoMF training and not
a Nyström kernel approximation (no landmark Gram inverse/whitening is applied).
Related primary sources:
https://hcai.at/publications/2022_protomf/
https://proceedings.neurips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def _latent_blocks(model):
    users, items = np.asarray(model.users), np.asarray(model.items)
    if (users.ndim != 1 or items.ndim != 1 or not pd.Index(users).is_unique
            or not pd.Index(items).is_unique):
        raise ValueError('model users/items must be unique ID vectors')
    uf, vf = np.asarray(model.user_factors), np.asarray(model.item_factors)
    config = model.config
    factors = int(config.get('factors', getattr(model, 'stats', {}).get('latent_factors', 32)))
    explicit = int(getattr(model, 'n_user_features', len(getattr(model, 'user_feature_names', ()))))
    begin = explicit if config.get('fit_features_together', False) else 0
    if (factors < 1 or begin < 0 or uf.ndim != 2 or vf.ndim != 2
            or uf.shape[0] != len(users) or vf.shape[0] != len(items)
            or uf.shape[1] != vf.shape[1] or begin+factors > uf.shape[1]):
        raise ValueError('model factors do not match the configured latent block and raw IDs')
    return users, items, uf[:, begin:begin+factors], vf[:, begin:begin+factors], begin


def _active_mask(model, name, size):
    values = getattr(model, name, None)
    result = np.ones(size, bool) if values is None else np.asarray(values, dtype=bool)
    if result.shape != (size,):
        raise ValueError(f'{name} must be a raw-ID-aligned boolean vector')
    return result


def _ordered_norms(vectors):
    """Row norms with fixed float64 factor-axis accumulation."""
    values = np.asarray(vectors, dtype=np.float64)
    squares = np.zeros(len(values), np.float64)
    for factor in range(values.shape[1]):
        squares += values[:, factor]*values[:, factor]
    return np.sqrt(squares)


def _ordered_dot_norms(vectors, anchors):
    """Batch-independent dots/norms; no GEMM/GEMV reduction dispatch.

    Every output cell accumulates factors in the same order in float64. This
    matters for tree splits: float32 BLAS rounding can change when inference
    has one query user whereas training materialization has hundreds.
    """
    values, basis = np.asarray(vectors, dtype=np.float64), np.asarray(anchors, dtype=np.float64)
    dots = np.zeros((len(values), len(basis)), np.float64)
    squares = np.zeros(len(values), np.float64)
    for factor in range(values.shape[1]):
        column = values[:, factor]
        dots += column[:, None]*basis[None, :, factor]
        squares += column*column
    return dots, np.sqrt(squares)


def select_anchors(model, pop, max_anchors=16, pool_size=1000):
    """Return frozen RAW item IDs from a prefix-popular, diverse latent pool.

    ``pop`` is aligned with model.items, or a Series keyed by raw IDs; extra
    Series IDs are ignored. Only active, finite, nonzero latent item rows are
    eligible. Pool order is decreasing popularity, then increasing raw item ID.
    Start with its first item, then maximize the nearest selected cosine
    distance. Numerical ties prefer that same stable pool order. Duplicate
    directions stop selection rather than adding redundant anchors.

    This function cannot infer data provenance: the caller must supply ONLY the
    first training-prefix model/popularity and persist the returned IDs.
    """
    if int(max_anchors) != max_anchors or int(pool_size) != pool_size or min(max_anchors, pool_size) < 1:
        raise ValueError('max_anchors and pool_size must be positive integers')
    _, items, _, vectors, _ = _latent_blocks(model)
    if isinstance(pop, pd.Series):
        if not pop.index.is_unique:
            raise ValueError('popularity Series must have unique raw item IDs')
        popularity = pop.reindex(items).fillna(0).to_numpy(dtype=np.float64)
    else:
        popularity = np.asarray(pop, dtype=np.float64)
    if popularity.shape != (len(items),) or not np.isfinite(popularity).all() or (popularity < 0).any():
        raise ValueError('popularity must be a finite, nonnegative vector aligned with raw items')
    active = _active_mask(model, 'active_items', len(items))
    norms = _ordered_norms(vectors)
    eligible = active & np.isfinite(vectors).all(axis=1) & np.isfinite(norms) & (norms > 1e-12)
    order = np.lexsort((items, -popularity))
    pool = order[eligible[order]][:int(pool_size)]
    if not len(pool):
        return items[:0].copy()
    unit = vectors[pool].astype(np.float64)/norms[pool, None]
    selected, nearest = [0], np.full(len(pool), np.inf)
    with threadpool_limits(limits=1, user_api='blas'):
        for _ in range(min(int(max_anchors), len(pool))-1):
            distance = np.clip(1-unit @ unit[selected[-1]], 0, 2)
            nearest = np.minimum(nearest, distance)
            nearest[selected] = -np.inf
            greatest = float(nearest.max())
            if greatest <= 1e-10:
                break
            # Preserve deterministic IDs under harmless float32 rotation noise.
            tied = np.flatnonzero(np.isclose(nearest, greatest, rtol=1e-6, atol=1e-8))
            selected.append(int(tied[0]))
    return items[pool[np.asarray(selected)]].copy()


class AnchorFeatures:
    """Current-prefix projections onto previously frozen item anchor IDs.

    ``build`` accepts raw user_id/item_id candidates, preserves their index and
    returns five float32 columns per anchor. Unknown candidate IDs raise.
    Missing, inactive or zero-vector anchors give zero geometry and available=0.
    Inactive users have zero user projections; inactive items have zero item
    cosines (and therefore zero products). No popularity/manual aggregates or
    labels enter this representation. Models must remain immutable after
    construction; construct a new wrapper for each fitted prefix.
    """

    def __init__(self, model, anchor_ids, chunk_rows=65536):
        users, items, self._users, self._items, begin = _latent_blocks(model)
        self._user_index, self._item_index = pd.Index(users), pd.Index(items)
        self.anchor_ids = np.asarray(anchor_ids).copy()
        if self.anchor_ids.ndim != 1 or not pd.Index(self.anchor_ids).is_unique:
            raise ValueError('anchor_ids must be a unique ordered raw-ID vector')
        if int(chunk_rows) != chunk_rows or chunk_rows < 1:
            raise ValueError('chunk_rows must be a positive integer')
        self.chunk_rows = int(chunk_rows)
        self._active_users = _active_mask(model, 'active_users', len(users))
        active_items = _active_mask(model, 'active_items', len(items))
        self.anchor_positions = self._item_index.get_indexer(self.anchor_ids)
        self.available = self.anchor_positions >= 0
        self._anchors = np.zeros((len(self.anchor_ids), self._items.shape[1]), np.float32)
        known = np.flatnonzero(self.available)
        if len(known):
            rows = self.anchor_positions[known]
            values = self._items[rows]
            norms = _ordered_norms(values)
            valid = active_items[rows] & np.isfinite(values).all(axis=1) & (norms > 1e-12)
            self.available[known] = valid
            self._anchors[known[valid]] = values[valid]
        self._anchor_norms = _ordered_norms(self._anchors)
        # Only items have a persistent projection cache (catalog x anchors).
        # User projections/norms are computed for unique query users per build.
        self._item_cosine = np.zeros((len(items), len(self.anchor_ids)), np.float32)
        with threadpool_limits(limits=1, user_api='blas'):
            for start in range(0, len(items), self.chunk_rows):
                stop = min(start+self.chunk_rows, len(items))
                values = self._items[start:stop]
                if not np.isfinite(values).all():
                    raise ValueError('item latent factors must be finite')
                dot, norms = _ordered_dot_norms(values, self._anchors)
                denom = norms[:, None]*self._anchor_norms[None, :]
                np.divide(dot, denom, out=self._item_cosine[start:stop], where=denom > 1e-12)
                self._item_cosine[start:stop][~active_items[start:stop]] = 0
        suffixes = ('user_dot', 'user_cos', 'item_cos', 'alignment_product', 'available')
        self.columns = tuple(f'learned_anchor_{item}_{name}' for item in self.anchor_ids for name in suffixes)
        self.feature_names = self.columns
        self.stats = {'anchors': len(self.anchor_ids), 'available_anchors': int(self.available.sum()),
                      'features': len(self.columns), 'latent_factors': self._items.shape[1],
                      'latent_offset': begin, 'source_cutoff': str(getattr(model, 'cutoff', 'unspecified')),
                      'item_projection_bytes': int(self._item_cosine.nbytes),
                      'owned_projection_bytes': int(self._item_cosine.nbytes+self._anchors.nbytes+
                                                    self._anchor_norms.nbytes+self.available.nbytes),
                      'user_projection_cache': 'query-only; no global user x anchor matrix',
                      'chunk_rows': self.chunk_rows,
                      'product': 'cos(user,anchor)*cos(item,anchor)',
                      'accumulation': 'fixed factor order in float64; final features float32',
                      'invariance': 'joint orthogonal rotations; prefix factors may otherwise change'}

    def build(self, candidates):
        if not {'user_id', 'item_id'}.issubset(candidates.columns):
            raise ValueError('candidates require raw user_id and item_id columns')
        user = self._user_index.get_indexer(candidates.user_id)
        item = self._item_index.get_indexer(candidates.item_id)
        if (user < 0).any() or (item < 0).any():
            raise ValueError('unknown candidate user/item ID; anchor features require raw-prefix IDs')
        output = np.zeros((len(candidates), len(self.columns)), np.float32)
        unique, inverse = np.unique(user, return_inverse=True)
        vectors = self._users[unique]
        if not np.isfinite(vectors).all():
            raise ValueError('query user latent factors must be finite')
        with threadpool_limits(limits=1, user_api='blas'):
            dots, norms = _ordered_dot_norms(vectors, self._anchors)
            dots[~self._active_users[unique]] = 0
            denom = norms[:, None]*self._anchor_norms[None, :]
            cosine = np.zeros_like(dots)
            np.divide(dots, denom, out=cosine, where=denom > 1e-12)
            dots, cosine = dots.astype(np.float32), cosine.astype(np.float32)
            for start in range(0, len(candidates), self.chunk_rows):
                stop = min(start+self.chunk_rows, len(candidates))
                block = output[start:stop].reshape(stop-start, len(self.anchor_ids), 5)
                block[:, :, 0] = dots[inverse[start:stop]]
                block[:, :, 1] = cosine[inverse[start:stop]]
                block[:, :, 2] = self._item_cosine[item[start:stop]]
                block[:, :, 3] = block[:, :, 1]*block[:, :, 2]
                block[:, :, 4] = self.available
        return pd.DataFrame(output, index=candidates.index, columns=self.columns, copy=False)
