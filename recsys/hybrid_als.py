"""Binary-prefix iALS with explicit features, following the RecTools recipe.

Reference: MobileTeleSystems/RecTools, commit
5f365cc546f3b54f39fe6367df26bd0b9178e9e1, ``implicit_als.py`` and example 5.
https://rectools.readthedocs.io/en/latest/examples/5_benchmark_iALS_with_features.html

For separate training, fit ordinary iALS, then independently regress the
interaction preferences onto the fixed features of the opposite side. For
joint training the vectors are U=[F_user, latent_U, paired_U] and
V=[paired_V, latent_V, F_item]. Each native ALS half-step optimizes ALL columns,
then restores the corresponding fixed columns, exactly as in RecTools. This
overwrite procedure is not an exact constrained-coordinate minimization.
With all-one features the dot product contains learned user and item biases.

Only binary user-item membership is used: positive confidence is alpha,
unobserved confidence is 1, and preferences are 1/0. The caller must supply a
strictly pre-cutoff matrix; this helper never accepts timestamps or labels.
The published benchmark's duration filter and watched-percent weights are
deliberately absent because those aggregate fields cannot be time-truncated.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits


def categorical_features(ids, metadata, fields, *, multiple=()):
    """Raw 0/1 columns, fitted only on the supplied prefix IDs, never normalized.

    Missing metadata gets an explicit category. ``multiple`` fields contain
    comma-separated tokens (KION genres). No vocabulary is learned from rows
    outside ``ids``. Column names and category order are deterministic.
    """
    ids = np.asarray(ids)
    if metadata is None:
        raise ValueError('metadata feature mode requires metadata frames')
    if not metadata.index.is_unique:
        raise ValueError('metadata index must contain unique IDs')
    missing = set(fields) - set(metadata.columns)
    if missing:
        raise ValueError(f'missing metadata fields: {sorted(missing)}')
    frame = metadata.reindex(ids)
    columns, names = [], []
    for field in fields:
        values = frame[field].fillna('<missing>').astype(str)
        if field in multiple:
            tokens = [[s.strip().casefold() for s in value.split(',') if s.strip()]
                      or ['<missing>'] for value in values]
        else:
            tokens = [[value] for value in values]
        vocabulary = sorted({token for row in tokens for token in row})
        lookup = {value: j for j, value in enumerate(vocabulary)}
        encoded = np.zeros((len(ids), len(vocabulary)), np.float32)
        rows = np.repeat(np.arange(len(ids)), [len(set(row)) for row in tokens])
        cols = [lookup[value] for row in tokens for value in sorted(set(row))]
        encoded[rows, cols] = 1
        columns.append(encoded)
        names.extend(f'{field}={value}' for value in vocabulary)
    return (np.concatenate(columns, axis=1) if columns else
            np.empty((len(ids), 0), np.float32)), tuple(names)


def _explicit_features(values, size):
    if values is None:
        return np.empty((size, 0), np.float32)
    if sparse.issparse(values):
        values = values.toarray()
    result = np.asarray(values, dtype=np.float32, order='C')
    if result.ndim != 2 or len(result) != size or not np.isfinite(result).all():
        raise ValueError('explicit features must be finite, aligned 2D matrices')
    return result


class HybridALS:
    """Reusable prefix-only factor model. Scores take matrix ROW POSITIONS.

    Config: factors=32, iterations=10, alpha=10, regularization=.5, seed=32,
    threads=1, features='bias' ('none'/'bias'/'metadata'),
    fit_features_together=False, use_cg=True, cg_steps=3, max_batch=1024.
    ``none`` is ordinary implicit ALS. ``metadata`` uses sex/age/income and
    genres/content_type, the fields in the RecTools KION benchmark. Explicit
    ``user_features`` / ``item_features`` arguments override that construction.
    No identity one-hots, features from future-only IDs, or labels are added.
    """

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.user_factors = self.item_factors = None

    def fit(self, x, *, users=None, items=None, user_metadata=None,
            item_metadata=None, user_features=None, item_features=None):
        from implicit.cpu.als import AlternatingLeastSquares

        started = time.perf_counter()
        factors = int(self.config.get('factors', 32))
        iterations = int(self.config.get('iterations', 10))
        alpha = float(self.config.get('alpha', 10))
        regularization = float(self.config.get('regularization', .5))
        if (factors < 1 or iterations < 1 or not np.isfinite(alpha) or alpha <= 0
                or not np.isfinite(regularization) or regularization <= 0):
            raise ValueError('factors, iterations, alpha and regularization must be positive')
        binary = sparse.csr_matrix(x, dtype=np.float32, copy=True)
        binary.sum_duplicates()
        binary.eliminate_zeros()
        if not np.isfinite(binary.data).all() or (binary.data < 0).any():
            raise ValueError('x must contain finite, nonnegative prefix memberships')
        binary.data.fill(1)
        binary.sort_indices()
        n_users, n_items = binary.shape
        if not n_users or not n_items or not binary.nnz:
            raise ValueError('x must contain at least one user, item and interaction')
        self.users = np.arange(n_users) if users is None else np.asarray(users).copy()
        self.items = np.arange(n_items) if items is None else np.asarray(items).copy()
        if (self.users.shape != (n_users,) or self.items.shape != (n_items,)
                or len(np.unique(self.users)) != n_users
                or len(np.unique(self.items)) != n_items):
            raise ValueError('users/items must be unique IDs aligned with x')
        mode = self.config.get('features', 'bias')
        self.user_feature_names = self.item_feature_names = ()
        if user_features is None and item_features is None:
            if mode == 'none':
                pass
            elif mode == 'bias':
                user_features = np.ones((n_users, 1), np.float32)
                item_features = np.ones((n_items, 1), np.float32)
                self.user_feature_names = self.item_feature_names = ('constant_bias',)
            elif mode == 'metadata':
                user_features, self.user_feature_names = categorical_features(
                    self.users, user_metadata, ['sex', 'age', 'income'])
                item_features, self.item_feature_names = categorical_features(
                    self.items, item_metadata, ['genres', 'content_type'], multiple=['genres'])
            else:
                raise ValueError("features must be 'none', 'bias' or 'metadata'")
        fu = _explicit_features(user_features, n_users)
        fi = _explicit_features(item_features, n_items)
        du, di = fu.shape[1], fi.shape[1]
        model = AlternatingLeastSquares(
            factors=factors, iterations=iterations, alpha=alpha,
            regularization=regularization, dtype=np.float32,
            random_state=int(self.config.get('seed', 32)),
            num_threads=int(self.config.get('threads', 1)),
            use_cg=bool(self.config.get('use_cg', True)))
        model.cg_steps = int(self.config.get('cg_steps', 3))
        if model.cg_steps < 1:
            raise ValueError('cg_steps must be positive')
        together = bool(self.config.get('fit_features_together', False)) and (du + di > 0)
        with threadpool_limits(limits=1, user_api='blas'):
            if together:
                # Float64 generation followed by float32 cast matches RecTools.
                rng = np.random.default_rng(int(self.config.get('seed', 32)))
                uf = np.concatenate((fu, rng.random((n_users, factors)) * .01,
                                     np.zeros((n_users, di))), axis=1).astype(np.float32)
                vf = np.concatenate((np.zeros((n_items, du)),
                                     rng.random((n_items, factors)) * .01, fi), axis=1).astype(np.float32)
                confidence = alpha * binary
                transposed = confidence.T.tocsr()
                for _ in range(iterations):
                    model.solver(confidence, uf, vf, regularization, model.num_threads)
                    if du:
                        uf[:, :du] = fu
                    model.solver(transposed, vf, uf, regularization, model.num_threads)
                    if di:
                        vf[:, -di:] = fi
            else:
                model.fit(binary, show_progress=False)
                user_chunks, item_chunks = [model.user_factors], [model.item_factors]
                confidence = alpha * binary
                # The opposite-side feature matrices remain fixed for this
                # half-step; no second alpha multiplication or latent residual.
                if du:
                    paired = np.zeros((n_items, du), np.float32)
                    model.solver(confidence.T.tocsr(), paired, fu,
                                 regularization, model.num_threads)
                    user_chunks.append(fu)
                    item_chunks.append(paired)
                if di:
                    paired = np.zeros((n_users, di), np.float32)
                    model.solver(confidence, paired, fi,
                                 regularization, model.num_threads)
                    user_chunks.append(paired)
                    item_chunks.append(fi)
                uf = np.concatenate(user_chunks, axis=1)
                vf = np.concatenate(item_chunks, axis=1)
        if not np.isfinite(uf).all() or not np.isfinite(vf).all():
            raise FloatingPointError('ALS produced non-finite factors')
        self.user_factors, self.item_factors = uf, vf
        self.n_user_features, self.n_item_features = du, di
        self.stats = {'fit_seconds': time.perf_counter() - started,
                      'users': n_users, 'items': n_items, 'unique_pairs': binary.nnz,
                      'latent_factors': factors, 'user_features': du, 'item_features': di,
                      'total_factors': uf.shape[1],
                      'fit_features_together': bool(together),
                      'stored_matrix_bytes': int(uf.nbytes + vf.nbytes),
                      'confidence': 'observed=alpha, missing=1; binary preferences',
                      'optimizer': 'RecTools overwrite' if together else 'independent paired features'}
        return self

    def scores(self, user_positions):
        """Dense user-by-item scores; -1 denotes an unknown user (all zeros)."""
        if self.user_factors is None:
            raise RuntimeError('fit must precede scores')
        positions = np.asarray(user_positions)
        if positions.ndim != 1 or (positions.size and positions.dtype.kind not in 'iu'):
            raise ValueError('user_positions must be an integer vector')
        positions = positions.astype(np.int64)
        if (positions < -1).any() or (positions >= len(self.users)).any():
            raise IndexError('user position out of range; only -1 denotes unknown')
        if len(positions) > int(self.config.get('max_batch', 1024)):
            raise ValueError('scores batch exceeds max_batch')
        result = np.zeros((len(positions), len(self.items)), np.float32)
        valid = positions >= 0
        result[valid] = self.user_factors[positions[valid]] @ self.item_factors.T
        return result


class HybridLightFM:
    """LightFM with identity plus unnormalized KION metadata features.

    The sparse feature construction follows RecTools' LightFM wrapper. Static
    side information can share representations across identities. Inference
    stores the summed vectors and biases, dropping optimizer accumulators.
    Config: factors=64, epochs=10, loss='warp', learning_rate=.01,
    user_alpha=6e-5, item_alpha=2e-5, max_sampled=150, seed=42, threads=1,
    features='metadata' ('none' is the identity-only control), max_batch=1024.
    """

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.user_factors = self.item_factors = None

    def fit(self, x, *, users=None, items=None, user_metadata=None, item_metadata=None,
            callback=None):
        from lightfm import LightFM

        started = time.perf_counter()
        x = sparse.csr_matrix(x, dtype=np.float32, copy=True)
        x.sum_duplicates()
        x.eliminate_zeros()
        if not x.nnz or not np.isfinite(x.data).all() or (x.data < 0).any():
            raise ValueError('x must contain finite, nonnegative prefix memberships')
        x.data.fill(1)
        x.sort_indices()
        n_users, n_items = x.shape
        self.users = np.arange(n_users) if users is None else np.asarray(users).copy()
        self.items = np.arange(n_items) if items is None else np.asarray(items).copy()
        if (self.users.shape != (n_users,) or self.items.shape != (n_items,)
                or len(np.unique(self.users)) != n_users
                or len(np.unique(self.items)) != n_items):
            raise ValueError('users/items must be unique IDs aligned with x')
        mode = self.config.get('features', 'metadata')
        self.user_feature_names = self.item_feature_names = ()
        fu = sparse.eye(n_users, format='csr', dtype=np.float32)
        fi = sparse.eye(n_items, format='csr', dtype=np.float32)
        if mode == 'metadata':
            explicit, self.user_feature_names = categorical_features(
                self.users, user_metadata, ['sex', 'age', 'income'])
            fu = sparse.hstack((fu, sparse.csr_matrix(explicit)), format='csr')
            explicit, self.item_feature_names = categorical_features(
                self.items, item_metadata, ['genres', 'content_type'], multiple=['genres'])
            fi = sparse.hstack((fi, sparse.csr_matrix(explicit)), format='csr')
            del explicit
        elif mode != 'none':
            raise ValueError("LightFM features must be 'none' or 'metadata'")
        factors, epochs = int(self.config.get('factors', 64)), int(self.config.get('epochs', 10))
        threads = int(self.config.get('threads', 1))
        if min(factors, epochs, threads) < 1:
            raise ValueError('factors, epochs and threads must be positive')
        model = LightFM(
            no_components=factors, loss=self.config.get('loss', 'warp'),
            learning_rate=float(self.config.get('learning_rate', .01)),
            user_alpha=float(self.config.get('user_alpha', 6e-5)),
            item_alpha=float(self.config.get('item_alpha', 2e-5)),
            max_sampled=int(self.config.get('max_sampled', 150)),
            random_state=int(self.config.get('seed', 42)))
        with threadpool_limits(limits=1, user_api='blas'):
            if callback is None:
                model.fit(x, user_features=fu, item_features=fi, epochs=epochs,
                          num_threads=threads, verbose=False)
            else:
                for epoch in range(1, epochs + 1):
                    model.fit_partial(x, user_features=fu, item_features=fi, epochs=1,
                                      num_threads=threads, verbose=False)
                    self.user_biases, self.user_factors = model.get_user_representations(fu)
                    self.item_biases, self.item_factors = model.get_item_representations(fi)
                    self.stats = {'epochs': epoch, 'fit_seconds': time.perf_counter()-started}
                    callback(self, epoch)
            self.user_biases, self.user_factors = model.get_user_representations(fu)
            self.item_biases, self.item_factors = model.get_item_representations(fi)
        self.stats = {'fit_seconds': time.perf_counter() - started,
                      'users': n_users, 'items': n_items, 'unique_pairs': x.nnz,
                      'latent_factors': factors, 'epochs': epochs,
                      'user_features': len(self.user_feature_names),
                      'item_features': len(self.item_feature_names),
                      'feature_normalization': 'none; identity plus raw 0/1 metadata',
                      'stored_matrix_bytes': int(sum(a.nbytes for a in [
                          self.user_factors, self.item_factors, self.user_biases, self.item_biases]))}
        return self

    def scores(self, user_positions):
        """Dot product plus BOTH learned bias terms; -1 gives an unknown zero row."""
        if self.user_factors is None:
            raise RuntimeError('fit must precede scores')
        positions = np.asarray(user_positions)
        if positions.ndim != 1 or (positions.size and positions.dtype.kind not in 'iu'):
            raise ValueError('user_positions must be an integer vector')
        positions = positions.astype(np.int64)
        if (positions < -1).any() or (positions >= len(self.users)).any():
            raise IndexError('user position out of range; only -1 denotes unknown')
        if len(positions) > int(self.config.get('max_batch', 1024)):
            raise ValueError('scores batch exceeds max_batch')
        result = np.zeros((len(positions), len(self.items)), np.float32)
        valid = positions >= 0
        result[valid] = (self.user_factors[positions[valid]] @ self.item_factors.T +
                         self.user_biases[positions[valid], None] + self.item_biases[None, :])
        return result
