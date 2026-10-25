"""Prefix-only collaborative retrieval with bounded item-item working memory.

Implicit BM25 item inner products and RP3beta use block products pruned to top-k.
No global dense item-item Gram matrix is built. Optional EASE restricts its
dense Gram matrix to a configurable prefix-popular subset of the catalog.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from scipy import sparse


def _row_topk(matrix, k, diagonal_offset=0):
    matrix = matrix.tocsr()
    data, indices, indptr = [], [], [0]
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row:row + 2]
        vals, cols = matrix.data[start:end], matrix.indices[start:end]
        valid = (cols != row + diagonal_offset) & np.isfinite(vals) & (vals > 0)
        vals, cols = vals[valid], cols[valid]
        # Stable ties matter for reproducibility; item columns are sorted IDs.
        order = np.lexsort((cols, -vals))[:k]
        # CSR indices sorted, without disturbing the membership selected above.
        order = order[np.argsort(cols[order])]
        data.extend(vals[order]); indices.extend(cols[order]); indptr.append(len(data))
    return sparse.csr_matrix((np.asarray(data, np.float32),
                              np.asarray(indices, np.int32),
                              np.asarray(indptr, np.int64)), shape=matrix.shape)


def _csr_bytes(x):
    return x.data.nbytes + x.indices.nbytes + x.indptr.nbytes


class CollaborativeIndex:
    """Train only on timestamp < cutoff. Scores preserve input user order.

    Config keys: topk=128, block_size=128, bm25_k1=1.2, bm25_b=.8,
    rp3_alpha=1, rp3_beta=.6, als=False, als_factors=64,
    als_iterations=15, als_alpha=20, seed=42, threads=1, max_batch=256.
    Optional lightfm=True enables WARP with lightfm_factors=128,
    lightfm_epochs=15, lightfm_user_alpha=6e-5, lightfm_item_alpha=2e-5,
    lightfm_learning_rate=.01 and lightfm_max_sampled=150.
    Optional ease=True enables EASE on ease_items=5000 most frequent prefix
    items, with ease_l2=1000. Its score key and candidate budget key are 'ease'.
    Unknown users get zero collaborative scores and the same popularity prior.
    Item metadata is intentionally unnecessary for these collaborative methods.
    """
    def __init__(self, history, cutoff, config=None):
        started = time.perf_counter()
        self.config = dict(config or {})
        self.cutoff = pd.Timestamp(cutoff)
        timestamps = pd.to_datetime(history.timestamp)
        before = timestamps < self.cutoff
        history = history.loc[before, ['user_id', 'item_id', 'timestamp']].copy()
        history['timestamp'] = timestamps.loc[before]
        if history.empty:
            raise ValueError('No history strictly before cutoff')
        self.users = np.sort(history.user_id.unique())
        self.items = np.sort(history.item_id.unique())
        self.uindex = {u: j for j, u in enumerate(self.users)}
        self.iindex = {v: j for j, v in enumerate(self.items)}
        pairs = history[['user_id', 'item_id']].drop_duplicates()
        rows = pd.Index(self.users).get_indexer(pairs.user_id)
        cols = pd.Index(self.items).get_indexer(pairs.item_id)
        self.x = sparse.csr_matrix((np.ones(len(pairs), np.float32), (rows, cols)),
                                  shape=(len(self.users), len(self.items)))
        self.x.sort_indices()
        self.degree = np.asarray(self.x.sum(axis=0)).ravel()
        recent = history.loc[history.timestamp >= self.cutoff - pd.Timedelta(days=30), 'item_id'].value_counts()
        self.pop = recent.reindex(self.items, fill_value=0).to_numpy(dtype=np.float32)
        length = np.asarray(self.x.sum(axis=1)).ravel()
        from implicit.nearest_neighbours import bm25_weight
        # Exact implicit.BM25Recommender weighting: documents are items,
        # terms are users, and similarity is an unnormalised inner product.
        # In particular, neither cosine normalisation nor query IDF is applied.
        weighted = bm25_weight(
            self.x.T, K1=self.config.get('bm25_k1', 1.2),
            B=self.config.get('bm25_b', .8)).T.tocsr()
        # P(item -> user) and P(user -> item), with RP3 transition powers.
        alpha = self.config.get('rp3_alpha', 1.0)
        piu = (sparse.diags(1 / np.maximum(self.degree, 1)) @ self.x.T).tocsr()
        pui = (sparse.diags(1 / np.maximum(length, 1)) @ self.x).tocsr()
        piu.data **= alpha
        pui.data **= alpha
        penalty = sparse.diags(np.maximum(self.degree, 1) ** -self.config.get('rp3_beta', .6))
        topk = int(self.config.get('topk', 128))
        block_size = int(self.config.get('block_size', 128))
        if topk < 1 or block_size < 1:
            raise ValueError('topk and block_size must be positive')
        bm25_parts, rp3_parts, cosine2_parts = [], [], []
        inverse_degree = sparse.diags(1 / np.maximum(self.degree, 1))
        for start in range(0, len(self.items), block_size):
            stop = min(start + block_size, len(self.items))
            bm25_parts.append(_row_topk(weighted[:, start:stop].T @ weighted, topk, start))
            rp3_parts.append(_row_topk((piu[start:stop] @ pui) @ penalty, topk, start))
            # Squared binary cosine for last-history-item ranking features.
            # Approximation: all prefix interactions (not capped last 100),
            # with only strongest cosine_topk edges retained per source item.
            cooc = self.x[:, start:stop].T @ self.x
            cosine2 = (sparse.diags(1 / np.maximum(self.degree[start:stop], 1)) @
                       cooc.power(2) @ inverse_degree)
            cosine2_parts.append(_row_topk(
                cosine2, int(self.config.get('cosine_topk', 256)), start))
        self.bm25 = sparse.vstack(bm25_parts, format='csr')
        self.rp3beta = sparse.vstack(rp3_parts, format='csr')
        self.cosine2 = sparse.vstack(cosine2_parts, format='csr')
        del bm25_parts, rp3_parts, cosine2_parts, weighted, piu, pui, cooc, cosine2
        self.als = None
        if self.config.get('als', False):
            from implicit.cpu.als import AlternatingLeastSquares
            self.als = AlternatingLeastSquares(
                factors=int(self.config.get('als_factors', 64)),
                iterations=int(self.config.get('als_iterations', 15)),
                regularization=float(self.config.get('als_regularization', .05)),
                alpha=float(self.config.get('als_alpha', 20)),
                random_state=int(self.config.get('seed', 42)),
                num_threads=int(self.config.get('threads', 1)), dtype=np.float32)
            self.als.fit(self.x, show_progress=False)
        self.lightfm = None
        if self.config.get('lightfm', False):
            from lightfm import LightFM
            self.lightfm = LightFM(
                loss='warp', no_components=int(self.config.get('lightfm_factors', 128)),
                user_alpha=float(self.config.get('lightfm_user_alpha', 6e-5)),
                item_alpha=float(self.config.get('lightfm_item_alpha', 2e-5)),
                learning_rate=float(self.config.get('lightfm_learning_rate', .01)),
                max_sampled=int(self.config.get('lightfm_max_sampled', 150)),
                random_state=int(self.config.get('seed', 42)))
            self.lightfm.fit(self.x, epochs=int(self.config.get('lightfm_epochs', 15)),
                             num_threads=int(self.config.get('threads', 1)), verbose=False)
        self.ease_matrix = None
        self.ease_columns = np.empty(0, dtype=np.int64)
        if self.config.get('ease', False):
            from scipy.linalg import cho_factor, cho_solve
            # Steck (WWW 2019), https://arxiv.org/abs/1905.03375.
            count = int(self.config.get('ease_items', 5000))
            regularization = float(self.config.get('ease_l2', 1000))
            if count < 1 or not np.isfinite(regularization) or regularization <= 0:
                raise ValueError('ease_items and ease_l2 must be positive')
            # Catalog selection depends exclusively on prefix interaction counts;
            # stable ID ties and sorted selected columns make it reproducible.
            self.ease_columns = np.sort(np.lexsort((self.items, -self.degree))[:count])
            selected_x = self.x[:, self.ease_columns].astype(np.float64)
            gram = (selected_x.T @ selected_x).toarray(order='F')
            diagonal = np.diag_indices_from(gram)
            gram[diagonal] += regularization
            # Float64 Cholesky and inverse avoid unstable single-precision solves.
            # Reuse the two dense arrays: factor and identity/inverse, then keep
            # only float32 B. No full-catalog dense Gram is allocated.
            factor = cho_factor(gram, lower=True, overwrite_a=True, check_finite=False)
            inverse = cho_solve(factor, np.eye(len(self.ease_columns), order='F'),
                                overwrite_b=True, check_finite=False)
            inverse /= -np.diag(inverse).copy()[None, :]
            inverse[diagonal] = 0
            self.ease_matrix = inverse.astype(np.float32)
            del selected_x, gram, factor, inverse
        factor_bytes = 0 if self.als is None else self.als.user_factors.nbytes + self.als.item_factors.nbytes
        if self.lightfm is not None:
            factor_bytes += sum(a.nbytes for a in [self.lightfm.user_embeddings,
                self.lightfm.item_embeddings, self.lightfm.user_biases, self.lightfm.item_biases])
        if self.ease_matrix is not None:
            factor_bytes += self.ease_matrix.nbytes + self.ease_columns.nbytes
        self.stats = {'fit_seconds': time.perf_counter() - started,
                      'prefix_events': len(history), 'unique_pairs': self.x.nnz,
                      'users': len(self.users), 'items': len(self.items),
                      'stored_matrix_bytes': _csr_bytes(self.x) + _csr_bytes(self.bm25) + _csr_bytes(self.rp3beta) + _csr_bytes(self.cosine2) + factor_bytes,
                      'bm25_edges': self.bm25.nnz, 'rp3beta_edges': self.rp3beta.nnz,
                      'cosine2_edges': self.cosine2.nnz,
                      'ease_items': len(self.ease_columns)}

    def _profiles(self, user_ids):
        positions = np.asarray([self.uindex.get(u, -1) for u in user_ids], dtype=np.int64)
        # Gather known rows through an indicator matrix, giving unknowns zero rows.
        known = np.flatnonzero(positions >= 0)
        selector = sparse.csr_matrix((np.ones(len(known), np.float32),
                                      (known, positions[known])),
                                     shape=(len(positions), len(self.users)))
        return selector @ self.x, positions

    def scores(self, user_ids):
        user_ids = list(user_ids)
        if len(user_ids) > int(self.config.get('max_batch', 256)):
            raise ValueError('scores batch exceeds max_batch; use candidates for automatic batching')
        profiles, positions = self._profiles(user_ids)
        result = {'bm25': (profiles @ self.bm25).toarray().astype(np.float32),
                  'rp3beta': (profiles @ self.rp3beta).toarray().astype(np.float32),
                  'pop': np.broadcast_to(self.pop, (len(user_ids), len(self.items))).copy()}
        if self.als is not None:
            factors = np.zeros((len(user_ids), self.als.user_factors.shape[1]), np.float32)
            valid = positions >= 0
            factors[valid] = self.als.user_factors[positions[valid]]
            result['als'] = factors @ self.als.item_factors.T
        if self.lightfm is not None:
            values = np.zeros((len(user_ids), len(self.items)), np.float32)
            valid = positions >= 0
            values[valid] = (self.lightfm.user_embeddings[positions[valid]] @
                            self.lightfm.item_embeddings.T +
                            self.lightfm.user_biases[positions[valid], None] +
                            self.lightfm.item_biases[None, :])
            result['lightfm'] = values
        if self.ease_matrix is not None:
            values = np.zeros((len(user_ids), len(self.items)), np.float32)
            values[:, self.ease_columns] = profiles[:, self.ease_columns] @ self.ease_matrix
            result['ease'] = values
        return result

    score_users = scores

    def candidates(self, users, budgets=None):
        budgets = dict(budgets or {'bm25': 60, 'rp3beta': 60, 'pop': 60,
                                   **({'als': 60} if self.als is not None else {}),
                                   **({'lightfm': 60} if self.lightfm is not None else {}),
                                   **({'ease': 60} if self.ease_matrix is not None else {})})
        users = sorted(set(users))
        rows = []
        batch_size = int(self.config.get('max_batch', 256))
        for start in range(0, len(users), batch_size):
            batch = users[start:start + batch_size]
            scores = self.scores(batch)
            unknown = set(budgets) - set(scores)
            if unknown:
                raise ValueError(f'Scorers unavailable: {sorted(unknown)}')
            profiles, _ = self._profiles(batch)
            for offset, uid in enumerate(batch):
                seen = profiles.getrow(offset).indices
                ranking, ranks = {}, {}
                for name, matrix in scores.items():
                    values = matrix[offset].copy()
                    values[seen] = -np.inf
                    order = np.lexsort((self.items, -values))
                    order = order[np.isfinite(values[order])]
                    # Sparse graph zero scores provide no evidence. Latent-model
                    # scores have arbitrary origins, so negative values are valid.
                    if name in {'bm25', 'rp3beta'}:
                        order = order[values[order] > 0]
                    if name == 'ease':
                        # Zero-filled scores outside the fitted EASE catalog
                        # must not outrank genuine negative EASE predictions.
                        order = order[np.isin(order, self.ease_columns)]
                    if name != 'pop' and uid not in self.uindex:
                        order = order[:0]
                    ranking[name] = order
                    ranks[name] = {int(i): rank + 1 for rank, i in enumerate(order)}
                selected = set()
                for name, budget in budgets.items():
                    if budget < 0:
                        raise ValueError('Candidate budgets must be non-negative')
                    selected.update(ranking[name][:budget].tolist())
                for item in sorted(selected):
                    row = {'user_id': uid, 'item_id': self.items[item]}
                    for name, matrix in scores.items():
                        row[name + '_score'] = float(matrix[offset, item])
                        row[name + '_rank'] = ranks[name].get(item, len(self.items) + 1)
                    row['cf_score'] = row['bm25_score']
                    row['cf_rank'] = row['bm25_rank']
                    rows.append(row)
        return pd.DataFrame(rows)
