"""Winner-inspired temporal features with global item and local user aggregation.

No duration/watch-percent data is accepted. Histories must be strict prefixes.
Expensive exploded relations are restricted to users present in the query.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse

MANUAL_WINDOWS = (1, 7, 30, 0)
AUTO_WINDOWS = (2, 3, 5, 10, 14, 21, 45, 60, 90)


def required_windows(columns):
    """Generated windows referenced by existing named feature columns."""
    if columns is None:
        return AUTO_WINDOWS
    wanted = set(columns)
    selected = []
    for window in AUTO_WINDOWS:
        names = {f'{entity}_{kind}_{window}'
                 for entity in ('user', 'item') for kind in ('count', 'share')}
        names.update(f'affinity_{relation}_{window}{suffix}'
                     for relation in ('genre', 'country', 'type') for suffix in ('', '_share'))
        names.update(f'{entity}_completed_engaged_count_{window}{suffix}'
                     for entity in ('user', 'item') for suffix in ('', '_share'))
        if wanted & names:
            selected.append(window)
    return tuple(selected)


def feature_request(store, columns, with_embeddings):
    if columns is not None and store.required_columns is not None:
        missing = set(columns) - set(store.required_columns)
        if missing:
            names = ', '.join(sorted(missing)[:5])
            raise ValueError(
                f'Feature request exceeds the constructor profile ({names}). '
                'Create a new store with the union of required_columns for all requests, '
                'or use required_columns=None in the constructor for a full profile.')
    columns = store.required_columns if columns is None else tuple(columns)
    if columns is not None:
        with_embeddings = with_embeddings and any(
            c.startswith(('learned_', 'latent_product_')) for c in columns)
    return columns, with_embeddings


def select_columns(frame, cats, groups, columns):
    if columns is None:
        return frame, cats, groups
    selected = [c for c in columns if c in frame]
    wanted = set(selected)
    return frame.loc[:, selected], [c for c in cats if c in wanted], {
        name: [c for c in values if c in wanted] for name, values in groups.items()}


def canonical(series):
    if pd.api.types.is_numeric_dtype(series):
        series = series.astype(float)
    return series.fillna('unknown').astype(str)


class TemporalFeatureStore:
    def __init__(self, events, items, users, index, windows=None, required_columns=None):
        self.required_columns = None if required_columns is None else tuple(required_columns)
        if required_columns is not None and windows is None:
            windows = required_windows(required_columns)
        self.index = index
        self.cutoff = index.cutoff
        self.history = events.loc[events.timestamp < self.cutoff, ['user_id', 'item_id', 'timestamp']].copy()
        self.items = items
        self.users = users
        self.windows = tuple(dict.fromkeys(MANUAL_WINDOWS + tuple(AUTO_WINDOWS if windows is None else windows)))
        self.item_pos = pd.Index(index.items)
        self.user_pos = pd.Index(index.users)
        self.hi = self.item_pos.get_indexer(self.history.item_id)
        self.hu = self.user_pos.get_indexer(self.history.user_id)
        if np.any(self.hi < 0) or np.any(self.hu < 0):
            raise ValueError('Collaborative index and feature-store prefix disagree')
        self.age = ((self.cutoff - self.history.timestamp).dt.total_seconds() / 86400).to_numpy()
        self.item_stats, self.user_stats = {}, {}
        for window in self.windows:
            mask = np.ones(len(self.age), bool) if not window else self.age <= window
            self.item_stats[window] = np.bincount(self.hi[mask], minlength=len(index.items)).astype(np.float32)
            self.user_stats[window] = np.bincount(self.hu[mask], minlength=len(index.users)).astype(np.float32)
        last = self.history.groupby('item_id').timestamp.agg(['min', 'max'])
        self.item_first = ((self.cutoff - last['min']).dt.total_seconds()/86400).reindex(index.items).to_numpy(np.float32)
        self.item_last = ((self.cutoff - last['max']).dt.total_seconds()/86400).reindex(index.items).to_numpy(np.float32)
        quant = self.history.groupby('item_id').timestamp.quantile(.95)
        self.item_q95 = ((self.cutoff - quant).dt.total_seconds()/86400).reindex(index.items).to_numpy(np.float32)
        ul = self.history.groupby('user_id').timestamp.max()
        self.user_last = ((self.cutoff - ul).dt.total_seconds()/86400).reindex(index.users).to_numpy(np.float32)
        # Last-week item popularity slope, ordinal time from oldest to newest.
        day = np.ceil(self.age).astype(int)
        recent = (day >= 1) & (day <= 7)
        daily = np.bincount(self.hi[recent]*7 + (7-day[recent]), minlength=len(index.items)*7).reshape(-1, 7)
        self.item_slope = (daily @ np.arange(-3, 4) / 28).astype(np.float32)
        # Proportions among preceding interactions only, smoothed by global prior.
        self.demo = {}
        for col in ['sex', 'age', 'income', 'kids_flg']:
            if col not in users:
                continue
            source = canonical(users[col])
            categories = sorted(set(source) - {'unknown'})
            mapped_users = pd.Series(index.users).map(source).fillna('unknown').to_numpy()
            event_values = mapped_users[self.hu]
            for token in categories:
                counts = np.bincount(self.hi[event_values == token], minlength=len(index.items)).astype(np.float32)
                prior = float(np.mean(event_values == token))
                self.demo[(col, token)] = (counts + 20*prior)/(self.item_stats[0] + 20)
        # Tokens map only prefix-eligible catalogue, avoiding future catalogue IDF.
        self.token_matrices = {}
        self.token_lists = {}
        for relation, col, limit in [('genre', 'genres', None), ('country', 'countries', None),
                                     ('director', 'directors', 3), ('actor', 'actors', 5), ('type', 'content_type', 1)]:
            if col not in items:
                continue
            lists = []
            for value in items[col].reindex(index.items).fillna('unknown').astype(str):
                tokens = sorted(set(t.strip() for t in value.split(',') if t.strip())) or ['unknown']
                lists.append(tokens if limit is None else tokens[:limit])
            tokens = {v:j for j,v in enumerate(sorted(set(t for row in lists for t in row)))}
            rows, cols = [], []
            for i, row in enumerate(lists):
                rows.extend([i]*len(row)); cols.extend(tokens[t] for t in row)
            matrix = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(index.items), len(tokens)))
            self.token_matrices[relation] = matrix
            self.token_lists[relation] = [matrix.getrow(i).indices for i in range(len(index.items))]
        if 'genre' in self.token_matrices:
            genre = self.token_matrices['genre']
            idf = np.log(len(index.items) / (np.asarray(genre.sum(axis=0)).ravel() + 1))
            g = genre @ sparse.diags(idf)
            norm = np.sqrt(np.asarray(g.power(2).sum(axis=1)).ravel())
            self.genre_normalized = (sparse.diags(1 / np.maximum(norm, 1e-12)) @ g).tocsr()
        else:
            self.genre_normalized = None

    def build(self, candidates, with_auto=True, with_history=True, with_embeddings=True, required_columns=None):
        required_columns, with_embeddings = feature_request(self, required_columns, with_embeddings)
        pairs = candidates.reset_index(drop=True)
        uid = self.user_pos.get_indexer(pairs.user_id)
        iid = self.item_pos.get_indexer(pairs.item_id)
        if np.any(uid < 0) or np.any(iid < 0):
            raise ValueError('Only prefix-observed warm user/item pairs are supported')
        unique_users = np.sort(pairs.user_id.unique())
        qpositions = self.user_pos.get_indexer(unique_users)
        local_u = pd.Index(unique_users).get_indexer(pairs.user_id)
        numeric, categorical, groups = {}, {}, defaultdict(list)
        for col in candidates:
            if col.endswith('_score') or col.endswith('_rank'):
                numeric[col] = pairs[col].to_numpy(np.float32)
                groups['retrieval'].append(col)
        for col in ['age', 'income', 'sex', 'kids_flg']:
            if col in self.users:
                categorical['user_' + col] = pairs.user_id.map(canonical(self.users[col])).fillna('unknown')
        for col in ['content_type', 'release_year', 'for_kids', 'age_rating', 'genres', 'countries', 'directors']:
            if col in self.items:
                categorical['item_' + col] = pairs.item_id.map(canonical(self.items[col])).fillna('unknown')
        categorical['item_identity'] = pairs.item_id.astype(str)
        for col, values in [('item_first_age', self.item_first[iid]), ('item_last_age', self.item_last[iid]),
                            ('item_q95_age', self.item_q95[iid]), ('item_slope7', self.item_slope[iid]),
                            ('user_last_age', self.user_last[uid])]:
            numeric[col] = values
            groups['manual'].append(col)
        windows = self.windows if with_auto else MANUAL_WINDOWS
        if required_columns is not None:
            needed = set(MANUAL_WINDOWS + required_windows(required_columns))
            windows = tuple(w for w in windows if w in needed)
        for window in windows:
            tag = str(window or 'all')
            group = 'manual' if window in MANUAL_WINDOWS else f'window_{window}'
            for entity, values in [('item', self.item_stats[window][iid]), ('user', self.user_stats[window][uid])]:
                col = f'{entity}_count_{tag}'
                numeric[col] = values; groups[group].append(col)
            for entity, values, all_count in [('item', self.item_stats[window][iid], self.item_stats[0][iid]),
                                               ('user', self.user_stats[window][uid], self.user_stats[0][uid])]:
                col = f'{entity}_share_{tag}'
                numeric[col] = values / np.maximum(all_count, 1); groups[group].append(col)
        for (col, token), proportion in self.demo.items():
            name = f'item_demographic_{col}_{token}'
            numeric[name] = proportion[iid]
            groups['manual'].append(name)
            match_name = f'match_demographic_{col}_{token}'
            numeric[match_name] = proportion[iid] * (categorical['user_' + col].to_numpy() == token)
            groups['manual'].append(match_name)
        # Sparse relation matrices restrict user history to queried users.
        x = self.index.x[qpositions]
        for relation, item_tokens in self.token_matrices.items():
            profiles = x @ item_tokens
            candidate_tokens = item_tokens[iid].tocoo()
            row = candidate_tokens.row
            profile_row = local_u[row]
            values = np.asarray(profiles[profile_row, candidate_tokens.col]).ravel()
            total = np.bincount(row, weights=values, minlength=len(pairs)).astype(np.float32)
            col = f'affinity_{relation}_all'
            numeric[col] = total
            numeric[col + '_share'] = total / np.maximum(np.asarray(profiles.sum(axis=1)).ravel()[local_u], 1)
            groups['manual'].extend([col, col + '_share'])
        # Timed affinity via only queried-user events: no global explode/join.
        qhist = self.history[self.history.user_id.isin(unique_users)]
        qhu = pd.Index(unique_users).get_indexer(qhist.user_id)
        qhi = self.item_pos.get_indexer(qhist.item_id)
        qage = ((self.cutoff - qhist.timestamp).dt.total_seconds()/86400).to_numpy()
        for window in windows:
            if window == 0:
                continue
            mask = qage <= window
            xw = sparse.csr_matrix((np.ones(mask.sum(), np.float32), (qhu[mask], qhi[mask])), shape=(len(unique_users), len(self.index.items)))
            group = 'manual' if window in MANUAL_WINDOWS else f'window_{window}'
            for relation in ['genre', 'country', 'type']:
                if relation not in self.token_matrices:
                    continue
                item_tokens = self.token_matrices[relation]
                profiles = xw @ item_tokens
                candidate_tokens = item_tokens[iid].tocoo()
                row = candidate_tokens.row
                values = np.asarray(profiles[local_u[row], candidate_tokens.col]).ravel()
                total = np.bincount(row, weights=values, minlength=len(pairs)).astype(np.float32)
                col = f'affinity_{relation}_{window}'
                numeric[col] = total
                numeric[col + '_share'] = total / np.maximum(np.asarray(profiles.sum(axis=1)).ravel()[local_u], 1)
                groups[group].extend([col, col + '_share'])
        if with_history:
            history_sorted = qhist.sort_values(['user_id', 'timestamp', 'item_id'], ascending=[True, False, True])
            for position in range(5):
                # nth preserves source row index; explicit rank gives stable ties.
                selected = history_sorted[history_sorted.groupby('user_id').cumcount().eq(position)]
                last_map = selected.set_index('user_id').item_id
                last = pairs.user_id.map(last_map)
                hp = self.item_pos.get_indexer(last.fillna(-1))
                valid = hp >= 0
                col = f'history_{position}_item'
                categorical[col] = last.fillna(-1).astype('int64').astype(str)
                pop = np.zeros(len(pairs), np.float32); pop[valid] = self.item_stats[0][hp[valid]]
                numeric[f'history_{position}_pop'] = pop
                for name, matrix in [('cosine2', getattr(self.index, 'cosine2', None)), ('genre', None)]:
                    value = np.zeros(len(pairs), np.float32)
                    if name == 'cosine2' and matrix is not None:
                        value[valid] = np.asarray(matrix[hp[valid], iid[valid]]).ravel()
                    elif name == 'genre' and self.genre_normalized is not None:
                        value[valid] = np.asarray(self.genre_normalized[hp[valid]].multiply(self.genre_normalized[iid[valid]]).sum(axis=1)).ravel()
                    numeric[f'history_{position}_{name}'] = value
                groups['history'].extend([f'history_{position}_{s}' for s in ['pop', 'cosine2', 'genre']])
        if with_embeddings and self.index.als is not None:
            # Products learn relevance directions without hand-written categories.
            for dim in range(min(32, self.index.als.user_factors.shape[1])):
                col = f'latent_product_{dim}'
                numeric[col] = self.index.als.user_factors[uid, dim] * self.index.als.item_factors[iid, dim]
                groups['latent'].append(col)
        frame = pd.DataFrame({**{c: np.asarray(v, np.float32) for c,v in numeric.items()}, **categorical})
        return select_columns(frame, list(categorical), dict(groups), required_columns)
