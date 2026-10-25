import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def evaluate(pairs, scores, truth, k=10):
    """Macro user metrics; missed/cold positives stay in ideal denominator."""
    ranked = pairs[['user_id', 'item_id']].copy()
    ranked['score'] = scores
    ranked = ranked.sort_values(['user_id', 'score', 'item_id'], ascending=[True, False, True])
    ranked['rank'] = ranked.groupby('user_id').cumcount() + 1
    ranked = ranked.merge(truth, on=['user_id', 'item_id'], how='left').fillna({'label': 0})
    npos = truth.groupby('user_id').size()
    ranked = ranked[ranked.user_id.isin(npos.index)]
    top = ranked[ranked['rank'] <= k].copy()
    top['dcg'] = top.label / np.log2(top['rank'] + 1)
    top['ap'] = top.groupby('user_id').label.cumsum() * top.label / top['rank']
    by = pd.DataFrame(index=npos.index)
    by['positives'] = npos
    by['hits'] = top.groupby('user_id').label.sum().reindex(npos.index, fill_value=0)
    by['recall'] = by.hits / npos
    discounts = np.cumsum(1 / np.log2(np.arange(1, k+1) + 1))
    ideal = discounts[np.minimum(npos, k) - 1]
    by['ndcg'] = top.groupby('user_id').dcg.sum().reindex(npos.index, fill_value=0) / ideal
    by['map'] = top.groupby('user_id').ap.sum().reindex(npos.index, fill_value=0) / np.minimum(npos, k)
    by['candidate_recall'] = ranked.groupby('user_id').label.sum().reindex(npos.index, fill_value=0) / npos
    result = {f'{m}@{k}': float(by[m].mean()) for m in ['ndcg', 'recall', 'map']}
    result['candidate_recall'] = float(by.candidate_recall.mean())
    result['users'] = len(by)
    result['topk_item_coverage'] = int(top.item_id.nunique())
    result['auc'] = float(roc_auc_score(ranked.label, ranked.score)) if ranked.label.nunique() > 1 else None
    if np.all((ranked.score >= 0) & (ranked.score <= 1)):
        result['sampled_logloss'] = float(log_loss(ranked.label, ranked.score, labels=[0, 1]))
    return result, by.reset_index()


def paired_bootstrap(a, b, seed=2026, n_boot=2000):
    joined = a[['user_id', 'ndcg', 'recall']].merge(b[['user_id', 'ndcg', 'recall']], on='user_id', suffixes=('_a', '_b'))
    rng = np.random.default_rng(seed)
    result = {}
    for metric in ['ndcg', 'recall']:
        delta = (joined[metric + '_b'] - joined[metric + '_a']).to_numpy()
        means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(n_boot)])
        result[metric] = {'delta': float(delta.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist(), 'users': len(delta)}
    return result
