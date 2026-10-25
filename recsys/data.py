import pandas as pd






def labels_at(events, cutoff, horizon=7):
    """New-to-user watches, including cold items missed by retrieval in denominator."""
    hist = events[events.timestamp < cutoff]
    future = events[(events.timestamp >= cutoff) & (events.timestamp < cutoff + pd.Timedelta(days=horizon))]
    seen = pd.MultiIndex.from_frame(hist[['user_id', 'item_id']])
    future = future[~pd.MultiIndex.from_frame(future[['user_id', 'item_id']]).isin(seen)]
    return future[['user_id', 'item_id']].drop_duplicates().assign(label=1)
