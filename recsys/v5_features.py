"""Frozen-anchor learned geometry on top of the unchanged V4 feature store."""
from __future__ import annotations

import pandas as pd

from .anchor_features import AnchorFeatures
from .v2_features import feature_request, select_columns
from .v4_quality_features import QualityWinnerFeatureStore


class AnchorWinnerFeatureStore(QualityWinnerFeatureStore):
    """Add learned relational coordinates while retaining every V4 base feature.

    Anchor IDs must be chosen from the earliest training prefix and frozen by
    the experiment specification. Their factors are recomputed from each
    current prefix, using the same aligned representation model as V4.
    """
    def __init__(self, events, items, users, index, windows=None,
                 anchor_item_ids=None, representation_model=None, required_columns=None):
        if representation_model is None:
            raise ValueError('Anchor features require an aligned representation model')
        if anchor_item_ids is None or not len(anchor_item_ids):
            raise ValueError('Frozen training-prefix anchor IDs are required')
        super().__init__(events, items, users, index, windows,
                         representation_model=representation_model, required_columns=required_columns)
        needs_anchors = required_columns is None or any(c.startswith("learned_anchor_") for c in required_columns)
        self.anchor_item_ids = tuple(anchor_item_ids)
        self.anchor_features = AnchorFeatures(representation_model, anchor_item_ids) if needs_anchors else None

    def build(self, candidates, with_auto=True, with_history=True, with_embeddings=True, required_columns=None):
        required_columns, with_embeddings = feature_request(self, required_columns, with_embeddings)
        frame, cats, groups = super().build(candidates, with_auto, with_history, with_embeddings, required_columns=required_columns)
        needs_anchors = required_columns is None or any(c.startswith("learned_anchor_") for c in required_columns)
        if with_embeddings and needs_anchors:
            if self.anchor_features is None:
                self.anchor_features = AnchorFeatures(self.representation_model, self.anchor_item_ids)
            anchors = self.anchor_features.build(candidates)
            if set(anchors.columns) & set(frame.columns):
                raise ValueError('Anchor feature names collide with base feature names')
            anchors = anchors.reset_index(drop=True)
            anchors.index = frame.index
            frame = pd.concat([frame, anchors], axis=1)
            groups['latent'] = list(dict.fromkeys(groups.get('latent', []) + list(anchors.columns)))
        return select_columns(frame, cats, groups, required_columns)
