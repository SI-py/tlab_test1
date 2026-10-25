"""Uniform user-disjoint cohorts and calendar-day temporal labels."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import labels_at

@dataclass(frozen=True)
class V2Config:
    train_cutoffs: tuple[str, ...]
    validation_cutoffs: tuple[str, ...]
    test_cutoff: str
    observation_end: str
    horizon_days: int = 7
    n_train_users: int = 30_000
    n_validation_users: int = 10_000
    n_test_users: int = 30_000
    seed: int = 2027

    def validate(self):
        if self.horizon_days <= 0 or not self.train_cutoffs or not self.validation_cutoffs:
            raise ValueError('Positive horizon and nonempty train/validation dates required')
        sizes = [self.n_train_users, self.n_validation_users, self.n_test_users]
        if any(not isinstance(n, int) or n <= 0 for n in sizes):
            raise ValueError('All cohort sizes must be positive integers')
        dates = [pd.Timestamp(c) for c in (*self.train_cutoffs, *self.validation_cutoffs, self.test_cutoff)]
        end = pd.Timestamp(self.observation_end)
        if any(pd.isna(t) or t.tz is not None or t.normalize() != t for t in dates + [end]):
            raise ValueError('Expected timezone-naive calendar-day boundaries')
        horizon = pd.Timedelta(days=self.horizon_days)
        if any(a + horizon > b for a, b in zip(dates, dates[1:])):
            raise ValueError('Train, validation and test label intervals must be ordered and disjoint')
        if dates[-1] + horizon > end:
            raise ValueError('Test horizon extends beyond observation_end')


def _ids(values):
    array = np.asarray(list(values))
    if array.size and (not np.issubdtype(array.dtype, np.integer)):
        raise ValueError('User IDs must be integers')
    return np.unique(array.astype(np.int64))


def create_manifest(path, all_users, previously_exposed_users, config):
    """Sample role IDs without activity-filtering all_users.

    previously_exposed_users must include every v1 user whose evaluation was
    inspected. They may belong to train/dev, but never to the new final test.
    Test sampling precedes dev sampling, so changing dev sizes preserves test.
    """
    config.validate()
    universe, exposed = _ids(all_users), _ids(previously_exposed_users)
    if sum((config.n_train_users, config.n_validation_users, config.n_test_users)) > len(universe):
        raise ValueError('Not enough unique users for requested disjoint cohorts')
    test_pool = np.setdiff1d(universe, exposed)
    if config.n_test_users > len(test_pool):
        raise ValueError('Not enough previously unexposed users for final test')
    test_seed, dev_seed = np.random.SeedSequence(config.seed).spawn(2)
    test = np.sort(np.random.default_rng(test_seed).choice(test_pool, config.n_test_users, replace=False))
    dev = np.random.default_rng(dev_seed).permutation(np.setdiff1d(universe, test))
    cohorts = {'train': np.sort(dev[:config.n_train_users]),
               'validation': np.sort(dev[config.n_train_users:config.n_train_users + config.n_validation_users]),
               'test': test}
    manifest = {
        'schema_version': 2,
        'config': json.loads(json.dumps(asdict(config))), 'population_size': len(universe),
        'previously_exposed_users': exposed.tolist(),
        'cohorts': {role: ids.tolist() for role, ids in cohorts.items()},
        'retrieval_protocol': 'Transductive global timestamp < cutoff prefix is allowed; no future rows.',
        'supervised_protocol': 'Fit train users only; tune validation users only; test users never supervised.',
        'sampling_protocol': 'Uniform user IDs without activity filtering; retain every row for chosen users.',
    }
    _validate_manifest(manifest)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2))
    return manifest


def _validate_manifest(manifest):
    if manifest['schema_version'] != 2:
        raise ValueError('Unsupported protocol version')
    cfg = V2Config(**manifest['config'])
    cfg.validate()
    cohorts = manifest['cohorts']
    for role, expected in [('train', cfg.n_train_users), ('validation', cfg.n_validation_users), ('test', cfg.n_test_users)]:
        ids = _ids(cohorts[role])
        if len(ids) != expected or ids.tolist() != cohorts[role]:
            raise ValueError(f'Invalid, duplicated, or unsorted {role} cohort')
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if set(cohorts[a]) & set(cohorts[b]):
            raise ValueError(f'User overlap: {a}/{b}')
    exposed = _ids(manifest['previously_exposed_users'])
    if set(exposed) & set(cohorts['test']):
        raise ValueError('Previously evaluated users cannot enter final test')
    return cfg


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    _validate_manifest(manifest)
    return manifest


def cohort_history(events, manifest, role, cutoff=None):
    selected = events[events.user_id.isin(manifest['cohorts'][role])]
    return selected if cutoff is None else selected[selected.timestamp < pd.Timestamp(cutoff)]


def labels_for_split(events, manifest, role, cutoff):
    cfg = _validate_manifest(manifest)
    cutoff = pd.Timestamp(cutoff)
    dates = {'train': cfg.train_cutoffs, 'validation': cfg.validation_cutoffs, 'test': [cfg.test_cutoff]}[role]
    if cutoff not in [pd.Timestamp(c) for c in dates]:
        raise ValueError('Undeclared role/cutoff')
    return labels_at(cohort_history(events, manifest, role), cutoff, cfg.horizon_days)

