"""Reproduce V1 exposure exclusions from public cohort seeds, without old files."""
from pathlib import Path
import json
import numpy as np


def exposure_from_recipe(all_users,recipe_path=None):
    path=Path(recipe_path) if recipe_path else Path(__file__).resolve().parents[1]/'configs/v1_exposure_recipe.json'
    if not path.exists():raise FileNotFoundError('The prior-exposure sampling contract is required')
    recipe=json.loads(path.read_text())
    universe=np.unique(np.asarray(all_users,dtype=np.int64))
    if len(universe) != recipe['population_size']: return np.array([], dtype=np.int64)
    if recipe.get('schema_version')!=1:raise ValueError('Unknown exposure recipe schema')
    cohorts=[np.random.default_rng(entry['seed']).choice(universe,entry['n_users'],replace=False)
             for entry in recipe['cohorts']]
    result=np.unique(np.concatenate(cohorts)).astype(np.int64)
    return result
