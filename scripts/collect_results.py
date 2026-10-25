"""Collect numerical results for the article, without fitting or evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def clean_diagnostics(candidate, geometry):
    candidate = {key: candidate[key] for key in
                 ('scope', 'frozen_budgets', 'query_population', 'query_weeks',
                  'equal_week', 'weighting', 'oracle', 'restriction') if key in candidate}
    geometry = {key: value for key, value in geometry.items()
                if key not in ('provenance', 'anchor_item_ids', 'largest_absolute_cosine_pairs')}
    if 'projection_energy' in geometry:
        geometry['projection_energy'] = {
            key: value for key, value in geometry['projection_energy'].items()
            if key in ('formula', 'basis_construction', 'zero_vector_policy',
                       'all_active_items', 'selection_pool', 'pool_definition',
                       'popularity_cutoff', 'popularity_window_days',
                       'pool_size_requested', 'no_quality_implication')}
    return candidate, geometry


def collect(run, candidate=None, geometry=None, hardware=None):
    result = read(run / 'results.json')
    selection = read(run / 'selection.json')
    metadata = read(run / 'final_models.json')
    config = selection['config']
    protocol = config['protocol']
    validation = {name: choice['metrics'] for name, choice in selection['best'].items()}
    validation.update({name: selection[key]['metrics'] for name, key in
                       [('rank_fusion', 'rank_fusion'), ('primary', 'primary_mixture')]})
    models = {}
    choices = {}
    for name, choice in selection['best'].items():
        saved = metadata[name]
        choices[name] = {key: choice[key] for key in ('family', 'params', 'trees')}
        choices[name].update(features=len(choice['columns']), seeds=config['seeds'],
                             selected_window_groups=choice.get('selected_window_groups', []))
        models[name] = {
            'fit_seconds_all_seeds': saved['fit_seconds'],
            'model_bytes_all_seeds': sum((run / file).stat().st_size for file in saved['files'])}
    snapshots = []
    for file in sorted((run / 'snapshots').glob('*/stats.json')):
        stat = read(file)
        snapshots.append({
            'role': stat['role'], 'cutoff': stat['cutoff'],
            'snapshot_total_seconds': stat.get('seconds'),
            'base_snapshot_seconds': stat.get('base_seconds'),
            'anchor_extension_seconds': stat.get('extension_seconds'),
            'candidate_rows_before_sampling': stat.get('candidate_rows_before_sampling'),
            'training_rows_after_sampling': stat.get('training_rows_after_sampling'),
            'prefix_events': stat.get('index_stats', {}).get('prefix_events'),
            'recorded_cf_fit_seconds': stat.get('index_stats', {}).get('fit_seconds'),
            'recorded_extra_fit_seconds': {key: value.get('fit_seconds')
                for key, value in stat.get('extra_model_stats', {}).items()}})
    search = read(run / 'validation_search.json')
    candidate, geometry = clean_diagnostics(candidate or {}, geometry or {})
    return {
        'dataset': {'name': 'KION', 'rows': 5476251, 'users': 962179, 'items': 15706,
                    'item_scope': 'Items observed in interactions; the metadata catalog has 15963 rows.',
                    'date_range': ['2021-03-13', '2021-08-22'],
                    'source': 'https://github.com/irsafilo/KION_DATASET',
                    'unit': 'Aggregated user-item pair with its last watch date'},
        'protocol': {**{key: protocol[key] for key in
                       ('train_cutoffs', 'validation_cutoffs', 'test_cutoff', 'horizon_days', 'seed')},
                     'cohort_sizes': {role: protocol[f'n_{role}_users']
                                      for role in ('train', 'validation', 'test')},
                     'previously_exposed_users_excluded': 10992,
                     'caveat': 'New held-out users on a previously studied calendar week; not a new temporal holdout.'},
        'validation': validation, 'test': result['metrics'],
        'all_future_active_user_metrics': result.get('all_future_active_user_metrics', {}),
        'cold_user_fallback': result.get('cold_user_fallback', {}),
        'paired_bootstrap': result['paired_bootstrap'],
        'diagnostic_paired_bootstrap': result.get('diagnostic_paired_bootstrap', {}),
        'model_choices': choices, 'primary_mixture': selection['primary_mixture']['weights'],
        'best_automatic_variant': selection['best_automatic_variant'],
        'development_seed': config['seeds'][0],
        'costs': {'models': models, 'snapshots': snapshots,
                  'search_seconds': selection.get('search_seconds'),
                  'extension_search_seconds': selection.get('extension_search_seconds'),
                  'prediction_seconds_all_models': result.get('prediction_seconds_all_models'),
                  'prediction_wall_seconds_with_load_io': result.get('prediction_wall_seconds_with_load_io'),
                  'development_peak_resident_bytes': selection.get('peak_resident_bytes'),
                  'final_peak_resident_bytes': result.get('peak_resident_bytes'),
                  'note': 'Original local fit and joint prediction timings; shared fits are reused. Controlled selective-feature and full warm-request timings are reported separately in results/serving_benchmark.json.'},
        'row_counts': {'sampled_training_rows': selection.get('training_rows'),
                       'validation_candidate_rows': selection.get('validation_rows'),
                       'final_training_rows': result.get('final_training_rows'),
                       'test_candidate_rows': result['test_snapshot'].get('candidate_rows_before_sampling')},
        'segments': result.get('segments', {}),
        'candidate_validation': candidate, 'anchor_geometry': geometry,
        'matched_catboost_comparisons': search.get('matched_catboost_comparisons', []),
        'hardware': hardware or {},
        'statistical_scope': 'Paired user-bootstrap, 2000 draws, descriptive marginal percentile intervals. No multiplicity correction, training-variance estimate, new-period inference or predeclared non-inferiority margin.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('artifacts/final'))
    parser.add_argument('--output', type=Path, default=Path('results/experiment.json'))
    parser.add_argument('--candidate-diagnostics', type=Path)
    parser.add_argument('--anchor-diagnostics', type=Path)
    parser.add_argument('--hardware', type=Path)
    args = parser.parse_args()
    optional = lambda path: read(path) if path else {}
    evidence = collect(args.run, optional(args.candidate_diagnostics),
                       optional(args.anchor_diagnostics), optional(args.hardware))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
