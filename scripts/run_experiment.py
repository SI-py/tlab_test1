"""Reproduce the temporal feature experiment on the full public KION data."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'prepare', 'search', 'final'], default='all')
    parser.add_argument('--data', type=Path, default=Path('data/raw'))
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    parser.add_argument('--cache', type=Path, default=Path('artifacts/cache'))
    parser.add_argument('--config', type=Path, default=Path('configs/v4.json'))
    args = parser.parse_args()
    development = args.output / 'v4'
    final = args.output / 'v5'
    final_config = final / 'config.json'
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}

    def run(*command):
        print(' '.join(map(str, command)), flush=True)
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, env=env, check=True)

    def pipeline(phase, config, output):
        run('-m', 'recsys.pipeline', phase, '--config', config, '--output', output,
            '--data', args.data, '--cache', args.cache)

    if args.stage in ('all', 'prepare'):
        run('scripts/download_data.py', '--output', args.data)
        pipeline('prepare', args.config, development)
    if args.stage in ('all', 'search'):
        pipeline('search', args.config, development)
        run('-m', 'recsys.v5_extension', '--parent', development, '--output', final,
            '--config-output', final_config, '--data', args.data, '--cache', args.cache)
        pipeline('search', final_config, final)
    if args.stage in ('all', 'final'):
        pipeline('final', final_config, final)
        # Preserve the numerical evidence separately from expensive snapshots/models.
        # Saved article results are replaced only by an explicit full rerun.
        run('scripts/collect_results.py', '--run', final, '--output', 'results/rerun.json')


if __name__ == '__main__':
    main()
