"""Compare HDBSCAN size policies on identical paired subsamples in one job.

Run from text/. Output is separate from the archived paired_recovery experiment.
Only aggregate comparisons are exported; the existing runner stores replicate
records and exact fitted parameters for reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import paired_recovery as recovery
from scenario_pipeline import load_bn_analysis_config


def indicators(record: dict, scenarios: list[dict], edges: list[tuple[str, str]],
               tau: float, reference_n: int) -> dict[tuple[str, str], int]:
    matches = record['matches']
    selected = {tuple(edge) for edge in record['reconstructed_edges']}
    result = {}
    def available(names):
        return all(matches[name]['candidate'] is not None and matches[name]['jaccard'] >= tau
                   for name in names)
    for parent, child in edges:
        found = available([parent, child])
        edge = (matches[parent]['candidate'], matches[child]['candidate'])
        result['relationship', parent + '->' + child] = int(found and edge in selected)
    # Integer comparison avoids floating-point ambiguity at the support boundary.
    for scenario in scenarios:
        count = record['scenarios'][scenario['id']]['reconstructed_counts'][str(tau)]
        recurrent = count * reference_n >= recovery.REFERENCE_MIN_COUNT * record['sample_size']
        result['scenario', scenario['id']] = int(available(scenario['factors']) and recurrent)
    return result


def compare(args, policies):
    root = args.output.resolve()
    records = {}
    designs = {}
    for policy in policies:
        folder = root / policy
        design = json.loads((folder / 'design.json').read_text())
        designs[policy] = design
        records[policy] = [json.loads((folder / 'replicates' / f'replicate_{i:04d}.json').read_text())
                           for i in range(design['replicates'])]
        if any(r['design_hash'] != recovery._fingerprint(design) or r['replicate'] != i
               for i, r in enumerate(records[policy])):
            raise ValueError(f'Inconsistent replicate archive: {policy}')
    baseline = records['fixed']
    common_keys = ['replicates', 'fraction', 'seed', 'matching_thresholds',
                   'main_matching_threshold', 'reference_min_count', 'd_max', 'files_sha256']
    for policy in policies:
        if any(designs[policy][key] != designs['fixed'][key] for key in common_keys):
            raise ValueError(f'Different analysis inputs or sampling settings: {policy}')
        for before, after in zip(baseline, records[policy]):
            if before['accident_ids'] != after['accident_ids'] or before['fixed_edges'] != after['fixed_edges']:
                raise ValueError(f'Unpaired input or fixed graph for {policy}/{after["replicate"]}')
            if any(before['scenarios'][s]['fixed_count'] != after['scenarios'][s]['fixed_count']
                   for s in before['scenarios']):
                raise ValueError('Fixed scenario counts differ across policies')
    run_dir = args.run_dir.resolve()
    matrix = pd.read_parquet(run_dir / 'bn_results_exact/matrix/accident_factor_matrix.parquet')
    scenarios = recovery._scenarios(run_dir / 'bn_results_exact/scenarios/recurrent_scenarios_all.csv',
                                    set(matrix.columns) - {'accident_id'})
    edge_table = pd.read_csv(run_dir / 'bn_results_exact/network/bn_edges_full.csv')
    edges = [(str(r.parent_factor), str(r.child_factor)) for r in edge_table.itertuples()
             if bool(r.in_full_sample_BN)]
    rows = []
    for tau in designs['fixed']['matching_thresholds']:
        base = [indicators(r, scenarios, edges, tau, len(matrix)) for r in baseline]
        for policy in policies:
            current = [indicators(r, scenarios, edges, tau, len(matrix)) for r in records[policy]]
            for kind, name in base[0]:
                a = np.array([r[kind, name] for r in base], dtype=float)
                b = np.array([r[kind, name] for r in current], dtype=float)
                difference = b - a
                rows.append({'object_type': kind, 'object_id': name, 'policy': policy,
                    'matching_threshold': tau, 'n_replicates': len(a),
                    'fixed_parameter_reconstructed_recovery': float(a.mean()),
                    'policy_reconstructed_recovery': float(b.mean()),
                    'policy_minus_fixed_parameters': float(difference.mean()),
                    'paired_mcse': recovery._mcse(difference.tolist())})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / 'policy_comparison.csv', index=False)
    frame.groupby(['policy', 'matching_threshold', 'object_type']).agg(
        n_objects=('object_id', 'size'), median_recovery=('policy_reconstructed_recovery', 'median'),
        median_paired_change=('policy_minus_fixed_parameters', 'median')).to_csv(root / 'policy_summary.csv')
    recovery._write_json(root / 'comparison_audit.json', {
        'paired_replicates_checked': len(baseline), 'policies': policies,
        'reference_scenarios': len(scenarios), 'reference_relationships': len(edges),
        'difference_direction': 'policy recovery minus fixed-parameter reconstruction recovery',
        'rounding': 'nearest integer, half up; minimum cluster size >= 2, min_samples >= 1',
        'scaling': 'sampled role-specific factual units / full role-specific factual units',
        'source_code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(recovery.__file__)]}})
    print(f'Validated paired comparisons written to {root}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('recurrent_scenarios/config.yaml'))
    parser.add_argument('--dataset', default=recovery.DEFAULT_DATASET)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replicates', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--summarise-only', action='store_true')
    args = parser.parse_args()
    args.run_dir = args.run_dir or Path('recurrent_scenarios/runs/theme_discovery_audit') / args.dataset
    args.output = args.output or args.run_dir / 'paired_parameter_sensitivity'
    if args.output.resolve() == (args.run_dir / 'paired_recovery').resolve():
        raise ValueError('Use a separate output folder for parameter sensitivity')
    config = load_bn_analysis_config(args.config.resolve(), args.dataset, args.run_dir.resolve())
    policies = config.get('paired_recovery', {}).get('sensitivity', {}).get(
        'parameter_policies', list(recovery.PARAMETER_POLICIES))
    if (not policies or len(set(policies)) != len(policies) or 'fixed' not in policies
            or any(p not in recovery.PARAMETER_POLICIES for p in policies)):
        raise ValueError('Select distinct supported parameter policies, including fixed')
    for policy in policies:
        run_args = argparse.Namespace(config=args.config, dataset=args.dataset, run_dir=args.run_dir,
            output=args.output / policy, replicates=args.replicates, workers=args.workers,
            fraction=None, seed=None, start=0, count=None, parameter_policy=policy)
        if not args.summarise_only:
            recovery.run(run_args)
        recovery.summarise(run_args)
    compare(args, policies)


if __name__ == '__main__':
    main()
