from pathlib import Path
import pandas as pd
import json

ROOT = Path('text/recurrent_scenarios/runs/theme_discovery_audit/btp_carpentry_and_joinery')
paths = [
 'tables/corpus_summary_by_role.csv', 'tables/retained_factors_summary.csv',
 'tables/tchebycheff_selection_all_roles.csv', 'tables/seed_sensitivity_summary_all_roles.csv',
 'bn_results_exact/matrix/input_summary.csv', 'bn_results_exact/matrix/factor_prevalence.csv',
 'bn_results_exact/network/global_bn_summary.csv', 'bn_results_exact/network/bn_edges_full.csv',
 'bn_results_exact/network/bn_cpt_support_diagnostics.csv',
 'bn_results_exact/scenarios/scenario_threshold_summary.csv',
 'bn_results_exact/scenarios/scenario_bn_discrepancy.csv',
 'bn_results_exact/scenarios/recurrent_scenarios_all.csv',
 'bn_results_exact/scenarios/scenario_bn_cpt_sensitivity.csv',
 'paired_recovery/factor_recovery_by_role.csv',
 'paired_recovery/edge_recovery_tau_0.5.csv',
 'paired_recovery/scenario_recovery_tau_0.5.csv',
 'paired_recovery/article_scenario_recovery.csv',
 'discovery/A0/candidate_metrics.csv', 'discovery/A0/pareto_front.csv',
]
for name in paths:
    df=pd.read_csv(ROOT/name)
    print('\nFILE', name, 'SHAPE', df.shape)
    print('COLUMNS', ', '.join(df.columns))
    print(df.head(4).to_string(index=False, max_colwidth=45))
print('\nDESIGN AND REPLICATES')
records=[json.loads(f.read_text()) for f in (ROOT/'paired_recovery/replicates').glob('*.json')]
print('n',len(records),'ids',len(set(r['replicate'] for r in records)), 'samples',sorted(set(r['sample_size'] for r in records)), 'hashes',len(set(r['design_hash'] for r in records)))
print('record keys', list(records[0]))
print('factor counts', pd.DataFrame([r['n_reconstructed_factors'] for r in records]).describe().to_string())
for tau in [.4,.5,.6]:
    e=pd.read_csv(ROOT/f'paired_recovery/edge_recovery_tau_{tau}.csv')
    s=pd.read_csv(ROOT/f'paired_recovery/scenario_recovery_tau_{tau}.csv')
    print('\nTAU',tau)
    print('EDGES', e.select_dtypes('number').describe().to_string())
    print('SCENARIOS', s.select_dtypes('number').describe().to_string())
    print('edge declines', (e.paired_difference>0).sum(), 'scenario declines', (s.paired_difference>0).sum(), 'increases',(s.paired_difference<0).sum(), 'ties',(s.paired_difference==0).sum())
    print('scenario recovery >=.8', (s.reconstructed_recovery>=.8).sum(), '<.1',(s.reconstructed_recovery<.1).sum(), 'zero',(s.reconstructed_recovery==0).sum())
    if tau==.5:
        print('TOP SCENARIOS\n', s.sort_values('reconstructed_recovery',ascending=False).head(10).to_string(index=False,max_colwidth=50))
        print('EDGE TABLE\n',e.to_string(index=False))
        print('BY COUNT\n',s.groupby('reference_count')[['fixed_recovery','reconstructed_recovery','all_factor_recovery','scenario_given_factors']].agg(['count','median']).to_string())
