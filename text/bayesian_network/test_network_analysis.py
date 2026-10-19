"""Small synthetic acceptance checks for the BN/scenario core."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import json

import run_bayesian_analysis as pipeline

from network_analysis import (_connected_factor_set, admissible_arcs, bn_joint_probabilities,
    cpt_jeffreys, summarize_scenarios, fit_exact_bn, fit_greedy_bn, scenario_candidates,
    topological_order)

from run_bayesian_analysis import bootstrap


def synthetic_matrix(n=240, seed=7):
    rng=np.random.default_rng(seed)
    a0=rng.binomial(1,.5,n)
    a1=np.where(a0==1,rng.binomial(1,.75,n),rng.binomial(1,.2,n))
    b=np.where(a1==1,rng.binomial(1,.8,n),rng.binomial(1,.15,n))
    c=np.where(b==1,rng.binomial(1,.7,n),rng.binomial(1,.1,n))
    return pd.DataFrame({"A0_001":a0,"A1_001":a1,"B_001":b,"C_001":c})


def test_exact_graph_obeys_roles_and_parent_limit():
    x=synthetic_matrix(); roles={"A0_001":"A0","A1_001":"A1","B_001":"B","C_001":"C"}
    parents,edges,_=fit_exact_bn(x,roles,2)
    allowed=set(admissible_arcs(roles))
    assert all(edge in allowed for edge in edges)
    assert max(map(len,parents.values()))<=2
    assert topological_order(list(x.columns),edges)


def test_greedy_sensitivity_is_acyclic_and_limited():
    x=synthetic_matrix(); roles={"A0_001":"A0","A1_001":"A1","B_001":"B","C_001":"C"}
    parents,edges,_=fit_greedy_bn(x,roles,2)
    assert max(map(len,parents.values()))<=2
    assert topological_order(list(x.columns),edges)


def test_scenario_freeze_and_jeffreys_probabilities():
    x=synthetic_matrix(); roles={"A0_001":"A0","A1_001":"A1","B_001":"B","C_001":"C"}
    parents,edges,_=fit_exact_bn(x,roles,2)
    candidates=scenario_candidates(x,roles,edges,min_count=5)
    assert not candidates.empty
    for row in candidates.itertuples(index=False):
        factors=set(row.factor_ids.split(" | "))
        graph_edges=[edge for edge in edges if edge[0] in factors and edge[1] in factors]
        assert _connected_factor_set(tuple(factors),graph_edges)
    cpts,cpt_table=cpt_jeffreys(x,parents,.5)
    assert all(0 < p < 1 for p in cpts.values())
    probs=bn_joint_probabilities(candidates,list(x.columns),parents,cpts,edges,n_sim=2000,seed=42)
    assert probs.bn_joint_probability_mc.between(0,1).all()
    assert probs.loc[probs.scenario_id.str.contains("C_"),"bn_conditional_C_probability_mc"].between(0,1).all()


def test_connected_BC_scenario_descriptive_summary_and_conditional_prevalence():
    train=pd.DataFrame({"B_001":[1]*6+[0]*4,"C_001":[1]*5+[0]*5})
    roles={"B_001":"B","C_001":"C"}
    candidates=scenario_candidates(train,roles,[("B_001","C_001")],min_count=5)
    assert candidates.scenario_id.tolist()==["B_001+C_001"]
    observed=pd.DataFrame({"B_001":[1,1,1,0,0],"C_001":[1,0,1,1,0]})
    evaluated=summarize_scenarios(candidates,observed).iloc[0]
    assert evaluated["conditional_C_n"]==3
    assert evaluated["conditional_C_count"]==2
    assert evaluated["conditional_C_support"]==pytest.approx(2/3)
    assert evaluated["accident_count"]==2
    assert "p_value" not in evaluated.index
    assert "wilson_95_low" not in evaluated.index

    with_upstream=pd.DataFrame({"A0_001":[1]*6+[0]*4,"B_001":[1]*6+[0]*4,"C_001":[1]*5+[0]*5})
    roles_with_upstream={"A0_001":"A0","B_001":"B","C_001":"C"}
    candidates_with_upstream=scenario_candidates(
        with_upstream,roles_with_upstream,[("A0_001","B_001"),("B_001","C_001")],min_count=5)
    assert "B_001+C_001" not in candidates_with_upstream.scenario_id.tolist()


def test_explicit_id_join_excludes_other_role_and_audits_accidents(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); run=root/"run";run.mkdir(); monkeypatch.setattr(pipeline,"RUN_DIR",run,raising=False)
        units=pd.DataFrame({"accident_id":["a","a","b","c"],"fact_id":[1,2,1,1],"doc_id":[101,102,103,104],
          "sentence":["context","other","event","no valid role"],"pred_label":["A0","AUTRE","B","AUTRE"],"pred_ok":[True]*4})
        emb=pd.DataFrame({"doc_id":[101,103],"dim_0001":[1.,0.],"dim_0002":[0.,1.]})
        units.to_csv(root/"units.csv",index=False);emb.to_csv(root/"e5.csv",index=False)
        cfg={"data":{"units_path":str(root/"units.csv"),"embeddings_path":str(root/"e5.csv"),
          "accident_id_col":"accident_id","fact_id_col":"fact_id","text_col":"sentence","role_col":"pred_label",
          "valid_col":"pred_ok","embedding_id_col":"doc_id","unit_id_col":"doc_id","embedding_dimensions_prefix":"dim_",
          "normalize_embeddings":False,"fail_on_unmatched_units":True}}
        selected,vectors,audit=pipeline.load_inputs(cfg)
        assert selected._role.tolist()==["A0","B"]
        assert vectors.shape==(2,2)
        assert audit["excluded_other_role_units"]==2
        assert audit["accidents_without_valid_A0_A1_B_C_units"]==1
        assert json.loads((run/"audit_alignment.json").read_text())["matched_valid_units"]==2


def test_bootstrap_parallel_and_sequential_are_seed_reproducible(tmp_path,monkeypatch):
    x=synthetic_matrix(160);roles={"A0_001":"A0","A1_001":"A1","B_001":"B","C_001":"C"}
    _,edges,_=fit_exact_bn(x,roles,2)
    candidates=scenario_candidates(x,roles,edges,min_count=3)
    cfg={"network":{"bootstrap_replicates":4,"bootstrap_seed":9,"max_parents":2,
      "scenario_min_count":3,"jeffreys_alpha":.5,"scenario_mc_draws":1000,"bootstrap_parallel":True},
      "parallel":{"enabled":True,"n_workers":"auto","max_workers":2,"backend":"loky"},
      "clustering_selection":{"show_progress":False}}
    first=tmp_path/"parallel";first.mkdir();monkeypatch.setattr(pipeline,"RUN_DIR",first,raising=False)
    bootstrap(x,roles,candidates,cfg)
    cfg["network"]["bootstrap_parallel"]=False
    second=tmp_path/"sequential";second.mkdir();monkeypatch.setattr(pipeline,"RUN_DIR",second,raising=False)
    bootstrap(x,roles,candidates,cfg)
    pd.testing.assert_frame_equal(pd.read_csv(first/"bootstrap_edge_stability.csv"),pd.read_csv(second/"bootstrap_edge_stability.csv"))
    pd.testing.assert_frame_equal(pd.read_csv(first/"bootstrap_scenario_stability.csv"),pd.read_csv(second/"bootstrap_scenario_stability.csv"))


def test_full_matrix_network_run_writes_descriptive_outputs_only(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,"RUN_DIR",tmp_path,raising=False)
    matrix=synthetic_matrix(160)
    roles={"A0_001":"A0","A1_001":"A1","B_001":"B","C_001":"C"}
    config={"network":{"max_parents":2,"jeffreys_alpha":.5,"scenario_min_count":3,
      "scenario_two_b_sensitivity":True,"scenario_mc_draws":1000,"bootstrap_replicates":2,
      "bootstrap_seed":99,"bootstrap_parallel":False},
      "parallel":{"enabled":True,"n_workers":"auto","max_workers":2,"backend":"loky"},
      "clustering_selection":{"random_state":42,"show_progress":False}}
    pipeline.run_network(matrix,roles,config)
    result=pd.read_csv(tmp_path/"primary"/"scenario_descriptive_results.csv")
    assert not result.empty
    assert "support" in result and "bn_joint_probability_mc" in result
    assert not any("p_value" in column or "wilson" in column.lower() for column in result.columns)
    assert not any("validation" in path.name.lower() for path in tmp_path.rglob("*"))
    assert (tmp_path/"bootstrap_edge_stability.csv").is_file()
    assert (tmp_path/"bootstrap_scenario_stability.csv").is_file()


def test_full_corpus_matrix_and_audit_serializes_numpy_counts(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,"RUN_DIR",tmp_path,raising=False)
    units_with_other=pd.DataFrame({"_accident_id":["a","b","b"],"_fact_id":["1","2","3"],
      "_source_fact_id":["1","1","2"],"_role":["A0","B","AUTRE"]})
    dictionary=pd.DataFrame({"topic_id":["A0_000","B_000"],"role":["A0","B"],
      "llm_label":["work setting","fall event"],"llm_description":["context","event"]})
    with pytest.raises(ValueError,match="Only A0/A1/B/C"):
        pipeline.build_factor_matrix(units_with_other,np.array([0,0,-1]),dictionary)
    units=units_with_other.iloc[:2].copy()
    matrix,_,_=pipeline.build_factor_matrix(units,np.array([0,0]),dictionary)
    import json
    audit=json.loads((tmp_path/"audit_factor_matrix.json").read_text())
    assert audit["roles"]=={"A0":1,"B":1}
    assert len(matrix)==2 and matrix.loc["b","B_000"]==1
    assert audit["accidents"]==2
    assert (tmp_path/"accident_factor_matrix.csv").is_file()
    assert (tmp_path/"factor_cooccurrence.csv").is_file()
    assert not any("validation" in path.name for path in tmp_path.iterdir())
