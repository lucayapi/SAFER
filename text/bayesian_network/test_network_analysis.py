"""Small synthetic acceptance checks for the BN/scenario core."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import json

import run_bayesian_analysis as pipeline

from network_analysis import (admissible_arcs, bn_joint_probabilities,
    cpt_jeffreys, fit_exact_bn, fit_greedy_bn, scenario_candidates,
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
    candidates=scenario_candidates(x,roles,min_count=5)
    assert not candidates.empty
    parents,edges,_=fit_exact_bn(x,roles,2)
    cpts,cpt_table=cpt_jeffreys(x,parents,.5)
    assert all(0 < p < 1 for p in cpts.values())
    probs=bn_joint_probabilities(candidates,list(x.columns),parents,cpts,edges,n_sim=2000,seed=42)
    assert probs.bn_joint_probability_mc.between(0,1).all()


def test_explicit_id_join_excludes_other_role_and_audits_accidents(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); run=root/"run";run.mkdir(); monkeypatch.setattr(pipeline,"RUN_DIR",run,raising=False)
        units=pd.DataFrame({"accident_id":["a","a","b","c"],"fact_id":[1,2,3,4],
          "sentence":["context","other","event","no valid role"],"pred_label":["A0","AUTRE","B","AUTRE"],"pred_ok":[True]*4})
        emb=pd.DataFrame({"doc_id":[1,3],"dim_0001":[1.,0.],"dim_0002":[0.,1.]})
        units.to_csv(root/"units.csv",index=False);emb.to_csv(root/"e5.csv",index=False)
        cfg={"data":{"units_path":str(root/"units.csv"),"embeddings_path":str(root/"e5.csv"),
          "accident_id_col":"accident_id","fact_id_col":"fact_id","text_col":"sentence","role_col":"pred_label",
          "valid_col":"pred_ok","embedding_id_col":"doc_id","embedding_dimensions_prefix":"dim_",
          "normalize_embeddings":False,"fail_on_unmatched_units":True}}
        selected,vectors,audit=pipeline.load_inputs(cfg)
        assert selected._role.tolist()==["A0","B"]
        assert vectors.shape==(2,2)
        assert audit["excluded_other_role_units"]==2
        assert audit["accidents_without_valid_A0_A1_B_C_units"]==1
        assert json.loads((run/"audit_alignment.json").read_text())["matched_valid_units"]==2


def test_accident_split_is_disjoint_and_reproducible():
    frame=pd.DataFrame({"_accident_id":[str(i) for i in range(100)]})
    a,b=pipeline.split_accidents(frame,42,.7)
    aa,bb=pipeline.split_accidents(frame,42,.7)
    assert a==aa and b==bb and not a&b and len(a)==70 and len(b)==30


def test_bootstrap_parallel_and_sequential_are_seed_reproducible(tmp_path,monkeypatch):
    x=synthetic_matrix(160);roles={"A0_001":"A0","A1_001":"A1","B_001":"B","C_001":"C"}
    candidates=scenario_candidates(x,roles,min_count=3)
    cfg={"network":{"bootstrap_replicates":4,"bootstrap_seed":9,"max_parents":2,
      "scenario_min_train_count":3,"jeffreys_alpha":.5,"scenario_mc_draws":1000,"bootstrap_parallel":True},
      "parallel":{"enabled":True,"n_workers":"auto","max_workers":2,"backend":"loky"},
      "validation":{"show_progress":False}}
    first=tmp_path/"parallel";first.mkdir();monkeypatch.setattr(pipeline,"RUN_DIR",first,raising=False)
    bootstrap(x,roles,candidates,cfg)
    cfg["network"]["bootstrap_parallel"]=False
    second=tmp_path/"sequential";second.mkdir();monkeypatch.setattr(pipeline,"RUN_DIR",second,raising=False)
    bootstrap(x,roles,candidates,cfg)
    pd.testing.assert_frame_equal(pd.read_csv(first/"bootstrap_edge_stability.csv"),pd.read_csv(second/"bootstrap_edge_stability.csv"))
    pd.testing.assert_frame_equal(pd.read_csv(first/"bootstrap_scenario_stability.csv"),pd.read_csv(second/"bootstrap_scenario_stability.csv"))
