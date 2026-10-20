"""One-command descriptive factor and constrained Bayesian network analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE=Path(__file__).resolve().parent
TEXT_ROOT=HERE.parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(TEXT_ROOT/"recurrent_scenarios"))

import scenario_pipeline as sp
from factor_annotation import annotate_selected_factors, check_annotation_credentials
from network_analysis import (ROLES, bn_joint_probabilities, cpt_jeffreys,
    fit_exact_bn, fit_greedy_bn, summarize_scenarios,
    scenario_candidates)


def read_csv(path):
    try:return pd.read_csv(path,low_memory=False)
    except UnicodeDecodeError:return pd.read_csv(path,encoding="latin-1",low_memory=False)


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def normalized_id(series):
    # Keep string IDs intact but normalize CSV numeric IDs such as "12.0".
    return series.astype(str).str.strip().str.replace(r"\.0$","",regex=True)


def load_inputs(config):
    dc=config["data"]; up=Path(dc["units_path"]); ep=Path(dc["embeddings_path"])
    if not up.is_file():raise FileNotFoundError(f"Unit table not found: {up}")
    if not ep.is_file():
        raise FileNotFoundError(f"Required embeddings are missing: {ep}. Place the aligned embedding export at this path; this pipeline does not encode text or switch embedding models.")
    units=read_csv(up); emb=read_csv(ep)
    needed=[dc[k] for k in ("accident_id_col","fact_id_col","unit_id_col","text_col","role_col")]
    missing=[c for c in needed if c not in units]
    if missing:raise ValueError(f"Missing unit columns: {missing}")
    idcol=dc["embedding_id_col"]
    if idcol not in emb:raise ValueError(f"Embedding ID column {idcol!r} missing from {ep}")
    dimcols=[c for c in emb if str(c).startswith(dc["embedding_dimensions_prefix"])]
    if not dimcols:raise ValueError(f"No embedding dimensions beginning with {dc['embedding_dimensions_prefix']!r}")
    all_units=units.copy()
    all_units["_accident_id"]=normalized_id(all_units[dc["accident_id_col"]])
    # Use the configured unit key for the globally aligned embedding join;
    # preserve the source fact identifier separately for auditing.
    all_units["_source_fact_id"]=normalized_id(all_units[dc["fact_id_col"]])
    all_units["_fact_id"]=normalized_id(all_units[dc["unit_id_col"]])
    all_units["_role"]=all_units[dc["role_col"]].astype(str).str.strip().str.upper()
    all_units["_text"]=all_units[dc["text_col"]].fillna("").astype(str)
    validflag=all_units[dc["valid_col"]].astype(str).str.lower().isin(["true","1","yes","y"]) if dc.get("valid_col") in all_units else pd.Series(True,index=all_units.index)
    rolevalid=all_units["_role"].isin(ROLES)
    nonempty=all_units["_text"].str.strip().ne("")
    eligible=validflag & rolevalid & nonempty
    embed=emb.copy(); embed["_join_id"]=normalized_id(embed[idcol])
    all_units["_join_id"]=all_units["_fact_id"]
    if embed["_join_id"].duplicated().any():
        dups=embed.loc[embed["_join_id"].duplicated(keep=False),"_join_id"].head().tolist()
        raise ValueError(f"Embedding IDs are not unique; examples: {dups}")
    embed=embed.set_index("_join_id",drop=False)
    ids=set(embed.index)
    unmatched=all_units.loc[eligible & ~all_units["_join_id"].isin(ids),["_accident_id","_fact_id","_role"]].copy()
    matched=all_units.loc[eligible & all_units["_join_id"].isin(ids)].copy()
    source_accidents=set(all_units["_accident_id"].dropna())
    eligible_accidents=set(all_units.loc[eligible,"_accident_id"].dropna())
    no_role_accidents=sorted(source_accidents-eligible_accidents)
    pd.DataFrame({"accident_id":no_role_accidents,"reason":"no valid non-empty A0/A1/B/C factual unit"}).to_csv(RUN_DIR/"audit_excluded_accidents.csv",index=False)
    audit={"source_units":len(units),"excluded_other_role_units":int((all_units["_role"]=="AUTRE").sum()),
       "excluded_non_target_role_units":int((~all_units["_role"].isin((*ROLES,"AUTRE"))).sum()),
       "role_counts_before_filter":all_units["_role"].value_counts(dropna=False).to_dict(),
       "excluded_invalid_units":int((~validflag).sum()),"excluded_empty_text_units":int((validflag & rolevalid & ~nonempty).sum()),
       "valid_A0_A1_B_C_units_before_embedding_join":int(eligible.sum()),"matched_valid_units":len(matched),
       "unmatched_valid_units":len(unmatched),"unmatched_accidents":int(unmatched["_accident_id"].nunique()),
       "source_accidents":len(source_accidents),"accidents_without_valid_A0_A1_B_C_units":len(no_role_accidents),
       "extra_embedding_rows":int((~embed.index.isin(set(matched["_join_id"]))).sum()),
       "embedding_dimension":len(dimcols),"join":f"units.{dc['unit_id_col']} = embeddings.{idcol}",
       "unit_key_note":"The configured unit key is used for an explicit one-to-one embedding join."}
    unmatched.to_csv(RUN_DIR/"audit_unmatched_units.csv",index=False)
    (RUN_DIR/"audit_alignment.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    if dc.get("fail_on_unmatched_units",True) and len(unmatched):
        raise ValueError(f"{len(unmatched)} valid A0/A1/B/C units lack configured embeddings; see audit_unmatched_units.csv. No rows were silently dropped.")
    if matched["_fact_id"].duplicated().any():raise ValueError("fact_id must be unique after valid-role filtering")
    vectors=embed.loc[matched["_join_id"],dimcols].to_numpy(dtype=np.float32)
    if not np.isfinite(vectors).all():raise ValueError("Configured embedding vectors contain non-finite values")
    if dc.get("normalize_embeddings",True):vectors/=np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
    matched=matched.drop(columns="_join_id").reset_index(drop=True)
    return matched,vectors,audit


def make_prepared(units,vectors):
    return sp.PreparedData(units=units.reset_index(drop=True),embeddings=np.asarray(vectors),input_summary=pd.DataFrame())


def build_factor_matrix(units,labels,dictionary):
    if dictionary.empty or "topic_id" not in dictionary:
        raise RuntimeError("No non-noise factors were discovered; cannot build an accident-factor matrix")
    invalid_roles=sorted(set(units["_role"].astype(str))-set(ROLES))
    if invalid_roles:
        raise ValueError(f"Only A0/A1/B/C units may enter the factor matrix; found {invalid_roles}")
    factors=dictionary[["topic_id","role","llm_label","llm_description"]].copy()
    factors=factors.rename(columns={"topic_id":"factor_id"})
    factors.to_csv(RUN_DIR/"factor_dictionary.csv",index=False)
    accident_ids=sorted(units["_accident_id"].unique())
    factor_ids=factors["factor_id"].astype(str).tolist()
    fact_role=dict(zip(factors.factor_id.astype(str),factors.role.astype(str)))
    if len(labels)!=len(units):raise ValueError("Factor-label count does not match the full unit table")
    frame=units[["_accident_id","_fact_id","_source_fact_id","_role"]].copy()
    frame["factor_id"]=[f"{r}_{int(label):03d}" if label>=0 else "" for r,label in zip(units["_role"],labels)]
    assigned=set(frame.loc[frame.factor_id.ne(""),"factor_id"].astype(str))
    unknown=sorted(assigned-set(factor_ids))
    if unknown:raise ValueError(f"Unit assignments refer to factors absent from the LLM dictionary: {unknown[:10]}")
    mismatched=sorted(factor for factor in assigned if fact_role.get(factor)!=factor.split("_",1)[0])
    if mismatched:raise ValueError(f"Assigned factors do not match their unit roles: {mismatched[:10]}")
    frame.to_csv(RUN_DIR/"unit_factor_assignments.csv",index=False)
    matrix=pd.DataFrame(0,index=accident_ids,columns=factor_ids,dtype=np.int8)
    for accident,factor in zip(frame._accident_id,frame.factor_id):
        if factor in matrix.columns:matrix.at[accident,factor]=1
    matrix.index.name="accident_id"
    units_lab=units.copy();units_lab["_factor_id"]=frame["factor_id"].to_numpy()
    counts=matrix.sum(axis=0); stats=factors.copy();stats["accident_count"]=stats.factor_id.map(counts).fillna(0).astype(int);stats["accident_percent"]=100*stats.accident_count/len(matrix)
    rolecounts=units_lab.loc[units_lab["_factor_id"].ne("")].groupby(["_factor_id","_role"]).size().unstack(fill_value=0)
    rolecounts=rolecounts.reindex(columns=ROLES,fill_value=0);rolecounts.index.name="factor_id";rolecounts.to_csv(RUN_DIR/"factor_unit_role_distribution.csv")
    role_distribution=factors.groupby("role").size().reindex(ROLES,fill_value=0).rename("n_factors").to_frame()
    role_distribution["percent_factors"]=100*role_distribution["n_factors"]/max(1,len(factors))
    role_distribution.to_csv(RUN_DIR/"factor_role_distribution.csv")
    stats.to_csv(RUN_DIR/"factor_prevalence.csv",index=False)
    matrix.sum(axis=1).value_counts().sort_index().rename_axis("n_factors").rename("n_accidents").to_csv(RUN_DIR/"factors_per_accident.csv")
    matrix.T.dot(matrix).to_csv(RUN_DIR/"factor_cooccurrence.csv")
    matrix.to_csv(RUN_DIR/"accident_factor_matrix.csv")
    audit={"accidents":len(matrix),"units":len(units),
      "accidents_with_no_discovered_factor":int((matrix.sum(axis=1)==0).sum()),
      "units_assigned_to_noise":int((units_lab._factor_id=="").sum()),
      "n_factors":int(len(factor_ids)),"roles":{str(role):int(count) for role,count in stats.groupby("role").size().items()}}
    (RUN_DIR/"audit_factor_matrix.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    return matrix,fact_role,stats


def _scenario_catalog_audit(candidates, matrix):
    if candidates.empty:
        return {"n_accidents":len(matrix),"n_candidates":0,
                "n_accidents_with_any_candidate":0,"candidate_catalog_coverage":0.0}, []
    covered=np.zeros(len(matrix),dtype=bool)
    exact_groups={}
    for row in candidates.to_dict("records"):
        factors=str(row["factor_ids"]).split(" | ")
        matched=matrix[factors].all(axis=1).to_numpy(dtype=bool)
        covered |= matched
        signature=tuple(np.flatnonzero(matched).tolist())
        exact_groups.setdefault(signature,[]).append(row)
    redundancy=[]
    for signature, group in exact_groups.items():
        if len(group)<2:
            continue
        representative=sorted(group,key=lambda item:(-int(item["n_factors"]),str(item["scenario_id"])))[0]
        redundancy.append({"n_accidents":len(signature),
          "candidate_catalog_coverage":len(signature)/len(matrix) if len(matrix) else 0.0,
          "n_equivalent_candidates":len(group),
          "representative_by_maximum_specificity":representative["scenario_id"],
          "scenario_ids":" | ".join(sorted(str(item["scenario_id"]) for item in group))})
    summary={"n_accidents":len(matrix),"n_candidates":len(candidates),
      "n_accidents_with_any_candidate":int(covered.sum()),
      "candidate_catalog_coverage":float(covered.mean()) if len(covered) else 0.0,
      "n_exact_redundancy_groups":len(redundancy),
      "n_candidates_in_exact_redundancy_groups":sum(int(x["n_equivalent_candidates"]) for x in redundancy)}
    return summary,redundancy


def run_network(matrix,role_map,config):
    net=config["network"];alpha=float(net["jeffreys_alpha"])
    parents,edges,score=fit_exact_bn(matrix,role_map,int(net["max_parents"]))
    candidates=scenario_candidates(matrix,role_map,edges,min_count=int(net["scenario_min_count"]))
    candidates.to_csv(RUN_DIR/"scenario_candidates.csv",index=False)
    sensitivity=scenario_candidates(matrix,role_map,edges,min_count=int(net["scenario_min_count"]),two_b=True) if net.get("scenario_two_b_sensitivity",True) else pd.DataFrame()
    sensitivity.to_csv(RUN_DIR/"scenarios_two_B_sensitivity.csv",index=False)
    summary,redundancy=_scenario_catalog_audit(candidates,matrix)
    (RUN_DIR/"scenario_catalog_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    pd.DataFrame(redundancy,columns=["n_accidents","candidate_catalog_coverage",
      "n_equivalent_candidates","representative_by_maximum_specificity","scenario_ids"]).to_csv(
        RUN_DIR/"scenario_exact_redundancy.csv",index=False)
    results={}
    for label,fit_result in [("primary",(parents,edges,score)),
                             ("within_role_sensitivity",fit_greedy_bn(matrix,role_map,int(net["max_parents"])) )]:
        parents,edges,score=fit_result
        cpts,cpttable=cpt_jeffreys(matrix,parents,alpha)
        prefix=RUN_DIR/label;prefix.mkdir(exist_ok=True)
        spec={"model":"observed-factor Bayesian network","latent_variable":False,
          "structure_method":"exact decomposable local BIC" if label=="primary" else "greedy BIC edge addition with DAG cycle checks",
          "allowed_role_arcs":[["A0","A1"],["A0","B"],["A0","C"],["A1","B"],["A1","C"],["B","C"]],
          "within_role_arcs_allowed":label!="primary","maximum_direct_parents":int(net["max_parents"]),
          "CPT_prior":f"Beta({alpha},{alpha})","interpretation":"statistical dependence; not causal effects"}
        (prefix/"model_specification.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
        pd.DataFrame(edges,columns=["parent","child"]).to_csv(prefix/"edges.csv",index=False)
        score.to_csv(prefix/"local_scores.csv",index=False);cpttable.to_csv(prefix/"cpts_jeffreys.csv",index=False)
        if label=="primary":
            descriptive=summarize_scenarios(candidates,matrix)
            probs=bn_joint_probabilities(candidates,list(matrix.columns),parents,cpts,edges,n_sim=int(net["scenario_mc_draws"]),seed=int(config["clustering_selection"]["random_state"]))
            descriptive=descriptive.merge(probs,on="scenario_id",how="left")
            descriptive.to_csv(prefix/"scenario_descriptive_results.csv",index=False)
            results[label]=(parents,edges,cpts,candidates)
        else:
            results[label]=(parents,edges,cpts,pd.DataFrame())
    bootstrap(matrix,role_map,results["primary"][3],config)


def _bootstrap_replicate(task):
    """Run one independently seeded accident bootstrap; safe for loky workers."""
    b=int(task["replicate"]);values=task["values"];names=task["names"]
    rng=np.random.default_rng(int(task["bootstrap_seed"])+b)
    boot=pd.DataFrame(values[rng.integers(0,len(values),len(values))],columns=names)
    roles=task["role_map"]
    parents,edges,_=fit_exact_bn(boot,roles,int(task["max_parents"]))
    found=scenario_candidates(boot,roles,edges,min_count=int(task["min_count"]))
    cpts,_=cpt_jeffreys(boot,parents,float(task["alpha"]))
    probabilities={}
    candidates=task["candidates"]
    if candidates:
        probs=bn_joint_probabilities(pd.DataFrame(candidates),names,parents,cpts,edges,
          n_sim=int(task["n_sim"]),seed=int(task["bootstrap_seed"])+b)
        probabilities={str(r["scenario_id"]):float(r["bn_joint_probability_mc"]) for r in probs.to_dict("records")}
    return {"replicate":b,"edges":[list(e) for e in edges],
      "selected_scenarios":found.get("scenario_id",pd.Series(dtype=str)).astype(str).tolist(),
      "scenario_probabilities":probabilities}


def bootstrap(matrix,role_map,base_candidates,config):
    from joblib import Parallel, delayed
    nrep=int(config["network"]["bootstrap_replicates"]);seed=int(config["network"]["bootstrap_seed"])
    worker_dir=RUN_DIR/"bootstrap_replicates";worker_dir.mkdir(exist_ok=True)
    values=matrix.to_numpy(dtype=np.int8);names=list(matrix.columns)
    signature_payload={"matrix_sha256":hashlib.sha256(values.tobytes()).hexdigest(),"names":names,
      "roles":dict(role_map),"replicates":nrep,"seed":seed,"max_parents":int(config["network"]["max_parents"]),
      "min_count":int(config["network"]["scenario_min_count"]),"alpha":float(config["network"]["jeffreys_alpha"]),
      "n_sim":min(5000,int(config["network"]["scenario_mc_draws"])),
      "scenario_ids":base_candidates.get("scenario_id",pd.Series(dtype=str)).astype(str).tolist(),
      "network_code_sha256":sha256(HERE/"network_analysis.py"),
      "runner_code_sha256":sha256(HERE/"run_bayesian_analysis.py")}
    signature=hashlib.sha256(json.dumps(signature_payload,sort_keys=True).encode()).hexdigest()
    state_path=RUN_DIR/"bootstrap_state.json"
    try:old=json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError):old={}
    if old.get("signature")!=signature:
        for path in worker_dir.glob("replicate_*.json"):path.unlink()
    state_path.write_text(json.dumps({"signature":signature,"requested":nrep,"completed":0},indent=2),encoding="utf-8")
    results={}
    for path in worker_dir.glob("replicate_*.json"):
        try:
            row=json.loads(path.read_text(encoding="utf-8"));results[int(row["replicate"])]=row
        except (OSError,json.JSONDecodeError,KeyError,ValueError):
            path.unlink(missing_ok=True)
    pending=[b for b in range(nrep) if b not in results]
    tasks=[{"replicate":b,"values":values,"names":names,"role_map":dict(role_map),
      "bootstrap_seed":seed,"max_parents":int(config["network"]["max_parents"]),
      "min_count":int(config["network"]["scenario_min_count"]),"alpha":float(config["network"]["jeffreys_alpha"]),
      "n_sim":signature_payload["n_sim"],"candidates":base_candidates[["scenario_id","factor_ids"]].to_dict("records")} for b in pending]
    workers=sp.resolve_n_workers(config) if config["network"].get("bootstrap_parallel",True) else 1
    backend=str(config.get("parallel",{}).get("backend","loky"))
    print(f"[bootstrap] {len(results)} cached; running {len(pending)} replicates with {workers} workers",flush=True)
    def completed_iterator():
        if workers<=1:
            yield from (_bootstrap_replicate(task) for task in tasks)
        else:
            yield from Parallel(n_jobs=workers,backend=backend,batch_size=1,verbose=0,
              return_as="generator_unordered",max_nbytes="10M",mmap_mode="r")(
                delayed(_bootstrap_replicate)(task) for task in tasks)
    iterator=completed_iterator()
    try:
        from tqdm.auto import tqdm
        if config.get("clustering_selection",{}).get("show_progress",True):iterator=tqdm(iterator,total=len(pending),desc="BN bootstrap",unit="replicate")
    except ImportError:pass
    for row in iterator:
        b=int(row["replicate"]);path=worker_dir/f"replicate_{b:04d}.json";tmp=path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(row,separators=(",",":")),encoding="utf-8");tmp.replace(path)
        results[b]=row
        (RUN_DIR/"bootstrap_progress.json").write_text(json.dumps({"completed":len(results),"requested":nrep}),encoding="utf-8")
    if len(results)!=nrep:raise RuntimeError(f"Only {len(results)}/{nrep} bootstrap replicates are available")
    edge_hits={};scenario_hits={s:0 for s in base_candidates.get("scenario_id",[])};prob_sums={s:0.0 for s in scenario_hits}
    for b in range(nrep):
        row=results[b]
        for edge in row["edges"]:
            edge=tuple(edge);edge_hits[edge]=edge_hits.get(edge,0)+1
        found=set(row["selected_scenarios"])
        for sid in scenario_hits:scenario_hits[sid]+=int(sid in found)
        for sid,value in row["scenario_probabilities"].items():prob_sums[sid]+=float(value)
    possible=[]
    order={r:i for i,r in enumerate(ROLES)}
    for p in role_map:
        for c in role_map:
            if p!=c and order[role_map[p]]<order[role_map[c]]:
                possible.append((p,c))
    pd.DataFrame([{"parent":p,"child":c,"selection_frequency":edge_hits.get((p,c),0)/nrep,"selected_in":edge_hits.get((p,c),0),"replicates":nrep} for p,c in possible]).to_csv(RUN_DIR/"bootstrap_edge_stability.csv",index=False)
    pd.DataFrame([{"scenario_id":s,"selection_frequency":scenario_hits[s]/nrep,"mean_bn_probability":prob_sums[s]/nrep,"replicates":nrep} for s in scenario_hits]).to_csv(RUN_DIR/"bootstrap_scenario_stability.csv",index=False)
    state_path.write_text(json.dumps({"signature":signature,"requested":nrep,"completed":nrep,"status":"complete"},indent=2),encoding="utf-8")


def main():
    global RUN_DIR
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=HERE/"config.yaml");p.add_argument("--run-dir",type=Path);p.add_argument("--reestimate",action="store_true")
    p.add_argument("--stage",choices=("all","factors"),default="all",
                   help="Run the complete analysis or stop after factor annotation and matrix creation")
    args=p.parse_args(); config_path=args.config.resolve()
    with config_path.open(encoding="utf-8") as f:config=yaml.safe_load(f)
    for section,key in [("data","units_path"),("data","embeddings_path"),("topics","stopwords_file")]:
        val=Path(config[section][key]);config[section][key]=str((config_path.parent/val).resolve() if not val.is_absolute() else val)
    analysis_id=str(config.get("analysis_id","btp")).strip().lower()
    run_id=re.sub(r"[^a-z0-9_-]+","_",analysis_id).strip("_-") or "bayesian_analysis"
    RUN_DIR=args.run_dir.resolve() if args.run_dir else HERE/"runs"/(run_id+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    RUN_DIR.mkdir(parents=True,exist_ok=True)
    (RUN_DIR/"config_resolved.yaml").write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding="utf-8")
    # Write an initial manifest before expensive work so interrupted runs retain provenance.
    source_files=[HERE/"run_bayesian_analysis.py",HERE/"network_analysis.py",TEXT_ROOT/"recurrent_scenarios"/"scenario_pipeline.py",TEXT_ROOT/"recurrent_scenarios"/"factor_annotation.py"]
    initial_manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"status":"running","stage":args.stage,
      "config_sha256":sha256(RUN_DIR/"config_resolved.yaml"),"source_code_sha256":{p.name:sha256(p) for p in source_files},
      "input_sha256":{key:sha256(Path(config["data"][key])) for key in ("units_path","embeddings_path") if Path(config["data"][key]).is_file()}}
    (RUN_DIR/"manifest.json").write_text(json.dumps(initial_manifest,indent=2),encoding="utf-8")
    if not Path(config["data"]["embeddings_path"]).is_file():
        raise FileNotFoundError(f"Required embeddings are missing: {config['data']['embeddings_path']}. Place the configured aligned embedding export at this path; this pipeline does not encode text or switch embedding models.")
    if config.get("factor_annotation",{}).get("enabled"):check_annotation_credentials(config)
    units,vectors,audit=load_inputs(config)
    full_data=make_prepared(units,vectors);selections={}
    for role in ROLES:
        metrics=sp.evaluate_candidates(role,units,vectors,config,RUN_DIR,reestimate=args.reestimate)
        theme,summary=sp.evaluate_resampling_stability(role,units,vectors,config,RUN_DIR,metrics,reestimate=args.reestimate)
        merged=metrics.merge(summary[["configuration_id","stability"]],on="configuration_id",how="left")
        table,selected,rule=sp.select_configuration_for_role(merged)
        table.to_csv(RUN_DIR/"discovery"/role/"selection.csv",index=False)
        space_name = "umap" if config["screening"]["umap"].get("enabled", True) else "cluster_space"
        sp.materialize_selected_partition(role,units,selected,RUN_DIR,theme,cluster_space_name=space_name)
        selections[role]=selected
        print(f"[selection] {role}: {selected} ({rule})",flush=True)
    if config.get("factor_annotation",{}).get("enabled"):
        dictionary_path=annotate_selected_factors(full_data,selections,config,RUN_DIR)
    else:
        raise RuntimeError("This analysis requires factor_annotation.enabled=true")
    dictionary=read_csv(dictionary_path)
    factor_labels=np.full(len(units),-1,dtype=int)
    for role in ROLES:
        mask=units._role.eq(role).to_numpy()
        selected_labels=np.load(RUN_DIR/"discovery"/role/"selected"/"labels.npy").astype(int)
        if len(selected_labels)!=int(mask.sum()):
            raise RuntimeError(f"Selected full-corpus labels do not align with the {role} units")
        factor_labels[mask]=selected_labels
    matrix,role_map,stats=build_factor_matrix(units,factor_labels,dictionary)
    if args.stage=="all":
        run_network(matrix,role_map,config)
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"status":"complete","stage":args.stage,
      "config_sha256":sha256(RUN_DIR/"config_resolved.yaml"),
      "units_sha256":sha256(Path(config["data"]["units_path"])),"embeddings_sha256":sha256(Path(config["data"]["embeddings_path"])),
      "source_code_sha256":{p.name:sha256(p) for p in source_files},
      "input_audit":audit,"factor_selections":selections,
      "descriptive_scope":"all eligible accidents in the configured corpus"}
    if args.stage=="all":
        manifest["bootstrap_conditional_on"]="full-corpus factor discovery, LLM labels and unit-to-factor assignments; factors are held fixed across bootstrap repetitions"
    (RUN_DIR/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Completed. Results: {RUN_DIR}",flush=True)

if __name__=="__main__":
    try:
        main()
    except Exception as error:
        run_dir=globals().get("RUN_DIR")
        manifest_path=(Path(run_dir)/"manifest.json") if run_dir else None
        if manifest_path and manifest_path.is_file():
            try:
                manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update({"status":"failed","failure_type":type(error).__name__,"failure_message":str(error)[:1000]})
                manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
            except OSError:
                pass
        raise
