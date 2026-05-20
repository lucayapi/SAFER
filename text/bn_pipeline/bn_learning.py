"""Estimation des paramètres (CPD) et sauvegarde des modèles pgmpy."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd


def drop_constant_columns(df: pd.DataFrame, cols: List[str]) -> tuple[pd.DataFrame, List[str]]:
    use = []
    for c in cols:
        if c not in df.columns:
            continue
        if df[c].nunique(dropna=False) <= 1:
            continue
        use.append(c)
    return df[use].copy(), use


def fit_bn_parameters(
    model: Any,
    data: pd.DataFrame,
    estimator: str = "bayesian",
    equivalent_sample_size: int = 5,
) -> Any:
    from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator

    nodes = list(model.nodes())
    df = data[[c for c in nodes if c in data.columns]].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if estimator == "bayesian":
        try:
            model.fit(df, estimator=BayesianEstimator, equivalent_sample_size=int(equivalent_sample_size))
        except Exception:
            model.fit(df, estimator=MaximumLikelihoodEstimator)
    else:
        model.fit(df, estimator=MaximumLikelihoodEstimator)
    return model


def save_bn_pickle(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def export_cpds_to_dir(model: Any, out_dir: Path, prefix: str = "cpd") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cpd in model.get_cpds():
        name = cpd.variable
        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
        txt_path = out_dir / f"{prefix}_{safe}.txt"
        txt_path.write_text(str(cpd), encoding="utf-8")
        vals = cpd.values
        pd.DataFrame(vals.reshape(vals.shape[0], -1)).to_csv(
            out_dir / f"{prefix}_{safe}.csv", index=False
        )


def try_write_bif(model: Any, path: Path) -> bool:
    try:
        from pgmpy.readwrite import BIFWriter

        path.parent.mkdir(parents=True, exist_ok=True)
        writer = BIFWriter(model)
        writer.write_bif(str(path))
        return True
    except Exception:
        return False


def check_model_safe(model: Any) -> tuple[bool, str]:
    try:
        ok = model.check_model()
        return bool(ok), ""
    except Exception as e:
        return False, str(e)
