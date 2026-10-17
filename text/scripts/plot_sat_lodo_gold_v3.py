#!/usr/bin/env python3
r"""Compare the completed narrative_v3 SaT LODO folds on identical target texts.

From SAFER on Windows:
    .\text\.venv\Scripts\python.exe text\scripts\plot_sat_lodo_gold_v3.py

The script labels incomplete imports as partial and can be rerun after all six
target folders have been copied from the cluster. It never retrains a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "safer_sat_lodo" / "runs" / "sat12l_sm_gold_v3_rq1"
CORPORA = {
    "btp": "BTP",
    "metallurgie": "Métallurgie",
    "caou_chimie_plas": "Chimie-plastiques",
    "bois_textille": "Bois-textile",
    "service_temporaire": "Service temporaire",
    "nicollin": "Nicollin",
}
MODELS = ("M0", "M1", "M2")
COLORS = {"M0": "#8795a4", "M1": "#37699a", "M2": "#bc7047"}
ROLES = ("A0", "A1", "B", "C", "AUTRE")


def load_folds(run_root: Path) -> dict[str, dict]:
    folds = {}
    signatures = set()
    baseline_path = ROOT / "annotation" / "rq1_sat_v3_20261017" / "rq1_main_table.csv"
    with baseline_path.open(encoding="utf-8-sig", newline="") as stream:
        baseline = {row["target"]: row for row in csv.DictReader(stream)}
    for name in CORPORA:
        path = run_root / f"target_{name}" / "metrics.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["target"] != name or data["model"] != "sat-12l-sm" or \
                data["reference"] != "USER_CONFIRMED_NARRATIVE_V3_GOLD":
            raise ValueError(f"Résultat incompatible : {path}")
        signatures.add(data["bundle_manifest_sha256"])
        test = data["test"]
        if set(test) != set(MODELS) or len({test[m]["n_documents"] for m in MODELS}) != 1:
            raise ValueError(f"Modèles ou effectifs incohérents : {name}")
        for model in MODELS:
            for kind in ("span", "boundary"):
                item = test[model]
                tp, pred, ref = (int(item[f"{kind}_{field}"]) for field in ("tp", "pred", "ref"))
                expected = 2 * tp / (pred + ref) if pred + ref else 0.0
                if not math.isclose(expected, item[f"{kind}_f1"], abs_tol=1e-12):
                    raise ValueError(f"F1 incohérent : {name}/{model}/{kind}")
        if test["M0"]["n_documents"] != int(baseline[name]["n_documents"]):
            raise ValueError(f"Effectif M0 différent du RQ1 initial : {name}")
        for kind in ("span", "boundary"):
            if not math.isclose(test["M0"][f"{kind}_f1"],
                                float(baseline[name][f"M0_{kind}_f1"]), abs_tol=1e-12):
                raise ValueError(f"M0 ne reproduit pas RQ1 : {name}/{kind}")
        folds[name] = data
    if len(signatures) > 1:
        raise ValueError("Les rotations proviennent de paquets de données différents.")
    if not folds:
        raise FileNotFoundError(f"Aucun metrics.json complet dans {run_root}")
    return folds


def paired_ci(path: Path, target: str, expected: dict, reps: int) -> dict[str, tuple[float, float]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_model = {model: {row["document_id"]: row for row in rows if row["model"] == model}
                for model in MODELS}
    ids = sorted(by_model["M1"])
    if (not ids or len(rows) != 3 * len(ids)
            or any(set(by_model[model]) != set(ids) for model in MODELS)
            or len(ids) != expected["M1"]["n_documents"]):
        raise ValueError(f"Récits M0/M1/M2 non appariés : {target}")
    for model in MODELS:
        for kind in ("span", "boundary"):
            for key in ("tp", "pred", "ref"):
                total = sum(int(by_model[model][doc][f"{kind}_{key}"]) for doc in ids)
                if total != expected[model][f"{kind}_{key}"]:
                    raise ValueError(f"Comptage des récits incohérent : {target}/{model}/{kind}/{key}")
    rng = np.random.default_rng(20261017 + list(CORPORA).index(target))
    samples = rng.integers(0, len(ids), (reps, len(ids)))
    result = {}
    for kind in ("span", "boundary"):
        values = {}
        for model in ("M1", "M2"):
            matrix = np.array([[int(by_model[model][doc][f"{kind}_{key}"])
                                for key in ("tp", "pred", "ref")] for doc in ids], dtype=np.int64)
            totals = matrix[samples].sum(axis=1)
            values[model] = np.divide(2 * totals[:, 0], totals[:, 1] + totals[:, 2],
                                      out=np.zeros(reps, dtype=float),
                                      where=totals[:, 1] + totals[:, 2] > 0)
        gains = values["M2"] - values["M1"]
        result[kind] = (float(np.quantile(gains, 0.025)),
                        float(np.quantile(gains, 0.975)))
    return result


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scores(folds: dict[str, dict], out: Path, note: str) -> None:
    names = list(folds)
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(4, len(names) * 0.75 + 1.8)),
                             sharey=True, constrained_layout=True)
    for ax, (kind, title) in zip(axes, (("span", "Segments exacts"), ("boundary", "Frontières"))):
        for offset, model in zip((-0.24, 0, 0.24), MODELS):
            values = [folds[name]["test"][model][f"{kind}_f1"] for name in names]
            ax.barh(y + offset, values, height=0.22, color=COLORS[model], label=model)
        ax.set_xlim(0, 0.76)
        ax.set_xlabel("F1")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [CORPORA[name] for name in names])
    axes[0].invert_yaxis()
    axes[0].legend(title="Modèle", loc="lower right")
    fig.suptitle(f"SaT : M0, M1 calibré et M2 adapté — {note}", fontweight="bold")
    save(fig, out, "f1_M0_M1_M2")


def plot_gains(folds: dict[str, dict], ci: dict[str, dict], out: Path, note: str) -> None:
    names = list(folds)
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(4, len(names) * 0.75 + 1.8)),
                             sharey=True, constrained_layout=True)
    for ax, (kind, title) in zip(axes, (("span", "Segments exacts"), ("boundary", "Frontières"))):
        gains = [folds[name]["test"]["M2"][f"{kind}_f1"] -
                 folds[name]["test"]["M1"][f"{kind}_f1"] for name in names]
        lows = [ci[name][kind][0] for name in names]
        highs = [ci[name][kind][1] for name in names]
        ax.barh(y, gains, height=0.54, color=["#23957d" if value >= 0 else "#b75c57"
                                                for value in gains])
        ax.errorbar(gains, y, xerr=[[value - low for value, low in zip(gains, lows)],
                                      [high - value for value, high in zip(gains, highs)]],
                    fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.axvline(0, color="#303030", linewidth=1)
        extent = max(abs(v) for v in lows + highs + gains) * 1.45
        ax.set_xlim(-extent, extent * 1.23)
        for yy, value in zip(y, gains):
            ax.text(extent * 1.04, yy, f"{value:+.3f}", ha="left", va="center", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Différence de F1 (M2 − M1)")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [CORPORA[name] for name in names])
    axes[0].invert_yaxis()
    fig.suptitle(f"Gain de l'adaptation — IC 95 % appariés par récit — {note}", fontweight="bold")
    save(fig, out, "gain_M2_moins_M1")


def plot_errors(folds: dict[str, dict], out: Path, note: str) -> None:
    names = list(folds)
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(4, len(names) * 0.75 + 1.8)),
                             sharey=True, constrained_layout=True)
    for ax, (field, title) in zip(axes, (("fusion_rate_supported", "Fusions"),
                                        ("fragmentation_rate", "Fragmentations"))):
        for offset, model in zip((-0.16, 0.16), ("M1", "M2")):
            values = [folds[name]["test"][model][field] for name in names]
            ax.barh(y + offset, values, height=0.3, label=model, color=COLORS[model])
        ax.set_xlabel("Taux")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [CORPORA[name] for name in names])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right")
    fig.suptitle(f"Profils d'erreurs M1 et M2 — {note}", fontweight="bold")
    save(fig, out, "fusion_fragmentation_M1_M2")


def plot_roles(folds: dict[str, dict], out: Path, note: str) -> None:
    names = list(folds)
    matrix = np.array([[folds[name]["test"]["M2"]["role_exact_recall"][role] -
                        folds[name]["test"]["M1"]["role_exact_recall"][role]
                        for role in ROLES] for name in names])
    extent = max(0.01, float(np.abs(matrix).max()))
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, len(names) * 0.6 + 1.4)), constrained_layout=True)
    im = ax.imshow(matrix, cmap="RdBu", vmin=-extent, vmax=extent, aspect="auto")
    ax.set_xticks(range(len(ROLES)), ROLES)
    ax.set_yticks(range(len(names)), [CORPORA[name] for name in names])
    for i in range(len(names)):
        for j in range(len(ROLES)):
            ax.text(j, i, f"{matrix[i,j]:+.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Gain de rappel exact (M2 − M1)")
    ax.set_title(f"Rappel exact par rôle — {note}", fontweight="bold")
    save(fig, out, "gain_rappel_roles_M2_moins_M1")


def plot_loss(folds: dict[str, dict], run_root: Path, out: Path, note: str) -> None:
    fig, axes = plt.subplots(len(folds), 1, figsize=(9, max(3.4, len(folds) * 2.5)),
                             squeeze=False, constrained_layout=True)
    for ax, name in zip(axes[:, 0], folds):
        path = run_root / f"target_{name}" / "loss_history.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for metric, label, color in (("loss", "Entraînement", COLORS["M2"]),
                                     ("eval_loss", "Validation", COLORS["M1"])):
            current = [row for row in rows if row["metric"] == metric]
            ax.plot([float(row["epoch"]) for row in current],
                    [float(row["value"]) for row in current],
                    marker="o" if metric == "eval_loss" else None,
                    markersize=3, linewidth=1.2, label=label, color=color)
        ax.set_ylabel(CORPORA[name] + "\nperte")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
    axes[-1, 0].set_xlabel("Époque")
    fig.suptitle(f"LoRA : pertes d'entraînement et de validation — {note}", fontweight="bold")
    save(fig, out, "courbes_pertes_M2")


def write_report(folds: dict[str, dict], ci: dict[str, dict], out: Path, note: str) -> None:
    lines = [f"# Comparaison M2 (SaT + LoRA) contre M1 — {note}", "",
             "Référence : annotations narrative_v3 confirmées par l'utilisatrice. "
             "M1 et M2 utilisent les mêmes récits cibles ; leurs seuils sont choisis "
             "séparément sur la même validation des cinq corpus sources. "
             "Les espaces périphériques sont neutralisés dans les métriques.", "",
             "| Cible | Récits | Seuil M1 | Seuil M2 | F1 exact M1 | F1 exact M2 | "
             "Gain exact [IC 95 %] | F1 frontières M1 | F1 frontières M2 | "
             "Gain frontières [IC 95 %] |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, data in folds.items():
        test = data["test"]
        parts = []
        for kind in ("span", "boundary"):
            gain = test["M2"][f"{kind}_f1"] - test["M1"][f"{kind}_f1"]
            if name in ci:
                low, high = ci[name][kind]
                parts.append(f"{gain:+.4f} [{low:+.4f} ; {high:+.4f}]")
            else:
                parts.append(f"{gain:+.4f} [IC indisponible]")
        lines.append(f"| {CORPORA[name]} | {test['M1']['n_documents']} | "
                     f"{data['selection']['M1_threshold']:.5g} | "
                     f"{data['selection']['M2_threshold']:.5g} | "
                     f"{test['M1']['span_f1']:.4f} | {test['M2']['span_f1']:.4f} | "
                     f"{parts[0]} | {test['M1']['boundary_f1']:.4f} | "
                     f"{test['M2']['boundary_f1']:.4f} | {parts[1]} |")
    lines += ["", "## Erreurs de segmentation", "",
              "| Cible | Segments M1 | Segments M2 | Précision exacte M1 → M2 | "
              "Rappel exact M1 → M2 | Fusions M1 → M2 | Fragmentations M1 → M2 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, data in folds.items():
        m1, m2 = data["test"]["M1"], data["test"]["M2"]
        lines.append(f"| {CORPORA[name]} | {m1['span_pred']} | {m2['span_pred']} | "
                     f"{m1['span_precision']:.3f} → {m2['span_precision']:.3f} | "
                     f"{m1['span_recall']:.3f} → {m2['span_recall']:.3f} | "
                     f"{m1['fusion_rate_supported']:.3f} → {m2['fusion_rate_supported']:.3f} | "
                     f"{m1['fragmentation_rate']:.3f} → {m2['fragmentation_rate']:.3f} |")
    lines += ["", "Les IC sont issus d'un bootstrap apparié par récit à seuil fixé. "
              "Les taux de fusion et de fragmentation ont des dénominateurs distincts ; "
              "ils servent à comprendre les erreurs et ne sont pas interchangeables.", ""]
    if len(folds) < len(CORPORA):
        missing = ", ".join(CORPORA[name] for name in CORPORA if name not in folds)
        lines += [f"**Analyse incomplète : {missing} manquent dans la copie locale.** "
                  "Aucune moyenne macro des six corpus ni conclusion de généralisation "
                  "ne peut encore être calculée.", ""]
    (out / "rapport_M2_vs_M1.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap doit être >= 100")
    run_root = args.run_root.resolve()
    out = (args.out or run_root / "analyse_M2_vs_M1").resolve()
    out.mkdir(parents=True, exist_ok=True)
    folds = load_folds(run_root)
    note = f"{len(folds)}/6 corpus" + (" (partiel)" if len(folds) < 6 else "")
    ci = {}
    for name in folds:
        path = run_root / f"target_{name}" / "document_metrics.csv"
        if path.is_file():
            ci[name] = paired_ci(path, name, folds[name]["test"], args.bootstrap)
    rows = []
    for name, data in folds.items():
        test = data["test"]
        row = {"target": name, "n_documents": test["M1"]["n_documents"],
               "M1_threshold": data["selection"]["M1_threshold"],
               "M2_threshold": data["selection"]["M2_threshold"]}
        for kind in ("span", "boundary"):
            for model in MODELS:
                for field in ("precision", "recall", "f1"):
                    row[f"{model}_{kind}_{field}"] = test[model][f"{kind}_{field}"]
            row[f"gain_{kind}_f1"] = row[f"M2_{kind}_f1"] - row[f"M1_{kind}_f1"]
            if name in ci:
                row[f"gain_{kind}_ci95_low"], row[f"gain_{kind}_ci95_high"] = ci[name][kind]
        rows.append(row)
    with (out / "comparaison_M2_vs_M1.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    plot_scores(folds, out, note)
    if len(ci) == len(folds):
        plot_gains(folds, ci, out, note)
    plot_errors(folds, out, note)
    plot_roles(folds, out, note)
    plot_loss(folds, run_root, out, note)
    write_report(folds, ci, out, note)
    print(json.dumps({"output": str(out), "available_targets": list(folds),
                      "missing_targets": [name for name in CORPORA if name not in folds],
                      "complete": len(folds) == len(CORPORA)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
