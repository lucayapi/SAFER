"""Interprétations OpenAI des scénarios typiques BN (co-présence de topics)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from bn_pipeline.aggregate_bn_variables import _macro_topic_column_name
from bn_pipeline.bn_visualization import _macro_topic_from_node, _macro_topic_to_theme_summary

_SCENARIO_SYSTEM = (
    "Tu es expert en prévention des risques professionnels (accidents du travail, BTP, industrie). "
    "À partir d'une configuration de co-présence de facteurs/thèmes dans des récits d'accidents, "
    "rédige une interprétation courte (1 à 2 phrases) en français, orientée scénario de prévention. "
    "Réponds uniquement en JSON : {\"interpretation\": \"...\"}"
)


def build_topic_variable_label_map(themes_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    """Mappe ``macro_topic_A0_03`` → libellé lisible."""
    mt_map = _macro_topic_to_theme_summary(themes_df)
    out: Dict[str, str] = {}
    for (macro, tid), label in mt_map.items():
        var = _macro_topic_column_name(macro, tid)
        short = str(label).strip()
        if len(short) > 80:
            short = short[:77] + "…"
        out[var] = short or f"{macro} topic {tid}"
    return out


def _topics_present_to_chain(
    topics_present: str,
    label_map: Dict[str, str],
) -> str:
    raw = str(topics_present or "").strip()
    if not raw:
        return ""
    tokens = [t.strip() for t in re.split(r"\s*\+\s*", raw) if t.strip()]
    labels: List[str] = []
    for tok in tokens:
        if tok in label_map:
            labels.append(label_map[tok])
            continue
        mt = _macro_topic_from_node(tok)
        if mt is not None:
            var = _macro_topic_column_name(mt[0], mt[1])
            labels.append(label_map.get(var, f"{mt[0]} T{mt[1]}"))
        else:
            labels.append(tok.replace("macro_topic_", "").replace("_", " "))
    return " → ".join(labels)


def _macro_path_to_chain(macro_path: str) -> str:
    s = str(macro_path or "").strip()
    if not s:
        return ""
    return s.replace(" -> ", " → ").replace("->", "→")


def build_configuration_probable(
    row: pd.Series,
    label_map: Dict[str, str],
) -> str:
    """Chaîne lisible pour affichage notebook."""
    chain = _topics_present_to_chain(str(row.get("topics_present", "")), label_map)
    if chain:
        return chain
    return _macro_path_to_chain(str(row.get("macro_path", "")))


def _interpret_one_openai(
    client: Any,
    model: str,
    configuration: str,
    representative_sentences: str,
    *,
    temperature: float = 0.3,
) -> str:
    user = (
        "Configuration probable (co-présence de motifs) :\n"
        f"{configuration}\n\n"
        "Extraits représentatifs d'accidents :\n"
        f"{representative_sentences[:2500]}\n\n"
        "Propose une interprétation courte (scénario de prévention)."
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": _SCENARIO_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("interpretation"):
            return str(data["interpretation"]).strip()
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(content[start : end + 1])
            if isinstance(data, dict) and data.get("interpretation"):
                return str(data["interpretation"]).strip()
    return content[:500]


def enrich_scenarios_table(
    freq_df: pd.DataFrame,
    n_accidents: int,
    themes_df: Optional[pd.DataFrame] = None,
    *,
    enable_openai: bool = True,
    openai_model: Optional[str] = None,
    cache_path: Optional[Path] = None,
    max_rows: int = 15,
) -> pd.DataFrame:
    """
    Ajoute ``configuration_probable``, ``prob``, ``interpretation`` au tableau de scénarios.

    ``prob`` = support / n_accidents.
    """
    if freq_df.empty:
        return pd.DataFrame(
            columns=[
                "configuration_probable",
                "interpretation",
                "prob",
                "support",
                "macro_path",
                "topics_present",
            ]
        )

    label_map = build_topic_variable_label_map(themes_df)
    n_acc = max(1, int(n_accidents))
    rows: List[Dict[str, Any]] = []
    subset = freq_df.head(max_rows)

    cache: Dict[str, str] = {}
    if cache_path is not None and Path(cache_path).is_file():
        try:
            cached = pd.read_csv(cache_path)
            if "configuration_probable" in cached.columns and "interpretation" in cached.columns:
                for _, cr in cached.iterrows():
                    key = str(cr["configuration_probable"])
                    cache[key] = str(cr.get("interpretation", ""))
        except Exception:
            cache = {}

    client = None
    model = openai_model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if enable_openai and os.environ.get("OPENAI_API_KEY"):
        try:
            from scgm_text.openai_theme_labels import _get_client, load_openai_dotenv

            load_openai_dotenv()
            client = _get_client()
        except Exception as exc:
            print(f"[scenario_interpretation] OpenAI indisponible : {exc}")
            client = None
    elif enable_openai:
        print(
            "[scenario_interpretation] OPENAI_API_KEY absente — "
            "tableau sans colonne interpretation."
        )

    for _, row in subset.iterrows():
        config = build_configuration_probable(row, label_map)
        support = int(row.get("support", 0))
        prob = round(float(support) / n_acc, 2)
        interpretation = cache.get(config, "")
        if not interpretation and client is not None:
            rep = str(row.get("representative_sentences", ""))[:2500]
            try:
                interpretation = _interpret_one_openai(
                    client, model, config, rep
                )
            except Exception as exc:
                interpretation = f"(échec API: {exc})"
        rows.append(
            {
                "configuration_probable": config,
                "interpretation": interpretation,
                "prob": prob,
                "support": support,
                "macro_path": row.get("macro_path", ""),
                "topics_present": row.get("topics_present", ""),
                "scenario_id": row.get("scenario_id"),
            }
        )

    out = pd.DataFrame(rows)
    display_cols = ["configuration_probable", "interpretation", "prob"]
    return out[[c for c in display_cols if c in out.columns] + [c for c in out.columns if c not in display_cols]]


def export_scenario_interpretations(
    df: pd.DataFrame,
    out_path: Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
