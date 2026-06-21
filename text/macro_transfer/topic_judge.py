"""Évaluation LLM-judge de la pertinence des topics BERTopic intra-macro."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from macro_transfer.constants import MACRO_NAMES
from scgm_text.openai_theme_labels import (
    _get_client,
    _parse_json_content,
    load_openai_dotenv,
)

JUDGE_CRITERIA: Tuple[str, ...] = (
    "coherence_interne",
    "homogeneite_accidentologique",
    "alignement_role",
    "specificite",
    "nommabilite",
    "utilite_reconstruction_scenario",
)

VALID_VERDICTS = frozenset({"conserver", "fusionner", "scinder", "rejeter"})
VALID_PROBLEMES = frozenset(
    {
        "aucun",
        "trop_general",
        "heterogene",
        "mauvais_role",
        "trop_fragmente",
        "exemples_insuffisants",
    }
)

_ROLE_LABELS: Dict[str, str] = {
    "A0": "A0 : contexte ou situation de travail avant l'accident",
    "A1": "A1 : facteur défavorable, condition dangereuse ou défaillance",
    "B": "B : événement accidentel ou mécanisme immédiat",
    "C": "C : conséquence, lésion ou dommage",
}

_SYSTEM_PROMPT = (
    "Tu es un évaluateur spécialisé en analyse de récits d'accidents du travail. "
    "Réponds uniquement en JSON strict, sans markdown."
)

_USER_PROMPT_TEMPLATE = """On te donne un topic extrait automatiquement à partir d'unités factuelles courtes.

Une unité factuelle est un segment court du récit qui porte une information élémentaire utile à l'analyse de la dynamique de l'accident.

Ton objectif est d'évaluer si le topic correspond à un motif factuel clair, homogène et exploitable pour reconstruire une chaîne accidentelle.

Rôle attendu :
{ROLE}

Définition des rôles :

* A0 : contexte ou situation de travail avant l'accident
* A1 : facteur défavorable, condition dangereuse ou défaillance
* B : événement accidentel ou mécanisme immédiat
* C : conséquence, lésion ou dommage

Topic à évaluer :

* Taille du topic : {TOPIC_SIZE}
* Exemples représentatifs : {REPRESENTATIVE_EXAMPLES}
* Exemples aléatoires : {RANDOM_EXAMPLES}

Évalue le topic de 0 à 5 selon les critères suivants :

1. coherence_interne :
   Les unités du topic décrivent-elles globalement le même type d'information factuelle ?

2. homogeneite_accidentologique :
   Les unités renvoient-elles au même contexte, facteur, événement ou dommage ?

3. alignement_role :
   Le topic correspond-il bien au rôle attendu A0, A1, B ou C ?

4. specificite :
   Le topic est-il assez précis, ou trop général ?

5. nommabilite :
   Peut-on proposer un label court, clair et accidentologiquement pertinent à partir des exemples fournis ?

6. utilite_reconstruction_scenario :
   Le topic peut-il servir comme motif ou nœud dans une chaîne accidentelle de type A0 → A1 → B → C ?

Règles :

* N'évalue pas directement la prévention.
* Ne propose pas de recommandation de prévention.
* Ne déduis rien qui n'est pas visible dans les exemples fournis.
* Ne suppose pas l'existence d'un mécanisme si les exemples ne permettent pas de l'établir.
* Pénalise les topics trop génériques.
* Pénalise les topics qui mélangent plusieurs mécanismes.
* Pénalise les topics dont le label proposé serait trop vague.
* Si les exemples sont trop courts, trop vagues ou insuffisants, baisse les scores.
* Les exemples représentatifs peuvent être plus propres que le topic réel ; utilise aussi les exemples aléatoires pour juger l'homogénéité globale.

Réponds uniquement en JSON strict (sans score_global — il sera calculé côté code) :

{{
"label_propose": "...",
"coherence_interne": 0,
"homogeneite_accidentologique": 0,
"alignement_role": 0,
"specificite": 0,
"nommabilite": 0,
"utilite_reconstruction_scenario": 0,
"verdict": "conserver | fusionner | scinder | rejeter",
"justification_courte": "...",
"probleme_principal": "aucun | trop_general | heterogene | mauvais_role | trop_fragmente | exemples_insuffisants"
}}"""


def role_label_for_macro(macro: str) -> str:
    """Libellé rôle attendu pour le prompt judge."""
    key = str(macro).strip().upper()
    return _ROLE_LABELS.get(key, key)


def compute_score_global(scores: Mapping[str, Any]) -> float:
    """Moyenne des 6 critères numériques (0–5)."""
    vals = [float(scores[c]) for c in JUDGE_CRITERIA]
    return float(np.mean(vals))


def _format_examples_block(examples: Sequence[str]) -> str:
    if not examples:
        return "(aucun exemple disponible)"
    lines = []
    for i, ex in enumerate(examples, start=1):
        text = str(ex).strip()
        if not text:
            continue
        short = text[:800] + ("…" if len(text) > 800 else "")
        lines.append(f"{i}. {short}")
    return "\n".join(lines) if lines else "(aucun exemple disponible)"


def build_topic_judge_prompt(
    macro: str,
    topic_size: int,
    rep_examples: Sequence[str],
    random_examples: Sequence[str],
) -> str:
    """Construit le message utilisateur pour le judge LLM."""
    return _USER_PROMPT_TEMPLATE.format(
        ROLE=role_label_for_macro(macro),
        TOPIC_SIZE=int(topic_size),
        REPRESENTATIVE_EXAMPLES=_format_examples_block(rep_examples),
        RANDOM_EXAMPLES=_format_examples_block(random_examples),
    )


def _clamp_score(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Score invalide : {value!r}") from exc
    return float(max(0.0, min(5.0, x)))


def parse_judge_response(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Valide la réponse JSON et calcule ``score_global`` en Python."""
    out: Dict[str, Any] = {}
    for key in JUDGE_CRITERIA:
        if key not in data:
            raise KeyError(f"Critère manquant dans la réponse judge : {key!r}")
        out[key] = _clamp_score(data[key])

    label = str(data.get("label_propose", "")).strip()
    out["label_propose"] = label

    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"verdict invalide : {verdict!r}")
    out["verdict"] = verdict

    probleme = str(data.get("probleme_principal", "aucun")).strip().lower()
    if probleme not in VALID_PROBLEMES:
        probleme = "aucun"
    out["probleme_principal"] = probleme

    out["justification_courte"] = str(data.get("justification_courte", "")).strip()
    out["score_global"] = round(compute_score_global(out), 4)
    return out


def _split_top_sentences(raw: str) -> List[str]:
    parts = [p.strip() for p in str(raw).split(" || ") if p and str(p).strip()]
    return parts


def sample_topic_examples(
    assignments: pd.DataFrame,
    meta: pd.DataFrame,
    macro: str,
    topic_id: int,
    sentence_col: str,
    *,
    n_rep: int,
    n_random: int,
    themes_row: Optional[pd.Series] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[str], List[str]]:
    """Échantillonne exemples représentatifs et aléatoires pour un topic."""
    rng = rng or np.random.default_rng(42)
    macro_s = str(macro).strip()
    tid = int(topic_id)

    rep_examples: List[str] = []
    if themes_row is not None and pd.notna(themes_row.get("top_sentences")):
        rep_examples = _split_top_sentences(str(themes_row["top_sentences"]))[: max(1, n_rep)]

    sub = assignments.loc[
        (assignments["macro"].astype(str) == macro_s)
        & (pd.to_numeric(assignments["topic_id"], errors="coerce").fillna(-1).astype(int) == tid)
    ]
    doc_idx = sub["doc_idx"].astype(int).to_numpy() if "doc_idx" in sub.columns else np.array([], dtype=int)
    meta_idx = meta.reset_index(drop=True)
    all_sentences = [
        str(meta_idx.iloc[int(i)][sentence_col]).strip()
        for i in doc_idx
        if 0 <= int(i) < len(meta_idx)
    ]
    all_sentences = [s for s in all_sentences if s]

    if len(rep_examples) < n_rep and all_sentences:
        rep_pool = [s for s in all_sentences if s not in rep_examples]
        need = n_rep - len(rep_examples)
        if rep_pool:
            pick = rng.choice(len(rep_pool), size=min(need, len(rep_pool)), replace=False)
            rep_examples.extend(rep_pool[int(i)] for i in np.atleast_1d(pick))

    random_pool = [s for s in all_sentences if s not in rep_examples]
    n_rand = min(max(0, n_random), len(random_pool))
    if n_rand > 0:
        pick = rng.choice(len(random_pool), size=n_rand, replace=False)
        random_examples = [random_pool[int(i)] for i in np.atleast_1d(pick)]
    else:
        random_examples = []

    return rep_examples[:n_rep], random_examples


def judge_single_topic(
    client: Any,
    *,
    model: str,
    macro: str,
    topic_id: int,
    topic_size: int,
    rep_examples: Sequence[str],
    random_examples: Sequence[str],
    temperature: float = 0.2,
    request_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Un appel OpenAI pour évaluer un topic."""
    user = build_topic_judge_prompt(macro, topic_size, rep_examples, random_examples)
    create_kwargs: Dict[str, Any] = {}
    if request_timeout is not None:
        create_kwargs["timeout"] = float(request_timeout)

    resp = client.chat.completions.create(
        model=model,
        temperature=float(temperature),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        **create_kwargs,
    )
    raw = resp.choices[0].message.content or "{}"
    parsed = parse_judge_response(_parse_json_content(raw))
    parsed["macro"] = str(macro)
    parsed["topic_id"] = int(topic_id)
    parsed["n_units"] = int(topic_size)
    parsed["model"] = str(model)
    return parsed


def aggregate_judge_by_macro(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Agrégats judge par macro."""
    if scores_df.empty:
        return pd.DataFrame(
            columns=[
                "macro",
                "n_topics_judged",
                "mean_score_global",
                "median_score_global",
                "pct_conserver",
                "pct_rejeter",
            ]
        )
    rows: list[dict[str, Any]] = []
    for macro in MACRO_NAMES:
        sub = scores_df.loc[scores_df["macro"].astype(str) == macro]
        if sub.empty:
            continue
        sg = pd.to_numeric(sub["score_global"], errors="coerce")
        verdict = sub["verdict"].astype(str).str.lower()
        rows.append(
            {
                "macro": macro,
                "n_topics_judged": int(len(sub)),
                "mean_score_global": round(float(sg.mean()), 4),
                "median_score_global": round(float(sg.median()), 4),
                "pct_conserver": round(100.0 * float((verdict == "conserver").mean()), 1),
                "pct_rejeter": round(100.0 * float((verdict == "rejeter").mean()), 1),
            }
        )
    return pd.DataFrame(rows)


def topic_judge_scores_path(out_dir: Union[str, Path]) -> Path:
    return Path(out_dir) / "summary" / "topic_judge_scores.csv"


def topic_judge_macro_summary_path(out_dir: Union[str, Path]) -> Path:
    return Path(out_dir) / "summary" / "topic_judge_macro_summary.csv"


def load_topic_judge_scores(out_dir: Union[str, Path]) -> pd.DataFrame:
    path = topic_judge_scores_path(out_dir)
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_topic_judge_macro_summary(out_dir: Union[str, Path]) -> pd.DataFrame:
    path = topic_judge_macro_summary_path(out_dir)
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_topic_judge_evaluation(
    out_dir: Union[str, Path],
    meta: pd.DataFrame,
    assignments: pd.DataFrame,
    themes: pd.DataFrame,
    *,
    cfg: Optional[Mapping[str, Any]] = None,
    text_col: str = "sentence",
    seed: int = 42,
    client: Any = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Évalue tous les topics (``topic_id >= 0``) via LLM judge.

    Écrit ``summary/topic_judge_scores.csv`` et ``summary/topic_judge_macro_summary.csv``.
    """
    cfg = dict(cfg or {})
    if not cfg.get("enabled", True):
        return {"skipped": True, "reason": "topic_judge disabled"}

    out_dir = Path(out_dir)
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    scores_path = topic_judge_scores_path(out_dir)
    macro_path = topic_judge_macro_summary_path(out_dir)

    if scores_path.is_file() and not force:
        scores_df = pd.read_csv(scores_path)
        macro_df = load_topic_judge_macro_summary(out_dir)
        return {
            "scores_path": str(scores_path),
            "macro_summary_path": str(macro_path) if macro_path.is_file() else None,
            "n_topics": int(len(scores_df)),
            "cached": True,
        }

    load_openai_dotenv()
    cli = client or _get_client()
    model = str(cfg.get("model", "gpt-5-mini"))
    temperature = float(cfg.get("temperature", 0.2))
    n_rep = int(cfg.get("n_representative", 5))
    n_random = int(cfg.get("n_random", 5))
    min_n_units = int(cfg.get("min_n_units", 8))
    show_progress = bool(cfg.get("show_progress", True))
    timeout = cfg.get("request_timeout")
    request_timeout = float(timeout) if timeout is not None else None

    rng = np.random.default_rng(int(seed))
    themes = themes.copy()
    if themes.empty:
        empty = pd.DataFrame()
        empty.to_csv(scores_path, index=False)
        aggregate_judge_by_macro(empty).to_csv(macro_path, index=False)
        return {"scores_path": str(scores_path), "n_topics": 0}

    rows: list[dict[str, Any]] = []
    topic_rows = themes.loc[pd.to_numeric(themes["topic_id"], errors="coerce").fillna(-1).astype(int) >= 0]
    eligible_rows: list[pd.Series] = []
    for _, theme_row in topic_rows.iterrows():
        if int(theme_row.get("n_units", 0)) >= min_n_units:
            eligible_rows.append(theme_row)
    iterator: Any = eligible_rows
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(
            eligible_rows,
            desc="LLM judge topics",
            unit="topic",
            total=len(eligible_rows),
        )

    for theme_row in iterator:
        macro = str(theme_row["macro"])
        tid = int(theme_row["topic_id"])
        n_units = int(theme_row.get("n_units", 0))
        if n_units < min_n_units:
            continue
        rep_ex, rand_ex = sample_topic_examples(
            assignments,
            meta,
            macro,
            tid,
            text_col,
            n_rep=n_rep,
            n_random=n_random,
            themes_row=theme_row,
            rng=rng,
        )
        try:
            result = judge_single_topic(
                cli,
                model=model,
                macro=macro,
                topic_id=tid,
                topic_size=n_units,
                rep_examples=rep_ex,
                random_examples=rand_ex,
                temperature=temperature,
                request_timeout=request_timeout,
            )
            rows.append(result)
        except Exception as exc:
            rows.append(
                {
                    "macro": macro,
                    "topic_id": tid,
                    "n_units": n_units,
                    "label_propose": "",
                    **{c: np.nan for c in JUDGE_CRITERIA},
                    "score_global": np.nan,
                    "verdict": "rejeter",
                    "justification_courte": f"Erreur judge : {exc}",
                    "probleme_principal": "exemples_insuffisants",
                    "model": model,
                }
            )

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(scores_path, index=False)
    macro_df = aggregate_judge_by_macro(scores_df)
    macro_df.to_csv(macro_path, index=False)

    run_meta = {
        "model": model,
        "n_topics": int(len(scores_df)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
    }
    with open(summary_dir / "topic_judge_run.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    return {
        "scores_path": str(scores_path),
        "macro_summary_path": str(macro_path),
        "n_topics": int(len(scores_df)),
        "cached": False,
    }
