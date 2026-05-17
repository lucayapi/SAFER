"""Enrichissement des lignes de ``themes_by_z.csv`` via l’API OpenAI (contexte professionnel générique)."""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from tqdm.auto import tqdm

_SYSTEM_PROMPT = (
    "Tu nommes des topics issus d’un modèle hiérarchique (topic modeling) sur des segments "
    "de textes décrivant des situations de travail, des incidents ou des accidents en milieu "
    "professionnel ou industriel (domaine non imposé : adapte le vocabulaire aux extraits fournis). "
    "Chaque topic correspond à une composante latente z. "
    "Les macros dominantes possibles sur les segments du cluster sont parmi : A0, A1, B, C. "
    "Reste factuel, cohérent avec la macro dominante et les exemples fournis ; n’invente pas un secteur "
    "si les extraits n’en parlent pas ; pas de récit long. "
    "Réponds uniquement en JSON valide, sans markdown."
)

# Placeholders : z_id, dominant_macro, n_units, top_words, example_block, n_example_texts
_USER_TEMPLATE = """Topic latent (composante) z_id={z_id}
- Macro dominante sur les segments du cluster : {dominant_macro}
- Effectif segments : {n_units}
- Mots fréquents (TF-IDF) : {top_words}

Segments d’exemple (jusqu’à {n_examples_cap} segments les plus centraux du topic) :
{example_block}

Produis un JSON avec exactement les clés suivantes :
{{
  "theme_title": "titre très court (≤ 60 caractères), style intitulé de topic",
  "theme_summary": "une seule étiquette de topic en français : EXACTEMENT entre 6 et 10 mots, séparés par des espaces, sans ponctuation finale, sans guillemets. Ce libellé sera utilisé pour annoter une carte 2D de segments.",
  "theme_keywords": ["mot1", "mot2", "mot3", "mot4", "mot5"]
}}
Les 5 mots-clés doivent être compacts et alignés sur le contenu des extraits (sécurité, équipements, causes, conséquences, etc., selon ce qui est pertinent).
"""


def load_openai_dotenv() -> bool:
    """
    Charge ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` depuis des fichiers ``.env`` locaux.

    Cherche, dans l’ordre : ``./.env`` (cwd), racine du dépôt ``.env``, ``scgm_text/.env``.
    Les variables déjà définies dans l’environnement ne sont pas écrasées (``override=False``).

    Retourne ``True`` si au moins un fichier ``.env`` existant a été lu (ou tenté via dotenv).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    repo_root = Path(__file__).resolve().parent.parent
    scgm_text_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / ".env",
        repo_root / ".env",
        scgm_text_dir / ".env",
    ]
    loaded_any = False
    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False)
            loaded_any = True
    return loaded_any


def _split_example_sentences(top_sentences: str, n: int) -> List[str]:
    """Découpe ``top_sentences`` (séparateur `` || `` depuis topic_export) en au plus ``n`` extraits."""
    if not isinstance(top_sentences, str) or not top_sentences.strip():
        return []
    parts = [p.strip() for p in re.split(r"\s*\|\|\s*", top_sentences) if p.strip()]
    return parts[: max(0, int(n))]


_SUMMARY_PAD_WORDS = (
    "topic",
    "latent",
    "segments",
    "risques",
    "accidents",
    "sécurité",
    "équipements",
    "incidents",
)


def _clamp_theme_summary_words(text: str, lo: int = 6, hi: int = 10) -> str:
    """Force ``theme_summary`` à une étiquette de ``lo`` à ``hi`` mots (troncature ou padding discret)."""
    cleaned = re.sub(r"\s+", " ", (text or "").replace("\n", " ")).strip()
    words = [w for w in cleaned.split(" ") if w]
    if len(words) > hi:
        return " ".join(words[:hi]) + "…"
    seen_lower = {w.lower() for w in words}
    pad_idx = 0
    while len(words) < lo and pad_idx < len(_SUMMARY_PAD_WORDS):
        w = _SUMMARY_PAD_WORDS[pad_idx]
        pad_idx += 1
        if w.lower() not in seen_lower:
            words.append(w)
            seen_lower.add(w.lower())
    while len(words) < lo:
        words.append("analyse")
    return " ".join(words[:hi])


def _default_openai_timeout() -> float:
    raw = os.environ.get("OPENAI_TIMEOUT", "120")
    try:
        return max(5.0, float(raw))
    except ValueError:
        return 120.0


def _get_client(*, timeout: Optional[float] = None, max_retries: Optional[int] = None):
    load_openai_dotenv()
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Installez le paquet « openai » (voir requirements.txt).") from exc
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Variable d’environnement OPENAI_API_KEY absente. "
            "Placez OPENAI_API_KEY=... dans un fichier .env à la racine du dépôt ou dans scgm_text/.env "
            "(chargé automatiquement si python-dotenv est installé), ou exportez-la dans le shell. "
            "Ne commitez jamais la clé."
        )
    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout if timeout is not None else _default_openai_timeout(),
        "max_retries": max_retries
        if max_retries is not None
        else int(os.environ.get("OPENAI_MAX_RETRIES", "2")),
    }
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url.strip()
    return OpenAI(**kwargs)


def probe_openai_connectivity(*, timeout: float = 20.0) -> bool:
    """
    Vérifie l’accès réseau à l’API (login nodes / JupyterHub souvent sans Internet sortant).

    Retourne False en cas de timeout, DNS ou pare-feu — sans lever d’exception.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        load_openai_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        cli = _get_client(timeout=timeout, max_retries=0)
        cli.models.list()
        return True
    except Exception:
        return False


def _parse_json_content(content: str) -> Dict[str, Any]:
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


def _fallback_row_labels(row: pd.Series, *, summary_words_min: int, summary_words_max: int) -> Dict[str, str]:
    """Étiquettes dérivées des mots TF-IDF si l’API OpenAI est indisponible."""
    z_id = int(row["z_id"])
    macro = str(row.get("dominant_macro", ""))
    raw_words = str(row.get("top_words", "")).replace(";", " ").replace(",", " ")
    words = [w for w in raw_words.split() if w][:5]
    while len(words) < 5:
        words.append("")
    summary_seed = " ".join(w for w in words if w) or f"topic latent z{z_id}"
    return {
        "theme_title": f"z{z_id} {macro}".strip()[:60],
        "theme_summary": _clamp_theme_summary_words(summary_seed, summary_words_min, summary_words_max),
        "theme_keywords": ";".join(w for w in words if w),
    }


def _one_row(
    client: Any,
    model: str,
    temperature: float,
    row: pd.Series,
    *,
    n_example_texts: int,
    summary_words_min: int,
    summary_words_max: int,
    request_timeout: Optional[float] = None,
) -> Dict[str, str]:
    examples = _split_example_sentences(str(row.get("top_sentences", "")), n_example_texts)
    if not examples:
        example_block = "(aucun extrait disponible pour ce z)"
    else:
        lines = []
        for i, ex in enumerate(examples, start=1):
            short = ex[:800] + ("…" if len(ex) > 800 else "")
            lines.append(f"Exemple {i}: {short}")
        example_block = "\n".join(lines)

    user = _USER_TEMPLATE.format(
        z_id=int(row["z_id"]),
        dominant_macro=str(row.get("dominant_macro", "")),
        n_units=int(row.get("n_units", 0)),
        top_words=str(row.get("top_words", ""))[:4000],
        example_block=example_block,
        n_examples_cap=int(n_example_texts),
    )
    create_kwargs: Dict[str, Any] = {}
    if request_timeout is not None:
        create_kwargs["timeout"] = float(request_timeout)

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        **create_kwargs,
    )
    raw = resp.choices[0].message.content or "{}"
    data = _parse_json_content(raw)
    title = str(data.get("theme_title", "")).strip()
    summary_raw = str(data.get("theme_summary", "")).strip()
    summary = _clamp_theme_summary_words(summary_raw, summary_words_min, summary_words_max)
    kws = data.get("theme_keywords") or []
    if not isinstance(kws, list):
        kws = []
    kws = [str(x).strip() for x in kws if str(x).strip()][:5]
    while len(kws) < 5:
        kws.append("")
    return {
        "theme_title": title,
        "theme_summary": summary,
        "theme_keywords": ";".join(x for x in kws if x),
    }


def enrich_themes_by_z_openai(
    themes_csv: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    n_example_texts: int = 5,
    summary_words_min: int = 6,
    summary_words_max: int = 10,
    client: Any = None,
    show_progress: bool = True,
    skip_on_error: bool = False,
    request_timeout: Optional[float] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Lit ``themes_by_z.csv`` et écrit ``themes_by_z_openai.csv`` (mêmes colonnes + titres/résumé/mots-clés).

    ``theme_summary`` est contraint à une étiquelle de topic de ``summary_words_min`` à ``summary_words_max`` mots.
    """
    load_openai_dotenv()
    themes_path = Path(themes_csv)
    if not themes_path.is_file():
        raise FileNotFoundError(str(themes_path))
    frame = pd.read_csv(themes_path)
    required = {"z_id", "dominant_macro", "n_units", "top_words", "top_sentences"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {themes_path}: {sorted(missing)}")

    n_ex = max(1, min(int(n_example_texts), 20))
    lo = max(1, int(summary_words_min))
    hi = max(lo, int(summary_words_max))

    out_path = Path(output_csv) if output_csv else themes_path.with_name("themes_by_z_openai.csv")
    cli = client or _get_client()

    titles: List[str] = []
    summaries: List[str] = []
    kw_strings: List[str] = []
    rows = list(frame.iterrows())
    if max_rows is not None:
        rows = rows[: max(0, int(max_rows))]
    iterator = tqdm(rows, desc="OpenAI thèmes par z", unit="topic") if show_progress else rows
    failures = 0
    for _, row in iterator:
        try:
            parsed = _one_row(
                cli,
                model=model,
                temperature=temperature,
                row=row,
                n_example_texts=n_ex,
                summary_words_min=lo,
                summary_words_max=hi,
                request_timeout=request_timeout,
            )
        except Exception as exc:
            if not skip_on_error:
                raise
            failures += 1
            parsed = _fallback_row_labels(row, summary_words_min=lo, summary_words_max=hi)
            warnings.warn(
                f"z_id={row.get('z_id')}: API OpenAI indisponible ({type(exc).__name__}) — libellé local utilisé.",
                stacklevel=2,
            )
        titles.append(parsed["theme_title"])
        summaries.append(parsed["theme_summary"])
        kw_strings.append(parsed["theme_keywords"])

    if failures:
        print(
            f"[openai_theme_labels] {failures}/{len(rows)} topics sans API "
            f"(timeout réseau fréquent sur nœuds de calcul — lancez depuis le login ou OPENAI_BASE_URL)."
        )

    enriched = frame.copy()
    enriched["theme_title"] = titles
    enriched["theme_summary"] = summaries
    enriched["theme_keywords"] = kw_strings
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    return enriched


def _cli(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Enrichir themes_by_z.csv via OpenAI.")
    parser.add_argument("themes_csv", type=str, help="Chemin vers themes_by_z.csv")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument(
        "--n-example-texts",
        type=int,
        default=5,
        help="Nombre max d’extraits (segments) numérotés à fournir au modèle (découpe de top_sentences).",
    )
    parser.add_argument("--summary-words-min", type=int, default=6)
    parser.add_argument("--summary-words-max", type=int, default=10)
    args = parser.parse_args(argv)
    enrich_themes_by_z_openai(
        args.themes_csv,
        output_csv=args.output_csv,
        model=args.model,
        temperature=args.temperature,
        n_example_texts=args.n_example_texts,
        summary_words_min=args.summary_words_min,
        summary_words_max=args.summary_words_max,
    )
    print("OK:", args.output_csv or Path(args.themes_csv).with_name("themes_by_z_openai.csv"))


if __name__ == "__main__":
    _cli()
