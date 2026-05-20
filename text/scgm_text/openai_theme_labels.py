"""Enrichissement des lignes de ``themes_by_z.csv`` via l’API OpenAI (libellé court par topic)."""

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
    "Tu attribues des libellés courts à des topics intra-macro sur la sécurité au travail "
    "et les accidents. Chaque topic appartient à une macro (A0, A1, B ou C) dont la sémantique "
    "est précisée dans le message utilisateur. Le libellé doit respecter strictement cette macro. "
    "Réponds uniquement en JSON valide, sans markdown. Le libellé doit être en français."
)

_USER_TEMPLATE_WITH_MACRO = """{macro_block}

Voici un topic qui contient les extraits suivants :

{documents}

À partir de ces informations, propose un libellé court pour ce topic.
Le libellé doit être en français et refléter uniquement le thème de la macro indiquée ci-dessus.

Réponds en JSON : {{"libelle": "ton libellé court en français"}}"""

_USER_TEMPLATE_NO_MACRO = """Voici un topic qui contient les extraits suivants :

{documents}

À partir de ces informations, propose un libellé court pour ce topic.

Réponds en JSON : {{"libelle": "ton libellé court en français"}}"""


def build_documents_block(top_sentences: str, n_example_texts: int) -> str:
    """Formate les extraits (``top_sentences``, séparateur `` || ``) pour le prompt utilisateur."""
    examples = _split_example_sentences(top_sentences, n_example_texts)
    if not examples:
        return "(aucun extrait disponible pour ce topic)"
    lines = []
    for i, ex in enumerate(examples, start=1):
        short = ex[:800] + ("…" if len(ex) > 800 else "")
        lines.append(f"Extrait {i} :\n{short}")
    return "\n\n".join(lines)


def _macro_block_for_prompt(macro: Optional[str]) -> Optional[str]:
    """Bloc contexte macro ; ``None`` si macro absente ou inconnue (avec avertissement)."""
    if macro is None:
        return None
    mid = str(macro).strip()
    if not mid:
        return None
    from safer_core.macro_definitions import format_macro_context_for_prompt

    block = format_macro_context_for_prompt(mid)
    if block is None:
        warnings.warn(
            f"Macro inconnue pour le prompt OpenAI : {mid!r} — bloc macro omis.",
            stacklevel=3,
        )
    return block


def build_user_prompt(
    top_sentences: str,
    n_example_texts: int = 5,
    *,
    macro: Optional[str] = None,
) -> str:
    """Construit le message user (extraits + contexte macro optionnel)."""
    documents = build_documents_block(top_sentences, n_example_texts)
    macro_block = _macro_block_for_prompt(macro)
    if macro_block:
        return _USER_TEMPLATE_WITH_MACRO.format(
            macro_block=macro_block,
            documents=documents,
        )
    return _USER_TEMPLATE_NO_MACRO.format(documents=documents)


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


def _clamp_theme_summary_words(text: str, lo: int = 3, hi: int = 12) -> str:
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


def _keywords_from_label(label: str, row: pd.Series) -> str:
    words = [w for w in label.split() if w][:5]
    if len(words) < 5:
        raw_words = str(row.get("top_words", "")).replace(";", " ").replace(",", " ")
        for w in raw_words.split():
            if w and w not in words:
                words.append(w)
            if len(words) >= 5:
                break
    while len(words) < 5:
        words.append("")
    return ";".join(words[:5])


def _labels_from_api_response(
    data: Dict[str, Any],
    row: pd.Series,
    *,
    summary_words_min: int,
    summary_words_max: int,
) -> Dict[str, str]:
    """Mappe ``label`` (ou legacy ``theme_summary``) vers colonnes CSV."""
    raw_label = str(
        data.get("libelle") or data.get("label") or data.get("theme_summary") or ""
    ).strip()
    if not raw_label and data.get("theme_title"):
        raw_label = str(data.get("theme_title", "")).strip()
    summary = _clamp_theme_summary_words(raw_label, summary_words_min, summary_words_max)
    title = summary[:60]
    kws = data.get("theme_keywords")
    if isinstance(kws, list) and any(str(x).strip() for x in kws):
        kw_list = [str(x).strip() for x in kws if str(x).strip()][:5]
        while len(kw_list) < 5:
            kw_list.append("")
        kw_str = ";".join(kw_list)
    else:
        kw_str = _keywords_from_label(summary, row)
    return {
        "theme_title": title,
        "theme_summary": summary,
        "theme_keywords": kw_str,
    }


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
    raw_words = str(row.get("top_words", "")).replace(";", " ").replace(",", " ")
    words = [w for w in raw_words.split() if w][:5]
    summary_seed = " ".join(w for w in words if w) or f"topic {z_id}"
    summary = _clamp_theme_summary_words(summary_seed, summary_words_min, summary_words_max)
    return {
        "theme_title": summary[:60],
        "theme_summary": summary,
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
    macro_raw = row.get("dominant_macro") if row.get("dominant_macro") is not None else row.get("macro")
    macro_str = str(macro_raw).strip() if macro_raw is not None and str(macro_raw).strip() else None
    user = build_user_prompt(
        str(row.get("top_sentences", "")),
        n_example_texts,
        macro=macro_str,
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
    return _labels_from_api_response(
        data,
        row,
        summary_words_min=summary_words_min,
        summary_words_max=summary_words_max,
    )


def enrich_themes_by_z_openai(
    themes_csv: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    n_example_texts: int = 5,
    summary_words_min: int = 3,
    summary_words_max: int = 12,
    client: Any = None,
    show_progress: bool = True,
    skip_on_error: bool = False,
    request_timeout: Optional[float] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Lit ``themes_by_z.csv`` et écrit ``themes_by_z_openai.csv`` (mêmes colonnes + titres/résumé/mots-clés).

    ``theme_summary`` est un libellé court en français (``summary_words_min``–``summary_words_max`` mots).
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
        help="Nombre max d’extraits (segments) fournis comme documents (découpe de top_sentences).",
    )
    parser.add_argument("--summary-words-min", type=int, default=3)
    parser.add_argument("--summary-words-max", type=int, default=12)
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
