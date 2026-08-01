"""BERTopic representation OpenAI (libellés LLM intégrés au fit)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from safer_core.corpus_context import format_corpus_context_for_prompt
from safer_core.macro_definitions import format_macro_context_for_prompt
from macro_transfer.openai_utils import openai_chat_accepts_custom_temperature
from scgm_text.openai_theme_labels import _get_client, load_openai_dotenv

DEFAULT_FR_CHAT_PROMPT = """Contexte — accidents du travail

[Sector_CONTEXT]

Macro obligatoire pour ce topic :
[MACRO_CONTEXT]

Documents représentatifs du topic :
[DOCUMENTS]

Mots-clés c-TF-IDF du topic : [KEYWORDS]

Consignes :
- Propose un libellé court en français, strictement conforme à la macro.
- Les exemples fournis sont illustratifs : ne pas les recopier ni extrapoler hors du topic.
Réponds au format :
topic: <libellé en français>"""


def _include_macro_context(rep_cfg: Dict[str, Any]) -> bool:
    """
    ``include_corpus_context`` (défaut True) : injecte la sémantique macro A0–C
    (contexte de travail, facteurs contributifs, événement, conséquences) depuis
    ``configs/accident_macros.yaml`` — pas le secteur métallurgie/BTP.
    """
    return bool(rep_cfg.get("include_corpus_context", True))


def _include_sector_context(rep_cfg: Dict[str, Any]) -> bool:
    """``include_sector_context`` (défaut False) : contexte sectoriel BTP/métallurgie/caou."""
    return bool(rep_cfg.get("include_sector_context", False))


def build_tiktoken_tokenizer(model_name: str):
    """
    Retourne ``tiktoken.encoding_for_model(...)`` tel que dans la doc BERTopic.

    L'objet ``Encoding`` expose déjà ``encode`` / ``decode`` attendus par
    ``truncate_document``. Le bug initial venait d'un *callable* maison
    (``lambda text: enc.encode(text)``) qui ne correspondait à aucune branche.
    """
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _resolve_prompt(
    rep_cfg: Dict[str, Any],
    *,
    macro: Optional[str],
    corpus_id: Optional[str],
    anchor: Optional[Any],
) -> str:
    custom = rep_cfg.get("prompt")
    if custom is not None and str(custom).strip():
        template = str(custom).strip()
    else:
        template = DEFAULT_FR_CHAT_PROMPT

    include_macro = _include_macro_context(rep_cfg)
    include_sector = _include_sector_context(rep_cfg)

    sector_block = ""
    if include_sector and corpus_id:
        ctx_yaml = rep_cfg.get("corpus_context_file")
        ctx_sector = format_corpus_context_for_prompt(
            corpus_id,
            context_yaml_path=ctx_yaml,
            anchor=anchor,
        )
        if ctx_sector:
            sector_block = f"Corpus analysé (unités du run en cours) :\n{ctx_sector}"
        else:
            sector_block = f"Corpus analysé : {corpus_id}"
    elif include_sector and corpus_id is None:
        sector_block = "Corpus analysé : (non spécifié)"

    macro_block = ""
    if include_macro and macro:
        ctx = format_macro_context_for_prompt(macro, anchor=anchor)
        if ctx:
            macro_block = ctx
        else:
            macro_block = f"Macro : {macro}"
    elif macro and not include_macro:
        macro_block = f"Macro : {macro}"

    out = template.replace("[Sector_CONTEXT]", sector_block)
    out = out.replace("[CORPUS_CONTEXT]", sector_block)  # rétrocompat templates custom
    return out.replace("[MACRO_CONTEXT]", macro_block)


def representation_enabled(bertopic_cfg: Dict[str, Any]) -> bool:
    """True si ``bertopic.representation`` doit être utilisé."""
    rep = bertopic_cfg.get("representation") or {}
    if "enabled" in rep:
        return bool(rep["enabled"])
    return True


def build_representation_model(
    rep_cfg: Dict[str, Any],
    *,
    macro: Optional[str] = None,
    corpus_id: Optional[str] = None,
    anchor: Optional[Any] = None,
):
    """
    Instancie ``bertopic.representation.OpenAI`` ou ``None`` si désactivé.

    Paramètres lus depuis ``bertopic.representation`` dans le YAML.
    """
    if not rep_cfg or not rep_cfg.get("enabled", True):
        return None

    from bertopic.representation import OpenAI as BertopicOpenAI

    load_openai_dotenv()
    client = _get_client()

    model = str(rep_cfg.get("model", "gpt-4o-mini"))
    chat = bool(rep_cfg.get("chat", True))
    nr_docs = int(rep_cfg.get("nr_docs", 4))
    doc_length = rep_cfg.get("doc_length")
    if doc_length is not None:
        doc_length = int(doc_length)

    tokenizer: Any = rep_cfg.get("tokenizer")
    if tokenizer is None and doc_length is not None:
        tok_model = str(rep_cfg.get("tokenizer_model", model))
        tokenizer = build_tiktoken_tokenizer(tok_model)
    elif isinstance(tokenizer, str):
        if tokenizer in ("char", "whitespace", "vectorizer"):
            pass
        else:
            tokenizer = build_tiktoken_tokenizer(tokenizer)

    delay = rep_cfg.get("delay_in_seconds")
    delay_in_seconds = float(delay) if delay is not None else None
    diversity = rep_cfg.get("diversity")
    if diversity is not None:
        diversity = float(diversity)

    prompt = _resolve_prompt(
        rep_cfg, macro=macro, corpus_id=corpus_id, anchor=anchor
    )

    kwargs: Dict[str, Any] = {
        "client": client,
        "model": model,
        "prompt": prompt,
        "chat": chat,
        "nr_docs": nr_docs,
        "exponential_backoff": bool(rep_cfg.get("exponential_backoff", True)),
    }
    if delay_in_seconds is not None:
        kwargs["delay_in_seconds"] = delay_in_seconds
    elif bool(rep_cfg.get("delay_in_seconds_auto", True)):
        kwargs["delay_in_seconds"] = float(rep_cfg.get("default_delay_in_seconds", 0.75))
    if doc_length is not None:
        kwargs["doc_length"] = doc_length
    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer
    if diversity is not None:
        kwargs["diversity"] = diversity

    gen_kw = dict(rep_cfg.get("generator_kwargs") or {})
    if not openai_chat_accepts_custom_temperature(model):
        gen_kw.pop("temperature", None)
    if gen_kw:
        kwargs["generator_kwargs"] = gen_kw

    return BertopicOpenAI(**kwargs)
