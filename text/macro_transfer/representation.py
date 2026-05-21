"""BERTopic representation OpenAI (libellés LLM intégrés au fit)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from safer_core.macro_definitions import format_macro_context_for_prompt
from scgm_text.openai_theme_labels import _get_client, load_openai_dotenv

DEFAULT_FR_CHAT_PROMPT = """Contexte — accidents du travail (macro obligatoire) :
[MACRO_CONTEXT]

Documents représentatifs du topic :
[DOCUMENTS]

Mots-clés c-TF-IDF du topic : [KEYWORDS]

Propose un libellé court en français pour ce topic. Le libellé doit respecter strictement la macro ci-dessus.
Réponds au format :
topic: <libellé en français>"""


def build_tiktoken_tokenizer(model_name: str) -> Callable[[str], list]:
    """Tokenizer tiktoken pour ``doc_length`` (troncature par tokens)."""
    import tiktoken

    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    def _tokenize(text: str) -> list:
        return enc.encode(text or "")

    return _tokenize


def _resolve_prompt(rep_cfg: Dict[str, Any], *, macro: Optional[str], anchor: Optional[Any]) -> str:
    custom = rep_cfg.get("prompt")
    if custom is not None and str(custom).strip():
        template = str(custom).strip()
    else:
        template = DEFAULT_FR_CHAT_PROMPT
    macro_block = ""
    if macro:
        ctx = format_macro_context_for_prompt(macro, anchor=anchor)
        if ctx:
            macro_block = ctx
        else:
            macro_block = f"Macro : {macro}"
    return template.replace("[MACRO_CONTEXT]", macro_block)


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

    prompt = _resolve_prompt(rep_cfg, macro=macro, anchor=anchor)

    kwargs: Dict[str, Any] = {
        "client": client,
        "model": model,
        "prompt": prompt,
        "chat": chat,
        "nr_docs": nr_docs,
        "exponential_backoff": bool(rep_cfg.get("exponential_backoff", False)),
    }
    if delay_in_seconds is not None:
        kwargs["delay_in_seconds"] = delay_in_seconds
    if doc_length is not None:
        kwargs["doc_length"] = doc_length
    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer
    if diversity is not None:
        kwargs["diversity"] = diversity

    gen_kw = rep_cfg.get("generator_kwargs")
    if isinstance(gen_kw, dict):
        kwargs["generator_kwargs"] = gen_kw

    return BertopicOpenAI(**kwargs)
