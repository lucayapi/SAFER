import os
import time
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import torch
import gc
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


# =========================
# Helpers HF token
# =========================
def _setup_hf_env(hf_cache_folder="./hf_cache", hf_token=None):
    os.makedirs(hf_cache_folder, exist_ok=True)

    os.environ["HF_HOME"] = os.path.abspath(hf_cache_folder)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(
        os.environ["HF_HOME"], "sentence_transformers"
    )
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "transformers")

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token


def _hf_kwargs(hf_token=None):
    """
    Pour transformers: utiliser uniquement `token`.
    Ne pas passer `use_auth_token`, qui peut fuiter jusqu'au constructeur
    du modèle selon les versions et provoquer une erreur.
    """
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token
    return kwargs


def _load_sentence_transformer(model_name, cache_folder, device, hf_token=None):
    """
    Charge SentenceTransformer en essayant d'abord `token=`,
    puis fallback sur `use_auth_token=` pour compatibilité.
    """
    try:
        return SentenceTransformer(
            model_name,
            cache_folder=cache_folder,
            device=device,
            token=hf_token,
        )
    except TypeError:
        return SentenceTransformer(
            model_name,
            cache_folder=cache_folder,
            device=device,
            use_auth_token=hf_token,
        )


# =========================
# naive (iid)
# =========================
def encode_embeddings(
    docs,
    model_names,
    batch_size=16,
    cache_dir="EMFisherTopic/embeddings",
    hf_cache_folder="./hf_cache",
    hf_token=None,
    force_recompute=False,
    desc="Encodage embeddings",
    device=None,            # "cuda", "cpu", "cuda:0"
    normalize=False,        # True/False (L2 via sentence-transformers)
    variant_tag="naive",    # optionnel si tu l'utilises déjà
):
    """
    Encode une liste de textes `docs` avec plusieurs modèles SentenceTransformers
    et sauvegarde UNIQUEMENT en CSV (un fichier par modèle).

    Sorties:
      - {cache_dir}/{model}__{variant_tag}__{raw|l2}.csv
    """
    os.makedirs(cache_dir, exist_ok=True)
    _setup_hf_env(hf_cache_folder=hf_cache_folder, hf_token=hf_token)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    N = len(docs)
    doc_ids = np.arange(1, N + 1, dtype=np.int64)
    suffix = "l2" if normalize else "raw"
    safe_variant = variant_tag.replace("/", "__").replace(" ", "_")

    for name in tqdm(model_names, desc=desc):
        safe_name = name.replace("/", "__")
        csv_path = os.path.join(cache_dir, f"{safe_name}__{safe_variant}__{suffix}.csv")

        if (not force_recompute) and os.path.exists(csv_path):
            print(f"[SKIP] {name} -> fichier existant: {csv_path}")
            continue

        model = _load_sentence_transformer(
            name,
            cache_folder=os.environ["SENTENCE_TRANSFORMERS_HOME"],
            device=device,
            hf_token=hf_token,
        )

        if str(device).startswith("cuda"):
            try:
                model = model.to(torch.bfloat16)
            except Exception:
                pass
            try:
                backbone = model[0].auto_model
                if hasattr(backbone.config, "use_cache"):
                    backbone.config.use_cache = False
            except Exception:
                pass

        emb = model.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        ).astype(np.float32)

        dim_cols = [f"dim_{j}" for j in range(1, emb.shape[1] + 1)]
        df = pd.DataFrame(emb, columns=dim_cols)
        df.insert(0, "doc_id", doc_ids)
        df.to_csv(csv_path, index=False)

        del model
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        print(f"[DONE] {name} -> fichier: {csv_path} enregistré")


# =========================
# late chunking
# =========================
def _safe_max_seq_len(tokenizer, model):
    m1 = getattr(model.config, "max_position_embeddings", None)
    m2 = getattr(tokenizer, "model_max_length", None)
    cands = [x for x in (m1, m2) if isinstance(x, int) and 0 < x < 100000]
    return min(cands) if cands else 512


def _find_spans_sequential(summary, sentences):
    spans = []
    cursor = 0
    for s in sentences:
        pos = summary.find(s, cursor)
        if pos == -1:
            spans.append(None)
            continue
        spans.append((pos, pos + len(s)))
        cursor = pos + len(s)
    return spans


def _token_indices_for_span(offsets, span):
    if span is None:
        return None
    a, b = span
    idx = [i for i, (x, y) in enumerate(offsets) if (x < b and y > a)]
    return idx if idx else None


def _build_inputs_and_special_mask(tokenizer, ids_slice):
    """
    ids_slice: List[int] (séquence SANS tokens spéciaux)
    Retourne:
      - ids_with_specials: List[int]
      - special_mask: List[int] (1 si token spécial, 0 sinon)

    Version robuste:
    - essaie d'abord build_inputs_with_special_tokens
    - essaie ensuite prepare_for_model
    - si aucun token spécial n'existe, retourne la séquence brute
    """
    ids_slice = list(ids_slice)

    # 1) voie standard la plus robuste
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        try:
            ids_with_specials = tokenizer.build_inputs_with_special_tokens(ids_slice)

            if hasattr(tokenizer, "get_special_tokens_mask"):
                try:
                    special_mask = tokenizer.get_special_tokens_mask(
                        ids_slice,
                        already_has_special_tokens=False
                    )
                except Exception:
                    try:
                        special_mask = tokenizer.get_special_tokens_mask(
                            ids_with_specials,
                            already_has_special_tokens=True
                        )
                    except Exception:
                        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
                        special_mask = [1 if t in special_ids else 0 for t in ids_with_specials]
            else:
                special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
                special_mask = [1 if t in special_ids else 0 for t in ids_with_specials]

            if len(ids_with_specials) == len(special_mask):
                return ids_with_specials, special_mask
        except Exception:
            pass

    # 2) voie prepare_for_model
    if hasattr(tokenizer, "prepare_for_model"):
        try:
            out = tokenizer.prepare_for_model(
                ids_slice,
                add_special_tokens=True,
                return_special_tokens_mask=True,
                truncation=False,
                padding=False,
            )
            ids_with_specials = out["input_ids"]

            special_mask = out.get("special_tokens_mask")
            if special_mask is None:
                special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
                special_mask = [1 if t in special_ids else 0 for t in ids_with_specials]

            if len(ids_with_specials) == len(special_mask):
                return ids_with_specials, special_mask
        except Exception:
            pass

    # 3) fallback minimal: pas de tokens spéciaux
    return ids_slice, [0] * len(ids_slice)


@torch.no_grad()
def _encode_token_embeddings_late(text, tokenizer, model, device, overlap_tokens=64):
    enc_full = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    if "offset_mapping" not in enc_full:
        raise RuntimeError(
            "Tokenizer non-fast: return_offsets_mapping indisponible. "
            "Utilise AutoTokenizer(..., use_fast=True)."
        )

    input_ids_full = enc_full["input_ids"]
    offsets = enc_full["offset_mapping"]
    T = len(input_ids_full)

    H = model.config.hidden_size
    max_seq = _safe_max_seq_len(tokenizer, model)
    L = max_seq - 2

    overlap_tokens = int(max(0, min(overlap_tokens, L - 1)))

    if T <= L:
        ids_with_specials, special_mask = _build_inputs_and_special_mask(tokenizer, input_ids_full)
        input_ids = torch.tensor([ids_with_specials], device=device)
        attention_mask = torch.ones_like(input_ids, device=device)

        out = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[0]
        out_np = out.detach().cpu().float().numpy()

        special_mask = np.asarray(special_mask, dtype=bool)
        token_emb = out_np[~special_mask]
        if token_emb.shape[0] != T:
            raise RuntimeError(
                f"Incohérence late_simple: token_emb len={token_emb.shape[0]} vs T={T}."
            )

        stats = {
            "n_tokens": int(T),
            "model_max_seq": int(max_seq),
            "L_window": int(L),
            "method": "late_simple",
            "n_windows": 1,
            "overlap_tokens": int(overlap_tokens),
        }
        return token_emb.astype(np.float32), offsets, stats

    token_emb = np.empty((T, H), dtype=np.float32)

    iend = 0
    n_windows = 0
    stride = L - overlap_tokens

    while iend < T:
        istart = max(iend - overlap_tokens, 0)
        iend = min(istart + L, T)

        ids_slice = input_ids_full[istart:iend]

        ids_with_specials, special_mask = _build_inputs_and_special_mask(tokenizer, ids_slice)
        input_ids = torch.tensor([ids_with_specials], device=device)
        attention_mask = torch.ones_like(input_ids, device=device)

        out = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[0]
        out_np = out.detach().cpu().float().numpy()

        special_mask = np.asarray(special_mask, dtype=bool)
        out_core = out_np[~special_mask]
        if out_core.shape[0] != len(ids_slice):
            raise RuntimeError(
                f"Incohérence long_late: out_core len={out_core.shape[0]} vs slice={len(ids_slice)}."
            )

        if istart == 0:
            token_emb[istart:iend] = out_core
        else:
            drop = overlap_tokens
            token_emb[istart + drop:iend] = out_core[drop:]

        n_windows += 1

        if iend == T:
            break

    stats = {
        "n_tokens": int(T),
        "model_max_seq": int(max_seq),
        "L_window": int(L),
        "method": "long_late_concat_drop_overlap",
        "n_windows": int(n_windows),
        "stride": int(stride),
        "overlap_tokens": int(overlap_tokens),
    }
    return token_emb.astype(np.float32), offsets, stats


def encode_late_chunking_sentences(
    sent_df: pd.DataFrame,
    model_names,
    cache_dir="EMFisherTopic/embeddings",
    report_folder="reports_late",
    hf_cache_folder="./hf_cache",
    hf_token=None,
    device=None,
    overlap_tokens=64,
    center_mode=None,
    normalize=False,
    force_recompute=False,
):
    """
    Enregistre UNIQUEMENT des fichiers CSV.
    """
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(report_folder, exist_ok=True)
    _setup_hf_env(hf_cache_folder=hf_cache_folder, hf_token=hf_token)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sent_df = sent_df.reset_index(drop=True).copy()
    N = len(sent_df)
    doc_ids = np.arange(1, N + 1, dtype=np.int64)

    if "accident_id" not in sent_df.columns:
        raise ValueError("sent_df doit contenir une colonne 'accident_id'.")

    mapping_df = pd.DataFrame({
        "doc_id": doc_ids,
        "accident_id": sent_df["accident_id"].values,
    })

    groups = sent_df.groupby("accident_id", sort=False)

    if center_mode not in (None, "mean", "raw"):
        raise ValueError("center_mode doit être None, 'mean' ou 'raw'.")

    hf_kwargs = _hf_kwargs(hf_token)

    for name in tqdm(model_names, dynamic_ncols=True, desc="Late chunking models"):
        print(f"[DEBUG] Loading tokenizer/model for: {name}")

        safe_name = name.replace("/", "__")
        variant = f"late_ov{overlap_tokens}"
        if center_mode is not None:
            variant += f"__center-{center_mode}"
        variant += ("__l2" if normalize else "__raw")

        emb_base = os.path.join(cache_dir, f"{safe_name}__{variant}")
        emb_csv_path = emb_base + ".csv"
        rep_base = os.path.join(report_folder, f"{safe_name}__{variant}")

        if (not force_recompute) and os.path.exists(emb_csv_path):
            print(f"[SKIP] {name} -> fichier existant: {emb_csv_path}")
            continue

        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, **hf_kwargs)
        model = AutoModel.from_pretrained(name, **hf_kwargs).to(device)
        model.eval()

        dim = model.config.hidden_size
        emb = np.full((N, dim), np.nan, dtype=np.float32)

        report_rows = []
        for acc_id, g in tqdm(
            groups,
            leave=False,
            miniters=100,
            dynamic_ncols=True,
            desc=f"{safe_name} accidents"
        ):
            summary = g["accident_summary"].iloc[0]
            sentences = g["sentence"].tolist()
            row_idx = g.index.to_numpy()

            spans = _find_spans_sequential(summary, sentences)

            token_emb, offsets, stats = _encode_token_embeddings_late(
                summary, tokenizer, model, device=device, overlap_tokens=overlap_tokens
            )

            doc_vec = token_emb.mean(axis=0, keepdims=True).astype(np.float32)

            sent_vecs = []
            ok_rows = []
            n_missing = 0

            T = token_emb.shape[0]
            covered_mask = np.zeros(T, dtype=bool)
            total_sentence_tokens = 0

            for r, sp in zip(row_idx, spans):
                tok_idx = _token_indices_for_span(offsets, sp)
                if tok_idx is None:
                    n_missing += 1
                    continue

                tok_idx = np.asarray(tok_idx, dtype=int)
                v = token_emb[tok_idx].mean(axis=0)

                sent_vecs.append(v)
                ok_rows.append(r)

                covered_mask[tok_idx] = True
                total_sentence_tokens += int(len(tok_idx))

            if len(ok_rows) == 0:
                report_rows.append({
                    "model": name,
                    "accident_id": acc_id,
                    "n_sentences": int(len(sentences)),
                    "n_embedded": 0,
                    "n_missing_spans_or_tokens": int(n_missing),
                    "coverage_sentences": 0.0,
                    "avg_tokens_per_sentence": 0.0,
                    "pct_tokens_covered_by_sentences": 0.0,
                    **stats
                })
                continue

            sent_vecs = np.stack(sent_vecs, axis=0).astype(np.float32)

            if center_mode == "raw":
                sent_vecs = sent_vecs - doc_vec
            elif center_mode == "mean":
                mu = sent_vecs.mean(axis=0, keepdims=True)
                sent_vecs = sent_vecs - mu

            if normalize:
                norms = np.linalg.norm(sent_vecs, axis=1, keepdims=True)
                sent_vecs = sent_vecs / np.maximum(norms, 1e-12)

            emb[ok_rows] = sent_vecs

            n_emb = len(ok_rows)
            avg_tok = float(total_sentence_tokens / max(1, n_emb))
            pct_cov = float(covered_mask.mean())

            report_rows.append({
                "model": name,
                "accident_id": acc_id,
                "n_sentences": int(len(sentences)),
                "n_embedded": int(n_emb),
                "n_missing_spans_or_tokens": int(n_missing),
                "coverage_sentences": float(n_emb / max(1, len(sentences))),
                "avg_tokens_per_sentence": avg_tok,
                "pct_tokens_covered_by_sentences": pct_cov,
                **stats
            })

        dim_cols = [f"dim_{j:04d}" for j in range(dim)]
        df_emb = pd.DataFrame(emb, columns=dim_cols)
        df_emb.insert(0, "doc_id", doc_ids)
        df_emb.to_csv(emb_csv_path, index=False)

        report_df = pd.DataFrame(report_rows)
        xlsx_path = rep_base + "__reports.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            mapping_df.to_excel(writer, sheet_name="mapping", index=False)
            report_df.to_excel(writer, sheet_name="late_report", index=False)

        del tokenizer, model
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        print(f"[DONE] {name} -> fichiers embeddings + reports enregistrés")