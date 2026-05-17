import os
import time
import pandas as pd
import torch

import emblearn as embl
import embprompt


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


print(f"[{stamp()}] Job started")

# -----------------------
# Device
# -----------------------
cuda_ok = torch.cuda.is_available()
device = "cuda" if cuda_ok else "cpu"
print(f"[{stamp()}] torch.cuda.is_available() = {cuda_ok} -> device = {device}")

# -----------------------
# Load data
# -----------------------
data_path = "data/btp_sentence_accidents.csv"
print(f"[{stamp()}] Loading CSV: {data_path}")
sent_df = pd.read_csv(data_path)
print(f"[{stamp()}] Loaded: {len(sent_df):,} rows | columns={list(sent_df.columns)}")

# -----------------------
# Checks
# -----------------------
if "sentence" not in sent_df.columns:
    raise ValueError("La colonne 'sentence' est introuvable dans le CSV.")

summary_col = embprompt.resolve_summary_col(sent_df)
print(f"[{stamp()}] Summary column detected: {summary_col}")

# -----------------------
# Prepare docs + models
# -----------------------
docs = sent_df["sentence"].astype(str).tolist()
print(f"[{stamp()}] Prepared baseline docs: {len(docs):,}")

model_names = [
    #"almanach/moderncamembert-base",
    #"Qwen/Qwen3-Embedding-0.6B",
    #"google/embeddinggemma-300m",
    "models/gemmaft",
    "models/modernCamembertft",
    "models/qwenft",
]
print(f"[{stamp()}] Models ({len(model_names)}): {model_names}")

# Ne jamais committer de clé : définir HF_TOKEN ou HUGGING_FACE_HUB_TOKEN dans .env ou le shell.
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
cache_dir = "nembeddings/btp"
report_folder = "reports_late"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(report_folder, exist_ok=True)
print(f"[{stamp()}] cache_dir='{cache_dir}' (exists={os.path.exists(cache_dir)})")

# -----------------------
# Build prompt-based docs
# -----------------------
print(f"[{stamp()}] Building prompt-based docs...")

prefixed_docs = embprompt.build_all_prefixed_docs(
    sent_df=sent_df,
    variants=[
        "prefix_macro_classes",
        "prefix_subtype_full_classes",
    ],
    sentence_col="sentence",
    summary_col=summary_col,
    include_summary=True,
)

for k, v in prefixed_docs.items():
    print(f"[{stamp()}] Variant={k} -> {len(v):,} docs")

print(f"[{stamp()}] Prompt-based docs ready")

# -----------------------
# 1) Naive embeddings
# -----------------------
print(f"\n[{stamp()}] ===== 1/8 NAIVE embeddings =====")
embl.encode_embeddings(
    docs,
    model_names,
    device=device,
    normalize=True,
    batch_size=64,
    hf_token=HF_TOKEN,
    cache_dir=cache_dir,
    variant_tag="naive_sentence_only",
)
print(f"[{stamp()}] NAIVE done")



# -----------------------
# 3) Prefix macro + classes
# -----------------------
print(f"\n[{stamp()}] ===== 3/8 PREFIX macro + classes =====")
embl.encode_embeddings(
    prefixed_docs["prefix_macro_classes"],
    model_names,
    device=device,
    normalize=True,
    batch_size=64,
    hf_token=HF_TOKEN,
    cache_dir=cache_dir,
    variant_tag="prefix_macro_classes",
)
print(f"[{stamp()}] PREFIX macro + classes done")


# -----------------------
# 5) Prefix subtype + full classes
# -----------------------
print(f"\n[{stamp()}] ===== 5/8 PREFIX subtype + full classes =====")
embl.encode_embeddings(
    prefixed_docs["prefix_subtype_full_classes"],
    model_names,
    device=device,
    normalize=True,
    batch_size=64,
    cache_dir=cache_dir,
    variant_tag="prefix_subtype_full_classes",
)
print(f"[{stamp()}] PREFIX subtype + full classes done")

# -----------------------
# 6) Late chunking - no centering
# -----------------------
print(f"\n[{stamp()}] ===== 6/8 LATE embeddings (center_mode=None) =====")
embl.encode_late_chunking_sentences(
    sent_df,
    model_names=model_names,
    overlap_tokens=64,
    center_mode=None,
    normalize=True,
    hf_token=HF_TOKEN,
    report_folder=report_folder,
    cache_dir=cache_dir,
    device=device,
)
print(f"[{stamp()}] LATE None done")

## -----------------------
## 7) Late chunking - raw centering
## -----------------------
#print(f"\n[{stamp()}] ===== 7/8 LATE embeddings (center_mode='raw') =====")
#embl.encode_late_chunking_sentences(
#    sent_df,
#    model_names=model_names,
#    overlap_tokens=64,
#    center_mode="raw",
#    normalize=True,
#    hf_token=HF_TOKEN,
#    report_folder=report_folder,
#    cache_dir=cache_dir,
#    device=device,
#)
#print(f"[{stamp()}] LATE raw done")

# -----------------------
# 8) Late chunking - mean centering
# -----------------------
print(f"\n[{stamp()}] ===== 8/8 LATE embeddings (center_mode='mean') =====")
embl.encode_late_chunking_sentences(
    sent_df,
    model_names=model_names,
    overlap_tokens=64,
    center_mode="mean",
    normalize=True,
    hf_token=HF_TOKEN,
    report_folder=report_folder,
    cache_dir=cache_dir,
    device=device,
)
print(f"[{stamp()}] LATE mean done")

print(f"\n[{stamp()}] Job finished successfully")