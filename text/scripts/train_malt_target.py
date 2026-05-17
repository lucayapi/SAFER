import argparse
import os
import sys
from typing import Any, Dict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _as_repo_path(repo_root: str, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(repo_root, path))


def _flatten_config(data: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for section in ("run", "source", "target", "model", "temperatures", "em", "loss", "training"):
        block = data.get(section)
        if isinstance(block, dict):
            flat.update(block)
    for key, value in data.items():
        if key not in ("run", "source", "target", "model", "temperatures", "em", "loss", "training"):
            if not isinstance(value, dict):
                flat[key] = value
    return flat


from malt_text.malt_transfer import run_malt_training
from malt_text.utils import resolve_existing_path, resolve_target_embedding_csv
from scgm_text.utils_io import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MALT-EM on target corpus (SCGM-like EM transfer).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--source_checkpoint", type=str, default="runs/scgm_text_qwen06/best_model.pt")
    parser.add_argument("--target_data_csv", type=str, default="dataset/data_metallurgie.csv")
    parser.add_argument("--target_data_csv_alt", type=str, default="dataset/data_mettalurgie.csv")
    parser.add_argument("--target_emb_csv", type=str, default="embeddings/Qwen3-Embedding-0.6B_metallurgie.csv")
    parser.add_argument("--target_emb_csv_alt", type=str, default="embeddings/Qwen3-Embedding-0.6B_mettalurgie.csv")
    parser.add_argument("--output_dir", type=str, default="runs/malt_btp_to_mettalurgie_qwen06")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--num_cycles", type=int, default=10)
    parser.add_argument("--tau_macro", type=float, default=0.1)
    parser.add_argument("--tau_z", type=float, default=0.1)
    parser.add_argument("--tau_yz", type=float, default=0.1)
    parser.add_argument("--tau_div", type=float, default=0.1)
    parser.add_argument("--n_subclass", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--hiddim", type=int, default=128)
    parser.add_argument("--projection", type=str, default="linear", choices=["identity", "linear", "mlp"])
    parser.add_argument("--n_iter_estep", type=int, default=5)
    parser.add_argument("--sinkhorn_lmd", type=float, default=25.0)
    parser.add_argument("--em_q_mode", type=str, default="hard", choices=["hard", "soft"])
    parser.add_argument("--init_q_mode", type=str, default="source_scores")
    parser.add_argument("--beta_anchor", type=float, default=1.0)
    parser.add_argument("--beta_div", type=float, default=0.1)
    parser.add_argument("--beta_macro", type=float, default=0.5)
    parser.add_argument("--beta_balance", type=float, default=0.0)
    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    parser.add_argument("--macro_weight_mode", type=str, default="max_prob", choices=["max_prob", "threshold", "none"])
    parser.add_argument("--copy_source_projector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze_projector", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--init_mu_target", type=str, default="source", choices=["source", "random"])
    parser.add_argument("--init_nu", type=str, default="kmeans")
    parser.add_argument("--save_q_every_estep", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter_pred_ok", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable_anchor", action="store_true")
    parser.add_argument("--disable_div", action="store_true")
    parser.add_argument("--disable_macro", action="store_true")
    parser.add_argument("--disable_balance", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    raw = load_yaml_config(_as_repo_path(ROOT_DIR, args.config))
    flat = _flatten_config(raw) if any(k in raw for k in ("run", "training", "em")) else raw
    for key, value in flat.items():
        key_norm = key.replace("-", "_")
        if hasattr(args, key_norm):
            setattr(args, key_norm, value)
    return args


def main() -> None:
    args = apply_config(parse_args())
    if args.run_name:
        args.output_dir = os.path.join("runs", args.run_name)
    args.source_checkpoint = _as_repo_path(ROOT_DIR, args.source_checkpoint)
    args.output_dir = _as_repo_path(ROOT_DIR, args.output_dir)
    args.resolved_target_data_csv = resolve_existing_path(
        _as_repo_path(ROOT_DIR, args.target_data_csv),
        _as_repo_path(ROOT_DIR, args.target_data_csv_alt),
        "target data CSV",
        repo_root=ROOT_DIR,
    )
    args.resolved_target_emb_csv = resolve_target_embedding_csv(
        _as_repo_path(ROOT_DIR, args.target_emb_csv),
        _as_repo_path(ROOT_DIR, args.target_emb_csv_alt),
        repo_root=ROOT_DIR,
    )
    run_malt_training(args)


if __name__ == "__main__":
    main()
