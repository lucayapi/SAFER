import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _as_repo_path(repo_root: str, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(repo_root, path))


from malt_text.malt_transfer import run_malt_training
from malt_text.utils import resolve_existing_path, resolve_target_embedding_csv
from scgm_text.utils_io import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MALT target model from a SCGM source checkpoint.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--source_checkpoint", type=str, default="runs/scgm_text_qwen06/best_model.pt")
    parser.add_argument("--source_config", type=str, default="runs/scgm_text_qwen06/config.json")
    parser.add_argument("--target_data_csv", type=str, default="dataset/data_mettalurgie.csv")
    parser.add_argument("--target_data_csv_alt", type=str, default="dataset/data_metallurgie.csv")
    parser.add_argument("--target_emb_csv", type=str, default="embeddings/Qwen3-Embedding-0.6B_mettalurgie.csv")
    parser.add_argument("--target_emb_csv_alt", type=str, default="embeddings/Qwen3-Embedding-0.6B_metallurgie.csv")
    parser.add_argument("--output_dir", type=str, default="runs/malt_btp_to_mettalurgie_qwen06")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--tau_macro", type=float, default=0.1)
    parser.add_argument("--tau_z", type=float, default=0.1)
    parser.add_argument("--tau_yz", type=float, default=0.1)
    parser.add_argument("--tau_div", type=float, default=0.1)
    parser.add_argument("--beta_latent", type=float, default=1.0)
    parser.add_argument("--beta_anchor", type=float, default=1.0)
    parser.add_argument("--beta_div", type=float, default=0.1)
    parser.add_argument("--n_subclass", type=int, default=32)
    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    parser.add_argument("--freeze_projector", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--filter_pred_ok", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--latent_loss_mode", choices=["marginal", "sinkhorn"], default="marginal")
    parser.add_argument("--use_sinkhorn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sinkhorn_lmd", type=float, default=25.0)
    parser.add_argument("--disable_softmacro", action="store_true")
    parser.add_argument("--disable_latent", action="store_true")
    parser.add_argument("--disable_anchor", action="store_true")
    parser.add_argument("--disable_div", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    config = load_yaml_config(args.config)
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def main() -> None:
    args = apply_config(parse_args())
    args.source_checkpoint = _as_repo_path(ROOT_DIR, args.source_checkpoint)
    args.output_dir = _as_repo_path(ROOT_DIR, args.output_dir)
    args.resolved_target_data_csv = resolve_existing_path(
        _as_repo_path(ROOT_DIR, args.target_data_csv),
        _as_repo_path(ROOT_DIR, args.target_data_csv_alt),
        "target data CSV",
    )
    args.resolved_target_emb_csv = resolve_target_embedding_csv(
        _as_repo_path(ROOT_DIR, args.target_emb_csv),
        _as_repo_path(ROOT_DIR, args.target_emb_csv_alt),
        _as_repo_path(ROOT_DIR, "embeddings"),
    )
    run_malt_training(args)


if __name__ == "__main__":
    main()
