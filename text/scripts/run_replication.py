"""CLI des réplications multi-seeds (une tâche Slurm = un modèle × une seed)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from replication.runner import load_replication_config, run_replication, task_matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Réplication multi-seeds des modèles macro.")
    parser.add_argument("--config", default="output/replication_recipes/replication_config.yaml")
    parser.add_argument("--model", help="Identifiant modèle de la recette.")
    parser.add_argument("--training-seed", type=int, help="Seed d'entraînement de cette tâche.")
    parser.add_argument("--task-index", type=int, help="Indice de l'array Slurm (modèle × seed).")
    parser.add_argument("--print-task-count", action="store_true")
    parser.add_argument("--print-tasks", action="store_true")
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args(argv)
    config = load_replication_config(args.config)
    tasks = task_matrix(config)
    if args.print_task_count:
        print(len(tasks))
        return 0
    if args.print_tasks:
        for index, task in enumerate(tasks):
            print(f"{index}\t{task['model_id']}\t{task['training_seed']}")
        return 0
    if args.task_index is not None:
        if not 0 <= args.task_index < len(tasks):
            raise ValueError(f"task-index {args.task_index} invalide (0..{len(tasks)-1})")
        task = tasks[args.task_index]
        model_id, seed = task["model_id"], task["training_seed"]
    else:
        if not args.model or args.training_seed is None:
            parser.error("Fournir --task-index ou le couple --model / --training-seed.")
        model_id, seed = args.model, args.training_seed
    print(f"[replication] modèle={model_id} training_seed={seed}", flush=True)
    print(run_replication(args.config, model_id=model_id, training_seed=seed, refit=args.refit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
