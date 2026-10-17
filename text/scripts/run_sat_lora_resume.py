#!/usr/bin/env python3
"""Run the pinned upstream LoRA trainer, fixing its full-state checkpoint loader.

The vendored AdapterTrainer saves ``Model.state_dict()`` to pytorch_model.bin,
but its default resume method only looks for adapter subdirectories.  This
loader restores the exact saved model state; Hugging Face Trainer subsequently
restores optimizer, scheduler, RNG and trainer state as usual.
"""

from pathlib import Path

import torch
from wtpsplit.train.adaptertrainer import AdapterTrainer
from wtpsplit.train.transformers_trainer_imports import WEIGHTS_NAME


def load_full_checkpoint(self: AdapterTrainer, checkpoint: str) -> None:
    weights = Path(checkpoint) / WEIGHTS_NAME
    if not weights.is_file():
        raise FileNotFoundError(f"No full model state in checkpoint: {weights}")
    print(f"Restoring full model state from {weights}", flush=True)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    self.model.load_state_dict(state, strict=True)


def main() -> None:
    AdapterTrainer._load_from_checkpoint = load_full_checkpoint
    from wtpsplit.train.train_lora import main as upstream_main

    upstream_main()


if __name__ == "__main__":
    main()
