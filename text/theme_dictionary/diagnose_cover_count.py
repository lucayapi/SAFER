"""Count cover partitions for the richest observed basket without fitting EM.

Run from text/: python -m theme_dictionary.diagnose_cover_count
"""

from __future__ import annotations

import argparse
import json
import time

import yaml

from .model import count_cover_groups, mine_dictionary
from .run import load_inputs, resolve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="theme_dictionary/config.yaml")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--cap", type=int, default=10_000_000)
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    _, dictionary, baskets, audit = load_inputs(config)
    model = config["model"]
    candidates = mine_dictionary(
        baskets, dictionary["role"].tolist(),
        int(model["max_theme_length"]), int(model["min_observed_count"]),
    )
    basket = max(baskets, key=len)
    started = time.monotonic()
    count, capped = count_cover_groups(basket, candidates, args.k, args.cap)
    print(json.dumps({
        "accidents": audit["accidents"], "factors": audit["factors"],
        "initial_themes": len(candidates), "basket_size": len(basket),
        "contained_composite_themes": sum(
            len(theme.items) > 1 and set(theme.items) <= set(basket)
            for theme in candidates
        ),
        "k": args.k, "cover_group_count_or_lower_bound": count,
        "capped": capped, "seconds": round(time.monotonic() - started, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
