import json
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_numpy(array: np.ndarray, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    np.save(path, array)


def get_dim_columns(df: pd.DataFrame) -> List[str]:
    columns = [column for column in df.columns if column.startswith("dim_")]
    if not columns:
        raise ValueError("No dim_* columns found in embedding dataframe.")
    return sorted(columns, key=lambda name: int(name.split("_", 1)[1]))


def parse_bool_column(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "t"})


def create_doc_id_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "doc_id" not in frame.columns:
        frame["doc_id"] = np.arange(1, len(frame) + 1, dtype=np.int64)
    else:
        frame["doc_id"] = frame["doc_id"].astype(np.int64)
    return frame


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data
