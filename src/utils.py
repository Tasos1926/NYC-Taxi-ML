from __future__ import annotations

import json
import os
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_month_paths(months: list[str], data_dir: str) -> list[str]:
    return [os.path.join(data_dir, f"yellow_tripdata_{month}.parquet") for month in months]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict[str, Any], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)