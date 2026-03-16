from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


@dataclass(frozen=True)
class TrafficLightRules:
    missing_green_zone: tuple[float, float]
    missing_yellow_zone: tuple[float, float]
    missing_red_zone: tuple[float, float]
    cardinality_limit: int


@dataclass(frozen=True)
class AlgorithmPreferences:
    auto_imputation_strategy: str
    knn_k: int


@dataclass(frozen=True)
class IOSettings:
    input_path: str
    output_path: str


@dataclass(frozen=True)
class AutoPrepSettings:
    io: IOSettings
    traffic_light: TrafficLightRules
    algorithms: AlgorithmPreferences
    raw: dict[str, Any]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and return a plain dictionary."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def parse_settings(config_path: str | Path | None = None) -> AutoPrepSettings:
    """Parse known settings into typed dataclasses with sensible defaults."""
    data = load_config(config_path)

    io_cfg = data.get("io", {})
    tl_cfg = data.get("traffic_light", {})
    algo_cfg = data.get("algorithms", {})

    settings = AutoPrepSettings(
        io=IOSettings(
            input_path=str(io_cfg.get("input_path", "data/sample.csv")),
            output_path=str(io_cfg.get("output_path", "reports/processed_data.csv")),
        ),
        traffic_light=TrafficLightRules(
            missing_green_zone=tuple(tl_cfg.get("missing_green_zone", [0.0, 0.10])),
            missing_yellow_zone=tuple(tl_cfg.get("missing_yellow_zone", [0.11, 0.79])),
            missing_red_zone=tuple(tl_cfg.get("missing_red_zone", [0.80, 1.0])),
            cardinality_limit=int(tl_cfg.get("cardinality_limit", 50)),
        ),
        algorithms=AlgorithmPreferences(
            auto_imputation_strategy=str(algo_cfg.get("auto_imputation_strategy", "median")),
            knn_k=int(algo_cfg.get("knn_k", 5)),
        ),
        raw=data,
    )
    return settings
