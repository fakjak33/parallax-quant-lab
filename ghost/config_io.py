"""Save/load a full lab setup to YAML for reproducible, shareable runs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from .config import BacktestConfig, RiskConfig, CONFIG_DIR


def save_config(
    name: str,
    source: dict,
    strategies: list[dict],
    bt: BacktestConfig,
    risk: RiskConfig,
) -> Path:
    """Persist a lab configuration. ``source`` describes data selection,
    ``strategies`` is a list of {key, params}. Returns the written path."""
    payload = {
        "source": source,
        "strategies": strategies,
        "backtest": asdict(bt),
        "risk": asdict(risk),
    }
    path = CONFIG_DIR / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def load_config(name: str) -> dict:
    """Load a saved configuration into a dict with reconstructed dataclasses."""
    path = CONFIG_DIR / f"{name}.yaml" if not name.endswith(".yaml") else Path(name)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["backtest"] = BacktestConfig(**data.get("backtest", {}))
    data["risk"] = RiskConfig(**data.get("risk", {}))
    return data


def list_configs() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
