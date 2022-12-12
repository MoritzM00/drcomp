import json
import logging
import pickle
from pathlib import Path

from omegaconf import DictConfig

from drcomp import DimensionalityReducer

logger = logging.getLogger(__name__)


def save_model_from_cfg(model, cfg: DictConfig) -> None:
    base_path = Path(cfg.root_dir, cfg.model_dir, cfg.dataset.name)
    base_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(base_path, cfg.reducer._name_)
    save_model(model, model_path)


def save_model(model: DimensionalityReducer, path: Path) -> None:
    """Save a model to a file using pickle."""
    logger.debug(f"Saving model to {path.with_suffix('.pkl')}")
    with path.with_suffix(".pkl").open("wb") as f:
        pickle.dump(model, f)


def save_metrics_from_cfg(metrics: dict, cfg: DictConfig) -> None:
    base_path = Path(cfg.root_dir, cfg.metrics_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(base_path, f"{cfg.dataset.name}_{cfg.reducer._name_}.json")
    save_metrics(metrics, metrics_path)


def save_metrics(metrics: dict, path: Path) -> None:
    logger.debug(f"Saving metrics to {path}")
    json.dump(metrics, path.open("w"), indent=4, sort_keys=True)
