import json
import logging
import pickle
from pathlib import Path

from omegaconf import DictConfig

from drcomp import DimensionalityReducer
from drcomp.utils._pathing import get_metrics_dir, get_model_dir, get_model_path

logger = logging.getLogger(__name__)


def save_model_from_cfg(model, cfg: DictConfig) -> None:
    """Save a model to a file using a dict config object."""
    base = get_model_dir(cfg, for_dataset=True)
    base.mkdir(parents=True, exist_ok=True)
    model_path = get_model_path(cfg)
    save_model(model, model_path)


def save_model(model: DimensionalityReducer, path: Path) -> None:
    """Save a model to a file using pickle."""
    logger.debug(f"Saving model to {path}")
    with path.open("wb") as f:
        pickle.dump(model, f)


def save_metrics_from_cfg(metrics: dict, cfg: DictConfig) -> None:
    """Save metrics to a file using a dict config object."""
    metrics_dir = get_metrics_dir(cfg)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(metrics_dir, f"{cfg.dataset.name}_{cfg.reducer._name_}.json")
    logger.info(f"Saved metrics to {metrics_path}")
    save_metrics(metrics, metrics_path)


def save_metrics(metrics: dict, path: Path) -> None:
    """Save metrics to a file using json."""
    logger.debug(f"Saving metrics to {path}")
    json.dump(metrics, path.open("w"), indent=4, sort_keys=True)
