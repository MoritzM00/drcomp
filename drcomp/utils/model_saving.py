import pickle
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig

from drcomp import DimensionalityReducer


def save_model_from_cfg(model, cfg: DictConfig):
    base_path = Path(cfg.root_dir, cfg.model_dir, cfg.dataset.name)
    base_path.mkdir(parents=True, exist_ok=True)
    model_path = Path(base_path, cfg.reducer._name_)
    save_model(model, model_path)


def save_model(model: DimensionalityReducer, path: Path):
    """Save a model to a file using pickle."""
    logger.debug(f"Saving model to {path}")
    with path.with_suffix(".pkl").open("wb") as f:
        pickle.dump(model, f)
