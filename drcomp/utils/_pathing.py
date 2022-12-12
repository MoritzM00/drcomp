from pathlib import Path

from omegaconf import DictConfig


def get_model_dir(cfg: DictConfig, for_dataset=True) -> Path:
    path = Path(cfg.root_dir, cfg.model_dir)
    if for_dataset:
        path = Path(path, cfg.dataset.name)
    return path


def get_model_path(cfg: DictConfig) -> str:
    """Get the path to the model file."""
    base = get_model_dir(cfg, for_dataset=True)
    return Path(base, cfg.reducer._name_).with_suffix(".pkl")


def get_data_dir(cfg: DictConfig) -> Path:
    return Path(cfg.root_dir, cfg.data_dir)


def get_metrics_dir(cfg: DictConfig) -> Path:
    return Path(
        cfg.root_dir,
        cfg.metrics_dir,
    )
