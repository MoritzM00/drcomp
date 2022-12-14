import numpy as np
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from drcomp import DimensionalityReducer
from drcomp.utils._data_loading import load_dataset_from_cfg


def get_model_dataset_pair(
    reducer: str = "PCA", dataset: str = "SwissRoll"
) -> tuple[DimensionalityReducer, np.ndarray]:
    """Get a model and dataset pair configured by hydra for use in a notebook."""
    model: DimensionalityReducer = None
    dataset: np.ndarray = None
    with initialize_config_module(version_base="1.3", config_module="drcomp.conf"):
        cfg = compose(
            config_name="config.yaml",
            overrides=[f"reducer={reducer}", f"dataset={dataset}"],
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = instantiate(
            cfg.reducer,
            batch_size=cfg.dataset.batch_size,
            device=device,
            _convert_="object",
        )
        dataset = load_dataset_from_cfg(cfg)
    return model, dataset
