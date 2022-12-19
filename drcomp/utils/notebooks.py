import pickle

import numpy as np
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from drcomp import DimensionalityReducer
from drcomp.utils._data_loading import load_dataset_from_cfg
from drcomp.utils._pathing import get_model_path


def get_model_for_dataset(
    dataset: str = "SwissRoll",
    reducer: str = "PCA",
    root_dir: str = ".",
    from_pretrained: bool = False,
) -> DimensionalityReducer:
    """Get a model and dataset pair configured by hydra for use in a notebook."""
    model: DimensionalityReducer = None
    with initialize_config_module(version_base="1.3", config_module="drcomp.conf"):
        cfg = compose(
            config_name="config.yaml",
            overrides=[
                f"reducer={reducer}",
                f"dataset={dataset}",
                f"root_dir={root_dir}",
            ],
        )
        if from_pretrained:
            try:
                path = get_model_path(cfg)
                model = pickle.load(open(path, "rb"))
            except FileNotFoundError as e:
                raise ValueError(f"Could not find pretrained model at {path}.") from e

        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = instantiate(
                cfg.reducer,
                batch_size=cfg.dataset.batch_size,
                device=device,
                _convert_="object",
            )
    return model


def get_data_set(dataset: str, root_dir: str = "."):
    """Get a dataset configured by hydra for use in a notebook."""
    X: np.ndarray = None
    with initialize_config_module(version_base="1.3", config_module="drcomp.conf"):
        cfg = compose(
            config_name="config.yaml",
            overrides=[f"dataset={dataset}", f"root_dir={root_dir}"],
        )
        X, targets = load_dataset_from_cfg(cfg)
    return X, targets
