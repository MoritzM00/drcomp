import json
import pathlib
import pickle

import numpy as np
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from sklearn.base import BaseEstimator

from drcomp import DimensionalityReducer
from drcomp.utils._data_loading import load_dataset_from_cfg
from drcomp.utils._pathing import get_model_path, get_preprocessor_path


def load_metrics(dataset: str, reducer: str, dir="../metrics"):
    filename = f"{dataset}_{reducer}.json"
    path = pathlib.Path(dir, filename)
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found.")
    with path.open("r") as f:
        return json.load(f)


def load_all_metrics_for(
    dataset: str,
    reducers: list[str] = None,
    throw_on_missing: bool = True,
    dir: str = "../metrics",
):
    if reducers is None:
        reducers = ["ConvAE", "PCA", "KernelPCA", "AE", "LLE", "CAE"]
    metrics: dict[str, dict] = {}
    for reducer in reducers:
        try:
            metric = load_metrics(dataset, reducer, dir=dir)
        except FileNotFoundError:
            if throw_on_missing:
                raise
            else:
                continue
        metrics[reducer] = metric
    return metrics


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
            path = get_model_path(cfg)
            model = load_from_pretrained(path)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = instantiate(
                cfg.reducer,
                batch_size=cfg.dataset.batch_size,
                device=device,
                _convert_="object",
            )
    return model


def get_dataset(dataset: str, root_dir: str = "."):
    """Get a dataset configured by hydra for use in a notebook."""
    X: np.ndarray = None
    with initialize_config_module(version_base="1.3", config_module="drcomp.conf"):
        cfg = compose(
            config_name="config.yaml",
            overrides=[f"dataset={dataset}", f"root_dir={root_dir}"],
        )
        X, targets = load_dataset_from_cfg(cfg)
    return X, targets


def get_preprocessor(dataset: str, root_dir=".") -> BaseEstimator:
    """Get the fitted preprocessor for a dataset."""
    preprocessor: BaseEstimator
    with initialize_config_module(version_base="1.3", config_module="drcomp.conf"):
        cfg = compose(
            config_name="config.yaml",
            overrides=[f"dataset={dataset}", f"root_dir={root_dir}"],
        )
        try:
            path = get_preprocessor_path(cfg)
            preprocessor = load_from_pretrained(path)
        except FileNotFoundError:
            X, _ = load_dataset_from_cfg(cfg)
            preprocessor = instantiate(cfg.preprocessor, _convert_="object")
            preprocessor.fit(X)
    return preprocessor


def load_from_pretrained(path) -> BaseEstimator:
    """Load a model from the specified path using pickle."""
    model: BaseEstimator
    try:
        model = pickle.load(open(path, "rb"))
    except FileNotFoundError as e:
        raise ValueError(f"Could not find pretrained model at {path}.") from e
    return model
