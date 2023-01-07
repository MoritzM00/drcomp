import json
import pathlib
import pickle

import numpy as np
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from sklearn.base import BaseEstimator

from drcomp import DimensionalityReducer, MetricsDict
from drcomp.utils._data_loading import load_dataset_from_cfg
from drcomp.utils._pathing import get_model_path, get_preprocessor_path


def load_metrics(
    dataset: str, reducer: str, dir="../metrics", reduce_metric=None
) -> MetricsDict:
    """Load the metrics for a given dataset and reducer.

    Parameters
    ----------
    dataset : str
        The name of the dataset.
    reducer : str
        The name of the reducer.
    dir : str, default="../metrics"
        The directory to load the metrics from.
    reduction : str, default=None
        If not None, then reduce the metric by this function. Else, no reduction is applied.
        This must be a function, that maps an array of floats to a single scalar, for example, mean or max.

    Returns
    -------
    metrics : dict
        The metrics for the given dataset and reducer.
    """
    filename = f"{dataset}_{reducer}.json"
    path = pathlib.Path(dir, filename)
    metrics: MetricsDict
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found.")
    with path.open("r") as f:
        metrics = json.load(f)
    if reduce_metric is not None:
        metrics = {k: reduce_metric(v) for k, v in metrics.items()}
    return metrics


def load_all_metrics_for(
    dataset: str,
    reducers: list[str] = None,
    throw_on_missing: bool = True,
    dir: str = "../metrics",
    reduce_metric: str = None,
):
    """Load all metrics for a given dataset and for the given reducers with optional reduction (e.g. mean)

    Parameters
    ----------
    dataset : str
        The name of the dataset.
    reducers : list[str], default=None
        The list of reducers to load. If None, then all available reducers are loaded.
    throw_on_missing : bool, default=True
        If True, then raise a FileNotFoundError if the metric for a reducer is missing.
    dir : str, default="../metrics"
        The directory to load the metrics from.
    reduce_metric : str, default=None
        If not None, then reduce the metric by this function. Else, no reduction is applied.
        This must be a function, that maps an array of floats to a single scalar, for example, "np.mean" or "np.max".
    """
    if reducers is None:
        reducers = ["PCA", "KernelPCA", "LLE", "AE", "CAE", "ConvAE"]
    metrics: dict[str, dict] = {}
    for reducer in reducers:
        try:
            metric = load_metrics(
                dataset, reducer, dir=dir, reduce_metric=reduce_metric
            )
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


def get_preprocessor(dataset: str, root_dir=".", from_pretrained=True) -> BaseEstimator:
    """Get the preprocessor for a dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset to get the preprocessor for.
    root_dir : str, default="."
        The root directory where the project is located.
    from_pretrained : bool, default=True
        Whether to load the pretrained preprocessor or instantiate a new one. If True, attempt to load the pretrained model from the preprocessor directory."""
    preprocessor: BaseEstimator
    with initialize_config_module(version_base="1.3", config_module="drcomp.conf"):
        cfg = compose(
            config_name="config.yaml",
            overrides=[f"dataset={dataset}", f"root_dir={root_dir}"],
        )
        if from_pretrained:
            path = get_preprocessor_path(cfg)
            preprocessor = load_from_pretrained(path)
        else:
            preprocessor = instantiate(cfg.preprocessor, _convert_="object")
    return preprocessor


def load_from_pretrained(path) -> BaseEstimator:
    """Load a model from the specified path using pickle."""
    model: BaseEstimator
    try:
        model = pickle.load(open(path, "rb"))
    except FileNotFoundError as e:
        raise ValueError(f"Could not find pretrained model at {path}.") from e
    return model
