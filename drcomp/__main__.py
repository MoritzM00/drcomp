"""Script to train and evaluate a model on a dataset."""

import logging
import pickle
import time

import hydra
import torch
import torchinfo
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from sklearn.utils import resample

from drcomp import DimensionalityReducer, estimate_intrinsic_dimension
from drcomp.reducers import LLE, AutoEncoder
from drcomp.utils._data_loading import load_dataset_from_cfg
from drcomp.utils._pathing import get_model_path
from drcomp.utils._saving import (
    save_metrics_from_cfg,
    save_model_from_cfg,
    save_preprocessor_from_cfg,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train and evaluate a model on a dataset."""
    if cfg._skip_:
        logger.info(
            f"Skipping run {cfg.dataset.name} - {cfg.reducer._name_} because this combination of reducer and dataset is not compatible."
        )
        return
    try:
        train(cfg)
    except Exception:
        logger.exception("An exception occurred during training. Exiting.")


if __name__ == "__main__":
    main()


def train(cfg: DictConfig):
    """Train a model on a dataset."""
    # load the data
    train_time_start = time.time()
    logger.info(f"Loading dataset: {cfg.dataset.name}")
    X, _ = load_dataset_from_cfg(cfg)
    try:
        cfg.dataset.intrinsic_dim
    except MissingMandatoryValue:
        # estimate intrinsic dimensionality
        intrinsic_dim = estimate_intrinsic_dimension(X)
        logger.info(f"ML Estimate of intrinsic dimensionality: {intrinsic_dim}")
        cfg.dataset.intrinsic_dim = intrinsic_dim

    logger.debug(f"Config: {OmegaConf.to_yaml(cfg, resolve=True)}")

    reducer = instantiate_reducer(cfg)

    # preprocess the data
    preprocessor = hydra.utils.instantiate(cfg.preprocessor)
    logger.info(f"Preprocessing data with {preprocessor.__class__.__name__}.")
    X = preprocessor.fit_transform(X)
    save_preprocessor_from_cfg(preprocessor, cfg)

    # sample subset of data if necessary
    if (
        cfg.reducer._max_sample_size_ is not None  # reducer has constraints
        and X.shape[0] > cfg.reducer._max_sample_size_  # dataset too large
    ):
        logger.info(
            f"Sampling {cfg.reducer._max_sample_size_} samples from the dataset because of computational constraints of the reducer."
        )
        X = resample(X, n_samples=cfg.reducer._max_sample_size_)

    # data is flattened by default because most reducer expect it this way
    # only convolutional autoencoders expect the data to be in the shape of an image
    if not cfg.reducer._flatten_:
        X = X.reshape(X.shape[0], *cfg.dataset.image_size)

    if isinstance(reducer, AutoEncoder):
        input_size = (cfg.dataset.batch_size, *X.shape[1:])
        logger.debug(f"Input size of X_train (with Batch Size first): {input_size}")
        logger.info("Summary of AutoEncoder model:")
        stats = torchinfo.summary(reducer.module, input_size=input_size, verbose=0)
        logger.info(stats)

    # train the reducer if use_pretrained is false, else try to load the pretrained model
    reducer = fit_reducer(cfg, reducer, X)

    # evaluate the reducer
    if cfg.evaluate:
        evaluate(cfg, reducer, X)
    else:
        logger.info("Skipping evaluation because `evaluate` was set to False.")
    logger.info(f"Finished in {time.time() - train_time_start:.2f} seconds.")


def instantiate_reducer(cfg):
    # instantiate the reducer
    device = "cpu"
    if cfg.use_gpu and torch.cuda.is_available():
        device = "cuda"
    logger.debug(f"Using device (GPU support only for Autoencoders): {device}")
    logger.info(f"Using dimensionality reducer: {cfg.reducer._name_}")
    reducer = hydra.utils.instantiate(
        cfg.reducer,
        batch_size=cfg.dataset.batch_size,
        device=device,
        _convert_="object",
    )
    return reducer


def fit_reducer(cfg, reducer, X):
    failed = False
    if cfg.use_pretrained:
        logger.info(
            "Loading pretrained model because `use_pretrained` was set to True."
        )
        try:
            path = get_model_path(cfg)
            reducer = pickle.load(open(path, "rb"))
        except FileNotFoundError:
            failed = True
            logger.warning(f"Could not find pretrained model at {path}.")
    if not cfg.use_pretrained or failed:
        logger.info("Training model...")
        start = time.time()
        reducer.fit(X)
        end = time.time()
        logger.info(f"Training took {end - start:.2f} seconds.")
        logger.info("Saving model...")
        save_model_from_cfg(reducer, cfg)
    return reducer


def evaluate(cfg, reducer: DimensionalityReducer, X):
    logger.info("Evaluating model...")
    Y = None
    if isinstance(reducer, LLE):
        Y = reducer.lle.embedding_
    else:
        Y = reducer.transform(X)
    if X.shape[0] > cfg.max_evaluation_samples:
        logger.info(
            f"Sampling {cfg.max_evaluation_samples} samples from the dataset because of computational constraints of the evaluation."
        )
        X, Y = resample(X, Y, n_samples=cfg.max_evaluation_samples)
    start = time.time()
    metrics = reducer.evaluate(
        X=X, Y=Y, max_K=cfg.max_n_neighbors, as_builtin_list=True
    )
    end = time.time()
    logger.info(f"Evaluation took {end - start:.2f} seconds.")
    save_metrics_from_cfg(metrics, cfg)
