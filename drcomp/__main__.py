"""Script to train and evaluate a model on a dataset."""

import logging
import pickle
import pprint
import time
from pathlib import Path

import hydra
import torch
import torchinfo
from omegaconf import DictConfig, OmegaConf

from drcomp.reducers import AutoEncoder
from drcomp.utils._data_loading import load_dataset_from_cfg
from drcomp.utils._saving import save_metrics_from_cfg, save_model_from_cfg

logger = logging.getLogger(__name__)


def get_model_path(cfg: DictConfig) -> str:
    """Get the path to the model file."""
    return Path(
        cfg.root_dir, cfg.model_dir, cfg.dataset.name, cfg.reducer._name_
    ).with_suffix(".pkl")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train and evaluate a model on a dataset."""
    if cfg._skip_:
        logger.info(
            "Skipping this run because this combination of reducer and dataset is not compatible."
        )
        return

    logger.debug(OmegaConf.to_yaml(cfg, resolve=True))

    # instantiate the reducer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using device: {device}")
    logger.info(f"Using dimensionality reducer: {cfg.reducer._name_}")
    reducer = hydra.utils.instantiate(
        cfg.reducer,
        batch_size=cfg.dataset.batch_size,
        device=device,
        _convert_="object",
    )

    # load the data
    name = cfg.dataset.name
    logger.info(f"Loading dataset: {name}")
    X_train = load_dataset_from_cfg(cfg)

    if isinstance(reducer, AutoEncoder):
        logger.info("Summary of AutoEncoder model:")
        torchinfo.summary(reducer.module, input_size=X_train.shape[1:])

    # train the reducer if use_pretrained is false
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
            logger.error(f"Could not find pretrained model at {path}.")
    if not cfg.use_pretrained or failed:
        logger.info("Training model...")
        start = time.time()
        reducer.fit(X_train)
        end = time.time()
        logger.info(f"Training took {end - start:.2f} seconds.")
        logger.info("Saving model...")
        save_model_from_cfg(reducer, cfg)

    # evaluate the reducer
    logger.info("Evaluating model...")
    Y = reducer.transform(X_train)
    start = time.time()
    metrics = reducer.evaluate(X_train, Y)
    end = time.time()

    pp = pprint.PrettyPrinter(indent=2, stream=logger)
    logger.info(f"Evaluation took {end - start:.2f} seconds.")
    logger.info(pp.pformat(metrics))
    save_metrics_from_cfg(metrics, cfg)
    logger.info("Done.")


if __name__ == "__main__":
    main()
