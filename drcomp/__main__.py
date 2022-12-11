"""Script to train and evaluate a model on a dataset."""

import logging
import time

import hydra
import matplotlib.pyplot as plt
import torch
import torchsummary
from omegaconf import DictConfig

from drcomp.reducers import AutoEncoder
from drcomp.utils._data_loading import load_dataset_from_cfg
from drcomp.utils.model_saving import save_model_from_cfg

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train and evaluate a model on a dataset."""
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
    logger.info(f"Loading dataset {name} ...")
    X_train = load_dataset_from_cfg(cfg)

    if isinstance(reducer, AutoEncoder):
        logger.info("Summary of AutoEncoder model:")
        torchsummary.summary(reducer.module, input_size=X_train.shape[1:])

    # train the reducer
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
    Q = reducer.evaluate(X_train, Y, n_jobs=cfg.n_jobs)["coranking"]
    plt.matshow(Q)
    plt.show()
    logger.info("Done.")


if __name__ == "__main__":
    main()
