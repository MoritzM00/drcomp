"""Script to train and evaluate a model on a dataset."""

import pathlib
import pickle
import time

import hydra
import matplotlib.pyplot as plt
import torch
import torchsummary
from loguru import logger
from omegaconf import DictConfig

from drcomp.reducers import AutoEncoder
from drcomp.utils._data_loading import _load_mnist


def save_model(model, cfg: DictConfig):
    base_path = pathlib.Path(cfg.root_dir, cfg.model_dir, cfg.dataset.name)
    base_path.mkdir(parents=True, exist_ok=True)

    model_path = pathlib.Path(base_path, cfg.reducer._name_)

    with model_path.open("wb") as f:
        pickle.dump(model, f)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train and evaluate a model on a dataset."""
    # instantiate the reducer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using device: {device}")
    reducer = hydra.utils.instantiate(
        cfg.reducer,
        batch_size=cfg.dataset.batch_size,
        device=device,
        _convert_="object",
    )

    # load the data
    logger.info("Loading data...")
    X_train = _load_mnist(cfg, flatten=cfg.reducer._flatten_)
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
    save_model(reducer, cfg)

    # evaluate the reducer
    logger.info("Evaluating model...")
    Y = reducer.transform(X_train)
    Q = reducer.evaluate(X_train, Y, n_jobs=cfg.n_jobs)["coranking"]
    plt.matshow(Q)
    logger.info("Done.")


if __name__ == "__main__":
    main()
