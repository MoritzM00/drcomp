"""Script to train and evaluate a model on a dataset."""

import pathlib
import pickle
import time

import hydra
import torchsummary
from omegaconf import DictConfig

from drcomp.reducers import AutoEncoder
from drcomp.utils._data_loading import _load_mnist


def save_model(model, cfg: DictConfig):
    model_path = pathlib.Path(f"{cfg.root_dir}/{cfg.model_dir}/{cfg.reducer._name_}")
    with open(model_path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(model, f)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # instantiate the reducer
    reducer = hydra.utils.instantiate(
        cfg.reducer, batch_size=cfg.dataset.batch_size, _convert_="object"
    )

    # load the data
    print("Loading data...")
    X_train = _load_mnist(cfg, flatten=cfg.reducer._flatten_)
    if isinstance(reducer, AutoEncoder):
        print("Summary of AutoEncoder model:")
        torchsummary.summary(reducer.module, input_size=X_train.shape[1:])

    # train the reducer
    print("Training model...")
    start = time.time()
    reducer.fit(X_train)
    end = time.time()
    print(f"Training took {end - start} seconds.")

    # save the model if training was successful
    print("Saving model...")
    model_path = f"{cfg.root_dir}/{cfg.model_dir}/{cfg.reducer._name_}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(reducer, f)
    print("Done.")


if __name__ == "__main__":
    main()
