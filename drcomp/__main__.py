"""Script to train and evaluate a model on a dataset."""

import pickle

import hydra
import torchsummary
from omegaconf import DictConfig
from torchvision import datasets, transforms

from drcomp.reducers import AutoEncoder


def _load_mnist(cfg: DictConfig, flatten: bool = False):
    mnist_train = datasets.MNIST(
        root=cfg.data_dir, download=True, transform=transforms.ToTensor()
    )
    X_train = mnist_train.data.numpy().astype("float32")
    if flatten:
        X_train = X_train.reshape((X_train.shape[0], -1))
    else:
        X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
    return X_train


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    reducer = hydra.utils.instantiate(
        cfg.reducer, batch_size=cfg.dataset.batch_size, _convert_="object"
    )
    X_train = _load_mnist(cfg, flatten=cfg.reducer._flatten_)
    if isinstance(reducer, AutoEncoder):
        torchsummary.summary(reducer.module, input_size=X_train.shape[1:])
    reducer.fit(X_train)
    model_path = f"{cfg.root_dir}/{cfg.model_dir}/{cfg.reducer._name_}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(reducer, f)


if __name__ == "__main__":
    main()
