from omegaconf import DictConfig
from sklearn.datasets import fetch_lfw_people, make_swiss_roll
from torchvision import datasets, transforms

from drcomp.utils._pathing import get_data_dir


def load_dataset_from_cfg(cfg: DictConfig):
    dataset = cfg.dataset
    flatten = cfg.reducer._flatten_
    if dataset.name not in cfg.available_datasets:
        raise ValueError(
            f"Unknown dataset given: {dataset.name}. Must be one of {cfg.available_datasets}"
        )
    if dataset.name == cfg.available_datasets[0]:
        # MNIST
        return load_mnist(cfg, flatten=flatten)
    elif dataset.name == cfg.available_datasets[1]:
        # CIFAR10
        return load_cifar10(cfg, flatten=flatten)
    elif dataset.name == cfg.available_datasets[2]:
        # Swiss roll
        return load_swiss_roll(n_samples=dataset.n_samples, noise=dataset.noise)
    elif dataset.name == cfg.available_datasets[3]:
        # LFW People
        return load_lfw_people(cfg, flatten=flatten)
    else:
        raise ValueError("Unknown dataset given.")


def load_mnist(cfg: DictConfig, flatten: bool = False):
    mnist_train = datasets.MNIST(
        root=get_data_dir(cfg), download=True, transform=transforms.ToTensor()
    )
    X_train = mnist_train.data.numpy().astype("float32")
    if flatten:
        X_train = X_train.reshape((X_train.shape[0], -1))
    else:
        X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
    return X_train


def load_cifar10(cfg: DictConfig, flatten: bool = False):
    cifar_train = datasets.CIFAR10(
        root=get_data_dir(cfg), download=True, transform=transforms.ToTensor()
    )
    X_train = cifar_train.data.astype("float32")
    if flatten:
        X_train = X_train.reshape((X_train.shape[0], -1))
    else:
        X_train = X_train.transpose((0, 3, 1, 2))
    return X_train


def load_swiss_roll(n_samples: int = 3000, noise: float = 0.0):
    X, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
    return X.astype("float32")


def load_lfw_people(cfg: DictConfig, flatten: bool = False):
    X, _ = fetch_lfw_people(
        min_faces_per_person=cfg.dataset.min_faces_per_person,
        data_home=get_data_dir(cfg),
        return_X_y=True,
    )
    if flatten:
        X = X.reshape((X.shape[0], -1))
    return X
