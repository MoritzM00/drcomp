from omegaconf import DictConfig
from sklearn.datasets import (
    fetch_lfw_people,
    fetch_olivetti_faces,
    fetch_openml,
    make_swiss_roll,
)
from torchvision import datasets, transforms

from drcomp.utils._pathing import get_data_dir


def load_dataset_from_cfg(cfg: DictConfig):
    """Load the dataset specified in the config.

    Uses the `name` attribute of the `dataset` config to determine which dataset to load.
    The dataset is loaded from the directory specified in the `data_dir` attribute of the `dataset` config.
    The dataset is loaded using the `load_{name}` function, which accepts the `dataset` config as an argument, as well
    as the per-dataset-specific config dictionary. Datasets that are not saved to disk only accept the dataset config.
    """
    dataset_cfg = cfg.dataset  # config for the dataset
    name = dataset_cfg.name
    data_dir = get_data_dir(cfg)
    X = None
    targets = None
    if name not in cfg.available_datasets:
        raise ValueError(
            f"Unknown dataset given: {name}. Must be one of {cfg.available_datasets}"
        )
    if name == cfg.available_datasets[0]:
        # MNIST
        X, targets = load_mnist(data_dir, dataset_cfg)
    elif name == cfg.available_datasets[1]:
        # CIFAR10
        X, targets = load_cifar10(data_dir, dataset_cfg)
    elif name == cfg.available_datasets[2]:
        # Swiss roll
        X, targets = load_swiss_roll(dataset_cfg)
    elif name == cfg.available_datasets[3]:
        # LFW People
        X, targets = load_lfw_people(data_dir, dataset_cfg)
    elif name == cfg.available_datasets[4]:
        # Dorothea
        X, targets = load_dorothea(data_dir, dataset_cfg)
    elif name == cfg.available_datasets[5]:
        # Olivetti faces
        X, targets = load_olivetti_faces(data_dir, dataset_cfg)
    else:
        raise ValueError(f"Unknown dataset {name} given.")
    return X, targets


def load_mnist(data_dir, dataset_cfg: DictConfig):
    mnist_train = datasets.MNIST(
        root=data_dir, download=True, transform=transforms.ToTensor()
    )
    X_train = mnist_train.data.numpy().astype("float32")
    X_train = X_train.reshape((X_train.shape[0], -1))
    targets = mnist_train.targets.numpy().astype("int64")
    return X_train, targets


def load_cifar10(data_dir, dataset_cfg: DictConfig):
    cifar_train = datasets.CIFAR10(
        root=data_dir, download=True, transform=transforms.ToTensor()
    )
    X_train = cifar_train.data.numpy().astype("float32")
    X_train = X_train.reshape((X_train.shape[0], -1))
    targets = cifar_train.targets.numpy().astype("int64")
    return X_train, targets


def load_swiss_roll(dataset_cfg: DictConfig):
    try:
        n_samples = dataset_cfg.n_samples
        noise = dataset_cfg.noise
    except KeyError:
        raise ValueError(
            "Invalid dataset config. Swiss roll dataset requires `n_samples` and `noise` to be set in the config."
        )
    X, targets = make_swiss_roll(n_samples=n_samples, noise=noise)
    X = X.astype("float32")
    return X, targets


def load_lfw_people(data_dir, dataset_cfg: DictConfig):
    try:
        min_faces_per_person = dataset_cfg.min_faces_per_person
        resize = dataset_cfg.resize
    except KeyError:
        raise ValueError(
            "Invalid dataset config. LFW People dataset requires `min_faces_per_person` and `resize` to be set in the config."
        )
    X, targets = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person,
        data_home=data_dir,
        resize=resize,
        return_X_y=True,
        color=False,
    )
    X = X.reshape((X.shape[0], -1))
    return X, targets


def load_dorothea(data_dir, dataset_cfg: DictConfig):
    X, targets = fetch_openml(
        "Dorothea", parser="auto", data_home=data_dir, return_X_y=True
    )
    return X, targets


def load_olivetti_faces(data_dir, dataset_cfg: DictConfig):
    X, targets = fetch_olivetti_faces(
        data_home=data_dir, return_X_y=True, shuffle=False
    )
    return X, targets
