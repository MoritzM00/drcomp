from omegaconf import DictConfig
from sklearn.datasets import make_swiss_roll
from torchvision import datasets, transforms


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
        return load_swiss_roll(n_samples=dataset.n_samples, noise=dataset.noise).astype(
            "float32"
        )
    else:
        raise ValueError("Unknown dataset given.")


def load_mnist(cfg: DictConfig, flatten: bool = False):
    mnist_train = datasets.MNIST(
        root=cfg.data_dir, download=True, transform=transforms.ToTensor()
    )
    X_train = mnist_train.data.numpy().astype("float32")
    if flatten:
        X_train = X_train.reshape((X_train.shape[0], -1))
    else:
        X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
    return X_train


def load_cifar10(cfg: DictConfig, flatten: bool = False):
    cifar_train = datasets.CIFAR10(
        root=cfg.data_dir, download=True, transform=transforms.ToTensor()
    )
    X_train = cifar_train.data.astype("float32")
    if flatten:
        X_train = X_train.reshape((X_train.shape[0], -1))
    else:
        X_train = X_train.transpose((0, 3, 1, 2))
    return X_train


def load_swiss_roll(n_samples: int = 3000, noise: float = 0.0):
    X, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
    return X
