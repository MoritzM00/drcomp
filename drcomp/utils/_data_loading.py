from omegaconf import DictConfig
from torchvision import datasets, transforms


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
