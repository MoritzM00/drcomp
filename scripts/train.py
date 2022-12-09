"""Script to train and evaluate a model on a dataset."""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    train()
