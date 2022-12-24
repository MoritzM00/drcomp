import logging

import click
import hydra
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from omegaconf import DictConfig
from sklearn.utils import resample

from drcomp import DimensionalityReducer
from drcomp.plotting import plot_reconstructions, save_fig
from drcomp.utils._pathing import get_figures_dir
from drcomp.utils.notebooks import get_dataset, get_model_for_dataset, get_preprocessor

logger = logging.getLogger(__name__)


@click.command()
@click.argument("dataset", default="SwissRoll")
@click.argument("reducers", nargs=-1)
@click.option("--n_images", default=5)
@click.option("--save", default=False, is_flag=True, help="Save the plot to a file.")
@click.option(
    "--latex",
    default=False,
    is_flag=True,
    help="Save the plot in for use in LaTeX (pgf format).",
)
@click.option("--root_dir", default=".", help="The root directory for the project.")
def reconstruct(
    dataset: str, reducers: list[str], n_images, save: bool, latex: bool, root_dir: str
):
    """Compare the metrics of different dimensionality reduction methods."""
    cfg: DictConfig
    with hydra.initialize_config_module(
        version_base="1.3", config_module="drcomp.conf"
    ):
        cfg = hydra.compose(
            config_name="config.yaml",
            overrides=[f"root_dir={root_dir}", f"dataset={dataset}"],
        )

    if dataset not in cfg.available_datasets:
        raise ValueError(f"Invalid Dataset {dataset} given.")
    try:
        cfg.dataset.image_size
    except Exception:
        raise ValueError(f"Dataset {dataset} does not contain images.")
    models: dict[str, DimensionalityReducer] = {}
    for reducer in reducers:
        if reducer not in cfg.available_reducers:
            raise ValueError(f"Reducer {reducer} not available.")
        models[reducer] = get_model_for_dataset(
            dataset, reducer, root_dir=root_dir, from_pretrained=True
        )
        if not models[reducer].supports_inverse_transform:
            raise ValueError(f"Reducer {reducer} does not support reconstructions.")
    X, _ = get_dataset(dataset, root_dir=root_dir)
    preprocessor = get_preprocessor(dataset, root_dir=root_dir, from_pretrained=True)

    plt.style.use("science")
    images = resample(X, n_samples=n_images)
    channels, width, height = cfg.dataset.image_size
    fig, _ = plot_reconstructions(
        models, images, preprocessor, channels=channels, width=width, height=height
    )
    if save:
        logger.info(f"Saving figure for {dataset}.")
        save_fig(
            dir=get_figures_dir(cfg),
            fig=fig,
            name=f"{dataset}_reconstructions",
            latex=latex,
        )
    else:
        plt.show()


if __name__ == "__main__":
    reconstruct()
