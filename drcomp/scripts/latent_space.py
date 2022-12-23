import logging
import pathlib

import click
import hydra
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from omegaconf import DictConfig
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from drcomp.plotting import save_fig, visualize_2D_latent_space
from drcomp.utils._pathing import get_figures_dir
from drcomp.utils.notebooks import get_dataset, get_model_for_dataset

logger = logging.getLogger(__name__)


@click.command()
@click.argument("dataset", default="SwissRoll")
@click.argument("reducers", nargs=-1)
@click.option("--save", default=False, is_flag=True, help="Save the plot to a file.")
@click.option(
    "--latex",
    default=False,
    is_flag=True,
    help="Save the plot in for use in LaTeX (pgf format).",
)
@click.option("--root_dir", default=".", help="The root directory for the project.")
def visualize_latent_space(
    dataset: str, reducers: list[str], save: bool, latex: bool, root_dir: str
):
    """Visualize the latent space of different dimensionality reduction methods for a dataset."""
    cfg: DictConfig
    with hydra.initialize_config_module(
        version_base="1.3", config_module="drcomp.conf"
    ):
        cfg = hydra.compose(
            config_name="config.yaml",
            overrides=[f"root_dir={root_dir}"],
        )
    if dataset not in cfg.available_datasets:
        raise ValueError(
            f"Dataset {dataset} invalid. Must be one of {cfg.available_datasets}."
        )
    # load the data
    X, y = get_dataset(dataset, root_dir=root_dir)

    # load the reducers
    models = {}
    for reducer in reducers:
        if reducer not in cfg.available_reducers:
            raise ValueError(
                f"Reducer {reducer} invalid. Must be one of {cfg.available_reducers}."
            )
        logger.info(f"Loading reducer {reducer} for dataset {dataset}.")
        models[reducer] = get_model_for_dataset(
            dataset, reducer, root_dir=root_dir, from_pretrained=True
        )
        try:
            check_is_fitted(models[reducer])
        except NotFittedError:
            logging.error(
                f"Cannot visualize latent space of unfitted reducer {reducer}."
            )
            continue
    if not models:
        raise ValueError("No valid reducers were loaded.")
    fig, axs = plt.subplots(1, len(models), figsize=(10, 5))
    if len(models) == 1:
        axs = [axs]
    for i, (name, model) in enumerate(models.items()):
        visualize_2D_latent_space(model, X=X, y=y, ax=axs[i])
        axs[i].set_title(name)
    if save:
        logger.info(f"Saving figure for {dataset}.")
        dir = pathlib.Path(get_figures_dir(cfg), "latent_spaces")
        dir.mkdir(parents=True, exist_ok=True)
        save_fig(
            dir=dir,
            fig=fig,
            name=f"{dataset}_{name}",
            latex=latex,
        )
    else:
        plt.show()


if __name__ == "__main__":
    visualize_latent_space()
