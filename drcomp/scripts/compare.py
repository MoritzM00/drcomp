import logging

import click
import hydra
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from omegaconf import DictConfig

from drcomp.plotting import compare_metrics, save_fig
from drcomp.utils._pathing import get_figures_dir, get_metrics_dir
from drcomp.utils.notebooks import load_all_metrics_for

logger = logging.getLogger(__name__)


@click.command()
@click.argument("datasets", nargs=-1)
@click.option("--save", default=False, is_flag=True, help="Save the plot to a file.")
@click.option(
    "--latex",
    default=False,
    is_flag=True,
    help="Save the plot in for use in LaTeX (pgf format).",
)
@click.option("--root_dir", default=".", help="The root directory for the project.")
def compare(datasets: list[str], save: bool, latex: bool, root_dir: str):
    """Compare the metrics of different dimensionality reduction methods."""
    cfg: DictConfig
    with hydra.initialize_config_module(
        version_base="1.3", config_module="drcomp.conf"
    ):
        cfg = hydra.compose(
            config_name="config.yaml",
            overrides=[f"root_dir={root_dir}"],
        )
    for dataset in datasets:
        if dataset not in cfg.available_datasets:
            raise ValueError(f"Dataset {dataset} not available.")
    for dataset in datasets:
        metrics = load_all_metrics_for(
            dataset, throw_on_missing=False, dir=get_metrics_dir(cfg)
        )
        plt.style.use("science")
        fig, _ = compare_metrics(metrics)
        if save:
            logger.info(f"Saving figure for {dataset}.")
            save_fig(
                dir=get_figures_dir(cfg),
                fig=fig,
                name=f"{dataset}_comparison",
                latex=latex,
            )
        else:
            plt.show()


if __name__ == "__main__":
    compare()
