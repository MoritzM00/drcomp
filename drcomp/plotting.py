import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import IndexLocator

from drcomp import DimensionalityReducer, MetricsDict


def save_fig(
    dir,
    fig: mpl.figure.Figure,
    name: str,
    latex: bool = True,
    width=5.91,
    height=4.8,
    **kwargs,
) -> None:
    """Save a figure to a file in the given directory.

    Parameters
    ----------
    dir : str
        Directory where the figure will be saved.
    fig : matplotlib.figure.Figure
        The Figure to save.
    name : str
        Name of the file.
    latex : bool, default=True
        Whether to save the figure in a format suitable for LaTeX. If True, then use the `pgf` backend.
    width : float, default=5.91
        Width of the figure in inches.
    height : float, default=4.8
        Height of the figure in inches.
    **kwargs
        Additional keyword arguments to pass to `matplotlib.figure.Figure.savefig`.
    """
    format = "png"
    backend = None
    if latex:
        format = "pgf"
        backend = "pgf"
        plt.style.use("science")
        fig.set_size_inches(w=width, h=height)
        fig.tight_layout()
    base = pathlib.Path(dir)
    base.mkdir(parents=True, exist_ok=True)
    path = pathlib.Path(base, f"{name}.{format}")
    fig.savefig(path, format=format, backend=backend, **kwargs)


def plot_metric(metric, label: str, ylabel: str, ax=None) -> mpl.axes.Axes:
    """Plot a metric as a function of the number of neighbors.

    Parameters
    ----------
    metric : array-like of shape (n_neighbors,)
        Values for the metric.
    label : str
        The label for the metric, e.g. the name of the dimensionality reduction method.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, then create a new axes.

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the metric was plotted.
    """
    if ax is None:
        ax = plt.axes()
    k = len(metric)
    x = np.arange(1, k + 1)
    ax.plot(x, metric, label=label)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("$K$")
    ax.set_xlim(0, k + 2)
    ax.xaxis.set_major_locator(IndexLocator(20, offset=-1))
    return ax


def compare_metrics(metrics: dict[str, MetricsDict], figsize=(8, 8)):
    """Compare the metrics of different dimensionality reduction methods.

    Parameters
    ----------
    metrics: dict
        A dictionary mapping the name of a dimensionality reduction method to a dictionary containing the metrics.
    """
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(122)
    for name, metric in metrics.items():
        plot_metric(metric["trustworthiness"], label=name, ylabel="$T(K)$", ax=ax1)
        ax1.set_title("Trustworthiness")
        plot_metric(metric["continuity"], label=name, ylabel="$C(K)$", ax=ax2)
        ax2.set_title("Continuity")
        plot_metric(metric["lcmc"], label=name, ylabel="$LCMC(K)$", ax=ax3)
        ax3.set_title("LCMC")
    plt.legend(metrics.keys())
    plt.tight_layout()
    return fig, [ax1, ax2, ax3]


def plot_trustworthiness_continuity(t, c, ax=None):
    if len(t) != len(c):
        raise ValueError(
            "Trustworthiness and Continuity arrays must have the same length."
        )
    if ax is None:
        ax = plt.axes()
    x = np.arange(1, len(t) + 1)
    ax.plot(x, t, label="Trustworthiness")
    ax.plot(x, c, label="Continuity")
    ax.set_xlabel("Number of neighbors")
    ax.set_xlim(0, len(t) + 2)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(IndexLocator(10, offset=-1))
    ax.legend()
    return ax


def plot_lcmc(lcmc, ax=None):
    if ax is None:
        ax = plt.axes()
    x = np.arange(1, len(lcmc) + 1)
    ax.plot(x, lcmc)
    ax.set_xlabel("Number of neighbors")
    ax.set_ylabel("Local Continuity Meta Criterion")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(lcmc) + 2)
    ax.xaxis.set_major_locator(IndexLocator(10, offset=-1))
    return ax


def plot_reconstructions(
    models: dict[str, DimensionalityReducer],
    images,
    preprocessor,
    width: int,
    height: int,
    channels: int,
    cmap="gray",
    figsize=None,
):
    """Plot the reconstructions of the samples by the given models compared to the original images."""
    n_images = len(images)
    if figsize is None:
        figsize = (n_images, len(models) + 1)
    fig, axs = plt.subplots(len(models) + 1, n_images, figsize=figsize)
    flattened_size = width * height * channels
    assert np.shape(images) == (n_images, flattened_size)
    ground_truth = images.reshape(
        -1, height, width, channels
    )  # matplotlib expects channels last
    processed_images = preprocessor.transform(images)
    reconstructions = []
    for i, model in enumerate(models.values()):
        try:
            X_hat = model.reconstruct(processed_images)
        except ValueError:
            # Convolutional autoencoder expects images in channels first format
            X_hat = model.reconstruct(
                processed_images.reshape(-1, channels, height, width)
            )
            X_hat = X_hat.reshape(-1, flattened_size)
        X_hat = preprocessor.inverse_transform(X_hat)
        X_hat = X_hat.reshape(-1, height, width, channels)
        reconstructions.append(X_hat)
    for i in range(n_images):
        axs[0, i].imshow(ground_truth[i], cmap=cmap)
        axs[0, i].axis("off")
        for j in range(len(models)):
            axs[j + 1, i].imshow(reconstructions[j][i], cmap=cmap)
            axs[j + 1, i].axis("off")
    plt.tight_layout()
    return fig, axs


def visualize_2D_latent_space(
    reducer: DimensionalityReducer,
    X,
    title: str,
    y=None,
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    ax=None,
):
    """Visualize the 2D latent space of the data. The reducer must be fitted with an intrinsic dimension of less than two.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Samples.
    y : array-like of shape (n_samples,), optional
        Labels for the samples. Will be used to color the scatter plot.
    """
    if ax is None:
        ax = plt.axes()
    Y = reducer.transform(X)
    if reducer.intrinsic_dim == 1:
        ax.plot(Y, c=y)
    if reducer.intrinsic_dim == 2:
        ax.scatter(Y[:, 0], Y[:, 1], c=y)
    elif reducer.intrinsic_dim < 2:
        raise ValueError(
            "Cannot visualize a latent space with more than two dimensions."
        )
    ax.set_title(title)
    return ax
