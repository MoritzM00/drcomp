import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.ticker import IndexLocator

from drcomp import DimensionalityReducer


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


def visualize_2D_latent_space(
    reducer: DimensionalityReducer,
    X,
    y=None,
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    ax=None,
):
    """Visualize the 2D latent space of the data.

    If the intrinsic dimensionality is not 2, UMAP will be used to project it to two dimensions.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Samples.
    y : array-like of shape (n_samples,), optional
        Labels for the samples. Will be used to color the scatter plot.
    """
    Y = reducer.transform(X)
    if reducer.intrinsic_dim > 2:
        Y = umap.UMAP(
            n_components=2, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist
        ).fit_transform(Y)
    elif reducer.intrinsic_dim < 2:
        raise ValueError("Cannot visualize a latent space with less than 2 dimensions.")
    if ax is None:
        ax = plt.axes()
    ax.scatter(Y[:, 0], Y[:, 1], c=y)
    return ax
