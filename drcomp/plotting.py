import matplotlib.pyplot as plt
import umap

from drcomp import DimensionalityReducer


def plot_trustworthiness_continuity():
    pass


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
