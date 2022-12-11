"""Base class for dimensionality reduction algorithms."""

from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from skdim.id import MLE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import TSNE

from drcomp.metrics import compute_coranking_matrix


def estimate_intrinsic_dimension(X, K: int = 5) -> int:
    """Estimate the intrinsic dimensionality of the data."""
    dimension = MLE(K=K).fit_transform(X)
    return int(dimension)


class DimensionalityReducer(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """Base class for Dimensionality Reducers.

    It specifies the interface that all dimensionality reducers should implement.
    """

    def __init__(
        self, intrinsic_dim: int = 2, supports_inverse_transform: bool = False, **kwargs
    ) -> None:
        super().__init__()
        self.intrinsic_dim = intrinsic_dim
        self.supports_inverse_transform = supports_inverse_transform

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the dimensionality reducer with X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : ignored
            Sklearn API compatibility.
        Returns
        -------
        self : object
            Returns the instance itself.
        """

    @abstractmethod
    def transform(self, X) -> np.ndarray:
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The samples, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Projection of X in the first principal components, where `n_samples`
            is the number of samples and `n_components` is the number of the components.
        """

    def evaluate(self, X, Y=None, K: int = 5, n_jobs=None) -> dict:
        """Evaluate the quality of the Dimensionality Reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        Y : array-like of shape (n_samples, intrinsic_dim), optional
            The transformed samples in the latent space. If not provided, it will be computed with the `transform` method.
        K : int, default=5
            Number of nearest neighbors to consider for the evaluation measures.

        Returns
        -------
        dict
            A dictionary containing the evaluation measures.
        """
        if Y is None:
            Y = self.transform(X)
        Q = compute_coranking_matrix(Y, X, n_jobs=n_jobs)
        # t = trustworthiness(X, Y, n_neighbors=K)
        return {
            # "trustworthiness": t,
            "coranking": Q,
        }

    def inverse_transform(self, Y) -> np.ndarray:
        """Transform data back to its original space, if it is supported by this dimensionality reducer.

        In other words, return an input `X_original` whose transform would be X.

        Parameters
        ----------
        Y : array-like of shape (n_samples, intrinsic_dim)
            Transformed samples (i.e. the data in the latent space).

        Returns
        -------
        X_original : array-like of shape (n_samples, n_features)
            Original data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        """
        raise ValueError("Inverse transform is not supported for this reducer.")

    def reconstruct(self, X) -> np.ndarray:
        """Reconstruct the original data, if it is supported by this dimensionality reducer.

        Convenience method that chains calls to `inverse_transform` and `transform`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        X_hat : array-like of shape (n_samples, n_features)
            Reconstructed samples.
        """
        if not self.supports_inverse_transform:
            raise ValueError("Inverse transform is not supported for this reducer.")
        return self.inverse_transform(self.transform(X))

    def visualize_2D_latent_space(self, X, y=None):
        """Visualize the 2D latent space of the data.

        If the intrinsic dimensionality is not 2, t-SNE will be used to project it to two dimensions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        y : array-like of shape (n_samples,), optional
            Labels for the samples. Will be used to color the scatter plot.
        """
        Y = self.transform(X)
        if self.intrinsic_dim > 2:
            Y = TSNE(n_components=2).fit_transform(Y)
        elif self.intrinsic_dim < 2:
            raise ValueError(
                "Cannot visualize a latent space with less than 2 dimensions."
            )
        return plt.scatter(Y[:, 0], Y[:, 1], c=y)
