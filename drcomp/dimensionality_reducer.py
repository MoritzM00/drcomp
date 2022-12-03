"""Base class for dimensionality reduction algorithms."""

from abc import ABCMeta, abstractmethod

import numpy as np
from skdim.id import MLE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import trustworthiness


def estimate_intrinsic_dimension(X, K: int = 5) -> int:
    """Estimate the intrinsic dimensionality of the data."""
    dimension = MLE(K=K).fit_transform(X)
    return int(dimension)


class DimensionalityReducer(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """Base class for Dimensionality Reducers.

    It specifies the interface that all dimensionality reducers should implement.
    """

    def __init__(self) -> None:
        super().__init__()
        self.supports_inverse_transform = False
        self.intrinsic_dim = None

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

    def evaluate(self, X, Y=None, K: int = 5) -> dict:
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
        t = trustworthiness(X, Y, n_neighbors=K)
        return {
            "trustworthiness": t,
        }

    def inverse_transform(self, X) -> np.ndarray:
        """Transform data back to its original space, if it is supported by this dimensionality reducer.

        In other words, return an input `X_original` whose transform would be X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Samples, where `n_samples` is the number of samples
            and `n_components` is the number of components.

        Returns
        -------
        X_original : array-like of shape (n_samples, n_features)
            Original data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        """
        raise ValueError("Inverse transform is not supported for this reducer.")
