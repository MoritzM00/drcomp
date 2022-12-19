"""Base class for dimensionality reduction algorithms."""

import logging
from abc import ABCMeta, abstractmethod

import coranking
import numpy as np
from coranking.metrics import LCMC, continuity, trustworthiness
from skdim.id import MLE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample

logger = logging.getLogger(__name__)

MAX_CORANKING_DIMENSION: int = 5000


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

    def evaluate(self, X, max_K: int = None, as_builtin_list=False) -> dict:
        """Evaluate the quality of the Dimensionality Reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        Y : array-like of shape (n_samples, intrinsic_dim), optional
            The transformed samples in the latent space. If not provided, it will be computed with the `transform` method.
        max_K : int, default=5
            The maximum Number of nearest neighbors to consider for the evaluation measures.

        Returns
        -------
        dict
            A dictionary containing the evaluation measures. Arrays are numpy arrays if as_builtin_list is False, otherwise builtin lists.
        """
        n_samples = X.shape[0]
        if n_samples > MAX_CORANKING_DIMENSION:
            logging.info(
                f"Computing trustworthiness on a random subsample ({MAX_CORANKING_DIMENSION}) because the dataset is too large."
            )
            X = resample(X, n_samples=MAX_CORANKING_DIMENSION)
        Y = self.transform(X)
        Q = coranking.coranking_matrix(X, Y)
        t = trustworthiness(Q, min_k=1, max_k=max_K)
        c = continuity(Q, min_k=1, max_k=max_K)
        lcmc = LCMC(Q, min_k=1, max_k=max_K)
        if as_builtin_list:
            t = t.tolist()
            c = c.tolist()
            lcmc = lcmc.tolist()
        return {
            "trustworthiness": t,
            "continuity": c,
            "lcmc": lcmc,
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
