"""Base class for dimensionality reduction algorithms."""

import logging
from abc import ABCMeta, abstractmethod

import numpy as np
from skdim.id import MLE
from sklearn.base import BaseEstimator, TransformerMixin

from drcomp.metrics import (
    MetricsDict,
    compute_coranking_matrix,
    compute_quality_criteria,
)

logger = logging.getLogger(__name__)


def estimate_intrinsic_dimension(X, K: int = 5) -> int:
    """Estimate the intrinsic dimensionality of the data."""
    dimension = MLE(K=K).fit_transform(X)
    return int(dimension)


class DimensionalityReducer(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """Base class for Dimensionality Reducers.

    It specifies the interface that all dimensionality reducers should implement.
    """

    def __init__(
        self,
        intrinsic_dim: int = 2,
        supports_inverse_transform: bool = False,
        n_jobs: int = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.intrinsic_dim = intrinsic_dim
        self.supports_inverse_transform = supports_inverse_transform
        self.n_jobs = n_jobs

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

    def evaluate(
        self, X, Y=None, max_K: int = None, as_builtin_list: bool = False
    ) -> MetricsDict:
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
        if max_K is not None:
            # max_K must be smaller than n_samples - 1
            max_K = min(max_K, n_samples - 1)
        if n_samples > 5000:
            logger.warning(
                f"Computing the evaluation measures for {n_samples} samples may take up a long time and RAM. Consider downsampling the data to less than 5000 samples."
            )
        Y = self.transform(X)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
            Y = Y.reshape(Y.shape[0], -1)
        logger.debug(f"Using {self.n_jobs} jobs to calculate the coranking matrix.")
        Q = compute_coranking_matrix(X, Y, n_jobs=self.n_jobs)
        metrics = compute_quality_criteria(Q, max_K=max_K)
        if as_builtin_list:
            metrics = {k: v.tolist() for k, v in metrics.items()}
        return metrics

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
