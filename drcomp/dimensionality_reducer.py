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
    """Base class for Dimensionality Reducers."""

    @abstractmethod
    def transform(self, X) -> np.ndarray:
        """Reduce the dimensionality of the data."""

    def evaluate(self, X, K: int = 5) -> dict:
        """Evaluate the quality of the Dimensionality Reduction."""
        t = trustworthiness(X, self.transform(X), n_neighbors=K)
        return {
            "trustworthiness": t,
        }
