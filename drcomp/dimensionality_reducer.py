"""Base class for dimensionality reduction algorithms."""

from abc import ABCMeta, abstractmethod

import pyDRMetrics.pyDRMetrics as metrics
from skdim.id import MLE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def estimate_intrinsic_dimension(X) -> int:
    """Estimate the intrinsic dimensionality of the data."""
    dimension = MLE().fit_transform(X)
    return int(dimension)


class DimensionalityReducer(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """Base class for Dimensionality Reducers."""

    def __init__(self):
        """Initialize the DimensionalityReducer."""
        super().__init__()

    @abstractmethod
    def fit(self, X, y):
        """Fit the DimensionalityReducer to the data."""

    @abstractmethod
    def transform(self, X):
        """Reduce the dimensionality of the data."""

    def evaluate(self, X) -> dict:
        """Evaluate the quality of the Dimensionality Reduction."""
        check_is_fitted(self)
        drm = metrics.DRMetrics(X, self.transform(X))
        return {"trustworthiness": drm.T, "continuity": drm.C, "LCMC": drm.LCMC}
