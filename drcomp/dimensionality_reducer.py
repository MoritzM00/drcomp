"""Base class for dimensionality reduction algorithms."""

from abc import ABCMeta, abstractmethod

# from pyDRMetrics.pyDRMetrics import DRMetrics
from skdim.id import MLE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


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
        # drm = DRMetrics(X, self.transform(X))
        # return {"trustworthiness": drm.T, "continuity": drm.C, "LCMC": drm.LCMC}

    def _estimate_intrinsic_dimension(self, X) -> int:
        """Estimate the intrinsic dimensionality of the data."""
        return MLE().fit_transform(X)
