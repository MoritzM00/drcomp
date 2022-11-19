"""Base class for dimensionality reduction algorithms."""

from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
        # TODO: Search for metrics to evaluate the quality of the dimensionality reduction
        return {"random_number": np.random.random()}
