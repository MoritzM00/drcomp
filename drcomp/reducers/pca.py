"""PCA implementation of the DimensionalityReducer interface."""
from sklearn.decomposition import PCA as _PCA

from drcomp import DimensionalityReducer


class PCA(DimensionalityReducer):
    """Principal Component Analysis."""

    def __init__(self, n_components=2):
        """Initialize PCA."""
        super().__init__()
        self.n_components = n_components
        self.pca = _PCA(n_components=n_components)

    def fit(self, X, y=None):
        """Fit the model."""
        self.pca.fit(X, y)
        return self

    def transform(self, X):
        """Reduce the dimensionality of the data."""
        return self.pca.transform(X)
