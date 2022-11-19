"""Kernel PCA implementation of the DimensionalityReducer interface."""
from sklearn.decomposition import KernelPCA as _KernelPCA

from drcomp import DimensionalityReducer


class KernelPCA(DimensionalityReducer):
    """Kernel Principal Component Analysis."""

    def __init__(self, n_components: int = 2, kernel: str = "linear"):
        """Initialize PCA."""
        super().__init__()
        self.n_components = n_components
        self.kernel_pca = _KernelPCA(n_components=n_components, kernel=kernel)

    def fit(self, X, y=None):
        """Fit the model."""
        self.kernel_pca.fit(X, y)
        return self

    def transform(self, X):
        """Reduce the dimensionality of the data."""
        return self.kernel_pca.transform(X)
