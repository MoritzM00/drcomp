"""PCA implementation of the DimensionalityReducer interface."""
from sklearn.decomposition import PCA as _PCA

from drcomp import DimensionalityReducer


class PCA(DimensionalityReducer):
    """Principal Component Analysis."""

    def __init__(self, intrinsic_dim=2, **kwargs):
        super().__init__(intrinsic_dim=intrinsic_dim, supports_inverse_transform=True)
        self.pca = _PCA(n_components=intrinsic_dim)

    def fit(self, X, y=None, **kwargs):
        self.pca.fit(X, y)
        return self

    def transform(self, X):
        return self.pca.transform(X)

    def inverse_transform(self, Y):
        return self.pca.inverse_transform(Y)
