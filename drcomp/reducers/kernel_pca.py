"""Kernel PCA implementation of the DimensionalityReducer interface."""
import numpy as np
from sklearn.decomposition import KernelPCA as _KernelPCA

from drcomp import DimensionalityReducer


class KernelPCA(DimensionalityReducer):
    """Kernel Principal Component Analysis."""

    def __init__(
        self,
        intrinsic_dim: int = 2,
        kernel: str = "linear",
        fit_inverse_transform: bool = False,
    ):
        """Initialize PCA."""
        super().__init__()
        self.supports_inverse_transform = fit_inverse_transform
        self.intrinsic_dim = intrinsic_dim
        self.kernel_pca = _KernelPCA(
            intrinsic_dim=intrinsic_dim,
            kernel=kernel,
            fit_inverse_transform=fit_inverse_transform,
        )

    def fit(self, X, y=None):
        """Fit the model."""
        self.kernel_pca.fit(X, y)
        return self

    def transform(self, X) -> np.ndarray:
        return self.kernel_pca.transform(X)

    def inverse_transform(self, X) -> np.ndarray:
        return self.kernel_pca.inverse_transform(X)
