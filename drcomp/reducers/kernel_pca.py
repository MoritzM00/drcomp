"""Kernel PCA implementation of the DimensionalityReducer interface."""
import numpy as np
from sklearn.decomposition import KernelPCA as _KernelPCA

from drcomp import DimensionalityReducer


class KernelPCA(DimensionalityReducer):
    """Kernel Principal Component Analysis.


    Parameters
    ----------
    intrinsic_dim : int, default=2
        Intrinsically dimensionality of the data.

    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed'} \
            or callable, default='poly'
        Kernel used for PCA.

    gamma : float, default=None
        Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
        kernels. If ``gamma`` is ``None``, then it is set to ``1/n_features``.

    degree : int, default=3
        Degree for poly kernels. Ignored by other kernels.

    coef0 : float, default=1
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.

    kernel_params : dict, default=None
        Parameters (keyword arguments) and
        values for kernel passed as callable object.
        Ignored by other kernels.

    alpha : float, default=1.0
        Hyperparameter of the ridge regression that learns the
        inverse transform (when fit_inverse_transform=True).

    fit_inverse_transform : bool, default=False
        Learn the inverse transform for non-precomputed kernels
        (i.e. learn to find the pre-image of a point). This method is based
        on [2]_.

    eigen_solver : {'auto', 'dense', 'arpack', 'randomized'}, \
            default='auto'
        Select eigensolver to use. If `intrinsic_dim` is much
        less than the number of training samples, randomized (or arpack to a
        smaller extent) may be more efficient than the dense eigensolver.
        Randomized SVD is performed according to the method of Halko et al.

        auto :
            the solver is selected by a default policy based on n_samples
            (the number of training samples) and `intrinsic_dim`:
            if the number of components to extract is less than 10 (strict) and
            the number of samples is more than 200 (strict), the 'arpack'
            method is enabled. Otherwise the exact full eigenvalue
            decomposition is computed and optionally truncated afterwards
            ('dense' method).
        dense :
            run exact full eigenvalue decomposition calling the standard
            LAPACK solver via `scipy.linalg.eigh`, and select the components
            by postprocessing
        arpack :
            run SVD truncated to intrinsic_dim calling ARPACK solver using
            `scipy.sparse.linalg.eigsh`. It requires strictly
            0 < intrinsic_dim < n_samples
        randomized :
            run randomized SVD by the method of Halko et al. The current
            implementation selects eigenvalues based on their module; therefore
            using this method can lead to unexpected results if the kernel is
            not positive semi-definite.

    tol : float, default=0
        Convergence tolerance for arpack.
        If 0, optimal value will be chosen by arpack.

    max_iter : int, default=None
        Maximum number of iterations for arpack.
        If None, optimal value will be chosen by arpack.

    iterated_power : int >= 0, or 'auto', default='auto'
        Number of iterations for the power method computed by
        svd_solver == 'randomized'. When 'auto', it is set to 7 when
        `intrinsic_dim < 0.1 * min(X.shape)`, other it is set to 4.


    remove_zero_eig : bool, default=False
        If True, then all components with zero eigenvalues are removed, so
        that the number of components in the output may be < intrinsic_dim
        (and sometimes even zero due to numerical instability).
        When intrinsic_dim is None, this parameter is ignored and components
        with zero eigenvalues are removed regardless.

    random_state : int, RandomState instance or None, default=None
        Used when ``eigen_solver`` == 'arpack' or 'randomized'. Pass an int
        for reproducible results across multiple function calls.

    copy_X : bool, default=True
        If True, input X is copied and stored by the model in the `X_fit_`
        attribute. If no further changes will be done to X, setting
        `copy_X=False` saves memory by storing a reference.


    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

        """

    def __init__(
        self,
        intrinsic_dim: int = 2,
        kernel: str = "poly",
        fit_inverse_transform: bool = False,
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        remove_zero_eig=False,
        random_state=None,
        copy_X=True,
        n_jobs=None,
        **kwargs
    ):
        super().__init__(
            intrinsic_dim=intrinsic_dim,
            supports_inverse_transform=fit_inverse_transform,
        )
        self.kernel_pca = _KernelPCA(
            n_components=intrinsic_dim,
            kernel=kernel,
            fit_inverse_transform=fit_inverse_transform,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            alpha=alpha,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            iterated_power=iterated_power,
            remove_zero_eig=remove_zero_eig,
            random_state=random_state,
            copy_X=copy_X,
            n_jobs=n_jobs,
        )

    def fit(self, X, y=None):
        self.kernel_pca.fit(X, y)
        self.fitted_ = True
        return self

    def transform(self, X) -> np.ndarray:
        return self.kernel_pca.transform(X)

    def inverse_transform(self, Y) -> np.ndarray:
        return self.kernel_pca.inverse_transform(Y)
