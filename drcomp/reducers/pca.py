"""PCA implementation of the DimensionalityReducer interface."""
from sklearn.decomposition import PCA as _PCA
from sklearn.utils.validation import check_is_fitted

from drcomp import DimensionalityReducer


class PCA(DimensionalityReducer):
    """Principal Component Analysis.

    Parameters
    ----------
    intrinsic_dim : int,
        Number of dimensions to reduce to.

    copy : bool, default=True
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    whiten : bool, default=False
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.

    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        If auto :
            The solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        If full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        If arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < min(X.shape)
        If randomized :
            run randomized SVD by the method of Halko et al.


    tol : float, default=0.0
        Tolerance for singular values computed by svd_solver == 'arpack'.
        Must be of range [0.0, infinity).


    iterated_power : int or 'auto', default='auto'
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.
        Must be of range [0, infinity).


    n_oversamples : int, default=10
        This parameter is only relevant when `svd_solver="randomized"`.
        It corresponds to the additional number of random vectors to sample the
        range of `X` so as to ensure proper conditioning.

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Power iteration normalizer for randomized SVD solver.
        Not used by ARPACK.

    random_state : int, RandomState instance or None, default=None
        Used when the 'arpack' or 'randomized' solvers are used. Pass an int
        for reproducible results across multiple function calls.
    """

    def __init__(
        self,
        intrinsic_dim=2,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        n_oversamples=10,
        power_iteration_normalizer="auto",
        random_state=None,
        n_jobs=None,
        **kwargs
    ):
        super().__init__(
            intrinsic_dim=intrinsic_dim, supports_inverse_transform=True, n_jobs=n_jobs
        )
        self.pca = _PCA(
            n_components=intrinsic_dim,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            n_oversamples=n_oversamples,
            power_iteration_normalizer=power_iteration_normalizer,
            random_state=random_state,
        )

    def fit(self, X, y=None):
        self.pca.fit(X, y)
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.pca.transform(X)

    def inverse_transform(self, Y):
        check_is_fitted(self)
        return self.pca.inverse_transform(Y)
