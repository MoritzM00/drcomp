import numpy as np
from sklearn.manifold import LocallyLinearEmbedding

from drcomp import DimensionalityReducer


class LLE(DimensionalityReducer):
    """Locally Linear Embedding.

    Parameters
    ----------
    intrinsic_dim : int, default=2
        Number of coordinates for the manifold.

    n_neighbors : int, default=5
        Number of neighbors to consider for each point.

    reg : float, default=1e-3
        Regularization constant, multiplies the trace of the local covariance
        matrix of the distances.

    eigen_solver : {'auto', 'arpack', 'dense'}, default='auto'
        The solver used to compute the eigenvectors. The available options are:

        - `'auto'` : algorithm will attempt to choose the best method for input
          data.
        - `'arpack'` : use arnoldi iteration in shift-invert mode. For this
          method, M may be a dense matrix, sparse matrix, or general linear
          operator.
        - `'dense'`  : use standard dense matrix operations for the eigenvalue
          decomposition. For this method, M must be an array or matrix type.
          This method should be avoided for large problems.

        .. warning::
           ARPACK can be unstable for some problems.  It is best to try several
           random seeds in order to check results.

    tol : float, default=1e-6
        Tolerance for 'arpack' method
        Not used if eigen_solver=='dense'.

    max_iter : int, default=100
        Maximum number of iterations for the arpack solver.
        Not used if eigen_solver=='dense'.

    hessian_tol : float, default=1e-4
        Tolerance for Hessian eigenmapping method.
        Only used if ``method == 'hessian'``.

    modified_tol : float, default=1e-12
        Tolerance for modified LLE method.
        Only used if ``method == 'modified'``.

    neighbors_algorithm : {'auto', 'brute', 'kd_tree', 'ball_tree'}, \
                          default='auto'
        Algorithm to use for nearest neighbors search, passed to
        :class:`~sklearn.neighbors.NearestNeighbors` instance.

    random_state : int, RandomState instance, default=None
        Determines the random number generator when
        ``eigen_solver`` == 'arpack'. Pass an int for reproducible results
        across multiple function calls.

    n_jobs : int or None, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    """

    def __init__(
        self,
        intrinsic_dim=2,
        n_neighbors=5,
        reg=1e-3,
        eigen_solver="auto",
        tol=1e-6,
        max_iter=100,
        hessian_tol=1e-4,
        modified_tol=1e-12,
        neighbors_algorithm="auto",
        random_state=None,
        n_jobs=None,
        **kwargs
    ) -> None:
        super().__init__(intrinsic_dim, supports_inverse_transform=False)
        self._lle = LocallyLinearEmbedding(
            n_components=intrinsic_dim,
            n_neighbors=n_neighbors,
            method="standard",
            reg=reg,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            hessian_tol=hessian_tol,
            modified_tol=modified_tol,
            neighbors_algorithm=neighbors_algorithm,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X, y=None):
        self._lle.fit(X)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self._lle.fit_transform(X)

    def transform(self, X) -> np.ndarray:
        return self._lle.transform(X)
