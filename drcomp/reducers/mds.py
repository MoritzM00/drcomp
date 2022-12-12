import numpy as np
from sklearn.manifold import MDS as _MDS

from drcomp import DimensionalityReducer


class MDS(DimensionalityReducer):
    """Multidimensional scaling (MDS).

    Parameters
    ----------
    intrinsic_dim : int, default=2
        Number of dimensions in which to immerse the dissimilarities.
    metric : bool, default=True
        If ``True``, perform metric MDS; otherwise, perform nonmetric MDS.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.
    n_init : int, default=4
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress.
    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.
    verbose : int, default=0
        Level of verbosity.
    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence. The value of `eps` should be tuned separately depending
        on whether or not `normalized_stress` is being used.
    n_jobs : int, default=None
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
    dissimilarity : {'euclidean', 'precomputed'}, default='euclidean'
        Dissimilarity measure to use:
        - 'euclidean':
            Pairwise Euclidean distances between points in the dataset.
        - 'precomputed':
            Pre-computed dissimilarities are passed directly to ``fit`` and
            ``fit_transform``.
    normalized_stress : bool or "auto" default=False
        Whether use and return normed stress value (Stress-1) instead of raw
        stress calculated by default. Only supported in non-metric MDS.
    """

    def __init__(
        self,
        intrinsic_dim=2,
        metric=True,
        n_init=4,
        max_iter=300,
        verbose=0,
        eps=1e-3,
        n_jobs=None,
        random_state=None,
        dissimilarity="euclidean",
        normalized_stress=False,
        **kwargs
    ) -> None:
        super().__init__(intrinsic_dim, supports_inverse_transform=False)
        self._mds = _MDS(
            n_components=intrinsic_dim,
            metric=metric,
            n_init=n_init,
            max_iter=max_iter,
            verbose=verbose,
            eps=eps,
            n_jobs=n_jobs,
            random_state=random_state,
            dissimilarity=dissimilarity,
            normalized_stress=normalized_stress,
        )

    def fit(self, X, y=None):
        self._mds.fit(X)
        return self

    def transform(self, X) -> np.ndarray:
        return self._mds.transform(X)
