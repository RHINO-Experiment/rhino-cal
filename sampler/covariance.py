import numpy as np
from .reader import ObservationReader
from numpy.typing import NDArray
from scipy.sparse.linalg import cg
from scipy.linalg import solve

def params_space_oper_and_data(
    d: NDArray[np.floating],
    U: NDArray[np.floating],
    p: NDArray[np.floating],
    N_inv: NDArray[np.floating],
    mu: float | NDArray[np.floating] = 0.0,
    Ninv_sqrt: NDArray[np.floating] | None = None,
) -> (
    tuple[NDArray[np.floating], NDArray[np.floating]]
    | tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
):
    """
    Project the data model into parameter space for a heteroskedastic GLS.

    Given the data model ``d = (U p + mu)(1 + n)`` where ``n`` has covariance
    ``N``, this function constructs the parameter-space normal equations:

    * ``A = U^T Sigma_inv U``
    * ``b = U^T Sigma_inv (d - mu)``

    where ``Sigma_inv = diag(1/(Up+mu)) N_inv diag(1/(Up+mu))``.

    Parameters
    ----------
    d : NDArray[np.floating]
        Observed data vector of shape ``(M,)``.
    U : NDArray[np.floating]
        Projection (design) matrix of shape ``(M, N)``.
    p : NDArray[np.floating]
        Current parameter estimate of shape ``(N,)``.
    N_inv : NDArray[np.floating]
        Inverse noise covariance, shape ``(M, M)``.
    mu : float or NDArray[np.floating], optional
        Offset / mean term. Default is 0.0.
    Ninv_sqrt : NDArray[np.floating] or None, optional
        Square root of the inverse noise covariance (lower Cholesky factor),
        shape ``(M, M)``.  If provided, the function also returns
        ``U^T Sigma_inv_sqrt`` for sampling.

    Returns
    -------
    UTSigmaU : NDArray[np.floating]
        Parameter-space operator ``U^T Sigma_inv U``, shape ``(N, N)``.
    UTSigmaD : NDArray[np.floating]
        Parameter-space data vector ``U^T Sigma_inv (d - mu)``, shape ``(N,)``.
    UTSigma_sqrt : NDArray[np.floating], optional
        Returned only when ``Ninv_sqrt`` is not None.  Product
        ``U^T Sigma_inv_sqrt``, shape ``(N, M)``.

    References
    ----------
    Zhang et al. (2026), RASTI, rzag024.
    """
    D_p_inv = 1.0 / (U @ p + mu)
    sigma_inv = N_inv * np.outer(D_p_inv, D_p_inv)
    aux = U.T @ sigma_inv
    if Ninv_sqrt is None:
        return aux @ U, aux @ (d - mu)
    else:
        sigma_inv_sqrt = D_p_inv[:, np.newaxis] * Ninv_sqrt
        return aux @ U, aux @ (d - mu), U.T @ sigma_inv_sqrt

def estimated_inverse_covariance(N: np.ndarray,
                        d_prime: np.ndarray,
                        U: np.ndarray,
                        mu: np.ndarray,):
    """ 
    Returns the esitmated covariance on d given the operator U and vector mu

    d'' = Up + mu + delta

    d '' = d / T_sys

    sigma = <delta delta'>

    d = GT_sys(1 + n)

    d'' = d / T_sys = U_g @ p_g

    From Zhang et al. (2026), RASTI, rzag024.
    
    """
    # Starting Point
    p = np.linalg.lstsq(U, d_prime - mu)[0]

    # p_new = solve(U, d_prime - mu,)[0]
    # p = p_new
    # N is diagonal in this case.
    D_p = (U @ p + mu)
    Sigma = D_p * N * D_p

    return 1 / Sigma


def apply_mask(inverse_covariance: np.ndarray,
                data_flags: np.ndarray):
    """
    data_flags is set up so flagged data is 0 and unflagged is 1.
    """
    return inverse_covariance * data_flags
    
