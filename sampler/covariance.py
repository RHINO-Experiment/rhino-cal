import numpy as np
from reader import ObservationReader


def gls_vector_estimate(U:np.ndarray,
                        d:np.ndarray,
                        mu:np.ndarray,):
    p_0 = np.linalg.lstsq(U, d - mu)[0]
    # For now just go with the LS solution.
    return p_0

def estimated_inverse_covariance(N: np.ndarray,
                        d_prime: np.ndarray,
                        U: np.ndarray,
                        mu: np.ndarray):
    """ 
    Returns the esitmated covariance on d given the operator U and vector mu

    d'' = Up + mu + delta

    d '' = d / T_sys

    sigma = <delta delta'>

    d = GT_sys(1 + n)

    d'' = d / T_sys = U_g @ p_g

    From Zhang et al. (2026), RASTI, rzag024.
    
    """

    p_0 = gls_vector_estimate(U, d_prime, mu)

    # N is diagonal in this case.
    Up_0mu = U@p_0 + mu

    sigma = Up_0mu * N * Up_0mu # assumes all the same length

    return 1 / sigma


def apply_mask(inverse_covariance: np.ndarray,
                data_flags: np.ndarray):
    """
    data_flags is set up so flagged data is 0 and unflagged is 1.
    """
    return inverse_covariance * data_flags
    
