"""
Set up GCR and gibbs sample
"""

import numpy as np




def construct_gcr_lhs(linear_operator:np.ndarray,
                      inv_data_cov:np.ndarray,
                      inv_prior_cov:np.ndarray|None=None):
    """
    Construct the left hand side of the GCR equation
    """

    if inv_prior_cov is not None:
        if inv_prior_cov.shape == inv_data_cov.shape: # Prior is data shaped
            return (linear_operator.T @ (inv_prior_cov[:,None] * linear_operator)) +\
                (linear_operator.T @ (inv_data_cov[:,None] * linear_operator))
        else:
            return linear_operator.T @ (inv_data_cov[:,None] * linear_operator) + inv_prior_cov
    else:
        return linear_operator.T @ (inv_data_cov[:,None] * linear_operator)



def construct_gcr_rhs(linear_operator,
                      data,
                      inv_data_cov,
                      omega_data: np.ndarray,
                      priors: np.ndarray | None=None,
                      inv_prior_cov: np.ndarray | None=None,
                      omega_prior: np.ndarray | None=None):

    data_term = linear_operator.T @ (inv_data_cov * data) + linear_operator.T @ (np.sqrt(inv_data_cov)*data)

    if priors is not None:

    if data.shape != priors.shape:
        prior_term = ()

        #FIXME Continue here...