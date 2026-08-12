"""
Set up GCR and gibbs sample
"""

import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import cg
from covariance import estimated_inverse_covariance
from pathlib import Path

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

def construct_gcr_rhs(linear_operator: np.ndarray,
                      data: np.ndarray,
                      inv_data_cov:np.ndarray,
                      omega_data: np.ndarray,
                      priors: np.ndarray | None=None,
                      inv_prior_cov: np.ndarray | None=None,
                      omega_prior: np.ndarray | None=None,
                      weiner_solution: bool = False):
    """ 
    """
    if weiner_solution:
        omega_data = np.zeros_like(data)

    data_term = (linear_operator.T @ (inv_data_cov * data)) +\
        (linear_operator.T @ (np.sqrt(inv_data_cov)*omega_data))

    if priors is not None:
        if weiner_solution:
            omega_prior = np.zeros_like(priors)
        if data.shape != priors.shape:
            prior_term = (inv_prior_cov*priors) + (np.sqrt(inv_prior_cov)*omega_prior)
        else:
            prior_term = (linear_operator.T @ (inv_prior_cov*priors)) +\
                (linear_operator.T @ (np.sqrt(inv_prior_cov) * omega_prior))
    else:
        prior_term = np.zeros_like(data_term)

    return data_term + prior_term


def solve_gcr(A,
              b,
              linalg_init: bool = False,
              rtol=1e-10,
              atol=1e-8,
              maxiter=int(1e6)):
    """
    """
    if linalg_init:
        x0 = np.linalg.lstsq(A,b)
        return cg(A,b,x0=x0, maxiter=maxiter, rtol=rtol, atol=atol)[0]
    else:
        return cg(A,b,rtol=rtol, atol=atol)[0]


def construct_gain_data_cov(data:np.ndarray,
                            N:np.ndarray,
                            gain_operator:np.ndarray,
                            t_sys_linear_operators:list,
                            t_sys_operator_amplitudes:list):
    """
    Returns the data for the gain GCR step which is scaled by the total system temperature.

    t_sys_operators:
        a list of linear operators forming contributions to t_sys. e.g [t_nw, t_cal, t_ant etc.]
    """
    gain_data = np.zeros_like(data)
    for operator, amplitude in zip(t_sys_linear_operators, t_sys_operator_amplitudes):
        gain_data += operator @ amplitude

    gain_data_inv_cov = estimated_inverse_covariance(N,
                                                     gain_data,
                                                     np.zeros_like(gain_data))
    
    return gain_data, gain_data_inv_cov

def construct_t_sys_term_data_cov(data:np.ndarray,
                                  N:np.ndarray,
                                  linear_operator_contributer:np.ndarray,
                                  gain_linear_operator:np.ndarray,
                                  gain_amplitudes:np.ndarray,
                                  t_sys_mu_operator_list:list,
                                  t_sys_mu_operator_amplitudes:list):
    """
    Returns the data and estimated covariance for the t_sys contribution
    being sampled.

    d = G (X_ant@p_ant + X_unc@p_unc+...)

    d'' = (d / G) - mu

    mu = sum(X_i @ p_i) over all other contributions.

    """
    gains = gain_linear_operator@gain_amplitudes # get current gains

    mu = np.zeros_like(data)

    for operator, amplitude in zip(t_sys_mu_operator_list, t_sys_mu_operator_amplitudes):
        mu += operator @ amplitude

    scaled_data = (data / gains) - mu # scaled data

    scaled_inv_cov = estimated_inverse_covariance(N,
                                                  scaled_data,
                                                  linear_operator_contributer,
                                                  mu)

    return scaled_data, scaled_inv_cov



def sample_loop(data:np.ndarray,
                N:np.ndarray,
                gain_linear_operator:np.ndarray,
                ant_unknown_nw_operator:np.ndarray,
                cal_source_operator:np.ndarray,
                gain_priors: np.ndarray|None=None,
                gain_prior_cov: np.ndarray|None=None,
                ant_unknown_nw_priors: np.ndarray|None=None,
                ant_unknown_nw_prior_cov: np.ndarray|None=None,
                cal_source_priors: np.ndarray|None=None,
                cal_source_prior_cov: np.ndarray|None=None,
                checkpoint: int|None=None,
                checkpoint_savepath: Path|str = 'checkpoint/samplecheckpoint.npz',
                burn_in: bool = True,
                init_gains: np.ndarray | None=None,
                init_cal_source_amps: np.ndarray | None=None,
                init_ant_unknown_nw_amps: np.ndarray | None=None,
                n_iter: int=1000,
                linalg_init: bool = False,
                rtol=1e-10,
                atol=1e-8,
                maxiter=int(1e6)):
    """
    Gibbs samples the amplitudes in a loop

    The loop follows:
        - Antenna Temp unknowns and noise wave source step
        - Gain step
        - Calibrator step
    and repeats until n_iterations is met.

    This can be ran as a burn in stage if need be before starting a full
    gibbs run.
    """
    n_steps=3 # use this to set the seed
    idx = 0
    # Initialise
    if not burn_in:
        assert init_gains is not None
        assert init_cal_source_amps is not None
        assert init_ant_unknown_nw_amps is not None # Carry on from burn in or checkpoint
    else:
        # set initial amplitudes to ones -- dicey
        if cal_source_priors is None:
            init_cal_source_amps = np.ones_like(cal_source_operator[0])
        else:
            cal_source_lhs = construct_gcr_lhs(cal_source_operator,
                                               inv_data_cov=np.zeros_like(data),
                                               inv_prior_cov=cal_source_prior_cov)
            cal_source_rhs = construct_gcr_rhs(cal_source_operator,
                                               data=np.zeros_like(data),
                                               inv_data_cov=np.zeros_like(data),
                                               omega_data=np.zeros_like(data),
                                               priors=cal_source_priors,
                                               inv_prior_cov=1/cal_source_prior_cov,
                                               weiner_solution=True)
            init_cal_source_amps = solve_gcr(A=cal_source_lhs,
                                             b=cal_source_rhs,
                                             linalg_init=linalg_init,
                                             rtol=rtol,
                                             atol=atol,
                                             maxiter=maxiter)
        if gain_priors is None:
            init_gains = np.ones_like(gain_linear_operator[0]) #FIXME write functions to generate priors
        else:
            init_gains = gain_priors
        if ant_unknown_nw_priors is None:
            init_ant_unknown_nw_amps = np.ones_like(ant_unknown_nw_operator[0])
        else:
            init_ant_unknown_nw_amps = ant_unknown_nw_priors
        
        init_gains = gain_priors # assume for now there are some prior values FIXME
        #FIXME write a function to provide priors for the gains
    


    #FIXME sort out covariances if they aren't given
    inv_gain_prior_cov = 1/gain_prior_cov
    inv_cal_source_prior_cov = 1/cal_source_priors
    inv_ant_nw_prior_cov = 1/ant_unknown_nw_prior_cov

    cal_source_amps = init_cal_source_amps
    gain_amps = init_gains
    ant_unknown_nw_amps = init_ant_unknown_nw_amps

    while idx < n_iter:
        if idx == 0:
            p_cal_source = cal_source_amps
            p_gain = gain_amps
            p_ant_unknown_nw = ant_unknown_nw_amps # get array p_0
        else:
            p_cal_source = cal_source_amps[-1]
            p_gain = gain_amps[-1]
            p_ant_nw = ant_unknown_nw_amps[-1] # get idx i-1

        ## Antena Temp NW Step
        d_ant_nw, inv_cov_ant_nw = construct_t_sys_term_data_cov(data,
                                                                 N,
                                                                 ant_unknown_nw_operator,
                                                                 gain_linear_operator,
                                                                 p_gain,
                                                                 [cal_source_operator],
                                                                 [p_cal_source])
        ant_nw_lhs = construct_gcr_lhs(linear_operator=ant_unknown_nw_operator,
                                       inv_data_cov=inv_cov_ant_nw,
                                       inv_prior_cov=inv_ant_nw_prior_cov)
        ant_nw_rhs = construct_gcr_rhs(linear_operator=ant_unknown_nw_operator,
                                       data=d_ant_nw,
                                       inv_data_cov=inv_cov_ant_nw,
                                       omega_data=np.random.normal(size=d_ant_nw.size),
                                       priors=ant_unknown_nw_priors,
                                       inv_prior_cov=inv_ant_nw_prior_cov,
                                       omega_prior=np.random.normal(size=ant_unknown_nw_priors.size))

        p_ant_nw = solve_gcr(A=ant_nw_lhs,
                             b=ant_nw_rhs,
                             linalg_init=linalg_init,
                             rtol=rtol,
                             atol=atol,
                             maxiter=maxiter)
        del ant_nw_rhs, ant_nw_lhs, d_ant_nw, inv_cov_ant_nw # delete uneeded variables

        # Gains step
        d_gains_nw, inv_cov_gains = construct_gain_data_cov(data,
                                                            N,
                                                            gain_linear_operator,
                                                            [ant_unknown_nw_operator,
                                                             cal_source_operator],
                                                             [p_ant_nw,
                                                              p_cal_source])

        gains_lhs = construct_gcr_lhs(linear_operator=gain_linear_operator,
                                               inv_data_cov=inv_cov_gains,
                                               inv_prior_cov=inv_gain_prior_cov)
        gains_rhs = construct_gcr_rhs(linear_operator=ant_unknown_nw_operator,
                                               data=d_gains_nw,
                                               inv_data_cov=inv_cov_gains,
                                               omega_data=np.random.normal(size=d_gains_nw.size),
                                               priors=gain_priors,
                                               inv_prior_cov=inv_gain_prior_cov,
                                               omega_prior=np.random.normal(size=gain_priors.size))
        p_gain = solve_gcr(A=gains_lhs,
                                     b=gains_rhs,
                                     linalg_init=linalg_init,
                                     rtol=rtol,
                                     atol=atol,
                                     maxiter=maxiter)
        del gains_rhs, gains_lhs, d_gains_nw, inv_cov_gains

        # Cal source step

        d_cal, inv_cov_cal = construct_t_sys_term_data_cov(data,
                                                            N,
                                                            cal_source_operator,
                                                            gain_linear_operator,
                                                            p_gain,
                                                            [ant_unknown_nw_operator],
                                                            [p_ant_nw])
        calsource_nw_lhs = construct_gcr_lhs(linear_operator=cal_source_operator,
                                       inv_data_cov=inv_cov_cal,
                                       inv_prior_cov=inv_cal_source_prior_cov)
        calsource_nw_rhs = construct_gcr_rhs(linear_operator=cal_source_operator,
                                       data=d_cal,
                                       inv_data_cov=inv_cov_cal,
                                       omega_data=np.random.normal(size=d_cal.size),
                                       priors=cal_source_priors,
                                       inv_prior_cov=inv_cal_source_prior_cov,
                                       omega_prior=np.random.normal(size=cal_source_priors.size))
        
        p_cal_source = solve_gcr(A=calsource_nw_lhs,
                                     b=calsource_nw_rhs,
                                     linalg_init=linalg_init,
                                     rtol=rtol,
                                     atol=atol,
                                     maxiter=maxiter)
        del calsource_nw_rhs, calsource_nw_lhs, d_cal, inv_cov_cal # delete uneeded variables

        # add to the list
        cal_source_amps = np.concatenate((cal_source_amps, p_cal_source),
                                         axis=0)
        gain_amps = np.concatenate((gain_amps, p_gain),
                                   axis=0)
        ant_unknown_nw_amps = np.concatenate((ant_unknown_nw_amps,p_ant_nw),
                                             axis=0)

        if checkpoint is not None and idx % checkpoint==0:
            # save current chain at checkpoint
            np.savez_compressed(file=checkpoint_savepath,
                                cal_source_amps=cal_source_amps,
                                gain_amps=gain_amps,
                                ant_unknown_nw_amps=ant_unknown_nw_amps,
                                checkpoint_idx=idx)

    return {'cal_source_amps':cal_source_amps,
            'gain_amps':gain_amps,
            'ant_unknown_nw_amps':ant_unknown_nw_amps}
    

        






