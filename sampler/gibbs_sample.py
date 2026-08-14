"""
Set up GCR and gibbs sample
"""

import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import cg
from .covariance import estimated_inverse_covariance
from pathlib import Path

def construct_gcr_lhs(linear_operator:np.ndarray,
                      inv_data_cov:np.ndarray,
                      inv_prior_cov:np.ndarray|None=None,
                      prior_in_data_space=True):
    """
    Construct the left hand side of the GCR equation
    """
    # if inv_prior_cov is not None:
    #     if inv_prior_cov.shape == inv_data_cov.shape: # Prior is data shaped
    #         return (linear_operator.T @ (inv_prior_cov[:,None] * linear_operator)) +\
    #             (linear_operator.T @ (inv_data_cov[:,None] * linear_operator))
    #     else:
    #         return linear_operator.T @ (inv_data_cov[:,None] * linear_operator) + inv_prior_cov
    # else:
    #     return linear_operator.T @ (inv_data_cov[:,None] * linear_operator)

    lhs = linear_operator.T @ (inv_data_cov[:, None] * linear_operator)

    if inv_prior_cov is not None:
        if prior_in_data_space:
            lhs += linear_operator.T @ (inv_prior_cov[:,None] * linear_operator)
        else:
            lhs += inv_prior_cov
    return lhs


    

def construct_gcr_rhs(linear_operator: np.ndarray,
                      data: np.ndarray,
                      inv_data_cov:np.ndarray,
                      omega_data: np.ndarray,
                      priors: np.ndarray | None=None,
                      inv_prior_cov: np.ndarray | None=None,
                      omega_prior: np.ndarray | None=None,
                      prior_in_data_space=True,
                      weiner_solution: bool = False):
    """ 
    """
    if weiner_solution:
        omega_data = np.zeros_like(data)

    rhs = (
        linear_operator.T @ (inv_data_cov * data)
        + linear_operator.T @ (np.sqrt(inv_data_cov)*omega_data)
    )

    if priors is not None:
        if weiner_solution:
            omega_prior = np.zeros_like(priors)

        if prior_in_data_space:
            rhs += (
                linear_operator.T @ (inv_prior_cov * priors)
                + linear_operator.T @ (np.sqrt(inv_prior_cov)*omega_prior)
            )
        else:
            rhs += (
                (inv_prior_cov * priors) + (np.sqrt(inv_prior_cov)*omega_prior)
            )

    # data_term = (linear_operator.T @ (inv_data_cov * data)) +\
    #     (linear_operator.T @ (np.sqrt(inv_data_cov)*omega_data))


    # if priors is None:
    #     prior_term = np.zeros_like(data_term)

    # else:
    #     if weiner_solution:
    #         omega_prior = np.zeros_like(priors)
    #     if data.shape != priors.shape:
    #         prior_term = (inv_prior_cov*priors) + (np.sqrt(inv_prior_cov)*omega_prior)
    #     else:
    #         prior_term = (linear_operator.T @ (inv_prior_cov*priors)) +\
    #             (linear_operator.T @ (np.sqrt(inv_prior_cov) * omega_prior))

    # print(data_term)
    # print(prior_term)

    return rhs


def solve_gcr(A,
              b,
              linalg_init: bool = False,
              rtol=1e-10,
              atol=1e-8,
              maxiter=int(1e6)):
    """
    """
    if linalg_init:
        x0 = np.linalg.lstsq(A,b)[0]
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
    t_sys_tot = np.zeros_like(data)
    for operator, amplitude in zip(t_sys_linear_operators, t_sys_operator_amplitudes):
        t_sys_tot += operator @ amplitude

    gain_data = data / t_sys_tot

    gain_data_inv_cov = estimated_inverse_covariance(N,
                                                     gain_data,
                                                     gain_operator,
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


def t_sys_cont_gcr(data,
               N,
               X_cont,
               U_gain,
               p_gain,
               t_sys_mu_op_list,
               t_sys_mu_amp_list,
               seed,
               prior_cont=None,
               inv_prior_cont_cov=None,
               data_mask =None,
               linalg_init: bool = False,
               rtol=1e-10,
               atol=1e-8,
               maxiter=int(1e6),
               verbose: bool = True):
    d_cont, inv_cont_cov = construct_t_sys_term_data_cov(data=data,
                                                         N=N,
                                                         linear_operator_contributer=X_cont,
                                                         gain_linear_operator=U_gain,
                                                         gain_amplitudes=p_gain,
                                                         t_sys_mu_operator_list=t_sys_mu_op_list,
                                                         t_sys_mu_operator_amplitudes=t_sys_mu_amp_list)

    if data_mask is not None:
        inv_cont_cov *=  data_mask

    if verbose:
        print("d_cont ", d_cont)
        print('inv_cont_cov ', inv_cont_cov)

    cont_lhs = construct_gcr_lhs(X_cont,
                                 inv_data_cov=inv_cont_cov,
                                 inv_prior_cov=inv_prior_cont_cov)
    np.random.seed(seed)
    omega_data = np.random.normal(size=d_cont.shape)
    if prior_cont is not None:
        omega_prior = np.random.normal(size=prior_cont.shape)
    else:
        omega_prior = None
    
    cont_rhs = construct_gcr_rhs(linear_operator=X_cont,
                                 data=d_cont,
                                 inv_data_cov=inv_cont_cov,
                                 omega_data=omega_data,
                                 priors=prior_cont,
                                 inv_prior_cov=inv_prior_cont_cov,
                                 omega_prior=omega_prior,
                                 weiner_solution=False)

    del d_cont, inv_prior_cont_cov

    p_cont = solve_gcr(cont_lhs, cont_rhs,
                       linalg_init=linalg_init, rtol=rtol,
                       atol=atol, maxiter=maxiter)
    del cont_rhs, cont_lhs
    return p_cont


def gain_gcr(data,
             N,
             U_gain,
             X_sys_list,
             p_sys_list,
             seed,
             prior_gain=None,
             inv_prior_gain_cov=None,
             data_mask =None,
             linalg_init: bool = False,
             rtol=1e-10,
             atol=1e-8,
             maxiter=int(1e6),
             verbose: bool = True):
    d_gain, d_gain_inv_cov = construct_gain_data_cov(data=data,
                                                     N=N,
                                                     gain_operator=U_gain,
                                                     t_sys_linear_operators=X_sys_list,
                                                     t_sys_operator_amplitudes=p_sys_list)
    if data_mask is not None:
        d_gain_inv_cov *=  data_mask

    if verbose:
            print("d_gain ", d_gain)
            print('d_gain_inv_cov ', d_gain_inv_cov)

    gain_lhs = construct_gcr_lhs(U_gain,
                                     inv_data_cov=d_gain_inv_cov,
                                     inv_prior_cov=inv_prior_gain_cov,
                                     prior_in_data_space=False)
    np.random.seed(seed)
    omega_data = np.random.normal(size=d_gain.shape)

    if prior_gain is not None:
        omega_prior = np.random.normal(size=prior_gain.shape)
    else:
        omega_prior = None

    gain_rhs = construct_gcr_rhs(linear_operator=U_gain,
                                     data=d_gain,
                                     inv_data_cov=d_gain_inv_cov,
                                     omega_data=omega_data,
                                     priors=prior_gain,
                                     inv_prior_cov=inv_prior_gain_cov,
                                     omega_prior=omega_prior,
                                     weiner_solution=False,
                                     prior_in_data_space=False)
    
    del d_gain, d_gain_inv_cov

    p_gain = solve_gcr(gain_lhs, gain_rhs,
                           linalg_init=linalg_init, rtol=rtol,
                           atol=atol, maxiter=maxiter)
    del gain_lhs, gain_rhs
    return p_gain


def sample_loop(data: np.ndarray,
                N: np.ndarray,
                U_gain,
                X_ant_nw,
                X_cal_src,
                init_gain_amps,
                init_ant_nw_amps,
                init_cal_src_amps,
                gain_priors: np.ndarray | None=None,
                ant_nw_priors: np.ndarray | None = None,
                cal_src_priors: np.ndarray | None = None,
                inv_gain_prior_cov: np.ndarray | None=None,
                inv_ant_nw_prior_cov: np.ndarray | None=None,
                inv_cal_src_prior_cov: np.ndarray | None=None,
                data_mask: np.ndarray| None=None, # for inv cov
                checkpoint: int|None=None,
                checkpoint_savepath: Path|str = 'checkpoint/samplecheckpoint.npz',
                n_iter: int = 1000,
                linalg_init: bool = False,
                rtol=1e-10,
                atol=1e-8,
                maxiter=int(1e6),
                verbose: bool = True,
                progress_update=50,
                fix_cal_source: bool =False,
                ):
    n_steps = 3
    all_p_cal_src = init_cal_src_amps
    all_p_ant_nw = init_ant_nw_amps
    all_p_gain = init_gain_amps

    idx = 0

    while idx < n_iter:
        if idx == 0:
            p_cal_src = all_p_cal_src
            p_ant_nw = all_p_ant_nw
            p_gain = all_p_gain
        else:
            p_cal_src = all_p_cal_src[-1]
            p_ant_nw = all_p_ant_nw[-1]
            p_gain = all_p_gain[-1]

        # Antenna Noise Wave GCR Step
        p_ant_nw = t_sys_cont_gcr(data=data, N=N, X_cont=X_ant_nw,
                                  U_gain=U_gain, p_gain=p_gain,
                                  t_sys_mu_op_list=[X_cal_src],
                                  t_sys_mu_amp_list=[p_cal_src],
                                  seed=int(idx*n_steps),
                                  prior_cont=ant_nw_priors,
                                  inv_prior_cont_cov=inv_ant_nw_prior_cov,
                                  data_mask=data_mask,
                                  linalg_init=linalg_init,
                                  rtol=rtol, atol=atol, maxiter=maxiter,
                                  verbose=verbose)

        if verbose:
            print(f'p_ant_nw - {idx}')
            print(p_ant_nw)

        # Gain Sampling Step
        p_gain = gain_gcr(data=data, N=N, U_gain=U_gain,
             X_sys_list=[X_ant_nw, X_cal_src],
             p_sys_list=[p_ant_nw, p_cal_src],
             seed=int(n_steps*idx)+1,
             prior_gain=gain_priors,
             inv_prior_gain_cov=inv_gain_prior_cov,
             data_mask =data_mask,
             linalg_init=linalg_init,
             rtol=rtol,
             atol=atol,
             maxiter=maxiter,
             verbose=verbose)
        
        if verbose:
            print(f'p_gain - {idx}')
            print(p_gain)

        # Cal Source GCR Step
        # Can be fixed
        if not fix_cal_source:
            p_cal_src = t_sys_cont_gcr(data=data, N=N, X_cont=X_cal_src,
                                    U_gain=U_gain, p_gain=p_gain,
                                    t_sys_mu_op_list=[X_ant_nw],
                                    t_sys_mu_amp_list=[p_ant_nw],
                                    seed=int(idx*n_steps)+2,
                                    prior_cont=cal_src_priors,
                                    inv_prior_cont_cov=inv_cal_src_prior_cov,
                                    data_mask=data_mask,
                                    linalg_init=linalg_init,
                                    rtol=rtol, atol=atol, maxiter=maxiter,
                                    verbose=verbose)
        else:
            p_cal_src = init_cal_src_amps
        if verbose:
            print(f'p_cal_src - {idx}')
            print(p_cal_src)


        all_p_cal_src = np.vstack((all_p_cal_src, p_cal_src))
        all_p_gain = np.vstack((all_p_gain, p_gain))
        all_p_ant_nw = np.vstack((all_p_ant_nw,p_ant_nw))


        if progress_update is not None and idx % progress_update==0:
            print(f'Gibbs Sample - {idx} / {n_iter}')


        if checkpoint is not None and idx % checkpoint==0:
            np.savez_compressed(file=checkpoint_savepath,
                                all_p_cal_src=all_p_cal_src,
                                all_p_gain=all_p_gain,
                                all_p_ant_nw=all_p_ant_nw,
                                checkpoint_idx=idx)



        idx += 1
    if verbose:
        print('Finished Sampling...')
    return {"all_p_cal_src":all_p_cal_src,
            "all_p_gain":all_p_gain,
            "all_p_ant_nw":all_p_ant_nw}