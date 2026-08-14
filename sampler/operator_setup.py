"""
Module for setting up operators for the Gibbs sampling.
"""

import numpy as np
from .linear_basis import BasisConstructor
from .reader import ObservationReader, ReflectionReader


def generate_smooth_source_matrix(observation_data: ObservationReader,
                                  reflection_data: ReflectionReader,
                                  source_basis: BasisConstructor,
                                  source_label: str | list):
    """
    Construct the linear operator term for a target source
    with smooth variations.

    X_src = (sum_k theta_src_k cs_k)U_src

    Can be over multiple or single calibrators in the case of the
    ambient calibrators.

    """
    cs = np.zeros_like(observation_data.data_waterfall)
    if isinstance(source_label,str):
        source_label = [source_label] # convert to list so same code can be used
    for state in source_label:
        theta = observation_data.theta_dict[state]
        cs_state = reflection_data.cs_dict[state]
        theta = theta[np.newaxis]
        cs_state = cs_state[np.newaxis]
        cs += (theta.T @ cs_state)
    cs = cs.flatten()
    X_src = (source_basis.basis_matrix.T * cs).T
    return X_src

def generate_non_smooth_source_matrix(observation_data: ObservationReader,
                                      reflection_data: ReflectionReader,
                                      source_label: str):
    # Return as a 1d vector as it is a sparse matrix that is only diagonal
    """
    Constructs the time by time and frequency by frequency transfer matrix
    for a non smooth source such as the antenna temperatures where
    their timedependance might be effected by RFI etc.
    """
    theta = observation_data.theta_dict[source_label] # (ntimes,1)
    cs_state = reflection_data.cs_dict[source_label] # (n_freqs,1)

    n_state = np.sum(theta) * np.sum(observation_data.nfreqs) # number of state data points

    cs_state = cs_state[np.newaxis]
    theta = theta[np.newaxis]

    cs_state = (theta.T @ cs_state).flatten() # will have values where the source contributes and 0s elsewhere (nd,1)
    # is not zero where there are source values

    final_cs_state = np.zeros(shape=(int(len(cs_state)), n_state))
    
    current_index = 0
    for i, value in enumerate(cs_state):
        if value != 0:
            final_cs_state[i, current_index]=value
            current_index += 1 # fills out final_cst_state to place 
    return final_cs_state

def generate_noise_wave_transfer_matrix(observation_data: ObservationReader,
                                        reflection_data: ReflectionReader,
                                        unc_basis: BasisConstructor,
                                        cos_basis: BasisConstructor,
                                        sin_basis: BasisConstructor,
                                        rx_basis: BasisConstructor):
    
    """
    Construct the linear operator matrix for the noise wave
    and receiver terms

    X_nw = (X_unc, X_cos, X_sin, U_rx) shape (n_data, n_unc_params+n_cos_params+n_sin_params+n_rx_params)

    """
    unique_states = np.unique(observation_data.states_array)
    kappa_unc = np.zeros_like(observation_data.data_waterfall)
    kappa_cos = np.zeros_like(observation_data.data_waterfall)
    kappa_sin = np.zeros_like(observation_data.data_waterfall)

    for state in unique_states:
        theta = observation_data.theta_dict[state]
        kappa_unc_state = reflection_data.kappa_unc_dict[state]
        kappa_sin_state = reflection_data.kappa_sin_dict[state]
        kappa_cos_state = reflection_data.kappa_cos_dict[state]

        kappa_unc_state = kappa_unc_state[np.newaxis]
        kappa_sin_state = kappa_sin_state[np.newaxis]
        kappa_cos_state = kappa_cos_state[np.newaxis]
        theta = theta[np.newaxis]

        kappa_unc += (theta.T @ kappa_unc_state)
        kappa_cos += (theta.T @ kappa_cos_state)
        kappa_sin += (theta.T @ kappa_sin_state)
    
    kappa_unc = kappa_unc.flatten()
    kappa_cos = kappa_cos.flatten()
    kappa_sin = kappa_sin.flatten()

    X_unc = (unc_basis.basis_matrix.T * kappa_unc).T
    X_cos = (cos_basis.basis_matrix.T * kappa_cos).T
    X_sin = (sin_basis.basis_matrix.T * kappa_sin).T

    noise_wave_X = np.concatenate((X_unc,
                               X_cos,
                               X_sin,
                               rx_basis.basis_matrix),
                               axis=1)
    index_dict = {
        'unc': slice(0 ,unc_basis.basis_matrix.shape[1]),
        'cos': slice(unc_basis.basis_matrix.shape[1],
                     unc_basis.basis_matrix.shape[1] + cos_basis.basis_matrix.shape[1]),
        'sin': slice(unc_basis.basis_matrix.shape[1] + cos_basis.basis_matrix.shape[1],
                     unc_basis.basis_matrix.shape[1] + cos_basis.basis_matrix.shape[1] + sin_basis.basis_matrix.shape[1]),
        'rx': slice(unc_basis.basis_matrix.shape[1] + cos_basis.basis_matrix.shape[1] + sin_basis.basis_matrix.shape[1],
                    noise_wave_X.shape[1])
    } # dict of indices to extract amplitudes for source reconstruction

    return noise_wave_X, index_dict


def construct_joint_smooth_matrix(observation_data: ObservationReader,
                                  reflection_data: ReflectionReader,
                                  cal_list: list,
                                  basis_list: list):
    """
    Creates a joint transfer matrix given a list of calibrators and associated basis.
    The cal_list can be a list of lists in the case where the source temperature
    is the same for multiple calibrators.

    E.g
    cal_list -> [[open, short, load, long_open, long_short], 'heated_load]
    basis_list -> [ambient_t_src_basis, heated_t_src_basis]


    """
    assert len(cal_list) == len(basis_list), 'Cal List and Basis List must be equal lengths'

    X_joint = None
    for calibrators, basis in zip(cal_list, basis_list):
        X_src = generate_smooth_source_matrix(observation_data,
                                              reflection_data,
                                              basis,
                                              calibrators)
        if X_joint is None:
            X_joint = X_src
        else:
            X_joint = np.concatenate((X_joint, X_src), axis=1) # join X_src onto the exisiting smooth basis
    
    return X_joint


def scaled_data(d: np.ndarray,
                denominator: np.ndarray,
                offset: np.ndarray|None=None):
    """
    """
    assert d.shape == denominator.shape, 'data and denominator must be same shape'
    if offset is None: offset = np.zeros_like(d)
    return (d / denominator) - offset


def construct_system_temperature(X_sys,
                                 p_sys):
    return X_sys @ p_sys


def concat_and_index_operators(operator_list:list,
                               label_list:list):

    """
    A function for joining together operators 
    and including a dict of slices for easy
    ampltide recovery following sampling
    """
    combined_operator = np.concatenate(tuple(operator_list),axis=1) # combine operators

    slice_index_dict = {}

    start = 0

    for op, lab in zip(operator_list, label_list):
        end = start + len(op[0]) # gets length of the amplitdes
        slice_index_dict[lab] = slice(start, end)
        start = end

    return combined_operator, slice_index_dict