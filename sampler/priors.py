import numpy as np
from .linear_basis import BasisConstructor
from .reader import ObservationReader

def setup_probe_cal_priors(observation_obj: ObservationReader,
                     source_temp_dict: dict,
                     temp_std:float | int,
                     ):
    """
    Redundant

    Set up priors and prior covariance for the source temperatures
    monitered via the probe
    """

    tau = np.zeros_like(observation_obj.data_waterfall) # (ntimes, nfreqs)
    
    switch_array = observation_obj.states_array # (ntimes)

    # source_temp_dict is assigned as {'heated':['heated_load'], 'ambient':['open','load']}
    for therm, state_list in source_temp_dict.items():
        temperatures = observation_obj.temperatures[therm] # selects correct thermistor
        # (ntimes)
        if not isinstance(state_list, list):
            state_list = [state_list]
        for state in state_list:
            state_mask = np.where(switch_array == state)[0] # indices

            tau[state_mask,:]=temperatures[state_mask,None]
    
    S_inv = 1 / (np.ones_like(tau) * temp_std**2) # set up prior covariance 

    S_inv = np.where(tau == 0, 0, S_inv) # set prior covariance to 0 for unmonitered states

    return tau.flatten(), S_inv.flatten() # return as flat arrays



def setup_probe_cal_priors(observation_obj: ObservationReader,
                           source_temp_dict: dict,
                           temp_std:float | int,
                           ):
    """
    Set up the thermistor probe data and covariance vector and matrix
    to be inline with the observation data.

    The thermistor data vector has a length n_t and a diagonal
    covariance matrix also of length n_t.
    """
    tau = np.zeros_like(observation_obj.times)
    states_array = observation_obj.states_array # ntimes

    for therm, state_list in source_temp_dict.items():
        temperatures = observation_obj.temperatures[therm] # get correct thermistor measurement
        if not isinstance(state_list, list):
            state_list = [state_list]
        for state in state_list:
            state_mask = np.where(states_array==state)[0] # get indices
            tau[state_mask]=temperatures[state_mask]
    S_inv = 1 / (np.ones_like(tau) * temp_std**2)
    S_inv = np.where(tau==0, 0, S_inv)

    return tau.flatten(), S_inv.flatten()

def generate_smooth_thermistor_operator_matrix(observation_data: ObservationReader,
                                               source_basis: BasisConstructor,
                                               source_label: str | list):
    """
    Construct the thermistor operator matrix
    corresponding to the source matrix given by
    operator_setup.py - generate_smooth_source_matrix()


    Y_src = (sum_k theta_src_k)Omega_src

    Can be over multiple or single calibrators in the case of the
    ambient calibrators.

    """
    
    m = np.zeros_like(observation_data.times) # mtimes

    if isinstance(source_label,str):
        source_label = [source_label] # convert to list so same code can be used
    for state in source_label:
        theta = observation_data.theta_dict[state]
        m += theta
    m = m.flatten()
    Y_src = (source_basis.basis_matrix.T * m).T
    return Y_src