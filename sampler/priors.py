import numpy as np
from reader import ObservationReader


def setup_probe_cal_priors(observation_obj: ObservationReader,
                     source_temp_dict: dict,
                     temp_std:float | int,
                     ):
    """ 
    Set up priors and prior covariance for the source temperatures
    monitered via the probe
    """

    tau = np.zeros_like(observation_obj.data_waterfall) # (ntimes, nfreqs)
    
    switch_array = observation_obj.switch_array # (ntimes)

    # source_temp_dict is assigned as {'heated':['heated_load'], 'ambient':['open','load']}
    for therm, state_list in source_temp_dict.items():
        temperatures = observation_obj.temperatures[therm] # selects correct thermistor
        # (ntimes)
        if not isinstance(state_list, list):
            state_list = [state_list]
        for state in state_list:
            state_mask = np.where(switch_array == state)[0] # indices

            tau[state_mask,:]=temperatures[state_mask]
    
    S_inv = 1 / (np.ones_like(tau) * temp_std**2) # set up prior covariance 

    S_inv = np.where(tau == 0, 0, S_inv) # set prior covariance to 0 for unmonitered states

    return tau.flatten(), S_inv.flatten() # return as flat arrays
