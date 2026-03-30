"""
Script for reading in raw data and processing for noise wave extraction
"""

import numpy as np
import astropy.units as un
from .data_and_noise_covariance import return_data_and_variance

def create_nw_data_and_covariance_from_raw(waterfall,
                                           times,
                                           freqs,
                                           switch_times,
                                           switch_states,
                                           source_temperatures:dict,
                                           noise_wave_loads:list,
                                           gamma_src_dict:dict,
                                           gamma_rec:np.ndarray,
                                           internal_load_label='internal_load',
                                           noise_source_label='heated_load',
                                           dicke_switch_targets = ['internal_load',
                                                                   'heated_load'],
                                           switch_buffer=2*un.s,
                                           time_unit=un.s,
                                           freq_unit=un.MHz,
                                           assumed_temp_sens_variance=None):
    """Function to take in raw observational data and return
    the data for extracting noise-wave parameters.

    Returns data waterfall, freqs, times, labels
    """
    if not isinstance(times, un.Quantity):
        times *= time_unit
    if not isinstance(freqs, un.Quantity):
        freqs *= freq_unit
    if not isinstance(switch_times, un.Quantity):
        switch_times*=time_unit

    

    n_dicke_sources = len(dicke_switch_targets) + 1 # add one for target itself

    # makes an ordered list of loads
    final_states = [s for s in switch_states if s in noise_wave_loads]

    n_cycles = len(switch_states) // (len(dicke_switch_targets)+1) # number of full dicke-switches
    final_states = final_states[:n_cycles]

    # eliminate unfinished cycles

    switch_states = switch_states[:n_cycles*(len(dicke_switch_targets)+1)]
    switch_times = switch_times[:n_cycles*(len(dicke_switch_targets)+1)]

    switch_states = np.reshape(switch_states, shape=(n_cycles, n_dicke_sources))
    switch_times = np.reshape(switch_times, shape=(n_cycles, n_dicke_sources))

    final_times = np.empty(shape=(n_cycles,), dtype=un.Quantity)
    final_states = []
    data_waterfall = np.empty(shape=(n_cycles, len(freqs)))
    covar_waterfall = np.empty(shape=(n_cycles, len(freqs)))

    data_waterfall = []
    covar_waterfall = []
    final_times = []
    final_states = []

    for i, (dicke_switches, dicke_times) in enumerate(zip(switch_states, switch_times)):
        gamma_src = np.empty(shape=freqs.shape)
        p_src_array = 0
        p_src_times = 0

        p_l_array = 0
        p_l_times = 0

        p_ns_array = 0
        p_ns_times = 0

        src_temps = 0
        l_temps = 0
        ns_temps = 0


        if catch_false_loads(dicke_switches,
                             dicke_switch_targets,
                             noise_wave_loads):

            for j, (state, st_time) in enumerate(zip(dicke_switches, dicke_times)):
                # pass null states
                if state not in noise_wave_loads and state != internal_load_label and state != noise_source_label:
                    pass
                else:
                    if j != n_dicke_sources-1 and i != n_cycles-1:
                        times_mask = (times > st_time + switch_buffer) &\
                        (times < dicke_times[j+1] - switch_buffer)
                    elif j == n_dicke_sources-1 and i != n_cycles-1:
                        times_mask = (times > st_time + switch_buffer) &\
                        (times < switch_times[i+1, 0] - switch_buffer)
                    # final cycle
                    elif i == n_cycles-1 and j != n_dicke_sources-1:
                        times_mask = (times > st_time + switch_buffer) &\
                        (times < dicke_times[j+1] - switch_buffer)
                    else:
                        times_mask = (times > st_time + switch_buffer)

                    if state in noise_wave_loads:
                        final_states.append(state)
                        p_src_array = waterfall[times_mask]
                        p_src_times = times[times_mask]
                        gamma_src = gamma_src_dict[state]
                        src_temps = source_temperatures[state][times_mask]

                    elif state == internal_load_label:
                        p_l_array = waterfall[times_mask]
                        p_l_times = times[times_mask]
                        l_temps = source_temperatures[state][times_mask]
                    elif state == noise_source_label:
                        p_ns_array = waterfall[times_mask]
                        p_ns_times = times[times_mask]
                        ns_temps = source_temperatures[state][times_mask]
            
            data_vector, variance_vector, median_time = return_data_and_variance(p_src_array=p_src_array,
                                                                                    p_src_array_times=p_src_times,
                                                                                    t_src_array=src_temps,
                                                                                    p_l_array=p_l_array,
                                                                                    p_l_array_times=p_l_times,
                                                                                    t_l_array=l_temps,
                                                                                    p_ns_array=p_ns_array,
                                                                                    p_ns_array_times=p_ns_times,
                                                                                    t_ns_array=ns_temps,
                                                                                    gamma_rec=gamma_rec,
                                                                                    gamma_src=gamma_src,
                                                                                    freqs=freqs,
                                                                                    freq_unit=freq_unit,
                                                                                    time_unit=time_unit,
                                                                                    temp_sens_variance=assumed_temp_sens_variance)
            #data_waterfall[i] = data_vector
            data_waterfall.append(data_vector)
            #covar_waterfall[i] = variance_vector
            covar_waterfall.append(variance_vector)
            #final_times[i] = median_time
            final_times.append(median_time)
        else:
            pass
    
    data_waterfall = np.array(data_waterfall)
    covar_waterfall = np.array(covar_waterfall)
    final_times = np.array(final_times, dtype=un.Quantity)

    return data_waterfall, covar_waterfall, final_times, final_states

def catch_false_loads(switch_states,
                      dicke_switch_targets,
                      noise_wave_targets):
    """Function to identify if a cycle
    includes a noise-wave target
    """
    switch_states = [f for f in switch_states]
    for ds in dicke_switch_targets:
        switch_states.remove(ds)
    
    if switch_states[0] in noise_wave_targets:
        return True
    else:
        return False