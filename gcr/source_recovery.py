"""
Module for recovering T_src with extracted values
from the GCR of the NW parameters
"""

import numpy as np
import astropy.units as un
import matplotlib.pyplot as plt
from .transfer_matrix_construction import construct_src_transfer_matrix


def recover_source_temperatures(waterfall,
                                times,
                                freqs,
                                switch_times,
                                switch_states,
                                source_temperatures,
                                temperatures_times,
                                t_nw_vector,
                                gamma_src_dict,
                                internal_load_label,
                                noise_source_label,
                                gamma_rec,
                                freq_norm_func,
                                time_norm_func,
                                unc_poly_orders,
                                cos_poly_orders,
                                sin_poly_orders,
                                t_src_label = 'antenna',
                                dicke_switch_targets = ['internal_load',
                                                        'heated_load'],
                                dicke_switched_averaged_t_src = False,
                                switch_buffer=2*un.s,
                                time_unit=un.s,
                                freq_unit=un.MHz,):
    
    if not isinstance(times, un.Quantity):
        times *= time_unit
    if not isinstance(freqs, un.Quantity):
        freqs *= freq_unit
    if not isinstance(switch_times, un.Quantity):
        switch_times*=time_unit

    n_dicke_sources = len(dicke_switch_targets) + 1 # add one for the source 

    n_full_cycles = len(switch_states) // n_dicke_sources

    switch_states = switch_states[:n_full_cycles*n_dicke_sources]
    switch_times = switch_times[:n_full_cycles*n_dicke_sources]

    switch_states = np.reshape(switch_states, shape=(n_full_cycles, n_dicke_sources))
    switch_times = np.reshape(switch_times, shape=(n_full_cycles, n_dicke_sources))

    # create an array of p_l and p_ns and t_l and t_ns corresponding to len(switch_states)
    # then can use the index in the calculation of t_ant

    # calculate t_l and p_l


    t_l, p_l = [], []
    t_ns, p_ns = [], []

    for i, (dicke_switches, dicke_times) in enumerate(zip(switch_states, switch_times)):
        for j, (state, st_time) in enumerate(zip(dicke_switches, dicke_times)):
            # construct a time mask
            if j != n_dicke_sources-1 and i != n_full_cycles-1:
                times_mask = (times > st_time + switch_buffer) &\
                (times < dicke_times[j+1] - switch_buffer)
                temp_time_mask = (temperatures_times > st_time + switch_buffer) &\
                (temperatures_times < dicke_times[j+1] - switch_buffer)
            elif j == n_dicke_sources-1 and i != n_full_cycles-1:
                times_mask = (times > st_time + switch_buffer) &\
                (times < switch_times[i+1, 0] - switch_buffer)

                temp_time_mask = (temperatures_times > st_time + switch_buffer) &\
                (temperatures_times < switch_times[i+1, 0] - switch_buffer)
                    # final cycle
            elif i == n_full_cycles-1 and j != n_dicke_sources-1:
                times_mask = (times > st_time + switch_buffer) &\
                (times < dicke_times[j+1] - switch_buffer)

                temp_time_mask = (temperatures_times > st_time + switch_buffer) &\
                (temperatures_times < dicke_times[j+1] - switch_buffer)
            else:
                times_mask = (times > st_time + switch_buffer)
                temp_time_mask = (temperatures_times > st_time + switch_buffer)
            # ----

            if state == internal_load_label or state == noise_source_label:
                p_i_array = waterfall[times_mask]
                t_i_array = source_temperatures[state][temp_time_mask]
                # flag_step
                t_i = t_i_array.mean()
                p_i = p_i_array.mean(axis=0)
                if state == internal_load_label:
                    t_l.append(t_i)
                    p_l.append(p_i)
                else:
                    t_ns.append(t_i)
                    p_ns.append(p_i)


    t_l, p_l = np.array(t_l), np.array(p_l)
    t_ns, p_ns = np.array(t_ns), np.array(p_ns)

    source_waterfall = []
    source_times = []

    # extract t_src
    for i, (dicke_switches, dicke_times) in enumerate(zip(switch_states, switch_times)):
        if t_src_label not in dicke_switches:
            continue
        for j, (state, st_time) in enumerate(zip(dicke_switches, dicke_times)):
            # construct a time mask
            if j != n_dicke_sources-1 and i != n_full_cycles-1:
                times_mask = (times > st_time + switch_buffer) &\
                (times < dicke_times[j+1] - switch_buffer)
                temp_time_mask = (temperatures_times > st_time + switch_buffer) &\
                (temperatures_times < dicke_times[j+1] - switch_buffer)
            elif j == n_dicke_sources-1 and i != n_full_cycles-1:
                times_mask = (times > st_time + switch_buffer) &\
                (times < switch_times[i+1, 0] - switch_buffer)

                temp_time_mask = (temperatures_times > st_time + switch_buffer) &\
                (temperatures_times < switch_times[i+1, 0] - switch_buffer)
                    # final cycle
            elif i == n_full_cycles-1 and j != n_dicke_sources-1:
                times_mask = (times > st_time + switch_buffer) &\
                (times < dicke_times[j+1] - switch_buffer)

                temp_time_mask = (temperatures_times > st_time + switch_buffer) &\
                (temperatures_times < dicke_times[j+1] - switch_buffer)
            else:
                times_mask = (times > st_time + switch_buffer)
                temp_time_mask = (temperatures_times > st_time + switch_buffer)
            # ----

            if state == t_src_label:
                p_i_array = waterfall[times_mask]
                src_times_array = times[times_mask]

                src_transfer_matrix = construct_src_transfer_matrix(freq_array=freqs.value,
                                                                    time_array=src_times_array.value,
                                                                    freq_norm_func=freq_norm_func,
                                                                    time_norm_func=time_norm_func,
                                                                    unc_poly_orders=unc_poly_orders,
                                                                    cos_poly_orders=cos_poly_orders,
                                                                    sin_poly_orders=sin_poly_orders,
                                                                    gamma_rec=gamma_rec,
                                                                    gamma_src_label=t_src_label,
                                                                    switch_state_src_gamma_dict=gamma_src_dict)
                
                p_l_dicke = p_l[i]
                p_ns_dicke = p_ns[i]
                t_l_dicke = t_l[i]
                t_ns_dicke = t_ns[i]

                t_src = t_src_calc(p_src=p_i_array,
                                   p_l=p_l_dicke,
                                   p_ns=p_ns_dicke,
                                   t_l=t_l_dicke,
                                   t_ns=t_ns_dicke,
                                   gamma_rec=gamma_rec,
                                   gamma_src=gamma_src_dict[t_src_label],
                                   h_src=src_transfer_matrix,
                                   t_nw=t_nw_vector)
                
                source_waterfall.append(t_src)
                source_times.append(src_times_array)

    final_t_src = []
    final_t_src_times = []
    for l in source_waterfall:
        for p in l:
            final_t_src.append(p)

    for l in source_times:
        for p in l:
            final_t_src_times.append(p.value)

    source_waterfall, source_times = np.array(final_t_src), np.array(final_t_src_times)

    source_waterfall = source_waterfall.flatten()
    source_waterfall = source_waterfall.reshape((int(len(source_waterfall) / len(freqs)), len(freqs)))
    source_times = source_times.flatten()

    source_covariance= 0
    return source_waterfall, source_covariance, source_times


def t_src_calc(p_src,
               p_l,
               p_ns,
               t_l,
               t_ns,
               gamma_rec,
               gamma_src,
               h_src,
               t_nw):
    
    f_src = np.sqrt(1 - (np.abs(gamma_rec)**2)) / (1 - (gamma_rec*gamma_src))

    h_src_t_nw = h_src @ t_nw

    h_src_t_nw = h_src_t_nw.reshape(p_src.shape)

    q = (p_src - p_l) / (p_ns - p_l)

    t_src = (q*(t_ns - t_l)*(1 - (np.abs(gamma_rec)**2))) + (t_l*(1 - (np.abs(gamma_rec)**2))) - h_src_t_nw

    t_src = t_src / ((1 - (np.abs(gamma_src)**2)) * (np.abs(f_src)**2))

    return t_src