"""
Module for building Transfer Matrix for GCR
"""

import numpy as np
from gcr.basis_construction import polynomial_basis, array_normalisation, construct_basis_vectorised, null_normalisation, construct_fourier_basis


def construct_h_spectra(gamma_rec,
                        gamma_src_list=None,
                        switch_states=None,
                        switch_state_src_gamma_dict=None):
    """Constructs lists of h_spectra based on the gamma_src
    of the observing source and the gamma_rec of the receiver.

    Assumes len(gamma_rec) = len(gamma_src)
    """
    h_unc = []
    h_cos = []
    h_sin = []
    
    if gamma_src_list is not None:
        for gamma_src in gamma_src_list:
            F_src = np.sqrt(1 - (np.abs(gamma_rec)**2)) / (1 - (gamma_rec*gamma_src))
            alpha = np.angle(F_src * gamma_src)
            h_unc.append((np.abs(F_src)**2) * (np.abs(gamma_src)**2))
            h_cos.append(np.abs(gamma_src)*np.abs(F_src)*np.cos(alpha))
            h_sin.append(np.abs(gamma_src)*np.abs(F_src)*np.sin(alpha))
        return h_unc, h_cos, h_sin
    else:
        for state in switch_states:
            gamma_src = switch_state_src_gamma_dict[state]
            F_src = np.sqrt(1 - (np.abs(gamma_rec)**2)) / (1 - (gamma_rec*gamma_src))
            alpha = np.angle(F_src * gamma_src)
            h_unc.append((np.abs(F_src)**2) * (np.abs(gamma_src)**2))
            h_cos.append(np.abs(gamma_src)*np.abs(F_src)*np.cos(alpha))
            h_sin.append(np.abs(gamma_src)*np.abs(F_src)*np.sin(alpha))
        return np.array(h_unc), np.array(h_cos), np.array(h_sin)


def construct_h_spectra_corrected(gamma_rec,
                                  gamma_src_list=None,
                                  switch_states=None,
                                  switch_state_src_gamma_dict=None,
                                  internal_load_label='internal_load'):
    """Constructs lists of h_spectra based on the gamma_src
    of the observing source and the gamma_rec of the receiver.

    Assumes len(gamma_rec) = len(gamma_src)
    """
    h_unc = []
    h_cos = []
    h_sin = []

    gamma_l = switch_state_src_gamma_dict[internal_load_label]

    F_l = np.sqrt(1 - (np.abs(gamma_rec)**2)) / (1 - (gamma_rec*gamma_l))
    alpha_l = np.angle(F_l * gamma_l)

    if gamma_src_list is not None:
        for gamma_src in gamma_src_list:
            F_src = np.sqrt(1 - (np.abs(gamma_rec)**2)) / (1 - (gamma_rec*gamma_src))
            alpha = np.angle(F_src * gamma_src)

            unc = (np.power(np.abs(F_src), 2) * np.power(np.abs(gamma_src), 2)) -\
                    (np.power(np.abs(F_l), 2) * np.power(np.abs(gamma_l), 2))
            
            cos = (np.abs(gamma_src)*np.abs(F_src)*np.cos(alpha)) - \
                    (np.abs(gamma_l)*np.abs(F_l)*np.cos(alpha_l))
            
            sin = (np.abs(gamma_src)*np.abs(F_src)*np.sin(alpha)) - \
                    (np.abs(gamma_l)*np.abs(F_l)*np.sin(alpha_l))

            h_unc.append(unc)
            h_cos.append(cos)
            h_sin.append(sin)
        return h_unc, h_cos, h_sin
    else:
        for state in switch_states:
            gamma_src = switch_state_src_gamma_dict[state]
            F_src = np.sqrt(1 - (np.abs(gamma_rec)**2)) / (1 - (gamma_rec*gamma_src))
            alpha = np.angle(F_src * gamma_src)

            unc = (np.power(np.abs(F_src), 2) * np.power(np.abs(gamma_src), 2)) -\
                    (np.power(np.abs(F_l), 2) * np.power(np.abs(gamma_l), 2))
            
            cos = (np.abs(gamma_src)*np.abs(F_src)*np.cos(alpha)) - \
                    (np.abs(gamma_l)*np.abs(F_l)*np.cos(alpha_l))
            
            sin = (np.abs(gamma_src)*np.abs(F_src)*np.sin(alpha)) - \
                    (np.abs(gamma_l)*np.abs(F_l)*np.sin(alpha_l))

            h_unc.append(unc)
            h_cos.append(cos)
            h_sin.append(sin)
        return np.array(h_unc), np.array(h_cos), np.array(h_sin)


def construct_H_matrix(freq_array,
                       gamma_src_list,
                       gamma_rec,
                       unc_polyorder,
                       cos_polyorder,
                       sin_polyorder,
                       internal_load_label='internal_load',
                       use_corrected = False,
                       reference_frequency=None):
    """"""

    # n_params = unc_polyorder + cos_polyorder + sin_polyorder + 3
    # n_d = len(gamma_src_list) * len(freq_array) # number of data points. Assuming that only one cycle has been used

    if use_corrected:
        h_unc_list, h_cos_list, h_sin_list = construct_h_spectra_corrected(gamma_rec=gamma_rec,
                                                                           gamma_src_list=gamma_src_list,
                                                                           internal_load_label=internal_load_label)
    else:
        h_unc_list, h_cos_list, h_sin_list = construct_h_spectra(gamma_rec=gamma_rec, gamma_src_list=gamma_src_list)


    unc_basis = polynomial_basis(freq_array, unc_polyorder, reference_frequency)
    cos_basis = polynomial_basis(freq_array, cos_polyorder, reference_frequency)
    sin_basis = polynomial_basis(freq_array, sin_polyorder, reference_frequency)

    combined_basis = np.concatenate((unc_basis, cos_basis, sin_basis), axis=1)

    H = combined_basis # add onto the basis with each calibrator

    for i in range(len(gamma_src_list)):
        unc_h_applied = (unc_basis.T * h_unc_list[i]).T
        cos_h_appied = (cos_basis.T * h_cos_list[i]).T
        sin_h_applied = (sin_basis.T * h_sin_list[i]).T

        combined_h = np.concatenate((unc_h_applied, cos_h_appied, sin_h_applied), axis=1) # join the bases along the 1-axis (n_freqs x n_params)

        if i == 0:
            H = combined_h # establish axis
        else:
            H = np.concatenate((H, combined_h), axis=0) # join the new combined H along the 0-axis
        pass
    
    return H


def construct_transfer_matrix(freq_array:np.ndarray,
                           time_array:np.ndarray,
                           switch_states_array:np.ndarray,
                           unc_poly_orders:tuple,
                           sin_poly_orders:tuple,
                           cos_poly_orders:tuple,
                           switch_state_src_gamma_dict:dict,
                           gamma_rec:np.ndarray,
                           return_norm_funcs:bool = False,
                           internal_load_label='internal_load',
                           use_corrected=False,
                           basis_label='polynomial',
                           Lf:float = 60.,
                           Lt:float = 3600*24*4):
    """Constructs the transfer matrix H for describing
    T_unc, T_cos, T_sin in terms of frequency in terms
    of the data vector.

    freq_array - (n_times) np.ndarray of observational frequencies

    time_array - (n_times) np.ndarray of observational times coreresponding
    to the spectra

    switch_states_array - (n_times) np.ndarray of binary string values of the
    switch states

    unc_poly_orders - tuple of (n_freq_coeffs, n_time_coeffs) describing the 
    polynomial order of the T_unc

    sin_poly_orders - tuple of (n_freq_coeffs, n_time_coeffs) describing the 
    polynomial order of the T_sin

    cos_poly_orders - tuple of (n_freq_coeffs, n_time_coeffs) describing the 
    polynomial order of the T_cos

    switch_state_gamma_src_dict - dict of {'switch_state':np.ndarray(gamma_src)}
    corresponding to (n_freqs)

    gamma_rec - np.ndarray - (n_freqs) array of receiver s11 values 
    """

    n_unc_coeffs_freqs, n_unc_coeffs_time = unc_poly_orders
    n_sin_coeffs_freqs, n_sin_coeffs_time = sin_poly_orders
    n_cos_coeffs_freqs, n_cos_coeffs_time = cos_poly_orders

    # Array Normalisation
    if basis_label == 'polynomial':
        freq_norm, freq_norm_func = array_normalisation(freq_array)
        time_norm, time_norm_func = array_normalisation(time_array)
    else:
        freq_norm, freq_norm_func = null_normalisation(freq_array)
        time_norm, time_norm_func = null_normalisation(time_array)

    if basis_label == 'polynomial':
        unc_basis = construct_basis_vectorised(x=freq_norm,
                                   y=time_norm,
                                   n_x_coeffs=n_unc_coeffs_freqs,
                                   n_y_coeffs=n_unc_coeffs_time)
    
        sin_basis = construct_basis_vectorised(x=freq_norm,
                                   y=time_norm,
                                   n_x_coeffs=n_sin_coeffs_freqs,
                                   n_y_coeffs=n_sin_coeffs_time)
    
        cos_basis = construct_basis_vectorised(x=freq_norm,
                                   y=time_norm,
                                   n_x_coeffs=n_cos_coeffs_freqs,
                                   n_y_coeffs=n_cos_coeffs_time)
    elif basis_label == 'fourier':
        unc_basis = construct_fourier_basis(x=freq_norm,
                                   y=time_norm,
                                   n_x_coeffs=n_unc_coeffs_freqs,
                                   n_y_coeffs=n_unc_coeffs_time,
                                   Lx=Lf,
                                   Ly=Lt)
    
        sin_basis = construct_fourier_basis(x=freq_norm,
                                   y=time_norm,
                                   n_x_coeffs=n_sin_coeffs_freqs,
                                   n_y_coeffs=n_sin_coeffs_time,
                                   Lx=Lf,
                                   Ly=Lt)
    
        cos_basis = construct_fourier_basis(x=freq_norm,
                                   y=time_norm,
                                   n_x_coeffs=n_cos_coeffs_freqs,
                                   n_y_coeffs=n_cos_coeffs_time,
                                   Lx=Lf,
                                   Ly=Lt)
    else:
        raise ValueError(f"basis_label {basis_label} not recognised. Must be 'polynomial' or 'fourier'")
    
    if use_corrected:
        h_unc_spectra, h_cos_spectra, h_sin_spectra = construct_h_spectra_corrected(gamma_rec,
                                                                      switch_states=switch_states_array,
                                                                      switch_state_src_gamma_dict=switch_state_src_gamma_dict,
                                                                      internal_load_label=internal_load_label)
    else:
        h_unc_spectra, h_cos_spectra, h_sin_spectra = construct_h_spectra(gamma_rec,
                                                                        switch_states=switch_states_array,
                                                                        switch_state_src_gamma_dict=switch_state_src_gamma_dict)
    
    # flattened h_spectra of length n_data
    h_unc_spectra = h_unc_spectra.flatten()
    h_cos_spectra = h_cos_spectra.flatten()
    h_sin_spectra = h_sin_spectra.flatten()

    unc_basis = (unc_basis.T * h_unc_spectra).T
    cos_basis = (cos_basis.T * h_cos_spectra).T
    sin_basis = (sin_basis.T * h_sin_spectra).T

    h_matrix = np.concatenate((unc_basis,
                               cos_basis,
                               sin_basis),
                               axis=1)
    
    if return_norm_funcs:
        return h_matrix, (freq_norm_func, time_norm_func)
    else:
        return h_matrix
    

def construct_src_transfer_matrix(freq_array,
                                  time_array,
                                  freq_norm_func,
                                  time_norm_func,
                                  unc_poly_orders,
                                  cos_poly_orders,
                                  sin_poly_orders,
                                  gamma_rec,
                                  gamma_src_label,
                                  switch_state_src_gamma_dict:dict,
                                  internal_load_label='internal_load',
                                  use_corrected=False,
                                  basis_label='polynomial',
                                  Lf:float = 60.,
                                  Lt:float = 3600*24*4):
    freq_norm = freq_norm_func(freq_array)
    time_norm = time_norm_func(time_array)

    n_unc_coeffs_freqs, n_unc_coeffs_time = unc_poly_orders
    n_sin_coeffs_freqs, n_sin_coeffs_time = sin_poly_orders
    n_cos_coeffs_freqs, n_cos_coeffs_time = cos_poly_orders

    gamma_src_list = [gamma_src_label for _ in time_array] # fills list of gamma_src
    
    if basis_label == 'polynomial':
        unc_basis = construct_basis_vectorised(x=freq_norm,
                                    y=time_norm,
                                    n_x_coeffs=n_unc_coeffs_freqs,
                                    n_y_coeffs=n_unc_coeffs_time)
        
        sin_basis = construct_basis_vectorised(x=freq_norm,
                                    y=time_norm,
                                    n_x_coeffs=n_sin_coeffs_freqs,
                                    n_y_coeffs=n_sin_coeffs_time)
        
        cos_basis = construct_basis_vectorised(x=freq_norm,
                                    y=time_norm,
                                    n_x_coeffs=n_cos_coeffs_freqs,
                                    n_y_coeffs=n_cos_coeffs_time)
    elif basis_label == 'fourier':
        unc_basis = construct_fourier_basis(x=freq_norm,
                                    y=time_norm,
                                    n_x_coeffs=n_unc_coeffs_freqs,
                                    n_y_coeffs=n_unc_coeffs_time,
                                    Lx=Lf,
                                    Ly=Lt)
        
        sin_basis = construct_fourier_basis(x=freq_norm,
                                    y=time_norm,
                                    n_x_coeffs=n_sin_coeffs_freqs,
                                    n_y_coeffs=n_sin_coeffs_time,
                                    Lx=Lf,
                                    Ly=Lt)
        
        cos_basis = construct_fourier_basis(x=freq_norm,
                                    y=time_norm,
                                    n_x_coeffs=n_cos_coeffs_freqs,
                                    n_y_coeffs=n_cos_coeffs_time,
                                    Lx=Lf,
                                    Ly=Lt)
    else:
        raise ValueError(f"basis_label {basis_label} not recognised. Must be 'polynomial' or 'fourier'")

    if use_corrected:
        h_unc_spectra, h_cos_spectra, h_sin_spectra = construct_h_spectra_corrected(gamma_rec,
                                                                        switch_states=gamma_src_list,
                                                                        switch_state_src_gamma_dict=switch_state_src_gamma_dict,
                                                                        internal_load_label=internal_load_label)
    else:
        h_unc_spectra, h_cos_spectra, h_sin_spectra = construct_h_spectra(gamma_rec,
                                                                        switch_states=gamma_src_list,
                                                                        switch_state_src_gamma_dict=switch_state_src_gamma_dict)
    
    # flattened h_spectra of length n_data
    h_unc_spectra = h_unc_spectra.flatten()
    h_cos_spectra = h_cos_spectra.flatten()
    h_sin_spectra = h_sin_spectra.flatten()

    unc_basis = (unc_basis.T * h_unc_spectra).T
    cos_basis = (cos_basis.T * h_cos_spectra).T
    sin_basis = (sin_basis.T * h_sin_spectra).T

    h_matrix = np.concatenate((unc_basis,
                               cos_basis,
                               sin_basis),
                               axis=1)

    return h_matrix