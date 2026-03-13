import numpy as np

def construct_data_i(p_src,
                     p_l,
                     p_ns,
                     T_src,
                     T_l,
                     T_ns,
                     Gamma_rec,
                     Gamma_src):
    F_src = np.sqrt(1 - (np.abs(Gamma_rec)**2)) / (1 - (Gamma_rec*Gamma_src))
    
    d = ((p_src - p_l) / (p_ns - p_l)) * (T_ns - T_l) * (1 - (np.abs(Gamma_rec)**2)) - \
        (T_src * (1 - (np.abs(Gamma_src)**2)) * (np.abs(F_src)**2)) + \
        (T_l * (1 - (np.abs(Gamma_rec)**2)))

    return d

def construct_basic_H_matrix(gamma_rec, gamma_src_list):
    n_src = len(gamma_src_list)
    H = np.ones(shape=(n_src, 3))

    for i, gamma_src in enumerate(gamma_src_list):
        F_src = F_src = np.sqrt(1 - (np.abs(gamma_rec)**2)) / (1 - (gamma_rec*gamma_src))
        alpha = np.angle(F_src * gamma_src)
        H[i,0] = (np.abs(F_src)**2) * (np.abs(gamma_src)**2)
        H[i,1] = np.abs(gamma_src)*np.abs(F_src)*np.cos(alpha)
        H[i,2] = np.abs(gamma_src)*np.abs(F_src)*np.sin(alpha)
    return H


def quadrature_data_variance(p_src,
                             p_l,
                             p_ns,
                             T_src,
                             T_l,
                             T_ns,
                             Gamma_rec,
                             Gamma_src,
                             t_int,
                             delta_nu,
                             temp_sens_variance):
    F_src = np.sqrt(1 - (np.abs(Gamma_rec)**2)) / (1 - (Gamma_rec*Gamma_src))

    dd_dp_src_sqr = np.power( (T_ns -T_l) * (1 - np.power(np.abs(Gamma_rec),2)) / (p_ns - p_l) ,2)

    dd_dp_ns_sqr = np.power( ((p_src-p_l)*(T_ns-T_l) * (1 - np.power(np.abs(Gamma_rec), 2))) / np.power(p_ns-p_l, 2)  ,2)

    dd_dp_l_sqr = np.power(((T_ns-T_l) * (1 - np.power(np.abs(Gamma_rec) ,2)) * (p_src - p_ns) ) / np.power(p_ns - p_l, 2) ,2)

    dd_dt_src_sqr = np.power(1 - np.power(np.abs(Gamma_src),2) ,2) * np.power(np.abs(F_src), 4)

    dd_dt_ns_sqr = np.power(((p_src - p_l) * (1 - np.power(np.abs(Gamma_rec), 2))) / (p_ns - p_l) , 2)

    dd_dt_l_sqr = np.power((1 - np.power(np.abs(Gamma_rec),2)) - ((p_src - p_l) * (1 - np.power(np.abs(Gamma_rec),2)) / (p_ns - p_l)) ,2)

    var_p_src = (p_src**2) / (delta_nu*t_int)
    var_p_l = (p_l**2) / (delta_nu*t_int)
    var_p_ns = (p_ns**2) / (delta_nu*t_int)
    
    var_d = (var_p_src * dd_dp_src_sqr) + (var_p_ns * dd_dp_ns_sqr) + (var_p_l * dd_dp_l_sqr) + \
            (temp_sens_variance * dd_dt_src_sqr) + (temp_sens_variance * dd_dt_ns_sqr) + (temp_sens_variance * dd_dt_l_sqr)

    return var_d







def construct_d_vector_and_cov_matrix(p_src,
                             p_l,
                             p_ns,
                             T_src,
                             T_l,
                             T_ns,
                             Gamma_rec,
                             Gamma_src,
                             t_int,
                             delta_nu,
                             temp_sens_variance,
                             d_vector_only=False,
                             noise_covariance_only=False,
                             inverse_noise_covariance=True):
    """
    Produces data vector and noise-covariance matrix for input measurements

    Lengths of lists must be equal
    
    :param p_src: array-like measurements corresponding to each src
    :param p_l: Description
    :param p_ns: Description
    :param T_src: Description
    :param T_l: Description
    :param T_ns: Description
    :param Gamma_rec: Description
    :param Gamma_src: Description
    :param t_int: Description
    :param delta_nu: Description
    :param temp_sens_variance: Description
    :param d_vector_only: Description
    :param noise_covariance_only: Description
    """
    d = construct_data_i(p_src,
                             p_l,
                             p_ns,
                             T_src,
                             T_l,
                             T_ns,
                             Gamma_rec,
                             Gamma_src)
    
    if d_vector_only:
        return d
    # creates diagonal N matrix
    if not inverse_noise_covariance:
        noise_covariance = np.diag(quadrature_data_variance(p_src,
                                                        p_l,
                                                        p_ns,
                                                        T_src,
                                                        T_l,
                                                        T_ns,
                                                        Gamma_rec,
                                                        Gamma_src,
                                                        t_int,
                                                        delta_nu,
                                                        temp_sens_variance))
        
    else:
        noise_covariance = np.diag(1 / quadrature_data_variance(p_src,
                                                        p_l,
                                                        p_ns,
                                                        T_src,
                                                        T_l,
                                                        T_ns,
                                                        Gamma_rec,
                                                        Gamma_src,
                                                        t_int,
                                                        delta_nu,
                                                        temp_sens_variance))
        
    noise_covariance = quadrature_data_variance(p_src,
                                                        p_l,
                                                        p_ns,
                                                        T_src,
                                                        T_l,
                                                        T_ns,
                                                        Gamma_rec,
                                                        Gamma_src,
                                                        t_int,
                                                        delta_nu,
                                                        temp_sens_variance)

    if noise_covariance_only:
        return noise_covariance
    else:
        return d, noise_covariance


def polynomial_basis(freq_array, polynomial_order, reference_frequency=None):
    polynomial_order = polynomial_order+1
    if reference_frequency is not None:
        freq_array = freq_array / reference_frequency

    basis = np.vander(freq_array, polynomial_order)
    return basis


def return_spectrum_polynomial(fitted_params,
                               unc_polyorder,
                               cos_polyorder,
                               sin_polyorder, reference_freq=0):
    unc_params = fitted_params[:unc_polyorder+1]
    cos_params = fitted_params[unc_polyorder+1:unc_polyorder+sin_polyorder+2]
    sin_params = fitted_params[unc_polyorder+cos_polyorder+2:]

    def unc_func(freqs):
        uncf = np.poly1d(unc_params)
        if reference_freq is None:
            return uncf(freqs)
        else:
            return uncf(freqs / reference_freq)
    def cos_func(freqs):
        cosf = np.poly1d(cos_params)
        if reference_freq is None:
            return cosf(freqs)
        else:
            return cosf(freqs / reference_freq)
    def sin_func(freqs):
        sinf = np.poly1d(sin_params)
        if reference_freq is None:
            return sinf(freqs)
        else:
            return sinf(freqs / reference_freq)
    
    return unc_func, cos_func, sin_func



def construct_h_spectra(gamma_rec, gamma_src_list):
    h_unc = []
    h_cos = []
    h_sin = []
    
    for gamma_src in gamma_src_list:
        F_src = np.sqrt(1 - (np.abs(gamma_rec)**2)) / (1 - (gamma_rec*gamma_src))
        alpha = np.angle(F_src * gamma_src)
        h_unc.append((np.abs(F_src)**2) * (np.abs(gamma_src)**2))
        h_cos.append(np.abs(gamma_src)*np.abs(F_src)*np.cos(alpha))
        h_sin.append(np.abs(gamma_src)*np.abs(F_src)*np.sin(alpha))
    return h_unc, h_cos, h_sin




def construct_H_matrix(freq_array,
                       gamma_src_list,
                       gamma_rec,
                       unc_polyorder,
                       cos_polyorder,
                       sin_polyorder,
                       reference_frequency=None):
    """"""

    n_params = unc_polyorder + cos_polyorder + sin_polyorder + 3
    n_d = len(gamma_src_list) * len(freq_array) # number of data points. Assuming that only one cycle has been used

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



def construct_gcr_lhs(H_matrix, N_inv_1d, S_inv):
    # Assume N_inv is diagonal

    return H_matrix.T @ (N_inv_1d[:,None] * H_matrix) + S_inv


def construct_gcr_rhs(H_matrix, N_inv_1d, S_inv, d, T_nw0, omega_t, omega_d):
    data_term = (H_matrix.T @ (N_inv_1d * d)) +\
        H_matrix.T @ (np.sqrt(N_inv_1d)*omega_d)
    prior_term = (S_inv * T_nw0) + (np.sqrt(S_inv)*omega_t)

    return data_term + prior_term
