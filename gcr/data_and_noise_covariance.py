"""
Script for calculating the noise covariance for the data.
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as un
from rfi_flagging.rfi_flagging import convert_manual_frequency_mask, flag_waterfall_momentRFI

def construct_data(p_src,
                   p_l,
                   p_ns,
                   t_src,
                   t_l,
                   t_ns,
                   gamma_rec,
                   gamma_src):
    """Constructs the data vector for a given source"""
    f_src = np.sqrt(1 - (np.abs(gamma_rec)**2)) / (1 - (gamma_rec*gamma_src))

    d = ((p_src - p_l) / (p_ns - p_l)) * (t_ns - t_l) * (1 - (np.abs(gamma_rec)**2)) -\
        (t_src * (1 - (np.abs(gamma_src)**2)) * (np.abs(f_src)**2)) + \
        (t_l * (1 - (np.abs(gamma_rec)**2)))

    return d


def construct_data_corrected(p_src,
                   p_l,
                   p_ns,
                   t_src,
                   t_l,
                   t_ns,
                   gamma_rec,
                   gamma_src,
                   gamma_l,
                   gamma_ns):
    
    f_i = lambda gamma_i: np.sqrt(1 - (np.abs(gamma_rec)**2)) / (1 - (gamma_rec*gamma_i))

    c_i = lambda gamma_i: np.power(np.abs(f_i(gamma_i)), 2) * (1 - np.power(np.abs(gamma_i), 2))

    q = (p_src - p_l) / (p_ns - p_l)

    return  q*((c_i(gamma_ns)*t_ns) - (c_i(gamma_l)*t_l)) - (c_i(gamma_src)*t_src) + (c_i(gamma_l)*t_l)



def quadrature_data_variance_calc(p_src, # array like
                             p_l,
                             p_ns,
                             t_l,
                             t_ns,
                             gamma_rec,
                             gamma_src,
                             t_int_p_src,
                             t_int_p_l,
                             t_int_p_ns,
                             delta_nu,
                             temp_sens_variance):
    """Calculates the quadrature variance on the data sample.
    """
    f_src = np.sqrt(1 - (np.abs(gamma_rec)**2)) / (1 - (gamma_rec*gamma_src))

    dd_dp_src_sqr = np.power( (t_ns -t_l) * (1 - np.power(np.abs(gamma_rec),2)) / (p_ns - p_l) ,2)

    dd_dp_ns_sqr = np.power( ((p_src-p_l)*(t_ns-t_l) * (1 - np.power(np.abs(gamma_rec), 2))) / np.power(p_ns-p_l, 2)  ,2)

    dd_dp_l_sqr = np.power(((t_ns-t_l) * (1 - np.power(np.abs(gamma_rec) ,2)) * (p_src - p_ns) ) / np.power(p_ns - p_l, 2) ,2)

    dd_dt_src_sqr = np.power(1 - np.power(np.abs(gamma_src),2) ,2) * np.power(np.abs(f_src), 4)

    dd_dt_ns_sqr = np.power(((p_src - p_l) * (1 - np.power(np.abs(gamma_rec), 2))) / (p_ns - p_l) , 2)

    dd_dt_l_sqr = np.power((1 - np.power(np.abs(gamma_rec),2)) - ((p_src - p_l) * (1 - np.power(np.abs(gamma_rec),2)) / (p_ns - p_l)) ,2)

    var_p_src = (p_src**2) / (delta_nu*t_int_p_src)
    var_p_l = (p_l**2) / (delta_nu*t_int_p_l)
    var_p_ns = (p_ns**2) / (delta_nu*t_int_p_ns)
    
    var_d = (var_p_src * dd_dp_src_sqr) + (var_p_ns * dd_dp_ns_sqr) + (var_p_l * dd_dp_l_sqr) +\
            (temp_sens_variance * dd_dt_src_sqr) + (temp_sens_variance * dd_dt_ns_sqr) + (temp_sens_variance * dd_dt_l_sqr)
    return var_d


def quadrature_data_variance_calc_corrected(p_src, # array like
                             p_l,
                             p_ns,
                             t_l,
                             t_ns,
                             gamma_rec,
                             gamma_l,
                             gamma_ns,
                             gamma_src,
                             t_int_p_src,
                             t_int_p_l,
                             t_int_p_ns,
                             delta_nu,
                             temp_sens_variance):
    f_i = lambda gamma_i: np.sqrt(1 - (np.abs(gamma_rec)**2)) / (1 - (gamma_rec*gamma_i))

    c_i = lambda gamma_i: np.power(np.abs(f_i(gamma_i)), 2) * (1 - np.power(np.abs(gamma_i), 2))

    c_src, c_l, c_ns = c_i(gamma_src), c_i(gamma_l), c_i(gamma_ns)

    q = (p_src - p_l) / (p_ns - p_l)

    t_int = t_int_p_src + t_int_p_ns + t_int_p_l

    phi_src = t_int_p_src / t_int
    phi_l = t_int_p_l / t_int
    phi_ns = t_int_p_ns / t_int

    q_var = np.power(q,2) * (
        (np.power(p_src, 2) / (phi_src*np.power(p_src-p_l, 2)) ) + \
        (np.power(p_ns, 2) / (phi_ns * np.power(p_ns-p_l, 2)) ) + \
        ((np.power(p_l, 2)*np.power(p_src - p_ns, 2)) / (phi_l * (np.power(p_ns-p_l, 2)*np.power(p_src - p_l, 2))) )
    ) / (delta_nu * t_int)

    dd_dq_sqr = np.power((c_ns*t_ns) - (c_l*t_l), 2)
    dd_dtns_sqr = np.power(q*c_ns, 2)
    dd_dtl_sqr = np.power(c_l - (q*c_l), 2)
    dd_dtsrc_sqr = np.power(c_src, 2)

    d_var = (q_var * dd_dq_sqr) + (temp_sens_variance * dd_dtl_sqr) + \
          (temp_sens_variance * dd_dtns_sqr) + (temp_sens_variance* dd_dtsrc_sqr)
    return d_var
    


def return_data_and_variance(p_src_array,
                         p_src_array_times,
                         p_l_array,
                         p_l_array_times,
                         p_ns_array,
                         p_ns_array_times,
                         t_src_array,
                         t_l_array,
                         t_ns_array,
                         gamma_rec,
                         gamma_src,
                         gamma_l,
                         gamma_ns,
                         freqs,
                         dt,
                         frequency_prior_mask=None,
                         temp_sens_variance=None,
                         freq_unit=un.MHz,
                         time_unit=un.s,
                         use_corrected=False,
                         p_src_flagger=None,
                         p_src_flagger_params={'total_freq_threshold':0.4,
                                               'total_time_threshold':0.4},
                         p_l_flagger=None,
                         p_l_flagger_params={'total_freq_threshold':0.4,
                                               'total_time_threshold':0.4},
                         p_ns_flagger=None,
                         p_ns_flagger_params={'total_freq_threshold':0.4,
                                               'total_time_threshold':0.4}):
    """Utility function to calculate the data and expected 
    variance from blocks of the arrays.
    """
    ### Enter data flagging step here if needed
    delta_nu = (freqs[-1] - freqs[0]) / len(freqs)
    if not isinstance(freqs, un.Quantity):
        delta_nu *= freq_unit
    
    delta_nu = delta_nu.to(un.Hz)

    p_src, p_src_t_int = return_mean_and_effective_tint(p_src_array,
                                                        dt,
                                                        p_src_flagger,
                                                        p_src_flagger_params,
                                                        frequency_prior_mask)

    p_l, p_l_t_int = return_mean_and_effective_tint(p_l_array,
                                                    dt,
                                                    p_l_flagger,
                                                    p_l_flagger_params,
                                                    frequency_prior_mask)
    
    p_ns, p_ns_t_int = return_mean_and_effective_tint(p_ns_array,
                                                    dt,
                                                    p_ns_flagger,
                                                    p_ns_flagger_params,
                                                    frequency_prior_mask)

    t_src = np.mean(t_src_array)
    if temp_sens_variance is None:
        temp_sens_variance = np.std(t_src_array)

    t_l = np.mean(t_l_array)
    t_ns = np.mean(t_ns_array)

    temp_sens_variance = max(temp_sens_variance,
                             np.std(t_l_array),
                             np.std(t_ns_array))

    median_time = np.median(np.concatenate([p_src_array_times,
                                     p_ns_array_times,
                                     p_l_array_times]))

    if isinstance(median_time, un.Quantity):
        median_time = float(median_time.to(un.s) / un.s)

    if use_corrected:
        variance = quadrature_data_variance_calc_corrected(p_src, # array like
                             p_l,
                             p_ns,
                             t_l,
                             t_ns,
                             gamma_rec,
                             gamma_l,
                             gamma_ns,
                             gamma_src,
                             p_src_t_int,
                             p_l_t_int,
                             p_ns_t_int,
                             delta_nu,
                             temp_sens_variance)
        
        data = construct_data_corrected(p_src,
                   p_l,
                   p_ns,
                   t_src,
                   t_l,
                   t_ns,
                   gamma_rec,
                   gamma_src,
                   gamma_l,
                   gamma_ns)
        
    else:
        variance = quadrature_data_variance_calc(p_src,
                                                p_l,
                                                p_ns,
                                                t_l,
                                                t_ns,
                                                gamma_rec,
                                                gamma_src,
                                                p_src_t_int,
                                                p_l_t_int,
                                                p_ns_t_int,
                                                delta_nu,
                                                temp_sens_variance)

        data = construct_data(p_src,
                            p_l,
                            p_ns,
                            t_src,
                            t_l,
                            t_ns,
                            gamma_rec,
                            gamma_src)
    # These will be masked flagged arrays
    return data, variance, median_time


def return_mean_and_effective_tint(waterfall,
                                   dt,
                                   flagger=None,
                                   flagger_params=None,
                                   prior_mask=None,
                                   time_unit=un.s):
    """
    Returns the mean of the watefall spectra as well
    as the effective integration time.
    Inputs:
        waterfall - (np.ndarray)(n_times, n_freqs)
            watefall of spectra
        dt - (float or un.Quantity)
            effective integration time per sample
        flagger - (MomentRFI.IterativeSurfaceFitter)
        flagger_params - (dict)
        time_unot (un.Quantity)
    Outputs
        - mean_spectra (np.ma.array) (n_freqs)
        - effective_t_int (np.array) (n_freqs)
    """
    
    if flagger is None:
        if not np.ma.is_masked(waterfall):
            waterfall = np.ma.array(waterfall, mask=np.zeros_like(waterfall,
                                                                  dtype=bool))
    else:
        assert flagger_params is not None, 'Need Flagging params for flagging'
        if prior_mask is not None:
            prior_mask = convert_manual_frequency_mask(prior_mask,
                                                       waterfall.shape)
            
        mask = flag_waterfall_momentRFI(waterfall,
                                        flagger,
                                        prior_mask,
                                        flagger_params['total_freq_threshold'],
                                        flagger_params['total_time_threshold'])
        waterfall = np.ma.array(waterfall, mask=~mask)

    mean_spectra = np.mean(waterfall, axis=0)
    n_samp_eff = np.sum(~waterfall.mask, axis=0)
    effective_t_int = dt * n_samp_eff

    if not isinstance(effective_t_int, un.Quantity):
        effective_t_int *= time_unit
    effective_t_int = effective_t_int.to(un.s)

    return mean_spectra, effective_t_int