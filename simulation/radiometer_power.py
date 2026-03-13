"""
Module for calculating radiometer power
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as un

def compute_radiometer_power(t_src,
                             t_unc,
                             t_sin,
                             t_cos,
                             t_0,
                             gamma_rec,
                             gamma_src,
                             gain=1000,
                             add_noise=False,
                             t_int=1,
                             delta_nu=1e4):
    """Compute the radiometric noise power of a source given its
    temperature
    """

    f_src = np.sqrt(1 - (np.abs(gamma_rec)**2)) / (1 - (gamma_rec*gamma_src))

    p_src = gain * ((t_src * (1 - (np.abs(gamma_src)**2)) * np.abs(f_src)**2) + \
                    (t_unc * np.abs(gamma_src)**2 * np.abs(f_src)**2) + \
                    (t_cos * np.real(gamma_src * f_src)) + \
                    (t_sin * np.imag(gamma_src * f_src)) + \
                    t_0)
    if add_noise:

        if isinstance(t_int, un.Quantity):
            t_int = t_int.to(un.s)
        if isinstance(delta_nu,un.Quantity):
            delta_nu = delta_nu.to(un.Hz)
            
        delta_p = p_src / np.sqrt(t_int * delta_nu)
        noise = np.random.normal(loc=0, scale=delta_p)
        p_src = p_src + noise
        p_src = np.abs(p_src)

    return p_src