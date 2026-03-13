"""
Utility Functions for producing receiver simulations.
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as un
import h5py
from astropy.time import Time

def return_measurements_i(T_l,
                          T_ns,
                          T_src,
                          gamma_rec,
                          gamma_src_list,
                          gamma_ns,
                          gamma_l,
                          t_int,
                          delta_nu,
                          T_unc,
                          T_cos,
                          T_sin,
                          T_0,
                          gain,
                          temp_sens_variance,
                          add_noise=True,
                          obs_freqs=None):
    
    if obs_freqs is None:
        n_freqs = len(gamma_rec)
    else:
        n_freqs = len(obs_freqs)

    n_meas = len(gamma_src_list)

    p_src = np.ones(shape=(n_meas, n_freqs))
    p_ns = np.ones(shape=(n_meas, n_freqs))
    p_l = np.ones(shape=(n_meas, n_freqs))

    T_src_meas = np.ones(shape=(n_meas,))
    T_ns_meas = np.ones(shape=(n_meas,))
    T_l_meas = np.ones(shape=(n_meas,))

    for i in range(n_meas):
        p_src[i] = compute_radiometer_power(T_src,
                                            T_unc,
                                            T_sin,
                                            T_cos,
                                            T_0,
                                            gamma_rec,
                                            gamma_src_list[i],
                                            gain,
                                            add_noise=add_noise,
                                            t_int=t_int,
                                            delta_nu=delta_nu)
        p_ns[i] = compute_radiometer_power(T_ns,
                                           T_unc,
                                           T_sin,
                                           T_cos,
                                           T_0,
                                           gamma_rec,
                                           gamma_ns,
                                           gain,
                                           add_noise=add_noise,
                                           t_int=t_int,
                                           delta_nu=delta_nu)
        p_l[i] = compute_radiometer_power(T_l,
                                          T_unc,
                                          T_sin,
                                          T_cos,
                                          T_0,
                                          gamma_rec,
                                          gamma_l,
                                          gain,
                                          add_noise=add_noise,
                                          t_int=t_int,
                                          delta_nu=delta_nu)
        
        if add_noise:
            T_src_meas[i] = np.abs(np.random.normal(T_src, scale=np.sqrt(temp_sens_variance)))
            T_ns_meas[i] = np.abs(np.random.normal(T_ns, scale=np.sqrt(temp_sens_variance)))
            T_l_meas[i] = np.abs(np.random.normal(T_l, scale=np.sqrt(temp_sens_variance)))
        else:
            T_src_meas[i] = T_src
            T_l_meas[i] = T_l
            T_ns_meas[i]=T_ns
        pass
    return {"T_src":T_src_meas,
                "T_L":T_l_meas,
                "T_NS":T_ns_meas,
                "P_src":p_src,
                "P_L":p_l,
                "P_NS":p_ns}



def compute_radiometer_power(T_src,
                             T_unc,
                             T_sin,
                             T_cos,
                             T_0,
                             Gamma_rec,
                             Gamma_src,
                             gain=1000,
                             add_noise=False,
                             t_int=1,
                             delta_nu=1e4):
    
    F_src = np.sqrt(1 - (np.abs(Gamma_rec)**2)) / (1 - (Gamma_rec*Gamma_src))

    p_src = gain * ((T_src * (1 - (np.abs(Gamma_src)**2)) * np.abs(F_src)**2) + \
                    (T_unc * np.abs(Gamma_src)**2 * np.abs(F_src)**2) + \
                    (T_cos * np.real(Gamma_src * F_src)) + \
                    (T_sin * np.imag(Gamma_src * F_src)) + \
                    T_0)
    if add_noise:
        if isinstance(t_int, un.Quantity) and isinstance(delta_nu,un.Quantity):
            t_int = t_int.to(un.s)
            delta_nu = delta_nu.to(un.Hz)
        delta_p = p_src / np.sqrt(t_int * delta_nu)
        noise = np.random.normal(loc=0, scale=delta_p)
        p_src = p_src + noise
        p_src = np.abs(p_src)

    return p_src


class TerminatedCable:
    def __init__(self, physical_temperature, frequencies, termination='Open', epsilon=1,cable_length=10, mag_s12=1, termination_reflection_coeffs=1):
        self.termination_type = termination
        c = 299792458
        self.frequencies = frequencies
        self.temperatures = physical_temperature * np.ones(len(self.frequencies))
        if termination=='Open':
            self.reflection_coefficients = (mag_s12**2) * np.exp(-4j*np.pi*self.frequencies*cable_length*epsilon/c)
        elif termination=='Shorted':
            self.reflection_coefficients = -(mag_s12**2) * np.exp(-4j*np.pi*self.frequencies*cable_length*epsilon/c)
        else:
            self.reflection_coefficients = (mag_s12**2) * np.exp(-4j*np.pi*self.frequencies*cable_length*epsilon/c)* np.exp(-1j*np.pi) * termination_reflection_coeffs


def construct_data_i(p_src,
                     p_l,
                     p_ns,
                     T_src,
                     T_l,
                     T_ns,
                     Gamma_rec,
                     Gamma_src):

    F_src = np.sqrt(1 - (np.abs(Gamma_rec)**2)) / (1 - (Gamma_rec*Gamma_src))

    d = ((p_src * (1 - (np.abs(Gamma_rec)**2)) * (T_ns - T_l)) / \
        (p_ns - p_l)) - \
        (T_src * (1 - (np.abs(Gamma_src)**2)) * (np.abs(Gamma_src)**2) * (np.abs(F_src)**2))
    return d

def construct_new_data_i(p_src,
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
    variance = (temp_sens_variance * ((2 * (p_src**2 / ((p_ns - p_l)**2)) * ((1 - (np.abs(Gamma_rec)**2))**2)) + \
                                      ((1-(np.abs(Gamma_src)**2))**2)*(np.abs(F_src)**4))) +  \
                ((p_src**2 / ((p_ns - p_l)**2)) * ((T_ns - T_l)**2) * ((1 - (np.abs(Gamma_rec)**2))**2) * \
                 ((((p_ns+p_l)/(p_ns-p_l))**2)+1)) / (delta_nu*t_int)
    
    variance = (temp_sens_variance * ((2 * (p_src**2 / ((p_ns - p_l)**2)) * ((1 - (np.abs(Gamma_rec)**2))**2)) + \
                                      ((1-(np.abs(Gamma_src)**2))**2)*(np.abs(F_src)**4))) +  \
                (((p_src**2) / ((p_ns - p_l)**2)) * ((T_ns - T_l)**2) * ((1 - (np.abs(Gamma_rec)**2))**2) * \
                 (1+( ((p_ns**2) + (p_l**2)) / ((p_ns - p_l)**2) ))) / (delta_nu*t_int)

    return variance

def calculate_cable_params(length,
                           loss,
                           velocity_factor,
                           freq_array,
                           cable_s11 = 0,
                           cable_s22 = 0):
    phase = -(2*np.pi * length*freq_array) / (velocity_factor *3e8)
    
    s21 = s12 = loss * np.exp(1j * phase)


    return cable_s11, cable_s22, s12, s21


def get_gammas(termination_type:str,
               freq_array,
               termination_impedence=None,
               cable_params = None,
               characteristic_impedence = 50 # Ohms
               ):
    """"""
    if cable_params is None:
        int_s11, int_s22, int_s12, int_s21 = 0,0,1,1
    
    else:
        int_s11, int_s22, int_s12, int_s21 = cable_params

    if termination_type == 'open' or termination_type == 'Open':
        s11_final_end = np.ones_like(freq_array)
    elif termination_type == 'short' or termination_type ==  'Short':
        s11_final_end = -np.ones_like(freq_array)
    else:
        s11_final_end = np.ones_like(freq_array)*(termination_impedence - characteristic_impedence) \
            / (termination_impedence + characteristic_impedence)
    
    final_s11 = int_s11 + ((int_s12 * int_s21 * s11_final_end) / (1 - int_s22*s11_final_end)) # Monsalve

    return final_s11

def chebyshev_model_2d(
        freqs:np.ndarray,
        times:np.ndarray,
        n_freq_coeffs:int,
        n_time_coeffs:int,
        amp = float) -> np.ndarray:
    """"""

    coeffs = np.sqrt(amp) * np.random.randn(n_freq_coeffs,
                                            n_time_coeffs)
    
    xx = np.linspace(-1., 1., freqs.size)
    yy = np.linspace(-1, 1., times.size)

    x_grid, y_grid = np.meshgrid(xx, yy)

    return np.polynomial.chebyshev.chebval2d(x_grid,
                                             y_grid,
                                             coeffs)
