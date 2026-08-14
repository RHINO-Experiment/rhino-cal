"""
Module for the Receiver object.
"""
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import read_s2p, interp_vals_to_new_freq, write_s2p
import astropy.units as un

class Receiver:
    """ A container for the receiver parameters.
    """
    def __init__(self,
                 gamma_rec = None,
                 freqs = None,
                 freqs_unit = un.Hz,
                 gamma_rec_s2p_path = None,
                 gains_s2p_path = None,
                 t_unc_scale = 300,
                 t_sin_scale = 20,
                 t_cos_scale = 20,
                 t_0_scale = 300,
                 bandwidth = 20*un.MHz,
                 centre_freq = 70*un.MHz,
                 n_freqs = 1024):
        
        if gamma_rec is None and gamma_rec_s2p_path is None:
            raise ValueError('gamma_rec must be specified explicietly or read in by s2p file.')

        # Assign gamma_rec
        if gamma_rec is None and freqs is None:
            self.gamma_rec, _,_,_, self.freqs = read_s2p(gamma_rec_s2p_path)
        else:
            self.gamma_rec, self.freqs = gamma_rec, freqs*freqs_unit


        if gains_s2p_path is not None:
            _, _, self.gains,_, gain_freqs = read_s2p(gains_s2p_path)
            self.gains = np.abs(self.gains) ** 2
            
            self.gains = interp_vals_to_new_freq(self.freqs,
                                                 gain_freqs,
                                                 self.gains)


            
        else:
            self.gains = np.ones_like(self.freqs)

    
        if not isinstance(self.freqs, un.Quantity):
            self.freqs = self.freqs * freqs_unit

        self.t_unc_scale = t_unc_scale
        self.t_cos_scale = t_cos_scale
        self.t_sin_scale = t_sin_scale
        self.t_0_scale = t_0_scale

        
    
    def change_freqs(self,
                     new_freqs):
        """ Function to change gamma_rec and freqs to new set.
        """
        new_gamma_rec = interp_vals_to_new_freq(new_freqs,
                                                 self.freqs,
                                                 self.gamma_rec)
        new_gains = interp_vals_to_new_freq(new_freqs,
                                                 self.freqs,
                                                 self.gains)
        _polyfit = np.polyfit(new_freqs.to(un.MHz).value, new_gains, deg=5)
        _polyfunc = np.poly1d(_polyfit)
        new_gains = _polyfunc(new_freqs.to(un.MHz).value)

        _polyfit = np.polyfit(new_freqs.to(un.MHz).value, new_gamma_rec, deg=5)
        _polyfunc = np.poly1d(_polyfit)
        new_gamma_rec = _polyfunc(new_freqs.to(un.MHz).value)

        self.gamma_rec = new_gamma_rec
        self.gains = new_gains

        self.freqs = new_freqs
    
    def export_to_s2p(self,
                      save_dir):
        """Export the gamma_rec to an s2p file.
        """
        filepath = f'{save_dir}/gamma_rec.s2p'
        
        s11 = self.gamma_rec
        s21 = np.sqrt(1 - np.abs(s11)**2)
        s12 = s21
        s22 = s11

        write_s2p(filepath, self.freqs.to(un.Hz).value, s11, s21, s12, s22, fmt="RI", freq_unit="HZ", z0=50)

