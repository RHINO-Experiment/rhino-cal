"""
Module for the Receiver object.
"""

import numpy as np
from utils.utils import read_s2p, interp_vals_to_new_freq
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
        self.gamma_rec = interp_vals_to_new_freq(new_freqs,
                                                 self.freqs,
                                                 self.gamma_rec)
        self.gains = interp_vals_to_new_freq(new_freqs,
                                                 self.freqs,
                                                 self.gains)
        self.freqs = new_freqs

