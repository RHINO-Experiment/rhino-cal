"""
Module for the load class.
"""

import numpy as np
import astropy.units as un
from utils.utils import read_s2p, interp_vals_to_new_freq
from astropy.constants import c

class Load:
    """ Class for parameters of a given load.
    """
    def __init__(self,
                 physical_temperature = None,
                 gamma_src = None,
                 label = None,
                 freqs = None,
                 gamma_src_s2p_path = None,
                 termination_type = None,
                 termination_impedence = 50*un.Ohm,
                 effective_cable_length = None,
                 cable_loss = 1,
                 characteristic_impedence=50*un.Ohm,
                 enr=None):
        
        if label is None:
            self.label=''
        else:
            self.label=label

        if gamma_src_s2p_path is not None:
            self.gamma_src,_,_,_,self.freqs= read_s2p(gamma_src_s2p_path)
            self.freqs = self.freqs * un.Hz
        
        else:
            assert freqs is not None, 'Must Specify Freqs'
            if gamma_src is None:
                if not isinstance(freqs, un.Quantity):
                    freqs *= un.Hz
                
                self.freqs = freqs

                if termination_type == 'open':
                    s11_termination = np.ones(shape=freqs.shape) * 1. * un.dimensionless_unscaled
                
                elif termination_type == 'short':
                    s11_termination = np.ones(shape=freqs.shape) * -1. * un.dimensionless_unscaled
                
                elif termination_type == 'matched':
                    s11_termination = np.zeros(shape=freqs.shape) * un.dimensionless_unscaled

                else:
                    try:
                        if not isinstance(termination_impedence, un.Quantity):
                            termination_impedence*=un.Ohm
                        s11_termination = np.ones_like(freqs) * (termination_impedence - characteristic_impedence) \
                / (termination_impedence + characteristic_impedence)
                    except ValueError:
                        print('Must specify termination impedence')
                
                if effective_cable_length is not None:
                    
                    if not isinstance(effective_cable_length, un.Quantity):
                        effective_cable_length *= un.m
                    
                    phase = -(2*np.pi * effective_cable_length * self.freqs) / (c)
                    
                    phase = phase.to(un.dimensionless_unscaled)
                    phase = np.asarray(phase)
                    s21 = cable_loss * np.exp(1j * phase)
                    
                    print("s11_termination", s11_termination)
                    s11_termination = np.asarray(s11_termination)

                    self.gamma_src = s21 * s21 * s11_termination / (1 - (0*s11_termination))
                else:
                    self.gamma_src = s11_termination
            else:
                if enr is not None:
                    self.gamma_src = gamma_src * enr
                else:
                    self.gamma_src = gamma_src
        self.t_src = physical_temperature

    def change_freqs(self,
                     new_freqs):
        """ Function to change gamma_rec and freqs to new set.
        """

        self.gamma_src = interp_vals_to_new_freq(new_freqs,
                                                 self.freqs,
                                                 self.gamma_src)
        self.freqs = new_freqs