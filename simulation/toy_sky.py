import numpy as np
import astropy.units as un
from utils.utils import read_s2p, interp_vals_to_new_freq, write_s2p
from astropy.constants import c

class ToySky:
    """ Class for parameters of a Toy Sky.

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
                 enr=None,
                 synchrotron_amplitude_210=180,
                 synchrotron_beta=-2.6,):
        
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
                    
                    s11_termination = np.asarray(s11_termination)

                    self.gamma_src = s21 * s21 * s11_termination / (1 - (0*s11_termination))
                else:
                    self.gamma_src = s11_termination
            else:
                if enr is not None:
                    self.gamma_src = gamma_src * enr
                else:
                    self.gamma_src = gamma_src
        if freqs is not None:
            self.freqs = freqs
        self.synchrotron_amplitude_210 = synchrotron_amplitude_210
        self.synchrotron_beta = synchrotron_beta
        self.synchrotron_nu_210 = 210*un.MHz
        
        self.t_src = synchrotron_temperatures(self.freqs,
                                              T_210=self.synchrotron_amplitude_210,
                                              beta=self.synchrotron_beta,
                                              nu_210=self.synchrotron_nu_210)


    def change_freqs(self,
                     new_freqs):
        """ Function to change gamma_rec and freqs to new set.
        """

        self.gamma_src = interp_vals_to_new_freq(new_freqs,
                                                 self.freqs,
                                                 self.gamma_src)
        
        self.t_src = synchrotron_temperatures(new_freqs,
                                              T_210=self.synchrotron_amplitude_210,
                                              beta=self.synchrotron_beta,
                                              nu_210=self.synchrotron_nu_210)

        self.freqs = new_freqs
    
    def export_to_s2p(self,
                      save_dir):
        """Export the gamma_src to an s2p file.
        """
        filepath = f'{save_dir}/{self.label}_gamma_src.s2p'

        s11 = self.gamma_src
        s21 = np.sqrt(1 - np.abs(s11)**2)
        s12 = s21
        s22 = s11

        write_s2p(filepath, self.freqs.to(un.Hz).value, s11, s21, s12, s22, fmt="RI", freq_unit="HZ", z0=50)


def synchrotron_temperatures(freqs,
                           T_210=180,
                           beta=-2.6,
                           nu_210=210*un.MHz):
    """ Function to calculate the synchrotron temperatures at a given frequency.
    """
    if not isinstance(freqs, un.Quantity):
        freqs *= un.Hz
    
    T_synch = T_210 * np.power((freqs/nu_210), beta)
    
    return T_synch