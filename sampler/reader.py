"""
Module for reading in observation files.
"""


import h5py
import numpy as np
import astropy.units as un
from utils.utils import read_s2p, interp_vals_to_new_freq, assign_states, load_dict_from_group
from pathlib import Path


BASE_TEMP_INDEX_DICT = {'heated':1,
                        'ambient':0}

class ObservationReader:
    """
    Container for observational data.
    """
    def __init__(self,
                 hdf5_filepath:str | Path,
                 temp_index_dict: dict=BASE_TEMP_INDEX_DICT,
                 frequency_unit: un.Quantity=un.MHz,
                 time_unit: un.Quantity=un.s,
                 frequency_lims: tuple[un.Quantity, un.Quantity] | None=None,
                 temperature_unit: un.Quantity=un.Celsius):
        

        with h5py.File(hdf5_filepath, mode='r') as f:
            self.freqs = f['sdr/sdr_freqs'][:]*frequency_unit # Read in and assign units
            self.times = f['sdr/sdr_times'][:]*time_unit
            self.data_waterfall = f['sdr/sdr_waterfall'][:]
            self.switch_states = f['switches/switch_states'][:].astype(str)
            self.switch_times = f['switches/switch_times'][:]*time_unit
            self.temperatures = f['temperatures/temperatures'][:]#*temperature_unit
            self.temperature_times = f['temperatures/temperature_times'][:]*time_unit
            # Attempt reading in from config. Else use time and freq data.
            try:
                config_group = f['obs_config']
                self.obs_config = load_dict_from_group(config_group)
            except KeyError:
                self.obs_config = None

        # Sort out temperatures ---
        if temperature_unit == un.Celsius:
            self.temperatures += 273.15
            self.temperatures *= un.K
            
        else:
            self.temperatures *= un.K
        self.temperature_times = self.temperature_times.to(un.s).value # ensure unit conversion and return floats
        self.temperatures = self.temperatures.to(un.K).value

        # Sort out switches ---
        self.switch_times = self.switch_times.to(un.s).value
        self.switch_states = [str(s) for s in self.switch_states] # a list of switch states

        # Sort out data ---
        if frequency_lims is not None:
            freq_inclusion = (self.freqs >= frequency_lims[0]) & (self.freqs <- frequency_lims[-1]) # create a mask
            self.freqs = self.freqs[freq_inclusion]
            self.data_waterfall = self.data_waterfall[:,freq_inclusion]
            
        self.times = self.times.to(un.s).value
        self.freqs = self.freqs.to(un.Hz).value

        self.ntimes = len(self.times)
        self.nfreqs = len(self.freqs)
        self.ndata = self.ntimes * self.nfreqs

        # Assign switch states
        self.states_array = assign_states(self.times,
                                          self.switch_times,
                                          self.switch_states)
        
        # Set up Theta vectors
        self.theta_dict = {}
        for state in np.unique(self.states_array):
            theta = np.where(self.states_array == state, 1, 0) # (n_times, 1)
            self.theta_dict[state]=theta
        
        # Sort out temperature dict
        self.temperatures_dict = {}
        for label, index in temp_index_dict.items():
            temps = self.temperatures[:,index]
            self.temperatures_dict[label] = np.interp(self.times, self.temperature_times, temps)
        self.temperatures = self.temperatures_dict
        del self.temperatures_dict

        # Use the config file to assign white noise variance.
        try:
            delta_nu = self.obs_config['sdr']['bandwidth'] / self.obs_config['sdr']['nChannels']
            delta_t = self.obs_config['sdr']['sampleIntegrationTime']
        except:
            delta_nu = (self.freqs[1] - self.freqs[0]) / len(self.freqs)
            delta_t = (self.times[-1] - self.times[0]) / len(self.times)
        self.fractional_data_variance = 1 / (delta_nu * delta_t) # single float

        self.Nw = self.fractional_data_variance * np.ones_like(self.data_waterfall) # (n_times, n_freqs)
        
    def flag(self,
             flag_dict: dict,
             prior_freq_mask: np.ndarray | None = None,
             whole_channel_flag_threshold: float = 0.4,
             whole_time_flag_threshold: float = 0.5,
             time_axis: int = 0):
        from rfi_flagging.rfi_flagging import mask_by_state

        self.mask = mask_by_state(self.data_waterfall,
                                  self.states_array,
                                  flag_dict,
                                  prior_freq_mask,
                                  whole_channel_flag_threshold,
                                  whole_time_flag_threshold,
                                  time_axis)


### Class For Reflection Terms
class ReflectionReader:
    def __init__(self,
                 component_dict: dict,
                 output_frequencies: np.ndarray|un.Quantity,
                 output_frequency_unit: un.Quantity = un.Hz,
                 polynomial_interp=False,
                 polynomial_fitting_order=5):
        
        # Check if receiver is included
        assert 'receiver' in component_dict.keys(),  "receiver should be in component_dict"
        if not isinstance(output_frequencies, un.Quantity):
            output_frequencies *= output_frequency_unit

        output_frequencies = output_frequencies.to(un.Hz).value

        self.s11_dict = {}
        for component, s2p_filepath in component_dict.items():
            s11,_,_,_, freq = read_s2p(s2p_filepath)

            # cut out unneeded frequency data
            s11 = s11[(freq > min(output_frequencies)) & (freq < max(output_frequencies))]
            freq = freq[(freq > min(output_frequencies)) & (freq < max(output_frequencies))]

            if polynomial_interp:
                polyfit = np.polyfit(x=freq,
                                     y=s11,
                                     deg=polynomial_fitting_order)
                polyfunc = np.poly1d(polyfit)
                self.s11_dict[component] = polyfunc(output_frequencies)
            else:
                self.s11_dict[component] = np.interp(x=output_frequencies,
                                                     xp=freq,
                                                     fp=s11)

        # Set up componant noise wave spectra

        components = list(component_dict.keys())
        components.remove('receiver') # remove receiver leaving only targets

        gamma_rec = self.s11_dict['receiver']

        self.cs_dict = {}
        self.kappa_unc_dict = {}
        self.kappa_sin_dict = {}
        self.kappa_cos_dict = {}
        for comp in components:
            gamma_src = self.s11_dict[comp]
            self.cs_dict[comp]=cs(gamma_src, gamma_rec)
            self.kappa_unc_dict[comp]=kappa_unc(gamma_src, gamma_rec)
            self.kappa_sin_dict[comp]=kappa_sin(gamma_src, gamma_rec)
            self.kappa_cos_dict[comp]=kappa_cos(gamma_src, gamma_rec)



def F(gamma_src, gamma_rec):
    return np.sqrt(1 - np.power(np.abs(gamma_rec),2)) / (1 - (gamma_rec*gamma_src))

def cs(gamma_src, gamma_rec):
    return (1 - np.power(np.abs(gamma_src), 2))*np.power(np.abs(F(gamma_src, gamma_rec)),2)


def cs(gamma_src, gamma_rec):
    return (1 - np.power(np.abs(gamma_src), 2)) *  np.power(np.abs(F(gamma_src,gamma_rec)),2)

def kappa_unc(gamma_src, gamma_rec):
    return np.power(np.abs(gamma_src),2)*np.power(np.abs(F(gamma_src, gamma_rec)),2)

def kappa_cos(gamma_src, gamma_rec):
    return np.real(gamma_src*F(gamma_src, gamma_rec))

def kappa_sin(gamma_src, gamma_rec):
    return np.imag(gamma_src*F(gamma_src, gamma_rec))