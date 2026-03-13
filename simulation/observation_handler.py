"""
Module for the observation handler.
"""

import numpy as np
import astropy.units as un
from astropy.time import Time
import h5py
from .receiver import Receiver
from .loads import Load
from utils.utils import set_up_switch_cycle_indices, assign_states, chebyshev_model, chebyshev_model_2d
from .radiometer_power import compute_radiometer_power
import h5py

class ObservationHandler:
    """Handler Class for simulating the simulated receiver observations.

    Will return the RHINO style .hdf5 observation files to properly mimic overvations.
    """

    def __init__(self,
                 observation_length:un.Quantity,
                 receiver:Receiver,
                 bandwidth = None,
                 delta_t = None,
                 n_freqs = None,
                 n_times = None,
                 n_t_unc_freq_coeffs:int = 1,
                 n_t_sin_freq_coeffs:int = 1,
                 n_t_cos_freq_coeffs:int = 1,
                 n_t_unc_time_coeffs:int = 1,
                 n_t_sin_time_coeffs:int = 1,
                 n_t_cos_time_coeffs:int = 1,
                 n_t_0_freq_coeffs: int=1,
                 n_t_0_time_coeffs: int=1,
                 t_unc_amp = 10*un.K,
                 t_cos_amp = 10*un.K,
                 t_sin_amp = 10*un.K,
                 t_0_amp = 10*un.K,
                 temp_sens_variance = 0.01,
                 sources = None,
                 dicke_switch_sources = None,
                 dicke_switch_length = None,
                 centre_freq = None,
                 reference_time = Time(1768906853.5738704,
                                       format='unix',
                                         scale='utc')
                 ):
        # Validate Frequencies
        self.reference_time = reference_time.unix*un.s

        self._n_freq_calc(n_freqs,
                          receiver,
                          bandwidth,
                          centre_freq)
        # Validate Times
        self._n_time_calc(observation_length,
                          delta_t,
                          n_times)
        

        print('reference time')
        print(float(reference_time.unix)*un.s)

        print('self.times')
        print(self.times)

        #self.delta_t = (self.times[-1] - self.times[0]) / self.n_times
        self.delta_nu = (self.freqs[-1] - self.freqs[0]) / self.n_freqs

        self._coeff_handler(n_t_unc_freq_coeffs,
                            n_t_sin_freq_coeffs,
                            n_t_cos_freq_coeffs,
                            n_t_unc_time_coeffs,
                            n_t_sin_time_coeffs,
                            n_t_cos_time_coeffs,
                            n_t_0_freq_coeffs,
                            n_t_0_time_coeffs)

        # if receiver.freqs != self.freqs:
        #     receiver.change_freqs(self.freqs)

        self.data = np.empty(shape=(self.n_times, self.n_freqs))

        # set up sources for switching
        if not isinstance(sources, list):
            sources = [sources]
        # make source into a list for dicke switching if 1

        if dicke_switch_sources is None and len(sources) == 1:
            self.switch_states = None # Now switching
            self.switch_times = None
            self.states_array = []
            for _ in range(self.n_times):
                self.states_array.append(sources[0].label)
            self.states_array = np.array(self.states_array, dtype=self.states_array)
        else:
            source_labels = [s.label for s in sources]
            dicke_switch_list = ['source']
            for ds in dicke_switch_sources:
                dicke_switch_list.append(ds.label)
        
            self.switch_states, self.switch_times = set_up_switch_cycle_indices(self.times[-1]-self.times[0],
                                                                                dicke_switch_length,
                                                                                source_labels,
                                                                                dicke_switch_list)
            self.states_array = assign_states(times=self.times,
                                              change_times=self.switch_times,
                                              change_states=self.switch_states)


        self.t_unc = receiver.t_unc_scale + chebyshev_model_2d(self.freqs,
                                                               self.times,
                                                               n_t_unc_freq_coeffs,
                                                               n_t_unc_time_coeffs,
                                                               t_unc_amp)
        
        self.t_cos = receiver.t_cos_scale + chebyshev_model_2d(self.freqs,
                                                               self.times,
                                                               n_t_cos_freq_coeffs,
                                                               n_t_cos_time_coeffs,
                                                               t_cos_amp)
        
        self.t_sin = receiver.t_sin_scale + chebyshev_model_2d(self.freqs,
                                                               self.times,
                                                               n_t_sin_freq_coeffs,
                                                               n_t_sin_time_coeffs,
                                                               t_sin_amp)
        
        self.t_0 = receiver.t_0_scale + chebyshev_model_2d(self.freqs,
                                                           self.times,
                                                           n_t_0_freq_coeffs,
                                                           n_t_0_time_coeffs,
                                                           t_0_amp)
        


        ### Set up a dictionary for sources
        source_dict = {}
        for src in sources:
            source_dict[src.label] = src
        try:
            for src in dicke_switch_sources:
                source_dict[src.label] = src
        except Exception:
            pass
        
        # compute 

        for i, src_string in enumerate(self.states_array):
            source = source_dict[src_string]
            self.data[i,:] = compute_radiometer_power(t_src=source.t_src,
                                                      t_unc=self.t_unc[i],
                                                      t_sin=self.t_sin[i],
                                                      t_cos=self.t_cos[i],
                                                      t_0=self.t_0[i],
                                                      gamma_rec=receiver.gamma_rec,
                                                      gamma_src=source.gamma_src,
                                                      gain=receiver.gains,
                                                      add_noise=True,
                                                      t_int=self.delta_t,
                                                      delta_nu=self.delta_nu)
        

        self.temperature_dict = {}
        self.gamma_src_dict = {}
        for source in dicke_switch_sources:
            self.temperature_dict[source.label] = np.random.normal(loc=source.t_src,
                                                                   scale=np.sqrt(float(temp_sens_variance)),
                                                                   size=self.times.shape)
        for source in sources:
            self.temperature_dict[source.label] = np.random.normal(loc=source.t_src,
                                                                   scale=np.sqrt(float(temp_sens_variance)),
                                                                   size=self.times.shape)
            self.gamma_src_dict[source.label] = source.gamma_src
        
        self.gamma_rec = receiver.gamma_rec
        print('Done')


    def save_to_hdf5(self, filepath):
        """Export"""
    
        with h5py.File(filepath, 'w') as f:
            pass
    



    def _n_time_calc(self,
                     observation_length,
                     delta_t,
                     n_times):
        
        if [observation_length, delta_t, n_times] == [None, None, None]:
            raise ValueError('Must input time parameters')

        if observation_length is not None and delta_t is not None:
            if not isinstance(observation_length, un.Quantity):
                observation_length *= un.s
        
            if not isinstance(delta_t, un.Quantity):
                delta_t *= un.s
            self.delta_t = delta_t
            self.n_times = int(observation_length.to(un.s) / delta_t.to(un.s))
            self.times = np.linspace(0, observation_length, self.n_times) + self.reference_time

        elif observation_length is not None and n_times is not None:
            if not isinstance(observation_length, un.Quantity):
                observation_length *= un.s
            self.delta_t = observation_length.to(un.s) / n_times
            self.n_times = n_times
            self.times = np.linspace(0, observation_length*un.s, self.n_times) + self.reference_time

        elif n_times is not None and delta_t is not None:
            if not isinstance(delta_t, un.Quantity):
                delta_t *= un.s
            self.delta_t = delta_t
            self.n_times = n_times
            self.times = (np.arange(n_times)*self.delta_t) + self.reference_time
        else:
            raise ValueError("Invalid Time Values")

    def _n_freq_calc(self,
                     n_freqs,
                     receiver:Receiver,
                     bandwidth,
                     centre_freq):
        if [n_freqs, receiver, bandwidth] == [None, None, None]:
            raise ValueError('Must input frequency parameters')

        if receiver is not None:
            self.n_freqs = len(receiver.freqs)
            self.freqs = receiver.freqs

        else:
            try:
                if not isinstance(bandwidth, un.Quantity):
                    bandwidth *= un.Hz
                if not isinstance(centre_freq, un.Quantity):
                    centre_freq *= un.Hz
                
                self.freqs = np.linspace(centre_freq - (bandwidth/2),
                                         centre_freq + (bandwidth/2),
                                         n_freqs)
                self.n_freqs = n_freqs
            except:
                raise ValueError('Must have Frequency information')
            
    def _coeff_handler(self,
                       n_t_unc_freq_coeffs,
                       n_t_sin_freq_coeffs,
                       n_t_cos_freq_coeffs,
                       n_t_unc_time_coeffs,
                       n_t_sin_time_coeffs,
                       n_t_cos_time_coeffs,
                       n_t_0_freq_coeffs,
                       n_t_0_time_coeffs):
        coeff_dict = {"n_t_unc_freq_coeffs":n_t_unc_freq_coeffs,
                       "n_t_sin_freq_coeffs":n_t_sin_freq_coeffs,
                       "n_t_cos_freq_coeffs":n_t_cos_freq_coeffs,
                       "n_t_unc_time_coeffs":n_t_unc_time_coeffs,
                       "n_t_sin_time_coeffs":n_t_sin_time_coeffs,
                       "n_t_cos_time_coeffs":n_t_cos_time_coeffs,
                       "n_t_0_freq_coeffs":n_t_0_freq_coeffs,
                       "n_t_0_time_coeffs":n_t_0_time_coeffs}
        for label, coeff in coeff_dict.items():
            if coeff <= 0:
                raise ValueError(f'{label} must be >= 1')
    
    