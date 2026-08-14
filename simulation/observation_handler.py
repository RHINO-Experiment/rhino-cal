"""
Module for the observation handler.
"""

import numpy as np
import astropy.units as un
from astropy.time import Time
import h5py
from .receiver import Receiver
from .loads import Load
from utils.utils import set_up_switch_cycle_indices, assign_states, chebyshev_model, chebyshev_model_2d, read_s2p, interp_vals_to_new_freq
from .radiometer_power import compute_radiometer_power
from gcr.data_processing import create_nw_data_and_covariance_from_raw
from gcr.transfer_matrix_construction import construct_transfer_matrix

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
                 n_gain_freq_coeffs=1,
                 n_gain_time_coeffs=1,
                 gain_fluctuation_amp_frac = 0.1,
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
        
        # set up time and freq. varying gains
        self.gains = receiver.gains # nfreq
        self.gains = np.ones_like(self.t_0)*self.gains # make into an waterfall like array
        if n_gain_freq_coeffs or n_gain_time_coeffs is not None:
            assert n_gain_time_coeffs and n_gain_freq_coeffs is not None, 'Must give both time and frequency'
            gain_flucs = chebyshev_model_2d(self.freqs,
                                            self.times,
                                            n_gain_freq_coeffs,
                                            n_gain_time_coeffs,
                                            gain_fluctuation_amp_frac)
            self.gains *= gain_flucs # multiplicatinve time and freq variation

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
                                                      gain=self.gains[i],
                                                      add_noise=True,
                                                      t_int=self.delta_t,
                                                      delta_nu=self.delta_nu)
        

        self.temperature_dict = {}
        self.gamma_src_dict = {}
        for source in dicke_switch_sources:
            if source.label == 'antenna':
                self.temperature_dict[source.label] = np.random.normal(loc=0,
                                                                       scale=np.sqrt(float(temp_sens_variance)),
                                                                       size=self.times.shape)
            else:
                self.temperature_dict[source.label] = np.random.normal(loc=source.t_src,
                                                                   scale=np.sqrt(float(temp_sens_variance)),
                                                                   size=self.times.shape)
        for source in sources:
            if source.label == 'antenna':
                self.temperature_dict[source.label] = np.random.normal(loc=0,
                                                                       scale=np.sqrt(float(temp_sens_variance)),
                                                                       size=self.times.shape)
            else:
                self.temperature_dict[source.label] = np.random.normal(loc=source.t_src,
                                                                    scale=np.sqrt(float(temp_sens_variance)),
                                                                    size=self.times.shape)
            self.gamma_src_dict[source.label] = source.gamma_src

        
        self.gamma_rec = receiver.gamma_rec
        self.temp_sens_variance = temp_sens_variance
        print('Done')


    def save_to_hdf5(self,
                     filepath,
                     save_temps=['internal_load',
                                 'heated_load']):
        """Export the simulation data to .hdf5 in the RHINO style"""

        with h5py.File(filepath, 'w') as f:
            sdr_group = f.create_group('sdr')
            aux_sdr_group = f.create_group('aux_sdr')
            temperature_group = f.create_group('temperatures')
            switching_group = f.create_group('switches')
            config_group = f.create_group('obs_config')

            sdr_freqs = np.array(self.freqs)
            sdr_waterfall = self.data
            sdr_times = np.array(self.times)
            sdr_group.create_dataset('sdr_waterfall', data=sdr_waterfall, dtype=sdr_waterfall.dtype)
            sdr_group.create_dataset('sdr_freqs', data=sdr_freqs, dtype=sdr_freqs.dtype)
            sdr_group.create_dataset('sdr_times', data=sdr_times, dtype=sdr_times.dtype)

            switch_states = np.array(self.switch_states, dtype='S')
            switch_times = np.array(self.switch_times)
            switching_group.create_dataset('switch_states',
                                           data=switch_states)
            switching_group.create_dataset('switch_times',
                                           data=switch_times,
                                           dtype=switch_times.dtype)

            
            temps = np.array([self.temperature_dict[st] for st in save_temps]).T - 273.15 # coversion to celcius
            temp_times = np.array(self.times)
            temperature_group.create_dataset('temperatures',
                                             data=temps,
                                             dtype=temps.dtype)
            temperature_group.create_dataset('temperature_times',
                                             data=temp_times,
                                             dtype=temp_times.dtype)
            
            aux_sdr_group.create_dataset('aux_sdr_waterfall', dtype="f")
            aux_sdr_group.create_dataset('aux_sdr_freqs', dtype="f")
            aux_sdr_group.create_dataset('aux_sdr_times', dtype="f")

        with h5py.File(f"{filepath}_sim",
                       'w') as f:
            f.create_dataset('t_unc',
                             data=self.t_unc,
                             dtype=self.t_unc.dtype)
            f.create_dataset('t_sin',
                             data=self.t_sin,
                             dtype=self.t_sin.dtype)
            f.create_dataset('t_cos',
                             data=self.t_cos,
                             dtype=self.t_cos.dtype)
            f.create_dataset('t_0',
                             data=self.t_0,
                             dtype=self.t_0.dtype)
            f.create_dataset('temp_sens_variance',
                             data=self.temp_sens_variance)
            times = np.array(self.times)
            freqs = np.array(self.freqs)
            f.create_dataset('times',
                             data=times,
                             dtype=times.dtype)
            f.create_dataset('freqs',
                             data=freqs,
                             dtype=freqs.dtype)
            f.create_dataset('gains',
                             data=self.gains,
                             dtype=self.gains.dtype)


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