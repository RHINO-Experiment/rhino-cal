"""
Module for handling utlity functions.
"""
import numpy as np
import astropy.units as un
from astropy.time import Time

def read_s2p(filename, flipped_measurement=False):
    """
    Reads a Touchstone .s2p file and returns:
        freq       → (N,) numpy array of frequencies (Hz)
        s_params   → (N, 2, 2) numpy array of complex S-parameters
    """

    # Storage
    freq = []
    s11 = []
    s21 = []
    s12 = []
    s22 = []

    data_format = None  # RI, MA, DB
    freq_multiplier = 1.0

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Skip comments
            if not line or line.startswith("!"):
                continue

            # Option line beginning with "#"
            if line.startswith("#"):
                parts = line.upper().split()
                # Detect frequency units
                if "HZ" in parts:
                    freq_multiplier = 1.0
                elif "KHZ" in parts:
                    freq_multiplier = 1e3
                elif "MHZ" in parts:
                    freq_multiplier = 1e6
                elif "GHZ" in parts:
                    freq_multiplier = 1e9

                # Detect S-parameter format
                if "RI" in parts:
                    data_format = "RI"
                elif "MA" in parts:
                    data_format = "MA"
                elif "DB" in parts:
                    data_format = "DB"
                continue

            # Parse numeric data lines
            values = line.split()
            if len(values) != 9:
                continue  # Not a valid S2P data row

            f_raw = float(values[0]) * freq_multiplier
            p = list(map(float, values[1:]))

            # p = [S11_a, S11_b, S21_a, S21_b, S12_a, S12_b, S22_a, S22_b]
            if data_format == "RI":
                c = lambda a, b: a + 1j * b
            elif data_format == "MA":
                c = lambda mag, ang_deg: mag * np.exp(1j * np.deg2rad(ang_deg))
            elif data_format == "DB":
                c = lambda db, ang_deg: 10**(db/20) * np.exp(1j * np.deg2rad(ang_deg))
            else:
                raise ValueError("S2P format not recognized (missing RI/MA/DB).")

            freq.append(f_raw)
            s11.append(c(p[0], p[1]))
            s21.append(c(p[2], p[3]))
            s12.append(c(p[4], p[5]))
            s22.append(c(p[6], p[7]))

    # Convert to array
    freq = np.array(freq)
    s_params = np.zeros((len(freq), 2, 2), dtype=complex)

    s11 = np.array(s11)
    s12 = np.array(s12)
    s21 = np.array(s21)
    s22 = np.array(s22)

    s_params[:, 0, 0] = s11
    s_params[:, 0, 1] = s12
    s_params[:, 1, 0] = s21
    s_params[:, 1, 1] = s22

    if flipped_measurement:
        return s22,s21,s12,s11, freq
    else:
        return s11,s12,s21,s22, freq
    

def interp_vals_to_new_freq(new_freqs,
                            old_freqs,
                            old_values,
                            freqs_unit=un.Hz):
    """ Function to interpolate values to new frequencies
    """
    if not isinstance(new_freqs, un.Quantity):
            new_freqs *= freqs_unit

    if not isinstance(old_freqs, un.Quantity):
        old_freqs *= freqs_unit

    new_values = np.interp(np.asarray(new_freqs.to(un.Hz)),
                           np.asarray(old_freqs.to(un.Hz)),
                           old_values)
        
    return new_values

def set_up_switch_cycle_indices(observation_length:un.Quantity,
                                dicke_switch_length:un.Quantity,
                                source_list:list,
                                dicke_switch_list:list,
                                observation_start_time:Time = Time(1768906853.5738704,
                                                                   format='unix',
                                                                   scale='utc')):
    """Function to return the switch_states for a mock observation in the
    RHINO style format -> unix time
    """
    switch_targets = []
    switch_times = []
    t = observation_start_time
    t_end = observation_start_time + observation_length
    source_time = dicke_switch_length / len(dicke_switch_list)
    src_idx = 0
    while t < t_end:
        for d in dicke_switch_list:

            if t > t_end:
                pass
            else:
                if d == 'source':
                    switch_targets.append(source_list[src_idx])
                    switch_times.append(t.unix)
                    
                else:
                    switch_targets.append(d)
                    switch_times.append(t.unix)
                t += source_time
        src_idx += 1
        if src_idx >= len(source_list):
            src_idx = 0
    
    return switch_targets, switch_times

def states_array_generator(switch_states,
                           switch_times,
                           time_basis):
    """ Generates array of switch states corresponding to each spectra
    """
    switch_states = np.asarray(switch_states)
    switch_times = np.asarray(switch_times)
    time_basis = np.asarray(time_basis)

    idx = np.searchsorted(switch_times, time_basis, side='right') - 1

    states_array = np.empty(len(time_basis), dtype=switch_states.dtype)

    before_first = idx < 0

    states_array[before_first] = switch_states[0]
    valid = idx >= 0
    states_array[valid] = switch_states[idx[valid]] # creates an array of switch states corresponding to each spectra

    return states_array


def assign_states(times, change_times, change_states, default_state=None):
    """
    Parameters
    ----------
    times : array-like (ntimes,)
    change_times : array-like (n_changes,)
    change_states : array-like (n_changes,)
    default_state : value used before first change (optional)

    Returns
    -------
    states : ndarray (ntimes,)
    """
    times = np.asarray(times)
    change_times = np.asarray(change_times)
    change_states = np.asarray(change_states)


    # Find index of last change_time <= each time
    idx = np.searchsorted(change_times, times, side='right') - 1

    states = np.empty(len(times), dtype=change_states.dtype)

    # Handle times before first change
    before_first = idx < 0
    if default_state is not None:
        states[before_first] = default_state
    else:
        states[before_first] = change_states[0]

    # Assign valid indices
    valid = idx >= 0
    states[valid] = change_states[idx[valid]]

    return states

def chebyshev_model(x, ncoeffs, amp):
    """
    This convenience function produces a Chebyshev series model in 
    frequency/time/etc. to produce a model for various components 
    and systematic effects.
    
    Parameters:
        x (array_like):
            Time/freq. variable.
        ncoeffs (int):
            No. of Chebyshev coefficients to use.
        amp (float):
            Overall amplitude of the Chebyshev polynomial.
    
    Returns:
        fn (array_like):
            Chebyshev function. 
    """
    # Chebyshev parameters
    coeffs = amp * np.random.randn(ncoeffs)

    # Construct time- and frequency-dependent random Chebyshev functions
    xx = np.linspace(-1., 1., x.size)
    fn = np.polynomial.chebyshev.Chebyshev(coeffs)(xx)
    
    # Return values
    return fn


def chebyshev_model_2d(
        freqs:np.ndarray,
        times:np.ndarray,
        n_freq_coeffs:int,
        n_time_coeffs:int,
        amp = float) -> np.ndarray:
    """Compute the 2D Chebyshev Polynomial
    """
    coeffs = np.sqrt(amp) * np.random.randn(n_freq_coeffs,
                                            n_time_coeffs)
    
    xx = np.linspace(-1., 1., freqs.size)
    yy = np.linspace(-1, 1., times.size)

    x_grid, y_grid = np.meshgrid(xx, yy)

    return np.polynomial.chebyshev.chebval2d(x_grid,
                                             y_grid,
                                             coeffs)
    

def create_idx_dict(switch_states,
                    desired_list,
                    dicke_switch_list):
    """Creates a Dictionary of blocks of
    indices corresponding to a source
    """
    source_blocks_indices = {} 
    for cal_source in desired_list:
        source_blocks_indices[cal_source] = [] # set up the cal source dictionary
        for ds in dicke_switch_list:
            source_blocks_indices[f'{cal_source}_{ds}'] = [] # set up dicke switch measurements too
    indices = []
    current_src = 'f'
    for i, swst in enumerate(switch_states):
        # check to see if current source has changed
        if swst != current_src:
            if swst in dicke_switch_list:
                pass
            else:
                current_src = swst # switch the current source

        
        if current_src in desired_list: # adds to the next indices sublist is current source is a calibration source
            indices.append(i)
        else:
            pass
        
        if i == len(switch_states)-1: # check if last element
            if swst in dicke_switch_list and current_src in desired_list:
                source_blocks_indices[f'{current_src}_{swst}'].append(indices)
            elif current_src in desired_list:
                source_blocks_indices[current_src].append(indices)
            indices = []

        elif switch_states[i+1] != swst:
            if swst in dicke_switch_list and current_src in desired_list:
                source_blocks_indices[f'{current_src}_{swst}'].append(indices)
            elif current_src in desired_list:
                source_blocks_indices[current_src].append(indices)
            indices = []
        else:
            pass
    return source_blocks_indices




def generate_states_array(switch_states,
                          switch_times,
                          times):
    """ Generates an array of switch-state labels
    corresponding to the time array.
    """

    switch_states = np.asarray(switch_states)
    switch_times = np.asarray(switch_times)

    idx = np.searchsorted(switch_times, times, side='right') - 1

    states_array = np.empty(len(times), dtype=switch_states.dtype)

    before_first = idx < 0

    states_array[before_first] = switch_states[0]
    valid = idx >= 0
    states_array[valid] = switch_states[idx[valid]] # creates an array of switch states corresponding to each spectra


    return states_array