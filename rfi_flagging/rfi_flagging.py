import numpy as np
import MomentRFI.MomentRFI as MomentRFI

from MomentRFI.MomentRFI import IterativeSurfaceFitter


def flag_waterfall_momentRFI(waterfall,
                             fitter:IterativeSurfaceFitter,
                             prior_mask=None,
                             whole_channel_flag_threshold=0.4,
                             whole_time_flag_threshold=0.5,
                             ):
    """
    Goes through a waterfall and uses momentRFI to produce a mask.
    Also masks channels in which some fraction of masks have been detected
    for robustness.

    Input:
        waterfall (np.ndarray) (n_times, n_freqs)
            2D array of power spectra
        fitter (MomentRFI.IterfativeSurfaceFitter)
            The MomentRFI surface fitter.
        prior_channel_flags (np.ndarray) (n_times, n_freqs).
            Prior bool array of flags.
        whole_channel_flag_threshold (float)
            Threshold for fraction of channels flagged before
            an entire channel is flagged.
    Outputs:
        momentrfi_mask (np.ndarray, bool) (n_times, n_freqs)
            The momentRFI produced mask. True for unflagged
            and False for flagged.
        frequency_weights:
            Frequency channel weights for where entire channels
            have been flagged
        
    """
    # momentRFI pass, true for flagged, flase for unflagged bool originally
    momentrfi_mask = ~fitter.fit(waterfall, prior_mask=prior_mask)

    n_t = len(waterfall)
    n_freqs = len(waterfall[0])

    collapsed_mask_nfreq = np.sum(momentrfi_mask, axis=0) # n_freqs

    collapsed_mask_ntimes = np.sum(momentrfi_mask, axis=1)

    frequency_weights = np.where(collapsed_mask_nfreq < n_t*whole_channel_flag_threshold,
                                 False, True)
    time_weights = np.where(collapsed_mask_ntimes < n_freqs*whole_time_flag_threshold)

    momentrfi_mask *= np.tile(frequency_weights, (n_t, 1)) * np.tile(time_weights[:, None], (1, n_freqs))

    return momentrfi_mask


def convert_manual_frequency_mask(
        frequency_mask,
        waterfall_shape):
    """
    Inputs a preset frequency mask and scales to the same
    shape as a waterfall 
    """
    return np.tile(frequency_mask,(waterfall_shape[0],1))


def mask_by_state(data,
                  state_labels,
                  flag_dict,
                  prior_freq_mask,
                  whole_channel_flag_threshold=0.4,
                  whole_time_flag_threshold=0.5,
                  time_axis=0):
    """
    Splits a 2D NumPy array by state along the time axis, runs a flagging function
    on each state group, and recombines the resulting mask.

    Parameters
    ----------
    data : np.ndarray
        The input 2D array (e.g., shape (time, freq) or (freq, time)).
    state_labels : np.ndarray
        1D array of state labels corresponding to the time dimension.
    flag_dict : dict
        A dictionary of MomentRFI.
    time_axis : int, optional
        The axis representing time in `data` (default is 0).

    Returns
    -------
    full_mask : np.ndarray
        Recombined mask matching the shape and time ordering of `data`.
    """
    state_labels = np.asarray(state_labels)
    unique_states = np.unique(state_labels)
    full_mask = None

    for state in unique_states:
        # Find time indices matching the current state
        indices = np.where(state_labels == state)[0]
        if len(indices) == 0:
            continue

        # Extract the state sub-array along the specified time axis
        sub_data = np.take(data, indices, axis=time_axis)

        # Assign the correct flagger
        flagger = flag_dict[state]

        # define a prior mask from the frequency mask
        prior_mask = convert_manual_frequency_mask(prior_freq_mask,
                                                   sub_data.shape)

        # generate mask
        sub_mask = flag_waterfall_momentRFI(sub_data,
                                        flagger,
                                        prior_mask=prior_mask,
                                        whole_time_flag_threshold=whole_time_flag_threshold=,
                                        whole_channel_flag_threshold=whole_channel_flag_threshold)

        # Pre-allocate output array on first pass to match flag_func's return dtype
        if full_mask is None:
            full_mask = np.empty(data.shape, dtype=sub_mask.dtype)

        # Construct slicing tuple to write sub_mask back to full_mask
        slices = [slice(None)] * data.ndim
        slices[time_axis] = indices
        full_mask[tuple(slices)] = sub_mask

    return full_mask