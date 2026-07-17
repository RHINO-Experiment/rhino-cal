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

    return momentrfi_mask, frequency_weights    


def convert_manual_frequency_mask(
        frequency_mask,
        waterfall_shape):
    """
    Inputs a preset frequency mask and scales to the same
    shape as a waterfall 
    """
    return np.tile(frequency_mask,(waterfall_shape[0],1))