#! python3
"""
Analysis toolbox
"""

import numpy as np
from scipy.fftpack import fft, ifft
from scipy.ndimage.interpolation import shift


def sync_signals(stacked_sigs: np.ndarray) -> np.ndarray:
    '''
    Synchronize input signals

    Input:
        stacked_sigs: Input signal stacked
    Return:
        stacked_sigs_synced: synced signals stacked
    '''
    freq_sig_ref = fft(stacked_sigs[0, :])
    freq_sigs = []
    for idx in range(1, stacked_sigs.shape[0]):
        freq_sigs.append(fft(stacked_sigs[idx, :]))
    stacked_freq_sigs = np.stack(freq_sigs)
    shift_values = []
    for idx in range(stacked_freq_sigs.shape[0]):
        shift_values.append(np.argmax(
            np.abs(ifft(-freq_sig_ref.conjugate()*stacked_freq_sigs[idx, :]))))
    for idx in range(stacked_freq_sigs.shape[0]):
        if shift_values[idx] > stacked_freq_sigs[idx, :].shape[0]/2:
            shift_values[idx] = shift_values[idx] - stacked_freq_sigs[
                idx, :].shape[0]
    sigs_synced = []
    for idx in range(1, stacked_sigs.shape[0]-2):
        sh = shift(np.append(
            stacked_sigs[idx-1, :],
            np.append(stacked_sigs[idx, :], stacked_sigs[idx+1, :])
            ), -shift_values[idx], cval=0.0)
        sigs_synced.append(
            sh[stacked_sigs[idx, :].size:2*stacked_sigs[idx, :].size])
    stacked_sigs_synced = np.stack(sigs_synced)
    return stacked_sigs_synced
