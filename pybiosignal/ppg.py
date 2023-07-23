#! python3
"""
PPG analysis
"""

import filtering
import numpy as np
from scipy import signal


def peak_detection(
        sig: np.ndarray, fs: float, method: str = 'dominant_frequency'
) -> np.ndarray:
    '''
    PPG systolic peak detection

    Input:
        sig: Input signal,
        fs: Sampling frequency,
        method: Peak detection method
    Return:
        peak_idx: Index of systolic peaks
    '''
    filtering_win = 0.5  # window to filter the signal
    peak_idx_raw = np.array(
        signal.argrelextrema(sig, np.greater)).reshape(1, -1)[0]
    # Filter the signal based on the frequancy of heart rate
    freq, ppg_den = signal.periodogram(sig, fs)
    # Minimum heart rate frequency
    freq_min = np.where(freq >= 0.6)[0][0]
    # Maximum heart rate frequency
    freq_max = np.where(freq >= 3.0)[0][0]
    sig_freq = ppg_den[freq_min:freq_max]
    hr_freq = freq[freq_min:freq_max]
    # Define cut-off frequancy
    hrf = hr_freq[np.argmax(sig_freq)]
    if hrf - filtering_win > 0.6:
        hrfmin = hrf - filtering_win
    else:
        hrfmin = 0.6
    if hrf + filtering_win < 3.0:
        hrfmax = hrf + filtering_win
    else:
        hrfmax = 3.0
    filtered_signal = filtering.bandpass_filter(
        sig=sig, low_cutoff=hrfmin, high_cutoff=hrfmax,
        filt_design='butter', fs=fs, order=5,
        freq_response=False, zero_pole=False)
    # Peaks in filtered signal
    peak_idx_filtered = np.array(
        signal.argrelextrema(filtered_signal, np.greater)).reshape(1, -1)[0]
    # Find the correct peaks according to peaks of raw and filtered
    peak_idx = np.array([]).astype(int)
    for i in peak_idx_filtered:
        peak_idx = np.append(
            peak_idx, peak_idx_raw[np.abs(i - peak_idx_raw).argmin()])
    peak_idx = np.unique(peak_idx)
    return peak_idx


def trough_detection(sig: np.ndarray, fs: float) -> np.ndarray:
    '''
    PPG trough detection

    Input:
        sig: Input signal,
        fs: Sampling frequency,
    Return:
        trough_idx: Index of troughs
    '''
    peak_idx = peak_detection(sig=sig, fs=fs)
    trough_idx = np.array([]).astype(int)
    for i in range(peak_idx.shape[0]-1):
        trough_idx = np.append(
            trough_idx, np.argmin(
                sig[peak_idx[i]:peak_idx[i+1]]) + peak_idx[i])
    return trough_idx
