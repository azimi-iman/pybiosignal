#! python3
"""
Analysis toolbox
"""

from typing import Tuple

import data_toolbox
import filtering
import neurokit2 as nk
import numpy as np
from scipy import signal, stats
from scipy.fftpack import fft, ifft
from scipy.ndimage.interpolation import shift


def sync_signals(stacked_sigs: np.ndarray) -> np.ndarray:
    '''
    Synchronize input signals

    Inputs:
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


def template_matching(
        stacked_sigs: np.ndarray,
        template: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Template matching

    Inputs:
        stacked_sigs: Input signals (stacked)

    Returns:
        cross_corr: Cross correlation between the inputs and template
        mean_abs_error: Mean Absolute Error between the inputs and template
        pearson_corr: Pearson correlation between the inputs and template
    """
    if template is None:
        template = np.mean(stacked_sigs, axis=0)
    cross_corr = np.array([np.max(signal.correlate(
        stacked_sigs[k, :], template, mode='same')) for k in range(
            stacked_sigs.shape[0])])
    mean_abs_error = np.array([np.mean(np.abs(
        stacked_sigs[k, :] - template)) for k in range(
            stacked_sigs.shape[0])])
    pearson_corr = np.array([stats.pearsonr(
        stacked_sigs[k, :], template)[0] for k in range(
            stacked_sigs.shape[0])])
    return (cross_corr, mean_abs_error, pearson_corr)


def zero_crossing_rate(sig: np.ndarray, fs: float) -> int:
    """Zero crossing rate

    Inputs:
        sig: Input signal
        fs: Sampling frequency

    Returns:
        zcr: Number of times the signal cross baseline (i.e., zero line)
    """
    sig = np.ravel(sig)
    # Remove baseline wandering (if any)
    filtered_signal = filtering.highpass_filter(
        sig=sig, fs=fs, order=5, cutoff=0.5)
    sign_changed = ((np.roll(
        np.sign(filtered_signal), 1) - np.sign(
            filtered_signal)) != 0).astype(int)
    sign_changed[0] = 0
    zcr = np.sum(sign_changed)
    return zcr


def turns_count(sig: np.ndarray, fs: float, height: float = None) -> int:
    """Zero crossing rate

    Inputs:
        sig: Input signal
        height: Count turns more/less than this value. If any, it
            must be between 0 and 1
    Returns:
        tc: Number of times the signal turns
    """
    filtered = filtering.highpass_filter(
        sig=sig, fs=fs, order=5, cutoff=0.5)
    normalized = (filtered-np.min(filtered))/(np.max(
        filtered)-np.min(filtered))
    exterma = signal.find_peaks(np.abs(normalized), height=height)[0]
    tc = exterma.size
    return tc


def shannon_entropy_extract(
    sig: np.ndarray,
    bins: int = 50,
    base: int = 2
) -> float:
    """Extract Shannon entropy

    Inputs:
        sig: Input signal
        bins: Number of histogram bins
        base: Base of the log of the entropy

    Returns:
        shannon_entropy: Shannon entropy
    """
    #
    hist = np.histogram(sig, bins)[0]
    shannon_entropy = round(stats.entropy(hist/hist.sum(), base=base), 2)
    return shannon_entropy


def entropy_extract(
    sig: np.ndarray,
    fs: float,
    bins: int = 50,
    base: int = 2
) -> float:
    """Extract Shannon entropy

    Inputs:
        sig: Input signal
        fs: Sampling frequency
        bins: Number of histogram bins
        base: Base of the log of the entropy

    Returns:
        shannon_entropy: Shannon entropy
    """
    ent = {}
    ent['shannon_entropy'] = shannon_entropy_extract(sig, bins, base)
    _, pxx = signal.periodogram(sig, fs)
    ent['spectral_entropy'] = stats.entropy(pxx/np.sum(pxx))
    ent['approximate_entropy'] = nk.entropy_approximate(sig, corrected=True)
    ent['sample_entropy'] = nk.entropy_sample(sig)
    ent['multiscale_entropy'] = nk.entropy_multiscale(sig)
    ent = {key: round(ent[key], 2) for key in ent}
    return ent


if __name__ == "__main__":
    # Import ECG data
    # Parse directory path and input file name from the
    #   input_arguments.txt. Replace with your path
    ARGUMENT_PATH = 'input_arguments.txt'
    parsed_arguments = data_toolbox.parse_arguments(ARGUMENT_PATH)
    (sigs, fs) = data_toolbox.read_ecg_mitdb(
        path=parsed_arguments['data_path'],
        file_name=parsed_arguments['filename']
    )
    stk_sigs = data_toolbox.segemntation_fix(
        sig=sigs[:, 0], fs=fs, win_length=30.0)
