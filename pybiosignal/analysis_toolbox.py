#! python3
"""
Analysis toolbox
"""

from typing import Tuple, List

import data_toolbox
import filtering
import neurokit2 as nk
import numpy as np
from scipy import signal, stats, interpolate
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


def statistical_moments_extract(sig: np.ndarray) -> dict:
    """Extract statistical moments

    Inputs:
        sig: Input signal
    Returns:
        moments: A dictionary of moments
    """
    sig = np.ravel(sig)
    moments = {}
    moments['1'] = np.mean(sig)
    moments['2'] = np.var(sig)
    moments['3'] = stats.skew(sig)
    moments['4'] = stats.kurtosis(sig)
    return moments


def zero_crossing_rate_extract(sig: np.ndarray, fs: float) -> int:
    """Zero crossing rate extraction

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


def turns_count_extract(
    sig: np.ndarray, fs: float, height: float = None
) -> int:
    """Turn counts extraction

    Inputs:
        sig: Input signal
        height: Count turns more/less than this value. If any, it
            must be between 0 and 1
    Returns:
        tc: Number of times the signal turns
    """
    sig = np.ravel(sig)
    filtered = filtering.highpass_filter(
        sig=sig, fs=fs, order=5, cutoff=0.5)
    normalized = (filtered-np.min(filtered))/(np.max(
        filtered)-np.min(filtered))
    exterma = signal.find_peaks(np.abs(normalized), height=height)[0]
    tc = exterma.size
    return tc


def power_extract(sig: np.ndarray, fs: float) -> float:
    """Extract power of the signal

    Inputs:
        sig: Input signal
        fs: sampling frequency
    Returns:
        tc: Number of times the signal turns
    """
    sig = np.ravel(sig)
    power = np.trapz(sig**2)/(sig.size/fs)
    return power


def rms_extract(
    sig: np.ndarray,
) -> float:
    '''
    Extract root mean square

    Input:
        sig: Input signal,
    Return:
        rms: Root mean square
    '''
    sig = np.ravel(sig)
    rms = np.sqrt(np.mean(sig**2))
    return rms


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
) -> dict:
    """Extract multiple entropy values

    Inputs:
        sig: Input signal
        fs: Sampling frequency
        bins: Number of histogram bins
        base: Base of the log of the entropy

    Returns:
        ent: Multiple entropy values
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


def envelope_extract(sig, order=1):
    """Extract envelope of signal

    Input:
        sig: Input signal
        order: Length of window, select more than 1 if the
            size of the signal is big
    Returns:
        max_enevelop: High envelope of input signal
        min_enevelop: Low envelope of input signal
    """
    sig = np.ravel(sig)
    max_local = (np.diff(np.sign(np.diff(sig))) < 0).nonzero()[0] + 1
    min_local = (np.diff(np.sign(np.diff(sig))) > 0).nonzero()[0] + 1
    # Find indices
    max_enevelop_idx = max_local[[k+np.argmax(
        sig[max_local[k:k+order]]) for k in range(len(max_local), order)]]
    min_enevelop_idx = min_local[[i+np.argmin(
        sig[min_local[i:i+order]]) for i in range(len(min_local), order)]]
    # Interpolate
    max_enevelop_idx = np.append(0, max_enevelop_idx)
    max_enevelop_idx = np.append(max_enevelop_idx, sig.size)
    min_enevelop_idx = np.append(0, min_enevelop_idx)
    min_enevelop_idx = np.append(min_enevelop_idx, sig.size)
    max_enevelop_func = interpolate.interp1d(
        np.arange(sig.size)[max_enevelop_idx],
        sig[max_enevelop_idx],
        kind='cubic',
        bounds_error=False,
        fill_value=0.0
    )
    min_enevelop_func = interpolate.interp1d(
        np.arange(sig.size)[min_enevelop_idx],
        sig[min_enevelop_idx],
        kind='cubic',
        bounds_error=False,
        fill_value=0.0
    )
    max_enevelop = max_enevelop_func(np.arange(sig.size))
    min_enevelop = min_enevelop_func(np.arange(sig.size))
    return (max_enevelop, min_enevelop)


def power_band_extract(
        sig: np.ndarray, fs: float, freq_band: List[float],
) -> float:
    """Extract powerband

    Input:
        sig: Input signal
        fs: Sampling frequency
        freq_band: A list include two variables: i.e., the start
            and end of the powerband
    Returns:
        power_band: High envelope of input signal
    """
    power_band = []
    freq, pxx = signal.periodogram(sig, fs)
    if fs/2 < freq_band[1]:
        freq_band[1] = fs/2
    if fs/2 < freq_band[0]:
        freq_band[0] = fs/2
    band_duration = freq_band[1] - freq_band[0]
    power_band = np.trapz(pxx[abs(
        freq - freq_band[0] - band_duration/2.0) <= band_duration/2.0])
    return power_band


def hjorth_parameters(sig: np.ndarray) -> Tuple[float, float, float]:
    """Extract Hjorth parameters

    Input:
        sig: Input signal

    Returns:
        activity: Hjorth Activity
        mobility: Hjorth Mobility
        complexity: Hjorth Complexity
    """
    activity = np.var(sig)
    mobility = np.sqrt(np.var(np.diff(sig))/np.var(sig))
    complexity = np.sqrt(np.var(
        np.diff(np.diff(sig)))/np.var(np.diff(sig)))/mobility
    return (activity, mobility, complexity)


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
