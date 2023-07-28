#! python3
"""
EEG analysis
"""

import numpy as np
import analysis_toolbox
import filtering


def eeg_power_bands(sig: np.ndarray, fs: float) -> dict:
    """Power bands features of EEG

    Input:
        sig: Input signal
        fs: sampling frquency
    Returns:
        power_bands: Dictionary of EEG power bands
    """
    power_bands = {}
    power_bands['delta'] = analysis_toolbox.power_band_extract(
        sig, fs, [0.5, 4])
    power_bands['theta'] = analysis_toolbox.power_band_extract(
        sig, fs, [4, 7])
    power_bands['alpha'] = analysis_toolbox.power_band_extract(
        sig, fs, [8, 12])
    power_bands['beta'] = analysis_toolbox.power_band_extract(
        sig, fs, [13, 30])
    power_bands['gamma'] = analysis_toolbox.power_band_extract(
        sig, fs, [30, 200])
    power_bands['sigma'] = analysis_toolbox.power_band_extract(
        sig, fs, [12, 16])
    power_bands['iso'] = analysis_toolbox.power_band_extract(
        sig, fs, [0, 0.5])
    power_bands['theta_alpha'] = power_bands['theta']/power_bands['alpha']
    power_bands['theta_beta'] = power_bands['theta']/power_bands['beta']
    power_bands['gamma_delta'] = power_bands['gamma']/power_bands['delta']
    return power_bands


def eeg_rhythm_signal(
        sig: np.ndarray, fs: float, rhythm_type: str,
) -> np.ndarray:
    """Extract EEG rhythm

    Input:
        sig: Input signal
        fs: sampling frquency
        rhythm_type: Type of the rhythm (i.e., "delta",
            "theta", "alpha", "beta", "gamma", "sigma", "iso")
    Returns:
        rhythm_signal: Requested rhythm signal
    """
    if rhythm_type == "delta":
        cutoff = [0.5, 4]
    elif rhythm_type == "theta":
        cutoff = [4, 7]
    elif rhythm_type == "alpha":
        cutoff = [8, 12]
    elif rhythm_type == "beta":
        cutoff = [13, 30]
    elif rhythm_type == "gamma":
        cutoff = [30, 200]
    elif rhythm_type == "sigma":
        cutoff = [12, 16]
    elif rhythm_type == "iso":
        cutoff = [0, 0.5]
    rhythm_signal = filtering.bandpass_filter(
        sig=sig, low_cutoff=cutoff[0], high_cutoff=cutoff[1], fs=fs, order=5)
    return rhythm_signal
