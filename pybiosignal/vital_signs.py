#! python3
"""
Vital signs extraction for cardiovascular signals (ECG, PPG, etc.)
"""

import data_toolbox
import ecg
import neurokit2 as nk
import numpy as np
from scipy import integrate, interpolate, signal


def hr_extract(peaks_idx: np.ndarray, fs: float) -> float:
    '''
    Heart rate extraction

    Input:
        peak_idx: Index of systolic peaks,
        fs: Sampling frequency,
    Return:
        heart_rate: Heart rate value
    '''
    # Calculate interbeat intervals
    ibi = np.diff(peaks_idx)/fs
    heart_rate = 60.0/ibi
    return heart_rate


def hrv_time_extract(peaks_idx: np.ndarray, fs: float) -> dict:
    '''
    Time-domain heart rate variability extraction

    Input:
        peak_idx: Index of systolic peaks,
        fs: Sampling frequency,
    Return:
        hrv_time: A dictonary of time-domain heart rate variability values
    '''
    hrv = {}
    # Calculate interbeat intervals in milliseconds
    ibi = np.diff(peaks_idx)/fs*1000
    hrv['rmssd'] = np.sqrt(np.mean(np.diff(ibi)**2))
    hrv['sdnn'] = np.std(ibi)
    hrv['avnn'] = np.mean(ibi)
    pnn50_count = 0
    for i in np.diff(ibi):
        if i > 50:
            pnn50_count += 1
    hrv['pnn50'] = pnn50_count/(ibi.size-1)*100
    hrv = {key: round(hrv[key], 2) for key in hrv}
    return hrv


def hrv_frequency_extract(peaks_idx: np.ndarray, fs: float) -> dict:
    '''
    Frequency-domain heart rate variability extraction

    Input:
        peak_idx: Index of systolic peaks,
        fs: Sampling frequency,
    Return:
        hrv_frequency: A dictonary of
            frequency-domain heart rate variability values
    '''
    hrv = {}
    # Calculate interbeat intervals in milliseconds
    ibi = np.diff(peaks_idx)/fs*1000
    ibi_interp_func = interpolate.interp1d(
        np.cumsum(ibi)/1000.0, ibi, kind='cubic')
    ibi_interpolated = ibi_interp_func(
        np.arange(1, np.max(np.cumsum(ibi)/1000.0), 1/fs))
    fxx, pxx = signal.periodogram(x=ibi_interpolated, fs=fs)
    # calculate power in each band
    hrv['ulf_power'] = integrate.trapz(
        pxx[(fxx >= 0.0) & (fxx < 0.0033)],
        fxx[(fxx >= 0.0) & (fxx < 0.0033)])
    hrv['vlf_power'] = integrate.trapz(
        pxx[(fxx >= 0.0033) & (fxx < 0.04)],
        fxx[(fxx >= 0.0033) & (fxx < 0.04)])
    hrv['lf_power'] = integrate.trapz(
        pxx[(fxx >= 0.04) & (fxx < 0.15)],
        fxx[(fxx >= 0.04) & (fxx < 0.15)])
    hrv['hf_power'] = integrate.trapz(
        pxx[(fxx >= 0.15) & (fxx < 0.4)],
        fxx[(fxx >= 0.15) & (fxx < 0.4)])
    hrv['ulf_peak'] = fxx[(fxx >= 0.0) & (fxx < 0.0033)][
        np.argmax(pxx[(fxx >= 0.0) & (fxx < 0.0033)])]
    hrv['vlf_peak'] = fxx[(fxx >= 0.0033) & (fxx < 0.04)][
        np.argmax(pxx[(fxx >= 0.0033) & (fxx < 0.04)])]
    hrv['lf_peak'] = fxx[(fxx >= 0.04) & (fxx < 0.15)][
        np.argmax(pxx[(fxx >= 0.04) & (fxx < 0.15)])]
    hrv['hf_peak'] = fxx[(fxx >= 0.15) & (fxx < 0.4)][
        np.argmax(pxx[(fxx >= 0.15) & (fxx < 0.4)])]
    hrv['lf_hf'] = hrv['lf_peak']/hrv['hf_peak']
    hrv = {key: round(hrv[key], 2) for key in hrv}
    return hrv


def hrv_nonlinear_extract(peaks_idx: np.ndarray, fs: float) -> dict:
    '''
    Nonlinear heart rate variability extraction

    Input:
        peak_idx: Index of systolic peaks,
        fs: Sampling frequency,
    Return:
        hrv_nonlinear: A dictonary of
            nonlinear-domain heart rate variability values
    '''
    hrv = {}
    # Calculate interbeat intervals in milliseconds
    ibi = np.diff(peaks_idx)/fs*1000
    hrv['sd1'] = np.sqrt(1/2*(np.std(np.diff(ibi))**2))
    hrv['sd2'] = np.sqrt(2*np.std(ibi)**2 - 1/2*(np.std(np.diff(ibi))**2))
    hrv['sd1_sd2'] = hrv['sd1']/hrv['sd2']
    hrv['s'] = np.pi*hrv['sd1']*hrv['sd2']
    hrv['ApEn'] = nk.entropy_approximate(ibi, corrected=True)
    hrv['SampEn'] = nk.entropy_sample(ibi)
    hrv = {key: round(hrv[key], 2) for key in hrv}
    return hrv


def hrv_extract(peaks_idx: np.ndarray, fs: float) -> dict:
    '''
    Heart rate variability extraction

    Input:
        peak_idx: Index of systolic peaks,
        fs: Sampling frequency,
    Return:
        hrv: A dictonary of heart rate variability values
    '''
    hrv_time = hrv_time_extract(peaks_idx, fs)  # Time-domain HRV
    hrv_frequency = hrv_frequency_extract(peaks_idx, fs)  # Freq-domain HRV
    hrv_nonlinear = hrv_nonlinear_extract(peaks_idx, fs)  # Nonlinear HRV
    hrv = {**hrv_time, **hrv_frequency, **hrv_nonlinear}
    return hrv


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
        sig=sigs[:, 0], fs=fs, win_length=60.0)
    peaks_idx = ecg.r_peak_detection(sig=stk_sigs[0, :], fs=fs)
    hrv = hrv_extract(peaks_idx=peaks_idx, fs=fs)
