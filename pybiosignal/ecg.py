#! python3
"""
ECG analysis
"""

import sys
from typing import Tuple

import biosppy
import data_toolbox
import filtering
# import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema


def r_peak_detection(
        sig: np.ndarray, fs: float, method: str = 'hamilton'
) -> np.ndarray:
    '''
    ECG R peak detection

    Input:
        sig: Input signal,
        fs: Sampling frequency,
        method: Peak detection method
    Return:
        rpeak_idx: Index of R peaks
    '''
    try:
        if method == 'hamilton':
            info = biosppy.signals.ecg.ecg(
                signal=np.ravel(sig), sampling_rate=fs, show=False)
            rpeak_idx = info['rpeaks']
        else:
            sys.exit('R peak detection method has not been defined!')
    except Exception:
        rpeak_idx = None
    return rpeak_idx


def cardiac_cycle_extraction(
    sig: np.ndarray, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Cardiac cycle boundaries detection

    Input:
        sig: Input signal,
        fs: Sampling frequency
    Return:
        cycle_start_idx: Index of start of cardiac cycles
        cycle_end_idx: Index of end of cardiac cycles
    '''
    rpeak_idx = r_peak_detection(sig=sig, fs=fs)
    cycle_start_idx = np.array(
        [int(rpeak_idx[k]-np.median(np.diff(rpeak_idx))*0.33) for k in range(
            1, rpeak_idx.size-1)])
    cycle_end_idx = np.array(
        [int(rpeak_idx[k]+np.median(np.diff(rpeak_idx))*0.66) for k in range(
            1, rpeak_idx.size-1)])
    return (cycle_start_idx, cycle_end_idx)


def qrs_complex_detection(
    sig: np.ndarray, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    QRS complex detection

    Input:
        sig: Input signal,
        fs: Sampling frequency
    Return:
        qrs_start_idx: Index of start of QRS complex
        qrs_end_idx: Index of end of QRS complex
        qtrough_idx: Index of end of Q
        strough_idx: Index of end of S
    '''
    rpeak_idx = r_peak_detection(sig=sig, fs=fs)
    filtered_signal = filtering.bandpass_filter(
        sig=sig, low_cutoff=0.5, high_cutoff=40.0,
        filt_design='butter', fs=fs, order=5,
        freq_response=False, zero_pole=False)
    start_idx = np.array(
        [int(rpeak_idx[k]-np.median(np.diff(rpeak_idx))*0.15) for k in range(
            1, rpeak_idx.size-1)])
    end_idx = np.array(
        [int(rpeak_idx[k]+np.median(np.diff(rpeak_idx))*0.15) for k in range(
            1, rpeak_idx.size-1)])
    rpeak_idx = rpeak_idx[1:]
    qtrough_idx = np.array([start_idx[k] + np.argmin(filtered_signal[
        start_idx[k]:rpeak_idx[k]]) for k in range(start_idx.size)])
    strough_idx = np.array([rpeak_idx[k] + np.argmin(filtered_signal[
        rpeak_idx[k]:end_idx[k]]) for k in range(start_idx.size)])
    qrs_start_idx = np.array([start_idx[k] + argrelextrema(filtered_signal[
        start_idx[k]:qtrough_idx[k]], np.greater)[0][-1] for k in range(
            qtrough_idx.size)])
    qrs_end_idx = np.array([strough_idx[k] + argrelextrema(filtered_signal[
        strough_idx[k]:end_idx[k]], np.greater)[0][0] for k in range(
            qtrough_idx.size)])
    return (qrs_start_idx, qrs_end_idx, qtrough_idx, strough_idx)


def t_wave_peak_detection(
    sig: np.ndarray, fs: float,
) -> np.ndarray:
    '''
    T wave detection

    Input:
        sig: Input signal,
        fs: Sampling frequency
    Return:
        tpeak_idx: Index of peak of T wave
    '''
    (_, qrs_end_idx, _, _) = qrs_complex_detection(
        sig=sig, fs=fs)
    (_, cycle_end_idx) = cardiac_cycle_extraction(
        sig=sig, fs=fs)
    filtered_signal = filtering.bandpass_filter(
        sig=sig, low_cutoff=0.5, high_cutoff=40.0,
        filt_design='butter', fs=fs, order=5,
        freq_response=False, zero_pole=False)
    tpeak_idx = np.array([qrs_end_idx[k] + np.argmax(filtered_signal[
        qrs_end_idx[k]:cycle_end_idx[k]]) for k in range(qrs_end_idx.size)])
    return tpeak_idx


def p_wave_peak_detection(
    sig: np.ndarray, fs: float,
) -> np.ndarray:
    '''
    P wave detection

    Input:
        sig: Input signal,
        fs: Sampling frequency
    Return:
        tpeak_idx: Index of peak of P wave
    '''
    (qrs_start_idx, _, _, _) = qrs_complex_detection(
        sig=sig, fs=fs)
    (cycle_start_idx, _) = cardiac_cycle_extraction(
        sig=sig, fs=fs)
    filtered_signal = filtering.bandpass_filter(
        sig=sig, low_cutoff=0.5, high_cutoff=40.0,
        filt_design='butter', fs=fs, order=5,
        freq_response=False, zero_pole=False)
    ppeak_idx = np.array([cycle_start_idx[k] + np.argmax(filtered_signal[
        cycle_start_idx[k]:qrs_start_idx[k]]) for k in range(
            qrs_start_idx.size)])
    return ppeak_idx


def ecg_components_extraction(
    sig: np.ndarray, fs: float,
) -> np.ndarray:
    '''
    ECG components extraction

    Input:
        sig: Input signal,
        fs: Sampling frequency
    Return:
        rpeak_idx: Index of peak of R peak,
        tpeak_idx: Index of peak of T wave,
        ppeak_idx: Index of peak of P wave,
        qrs_start_idx: Index of start of QRS complex,
        qrs_end_idx: Index of end of QRS complex,
    '''
    rpeak_idx = r_peak_detection(sig=sig, fs=fs)
    tpeak_idx = t_wave_peak_detection(sig=sig, fs=fs)
    ppeak_idx = p_wave_peak_detection(sig=sig, fs=fs)
    (qrs_start_idx, qrs_end_idx, _, _) = qrs_complex_detection(sig=sig, fs=fs)
    return (rpeak_idx, tpeak_idx, ppeak_idx, qrs_start_idx, qrs_end_idx)


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
    (rpeak_idx, tpeak_idx, ppeak_idx,
     qrs_start_idx, qrs_end_idx) = ecg_components_extraction(
        sig=stk_sigs[1, :], fs=fs)
