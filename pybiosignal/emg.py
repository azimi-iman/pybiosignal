#! python3
"""
EMG analysis
"""

import numpy as np
import analysis_toolbox


def signal_activity_extract(sig: np.ndarray, fs: float):
    '''
    Extract activitiy of EMG signal

    Input:
        sig: Input signal,
        fs: Sampling frequency,
    Return:
        rpeak_idx: Index of R peaks
    '''
    activity = {}
    activity['power'] = analysis_toolbox.power_extract(sig=sig, fs=fs)
    activity['rms'] = analysis_toolbox.rms(sig=sig)
    activity['turn_counts'] = analysis_toolbox.turns_count_extract(
        sig=sig, fs=fs)
    activity['zero_crossing'] = analysis_toolbox.zero_crossing_rate_extract(
        sig=sig, fs=fs)
    ent = analysis_toolbox.entropy_extract(
        sig=sig, fs=fs)
    activity = {**activity, **ent}
    activity = {key: round(activity[key], 2) for key in activity}
    return activity
