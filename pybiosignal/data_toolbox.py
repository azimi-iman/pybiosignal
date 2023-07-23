#! python3
"""
Data toolbox
"""

import os
import sys
from typing import Tuple

import numpy as np
import wfdb


def parse_arguments(path: str) -> dict:
    '''
    Parse argument file

    Input:
        path: Path to argument,
    Return:
        arguments: Arguments
    '''
    arguments = {}
    with open(path, 'r', encoding='UTF-8') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                key, value = line.split(": ", maxsplit=1)
                arguments[key] = value
    return arguments


def read_ecg_mitdb(path: str, file_name: str) -> Tuple[np.ndarray, float]:
    '''
    Import ECG from MIT-DB

    Input:
        path: Path to file,
        filename: Name of file,
    Return:
        sig: ecg signal
        fs: sampling frequency
    '''
    # Read the ECG record
    record = wfdb.rdrecord(os.path.join(path, file_name))
    # Get the ECG signal array and sampling frequency
    return (record.p_signal, record.fs)


def segemntation_fix(
        sig: np.ndarray,
        fs: float,
        win_length: float = 10) -> np.ndarray:
    '''
    Segment signals using a fixed window

    Input:
        sig: Input signal,
        fs: Sampling frequency,
        win_length: Length of the window in seconds,
    Return:
        stacked_sigs: segmented signals stacked
    '''
    sig = np.ravel(sig)
    if sig.size < win_length*fs:
        sys.exit('Signal length is less than window length')
    win_number = int(sig.size/(win_length*fs))
    stacked_sigs = np.stack(np.split(
        sig[:int(win_number*win_length*fs)], win_number))
    return stacked_sigs


if __name__ == "__main__":
    # Parse directory path and input file name from the
    #   input_arguments.txt. Replace with your path
    ARGUMENT_PATH = 'input_arguments.txt'
    parsed_arguments = parse_arguments(ARGUMENT_PATH)
    (sigs, fs) = read_ecg_mitdb(
        path=parsed_arguments['data_path'],
        file_name=parsed_arguments['filename']
    )
