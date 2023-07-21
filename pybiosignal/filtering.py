#! python3
"""
Filtering techniques
"""

import sys
from typing import Tuple, Union

import analysis_toolbox
import data_toolbox
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.pyplot import axhline, axvline
from scipy.signal import (bessel, butter, cheby1, cheby2, ellip, filtfilt,
                          firwin, freqz, iirnotch, tf2zpk)


def plot_frequency_response(
        b: np.ndarray, a: np.ndarray, fs: float = None):
    '''
    Plot frequency response of the filter

    Input:
        b: Numerator (Zeros) of the filter
        a: Denominator (Poles) of the filter
        fs: Sampling frequency,
    '''
    w, h = freqz(b, a, fs=fs)
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1 = plt.subplot(211)
    ax1.set_title('Frequency Response')
    ax1.set_ylabel('Amplitude')
    ax1.grid()
    ax1.plot(w, abs(h))
    ax2 = plt.subplot(212)
    ax2.set_ylabel('Angle (radians)')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.grid()
    ax2.plot(w, np.angle(h))
    plt.show()


def zero_pole_plot(b: np.ndarray, a: np.ndarray):
    '''
    Plot Zero-Pole plot of the filter

    Input:
        b: Numerator (Zeros) of the filter
        a: Denominator (Poles) of the filter
    '''
    (z, p, k) = tf2zpk(b, a)
    unit_circle = patches.Circle((0, 0), radius=1, fill=False,
                                 color='black', ls='solid', alpha=0.5)
    plt.gca().add_patch(unit_circle)
    axvline(0, color='black', alpha=0.5)
    axhline(0, color='black', alpha=0.5)
    # Plot the poles
    plt.plot(p.real, p.imag, 'x', markersize=8, color='red')
    # Plot the zeros
    plt.plot(z.real, z.imag, 'o', markersize=8,
             color='none', markeredgecolor='blue')
    plt.grid()
    plt.xlabel('Re(Z)')
    plt.ylabel('Im(Z)')
    plt.title('Zero-Pole Plot')
    r = 1.2 * np.amax(np.concatenate((abs(z), abs(p), [1])))
    plt.axis('scaled')
    plt.axis([-r, r, -r, r])
    plt.show()


def frequency_filter_design(
        filt_design: str,
        wn: Union[float, list],
        btype: str,
        fs: float = 100.0,
        rp: float = 5.0,
        rs: float = 40.0,
        order: int = 3,
        freq_response: bool = False,
        zero_pole: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Frequency filter design

    Input:
        filt_design: filter design ('butter',
            'cheby1', 'cheby2', 'ellip', 'bessel')
        wn: A scalar or length-2 sequence giving the critical frequencies,
        btype: filter type ('lowpass', 'highpass', 'bandpass', 'bandstop')
        fs: Sampling frequency,
        rp: Maximum ripple allowed below unity gain in the passband,
        rs: Minimum attenuation required in the stop band,
        order: Order of the filter,
        freq_response: If True, plot the frequency response,
        zero_pole: If True, plot the zero-pole plot,
    Return:
        b: Numerator (Zeros)
        a: Denominator (Poles)
    '''
    if filt_design == "butter":
        b, a = butter(
            N=order, Wn=wn,
            btype=btype, analog=False)
    elif filt_design == "cheby1":
        b, a = cheby1(
            N=order, rp=rp, Wn=wn,
            btype=btype, analog=False)
    elif filt_design == "cheby2":
        b, a = cheby2(
            N=order, rs=rs, Wn=wn,
            btype=btype, analog=False)
    elif filt_design == "ellip":
        b, a = ellip(
            N=order, rp=rp, rs=rs, Wn=wn,
            btype=btype, analog=False)
    elif filt_design == "bessel":
        b, a = bessel(
            N=order, Wn=wn,
            btype=btype, analog=False)
    elif filt_design == "FIR":
        b = firwin(numtaps=order+1, cutoff=wn, pass_zero=btype)
        a = 1
    else:
        sys.exit('Filter type has not been defined!')
    if freq_response:
        # Plot frequency response
        plot_frequency_response(b, a, fs)
    if zero_pole:
        # Plot Zero-Pole plot
        zero_pole_plot(b, a)
    return (b, a)


def highpass_filter(
        sig: np.ndarray,
        filt_design: str = 'butter',
        cutoff: float = 5.0,
        rp: float = 5.0,
        rs: float = 40.0,
        samp_freq: float = 100.0,
        order: int = 3,
        freq_response: bool = False,
        zero_pole: bool = False,
) -> np.ndarray:
    '''
    Highpass filter

    Input:
        sig: Input signal,
        filt_design: filter design ('butter',
            'cheby1', 'cheby2', 'ellip', 'bessel', 'FIR')
        cutoff: cut-off frequency,
        rp: Maximum ripple allowed below unity gain in the passband,
        rs: Minimum attenuation required in the stop band,
        samp_freq: Sampling frequency,
        order: Order of the filter,
        freq_response: If True, plot the frequency response,
        zero_pole: If True, plot the zero-pole plot,
    Return:
        filtered_signal: Filtered signal,
    '''
    btype = "highpass"
    # Compute the Nyquist frequency
    nyquist = 0.5 * samp_freq
    # Compute the normalized cutoff frequencies
    wn = cutoff / nyquist
    # Design filter. Obtain numerator and denominator
    (b, a) = frequency_filter_design(
        filt_design=filt_design, wn=wn, btype=btype,
        fs=samp_freq, rp=rp, rs=rs, order=order,
        freq_response=freq_response, zero_pole=zero_pole)
    # Apply the digital filter
    filtered_signal = filtfilt(b, a, np.ravel(sig))
    return filtered_signal


def lowpass_filter(
        sig: np.ndarray,
        filt_design: str = 'butter',
        cutoff: float = 40.0,
        rp: float = 5.0,
        rs: float = 40.0,
        samp_freq: float = 100.0,
        order: int = 3,
        freq_response: bool = False,
        zero_pole: bool = False,
) -> np.ndarray:
    '''
    Lowpass filter

    Input:
        sig: Input signal,
        filt_design: filter design ('butter',
            'cheby1', 'cheby2', 'ellip', 'bessel', 'FIR')
        cutoff: cut-off frequency,
        rp: Maximum ripple allowed below unity gain in the passband,
        rs: Minimum attenuation required in the stop band,
        samp_freq: Sampling frequency,
        order: Order of the filter,
        freq_response: If True, plot the frequency response,
        zero_pole: If True, plot the zero-pole plot,
    Return:
        filtered_signal: Filtered signal,
    '''
    btype = "lowpass"
    # Compute the Nyquist frequency
    nyquist = 0.5 * samp_freq
    # Compute the normalized cutoff frequencies
    wn = cutoff / nyquist
    # Design filter. Obtain numerator and denominator
    (b, a) = frequency_filter_design(
        filt_design=filt_design, wn=wn, btype=btype,
        fs=samp_freq, rp=rp, rs=rs, order=order,
        freq_response=freq_response, zero_pole=zero_pole)
    # Apply the digital filter
    filtered_signal = filtfilt(b, a, np.ravel(sig))
    return filtered_signal


def bandpass_filter(
        sig: np.ndarray,
        filt_design: str = 'butter',
        low_cutoff: float = 0.5,
        high_cutoff: float = 40.0,
        rp: float = 5.0,
        rs: float = 40.0,
        samp_freq: float = 100.0,
        order: int = 3,
        freq_response: bool = False,
        zero_pole: bool = False,
) -> np.ndarray:
    '''
    Bandpass filter

    Input:
        sig: Input signal,
        filt_design: filter design ('butter',
            'cheby1', 'cheby2', 'ellip', 'bessel', 'FIR')
        low_cutoff: Low cut-off frequency,
        high_cutoff: High cut-off frequency,
        rp: Maximum ripple allowed below unity gain in the passband,
        rs: Minimum attenuation required in the stop band,
        samp_freq: Sampling frequency,
        order: Order of the filter,
        freq_response: If True, plot the frequency response,
        zero_pole: If True, plot the zero-pole plot,
    Return:
        filtered_signal: Filtered signal,
    '''
    btype = "bandpass"
    # Compute the Nyquist frequency
    nyquist = 0.5 * samp_freq
    # Compute the normalized cutoff frequencies
    wn = [x / nyquist for x in [low_cutoff, high_cutoff]]
    # Design filter. Obtain numerator and denominator
    (b, a) = frequency_filter_design(
        filt_design=filt_design, wn=wn, btype=btype,
        fs=samp_freq, rp=rp, rs=rs, order=order,
        freq_response=freq_response, zero_pole=zero_pole)
    # Apply the digital filter
    filtered_signal = filtfilt(b, a, np.ravel(sig))
    return filtered_signal


def bandstop_filter(
        sig: np.ndarray,
        filt_design: str = 'butter',
        low_cutoff: float = 45.0,
        high_cutoff: float = 65.0,
        rp: float = 5.0,
        rs: float = 40.0,
        samp_freq: float = 100.0,
        order: int = 3,
        freq_response: bool = False,
        zero_pole: bool = False,
) -> np.ndarray:
    '''
    Bandstop filter

    Input:
        sig: Input signal,
        filt_design: filter design ('butter',
            'cheby1', 'cheby2', 'ellip', 'bessel', 'FIR')
        low_cutoff: Low cut-off frequency,
        high_cutoff: High cut-off frequency,
        rp: Maximum ripple allowed below unity gain in the passband,
        rs: Minimum attenuation required in the stop band,
        samp_freq: Sampling frequency,
        order: Order of the filter,
        freq_response: If True, plot the frequency response,
        zero_pole: If True, plot the zero-pole plot,
    Return:
        filtered_signal: Filtered signal,
    '''
    btype = "bandstop"
    # Compute the Nyquist frequency
    nyquist = 0.5 * samp_freq
    # Compute the normalized cutoff frequencies
    wn = [x / nyquist for x in [low_cutoff, high_cutoff]]
    # Design filter. Obtain numerator and denominator
    (b, a) = frequency_filter_design(
        filt_design=filt_design, wn=wn, btype=btype,
        fs=samp_freq, rp=rp, rs=rs, order=order,
        freq_response=freq_response, zero_pole=zero_pole)
    # Apply the digital filter
    filtered_signal = filtfilt(b, a, np.ravel(sig))
    return filtered_signal


def notch_filter(
        sig: np.ndarray,
        samp_freq: float,
        cutoff: float = 60.0,
        quality_factor: float = 30.0,
        freq_response: bool = False,
        zero_pole: bool = False,
) -> np.ndarray:
    '''
    Notch filter

    Input:
        sig: Input signal,
        cutoff: Frequency to be removed,
        quality_factor: Quality factor. Dimensionless
            parameter that characterizes notch filter
        freq_response: If True, plot the frequency response,
        zero_pole: If True, plot the zero-pole plot,
    Return:
        filtered_signal: Filtered signal,
    '''
    b, a = iirnotch(cutoff, quality_factor, samp_freq)
    filtered_signal = filtfilt(b, a, np.ravel(sig))
    if freq_response:
        # Plot frequency response
        plot_frequency_response(b, a, samp_freq)
    if zero_pole:
        # Plot Zero-Pole plot
        zero_pole_plot(b, a)
    return filtered_signal


def synchronized_averaging_filter(stacked_sigs: np.ndarray) -> np.ndarray:
    '''
    Synchronized averaging filter

    Input:
        stacked_sigs: Input signals (stacked),
    Return:
        filtered_signal: Filtered signal,
    '''
    filtered_signal = np.mean(
        analysis_toolbox.sync_signals(stacked_sigs), axis=0)
    return filtered_signal


def moving_average_filter(
    sig: np.ndarray,
    order: int = 8,
    b: list = None,
    mode: str = 'reflect',
) -> np.ndarray:
    '''
    Moving average filter

    Input:
        sig: Input signal,
        order: Order of the filter (Number of weights)
        b: Weights. Ignore if all weights are one
        mode: How the input array is extended beyond its boundaries
            'reflect' = extended by replicating the last n pixels
            'nearest' = extended by replicating the last pixel
            'zero' = extended by n zero
            'ignore' = nothing added
    Return:
        filtered_signal: Filtered signal,
    '''
    sig = np.ravel(sig)
    if b is None:
        b = [1] * order
        print(b)
    filtered_signal = np.array(
        [np.sum(b*sig[k-order:k]) for k in range(order, sig.size)])
    if mode == 'reflect':
        extnd = filtered_signal[-order:]
    elif mode == 'nearest':
        extnd = np.ones(order)*filtered_signal[-1]
    elif mode == 'zero':
        extnd = np.zeros(order)
    elif mode == 'ignore':
        extnd = []
    else:
        extnd = []
    filtered_signal = np.append(filtered_signal, extnd)
    return filtered_signal


def hanning_filter(sig: np.ndarray, mode: str = 'reflect') -> np.ndarray:
    '''
    Hanning filter

    Input:
        sig: Input signal,
        mode: How the input array is extended beyond its boundaries
            'reflect' = extended by replicating the last n pixels
            'nearest' = extended by replicating the last pixel
            'zero' = extended by n zero
            'ignore' = nothing added
    Return:
        filtered_signal: Filtered signal,
    '''
    filtered_signal = moving_average_filter(
        sig=sig, order=3, b=[0.25, 0.5, 0.25], mode=mode)
    return filtered_signal


def lms_gradient(
    sig: np.ndarray,
    ref: np.ndarray,
    weight: np.ndarray,
    iterations: int = 10000,
    learning_rate: float = 0.01,
    stopping_threshold: float = 1e-6
) -> np.ndarray:
    '''
    A gradient-based method to update weights

    Input:
        sig: Input signal,
        ref: Noise reference,
        weight: weights,
        iterations: Maximum iterations to update weights,
        learning_rate: Learning rate to update weights,
        stopping_threshold: Threshold to stop updating weights,
    Return:
        weight: weights,
    '''
    error = np.inf
    # Estimation of optimal weights
    for idx in range(iterations):
        # Calculating the error
        new_error = np.square(sig) - 2*sig*weight*ref + np.square(weight*ref)
        # If the change in cost is less than or equal to
        #   stopping_threshold we stop the gradient descent
        if np.max(np.abs(error-new_error)) <= stopping_threshold:
            break
        if np.max(np.abs(error)) < np.max(np.abs(new_error)):
            break
        # Updating the weights
        weight = weight + 2*learning_rate*np.sqrt(new_error)*ref
        error = new_error + 0.0
        # Printing the parameters for each 1000th iteration
        # print(f"Iteration {idx+1}: Cost {np.max(np.abs(error))} \
        #       , Weight, {np.mean(weight)}")
    return weight


def adaptive_filter(
        sig: np.ndarray,
        ref: np.ndarray,
        solver: str = 'lms_gradient',
        initial_weight: list = None,
        iterations: int = 10000,
        learning_rate: float = 0.01,
        stopping_threshold: float = 1e-6
) -> np.ndarray:
    '''
    Adaptive filter

    Input:
        sig: Input signal,
        ref: Noise reference,
        solver: Method to minimize the error,
        initial_weight: Initial weights,
        iterations: Maximum iterations to update weights,
        learning_rate: Learning rate to update weights,
        stopping_threshold: Threshold to stop updating weights,
    Return:
        filtered_signal: Filtered signal,
    '''
    sig = np.ravel(sig)
    ref = np.ravel(ref)
    if sig.size != ref.size:
        sys.exit('Adaptive filter: input signal and \
                  reference signal must have same size')
    if initial_weight is None:
        initial_weight = np.ones(ref.size)
    if solver == 'lms_gradient':
        weight = lms_gradient(
            sig=sig, ref=ref, weight=initial_weight,
            iterations=iterations, learning_rate=learning_rate,
            stopping_threshold=stopping_threshold)
    filtered_signal = sig - weight*ref
    return filtered_signal


def order_statistic_filter(
    sig: np.ndarray,
    order: int = 8,
    ftype: str = 'median',
    alpha: float = 0.1,
) -> np.ndarray:
    '''
    Order statistic filter

    Input:
        sig: Input signal,
        order: Order of the filter (window length)
        ftype: Type of the filter ('median', 'min', 'max', 'min-max',
            'alpha-trimmed-mean')
        alpha: Value to remove outliers (0 <= alpha < 0.5)
    Return:
        filtered_signal: Filtered signal,
    '''
    sig = np.ravel(sig)
    if ftype == 'median':
        filtered_signal = np.array(
            [np.median(sig[k-order:k]) for k in range(order, sig.size)])
    elif ftype == 'min':
        filtered_signal = np.array(
            [np.min(sig[k-order:k]) for k in range(order, sig.size)])
    elif ftype == 'max':
        filtered_signal = np.array(
            [np.max(sig[k-order:k]) for k in range(order, sig.size)])
    elif ftype == 'min-max':
        m_sig = np.array(
            [np.max(sig[k-order:k]) for k in range(order, sig.size)])
        filtered_signal = np.array(
            [np.max(m_sig[k-order:k]) for k in range(order, m_sig.size)])
    elif ftype == 'alpha-trimmed-mean':
        if alpha >= 0 and alpha < 0.5:
            filtered_signal = np.array([])
            for k in range(order, sig.size):
                low_bound = np.percentile(
                    sig[k-order:k], alpha*100.0)
                high_bound = np.percentile(
                    sig[k-order:k], (1-alpha)*100.0)
                win_trimmed = np.array([x for x in sig[
                        k-order:k] if low_bound <= x < high_bound])
                filtered_signal = np.append(
                    filtered_signal, np.mean(win_trimmed))
        else:
            sys.exit('Alpha must be 0 <= alpha < 0.5')
    else:
        sys.exit('Filter type has not been defined!')
    return filtered_signal


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
        sig=sigs[:, 0], samp_freq=fs, win_length=10.0)

    bandpass_filter(
        sig=sigs[:, 0], low_cutoff=10.0, high_cutoff=50.0,
        filt_design='butter', samp_freq=fs, order=5,
        freq_response=True, zero_pole=True)

    notch_filter(
        sig=sigs[:, 0], cutoff=10.0, samp_freq=fs,
        freq_response=True, zero_pole=True)
