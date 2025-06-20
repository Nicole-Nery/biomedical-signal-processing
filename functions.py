import numpy as np
from scipy.signal import square, sawtooth
import pandas as pd
from pyts.image import GramianAngularField
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
import antropy as ant
import streamlit as st


def generateSine(offset, amp, fs, tf, f):
    t = np.linspace(0, tf, int(fs * tf))
    sine = offset + amp * np.sin(2 * np.pi * f * t)
    df_sine = pd.DataFrame({'t': t, 'signal': sine})
    return df_sine

def generateSquare(offset, amp, fs, tf, f):
    t = np.linspace(0, tf, int(fs * tf))
    square_wave = offset + amp * square(2 * np.pi * f * t)
    df_square = pd.DataFrame({'t': t, 'signal': square_wave})
    return df_square

def generateTriangle(offset, amp, fs, tf, f):
    t = np.linspace(0, tf, int(fs * tf))
    triangular = offset + amp * sawtooth(2 * np.pi * f * t, width=0.5) 
    df_triangular = pd.DataFrame({'t': t, 'signal': triangular})
    return df_triangular

def generateRandomSignal(offset, amp, fs, tf, distribution):
    t = np.linspace(0, tf, int(fs * tf))

    if distribution == 'normal':
        signal = np.random.normal(0, 1, size=t.shape) # Mean = 0 and Standard Deviation = 1
    elif distribution == 'uniform':
        signal = np.random.uniform(-1, 1, size=t.shape) # Values are uniformly distributed between -1 and 1, 
                                                        # meaning every number in this interval has the same probability of being chosen
    elif distribution == 'binomial':        
        signal = np.random.binomial(n=1, p=0.5, size=t.shape) * 2 - 1   # n=1 means one trial per example
                                                                        # p=0.5 means a 50% chance of getting a 1 (success)
                                                                        # The result is then transformed to -1 or 1 by (0 or 1)*2-1
    signal = offset + amp * signal
    df_signal_aleatorio = pd.DataFrame({'t': t, 'signal': signal})
    return df_signal_aleatorio

def addNoise (df_signal, snr_dB):
    t = df_signal['t']
    signal = df_signal['signal']

    # Calculate the noise power needed to achieve the desired Signal-to-Noise Ratio (SNR)
    # SNR (in linear scale) = signal_power / noise_power
    # signal_power is estimated using the variance of the signal
    # Rearranged to: noise_power = signal_power / SNR
    # Since SNR is given in dB, convert it to linear scale using 10^(SNR_dB / 10)
    noise_power = np.var(signal) / (10**(snr_dB / 10))

    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape) # White Gaussian noise (Mean = 0, Standard Deviation = sqrt(noise_power))
                                                                         # This gives noise with the power that matches the desired SNR
    signal_with_noise = signal + noise
    df_signal_with_noise = pd.DataFrame({'t': t, 'signal': signal_with_noise}) 
    return df_signal_with_noise

def addTrend (df_signal, tipo, a, b, c, offset, amp, f):
    t = df_signal['t']
    signal = df_signal['signal']

    if tipo == 'linear':
        trend = a * t + b
    elif tipo == 'quadratic':
        trend = a * t**2 + b * t + c
    elif tipo == 'sinusoidal':
        trend = offset + amp * np.sin(2 * np.pi * f * t)

    signal_with_trend = signal + trend
    df_with_trend = pd.DataFrame({'t': t, 'signal': signal_with_trend})
    return df_with_trend

def addDiscontinuity(df_signal, t_jump, jump_value):
    t = df_signal['t']
    signal = df_signal['signal'].copy()

    # Adds a jump (discontinuity) in the signal from time t_jump onward
    signal[t >= t_jump] += jump_value
    df_discontinuity = pd.DataFrame({'t': t, 'signal': signal})
    return df_discontinuity

def addSuddenChange(df_signal, t_change, new_amp):
    t = df_signal['t']
    signal = df_signal['signal'].copy()

    # Calculates the amplitude before the change, to scale the rest of the signal
    old_max = np.max(np.abs(signal[t < t_change]))
    factor = new_amp / old_max if old_max != 0 else 1

    # Applies the scaling factor to the signal from t_change onward
    signal[t >= t_change] *= factor

    df_mudanca = pd.DataFrame({'t': t, 'signal': signal})
    return df_mudanca

def applyGAF(df, chosen_method='summation'):
    signal = df['signal']
    
    # Reshapes the signal to the format expected by the GAF transformer (1 sample, many time points)
    X = np.array(signal).reshape(1, -1) 

    gaf = GramianAngularField(method=chosen_method)
    X_gaf = gaf.fit_transform(X)
    return X_gaf[0] 

# Mean Absolute Value
def compute_mav(signal):
    return round(np.mean(np.abs(signal)), 6)

# Mean Absolute Value of First Differences
def compute_mavfd(signal):
    diffs = np.diff(signal)
    return round(np.mean(np.abs(diffs)), 6)

# Mean Absolute Value of Second Differences
def compute_mavsd(signal):
    diffs = np.abs(signal[2:] - signal[:-2])
    return round(np.mean(diffs), 6)

# Root Mean Square
def compute_rms(signal):
    return round(np.sqrt(np.mean(np.square(signal))), 6)

# Peak Value
def compute_peak_value(signal):
    return round(np.max(signal), 6)

# Zero Crossing
def compute_zero_crossings(signal):
    signs = np.sign(signal)
    crossings = np.where(np.diff(signs))[0]
    return round(len(crossings), 6)

# Mean Frequency
def compute_mean_frequency(freqs, mag):
    return round(np.sum(freqs * mag) / np.sum(mag), 6)

# Peak Frequency
def compute_peak_frequency(freqs, mag):
    return round(freqs[np.argmax(mag)], 6)

# Frequency at 50% Power
def compute_f50(freqs, mag):
    cumsum = np.cumsum(mag)
    target = np.sum(mag) * 0.5
    return round(freqs[np.argmin(np.abs(cumsum - target))], 6)

# Frequency at 80% Power
def compute_f80(freqs, mag):
    cumsum = np.cumsum(mag)
    target = np.sum(mag) * 0.8
    return round(freqs[np.argmin(np.abs(cumsum - target))], 6)

# Power in 3.5–7.5 Hz Band
def compute_band_power_3_5_to_7_5(freqs, mag):
    indices = (freqs > 3.5) & (freqs < 7.5)
    return round(np.sum(mag[indices]), 6)

# Fuzzy Entropy
def compute_fuzzy_entropy(signal, dim=2, r=0.15):
    def ufunc(d, r):
        return np.exp(-(d / r) ** 2)

    N = len(signal)
    correl = []
    dataMat = np.zeros((dim + 1, N - dim))

    for i in range(dim + 1):
        dataMat[i, :] = signal[i:N - dim + i] - np.mean(signal[i:N - dim + i])

    for m in range(dim, dim + 2):
        cont = []
        tempMat = dataMat[:m, :]

        for i in range(N - dim):
            if i >= N - dim - 1:
                dist = 0
                D = ufunc(dist, r)
                cont.append(np.sum(D) / (N - dim - 1))
            else:
                diffs = tempMat[:, i + 1:] - tempMat[:, [i]]
                dist = np.max(np.abs(diffs), axis=0)
                D = ufunc(dist, r)
                cont.append(np.sum(D) / (N - dim - 1))

        correl.append(np.mean(cont))

    fuzzy_en = np.log(correl[0] / correl[1]) if correl[1] != 0 else 0
    return round(fuzzy_en, 6)

# Approximate Entropy (requires antropy)
def compute_approximate_entropy(signal, dim=2):
    return round(ant.app_entropy(signal, order=dim, metric='chebyshev'), 6)

# Variance
def compute_variance(signal):
    return round(np.var(signal, ddof=1), 6)

# Range
def compute_range(signal):
    return round(np.ptp(signal), 6)

# Interquartile Range
def compute_iqr(signal):
    return round(pd.Series(signal).quantile(0.75) - pd.Series(signal).quantile(0.25), 6)

# Skewness
def compute_skewness(signal):
    return round(skew(signal), 6)

# Kurtosis
def compute_kurtosis(signal):
    return round(kurtosis(signal), 6)


# Hjorth Parameter Functions
def activity(x):
    return np.var(x)

def mobility(x):
    return np.sqrt(np.var(np.diff(x)) / np.var(x)) if np.var(x) != 0 else 0

def complexity(x):
    num = mobility(np.diff(x))
    denom = mobility(x)
    return num / denom if denom != 0 else 0


def extract_features(df_signal):
    t = df_signal['t'].values
    signal = df_signal['signal'].values
    sampling_rate = 1 / (t[1] - t[0])

    N = len(signal)
    freqs = fftfreq(N, d=1/sampling_rate)[:N//2]
    fft_magnitude = np.abs(fft(signal))[:N//2]

    features = {
        'MAV': compute_mav(signal),
        'MAVFD': compute_mavfd(signal),
        'MAVSD': compute_mavsd(signal),
        'RMS': compute_rms(signal),
        'PEAK_VALUE': compute_peak_value(signal),
        'ZERO_CROSSINGS': compute_zero_crossings(signal),
        'MEAN_FREQUENCY': compute_mean_frequency(freqs, fft_magnitude),
        'PEAK_FREQUENCY': compute_peak_frequency(freqs, fft_magnitude),
        'F50': compute_f50(freqs, fft_magnitude),
        'F80': compute_f80(freqs, fft_magnitude),
        'BAND_POWER_3_5_TO_7_5': compute_band_power_3_5_to_7_5(freqs, fft_magnitude),
        'FUZZY_ENTROPY': compute_fuzzy_entropy(signal),
        'APPROX_ENTROPY': compute_approximate_entropy(signal),
        'VARIANCE': compute_variance(signal),
        'RANGE': compute_range(signal),
        'IQR': compute_iqr(signal),
        'SKEWNESS': compute_skewness(signal),
        'KURTOSIS': compute_kurtosis(signal),
        'ACTIVITY': activity(signal),
        'MOBILITY': mobility(signal),
        'COMPLEXITY': complexity(signal),
    }

    return features


def extract_features_windowed(df_signal, window_size_sec, overlap_percent=0.5):
    t = df_signal['t'].values
    signal = df_signal['signal'].values
    sampling_rate = 1 / (t[1] - t[0])
    samples_per_window = int(window_size_sec * sampling_rate)

    step_size = int(samples_per_window * (1 - overlap_percent))

    all_features = []
    i = 0
    window_count = 1

    while (i + samples_per_window) <= len(signal):
        start = i
        end = start + samples_per_window
        window_signal = signal[start:end]
        window_time = t[start:end]

        if len(window_signal) == 0 or len(window_time) == 0:
            i += step_size
            continue

        df_window = pd.DataFrame({
            't': window_time,
            'signal': window_signal
        })

        features = extract_features(df_window)
        features['Window'] = window_count
        features['Mean_Time'] = window_time.mean()
        all_features.append(features)

        window_count += 1
        i += step_size

    df_features = pd.DataFrame(all_features)

    # Setting Window and Mean_Time as the first columns
    cols = ['Window', 'Mean_Time'] + [c for c in df_features.columns if c not in ['Window', 'Mean_Time']]
    df_features = df_features[cols].reset_index(drop=True)

    return df_features


