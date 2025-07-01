import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pickle
from scipy.stats import linregress
plt.rcParams['font.sans-serif'] = ['SimHei']  # Specify default font
    # Set English font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of the negative sign '-' showing as a square when saving images
    # Set global font size
plt.rcParams.update({'font.size': 14})
np.set_printoptions(threshold=10000, linewidth=10000, suppress=True)  # Set np output to be more visually appealing

def ztest(x, m, sigma, alpha=0.05, alternative='two-sided'):
    # Error checking
    if not np.isscalar(m):
        raise ValueError("M must be a scalar")
    if not np.isscalar(sigma) or sigma < 0:
        raise ValueError("Sigma must be a non-negative scalar")
    # Calculate necessary values
    xmean = np.mean(x)
    samplesize = 1
    ser = sigma / np.sqrt(samplesize)
    zval = (xmean - m) / ser
    # Compute p-value
    if alternative == 'two-sided':
        p = 2 * norm.cdf(-np.abs(zval))
    elif alternative == 'greater':
        p = norm.cdf(-zval)
    elif alternative == 'less':
        p = norm.cdf(zval)
    else:
        raise ValueError("Invalid alternative hypothesis")

    # Compute confidence interval if necessary
    if alternative == 'two-sided':
        crit = norm.ppf(1 - alpha / 2) * ser
        ci = (xmean - crit, xmean + crit)
    elif alternative == 'greater':
        crit = norm.ppf(1 - alpha) * ser
        ci = (xmean - crit, np.inf)
    elif alternative == 'less':
        crit = norm.ppf(1 - alpha) * ser
        ci = (-np.inf, xmean + crit)

    # Determine if the null hypothesis can be rejected
    h = 1 if p <= alpha else 0
    return h,p

def compute_window_entropy(data: np.ndarray, M: int) -> np.ndarray:

    # Ensure the input is a 2D array (N, 1)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.shape[1] != 1:
        raise ValueError("Input data must be an array of shape (N, 1)")
    
    N = data.shape[0]
    if M >= N:
        raise ValueError("Time window length M must be less than data length N")

    entropies = []

    # Pre-calculate log(M) to avoid repeated computation
    logM = np.log(M)

    for t in range(M, N):
        # Construct two adjacent windows W_t and W_{t+1}
        W_t = data[t - M:t, 0]
        W_tp1 = data[t - M + 1:t + 1, 0]

        # Calculate their respective standard deviations
        SD_W_t = np.std(W_t, ddof=1)  # Use unbiased estimation
        SD_W_tp1 = np.std(W_tp1, ddof=1)

        # Calculate the rate of change term
        delta_SD = np.abs(SD_W_tp1 - SD_W_t) / logM

        # Calculate the probability P_i corresponding to each element in W_t
        sum_W_t = np.sum(W_t)
        if sum_W_t == 0:
            P = np.full(M, 1.0 / M)  # If the sum is 0, avoid division by zero, use a uniform distribution
        else:
            P = W_t / sum_W_t
        
        # Calculate -sum(P_i log P_i)
        epsilon = 1e-12
        entropy_term = -np.sum(P * np.log(P + epsilon))

        # Calculate the final H(t)
        H_t = delta_SD * entropy_term
        entropies.append(H_t)

    entropies = np.array(entropies)

    # --------- Core modification part ---------
    # Replace all NaNs with 0
    entropies = np.where(np.isnan(entropies), 0, entropies)

    return entropies.tolist()

# -------------------------------
# Testing part
# -------------------------------

T=40

# Read the 'Cases' column from the CSV file as simulated data
file_path = f'./Simulated_Data/processed_ts_T{T}.csv'
# Use pandas to read data, taking only the 'Cases' column
data = pd.read_csv(file_path, usecols=['Cases'])
R0_data = pd.read_csv(file_path, usecols=['R0']).values  # Use directly as a one-dimensional array
# Convert to a numpy array and reshape to (N, 1)
simulated_data = data.values.reshape(-1, 1)
cases_series = simulated_data.flatten()
# Load the pkl file
with open("./Simulated_Data/delta_by_T.pkl", "rb") as f:
    delta_by_T = pickle.load(f)



# Set window length
M = 100
# Get data length N
N = len(simulated_data)


#print(len(H_values))#N-M


# Other value calculations
def compute_shannon_entropy_vec(windows):
    """
    Input shape: (N, M), N windows of length M.
    Output: Shannon entropy for each window.
    """
    entropies = []
    for window in windows:
        vals, counts = np.unique(window, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-12))  # Add 1e-12 to prevent log(0)
        entropies.append(entropy)
    return np.array(entropies)

def compute_kolmogorov_complexity_vec(windows):
    """
    Input: A 2D array composed of N windows.
    Output: Kolmogorov complexity (LZ complexity) for each window.
    """
    complexities = []
    for window in windows:
        # Detrending (linear residuals)
        x = np.arange(len(window))
        slope, intercept, *_ = linregress(x, window)
        detrended = window - (slope * x + intercept)

        # Convert to a binary sequence
        bit_seq = ''.join(['1' if v > 0 else '0' for v in detrended])

        # LZ complexity estimation
        complexities.append(lempel_ziv_complexity(bit_seq))
    return np.array(complexities)

def lempel_ziv_complexity(s):
    """Lempel-Ziv complexity estimation, used for Kolmogorov complexity approximation"""
    i, c, l = 0, 1, 1
    n = len(s)
    k = 1
    while True:
        if i + k == n:
            c += 1
            break
        if s[i:i + k] == s[l:l + k]:
            k += 1
        else:
            i += 1
            if i == l:
                c += 1
                l += k
                i = 0
                k = 1
            else:
                k = 1
    return c

def sliding_windows(arr, M):
    """
    Use stride tricks to generate sliding windows, shape (N, M)
    """
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(arr, window_shape=M)

def compute_entropy_and_complexity_fast(time_series, M):
    windows = sliding_windows(time_series, M)  # Shape (N, M)
    H_vals = compute_shannon_entropy_vec(windows)
    KC_vals = compute_kolmogorov_complexity_vec(windows)
    return H_vals, KC_vals

def compute_mean(window):
    """
    Calculate the mean within the sliding window
    """
    return np.mean(window)

def compute_variance(window, mean):
    """
    Calculate the variance within the sliding window
    """
    return np.var(window)

def compute_dispersion_index(variance, mean):
    """
    Calculate the dispersion index (variance / mean)
    """
    return variance / mean if mean != 0 else np.inf

def compute_autocorrelation(window, mean):
    """
    Calculate the autocorrelation within the sliding window (lag 1)
    """
    n = len(window)
    numerator = np.sum((window[:-1] - mean) * (window[1:] - mean))
    denominator = np.sum((window - mean) ** 2)
    return numerator / denominator if denominator != 0 else 0

def compute_correlation_time(window, mean):
    """
    Calculate the correlation time within the sliding window (by fitting the autocorrelation function)
    """
    n = len(window)
    ac = np.array([compute_autocorrelation(window[:i+1], mean) for i in range(1, n)])
    
    # Fit AC(t) = exp(-t / tau)
    log_ac = np.log(ac[ac > 0])  # Only keep autocorrelation values greater than 0
    time_lags = np.arange(1, len(log_ac) + 1)
    
    # Unpack only slope and intercept
    slope, intercept, _, _, _ = linregress(time_lags, log_ac)
    
    # The correlation time is the time constant tau
    tau = -1 / slope if slope < 0 else np.inf
    return tau


def sliding_windows(arr, M):
    """
    Use stride tricks to generate sliding windows, shape (N, M)
    """
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(arr, window_shape=M)

def compute_statistics(time_series, M):
    windows = sliding_windows(time_series, M)  # Shape (N, M)
    
    # Initialize result arrays
    means, variances, dispersion_indices, autocorrelations, correlation_times = [], [], [], [], []
    
    for window in windows:
        mean = compute_mean(window)
        variance = compute_variance(window, mean)
        dispersion_index = compute_dispersion_index(variance, mean)
        autocorrelation = compute_autocorrelation(window, mean)
        correlation_time = compute_correlation_time(window, mean)
        
        means.append(mean)
        variances.append(variance)
        dispersion_indices.append(dispersion_index)
        autocorrelations.append(autocorrelation)
        correlation_times.append(correlation_time)
    
    return np.array(means), np.array(variances), np.array(dispersion_indices), np.array(autocorrelations), np.array(correlation_times)

#------------------Value Calculation-----------------------------
# Calculate window entropy
H_values = compute_window_entropy(simulated_data, M)
H_vals, KC_vals = compute_entropy_and_complexity_fast(cases_series, M)
means, variances, dispersion_indices, autocorrelations, correlation_times = compute_statistics(cases_series, M)

print(len(H_values),len(H_vals),len(KC_vals),len(means),len(variances),len(dispersion_indices),len(autocorrelations),len(correlation_times))#420 421 421 421 421 421 421 421




It_p_list = []
Twindow=99
for i in range(Twindow,len(H_values)):
    It_p_temp = H_values[i - Twindow:i]
    It_p_temp_np = np.array(It_p_temp)
    hh, pp = ztest(H_values[i], np.mean(It_p_temp_np), np.std(It_p_temp_np, ddof=1))
    It_p_list.append(pp)
It_p_list = np.nan_to_num(It_p_list, nan=0)




