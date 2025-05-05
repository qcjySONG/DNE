import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pickle
from scipy.stats import linregress
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    # 设置英文字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    # 设置全局字体大小
plt.rcParams.update({'font.size': 14})
np.set_printoptions(threshold=10000, linewidth=10000, suppress=True)  # 设置使np输出时更美观

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

    # 保证输入是二维数组 (N, 1)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.shape[1] != 1:
        raise ValueError("输入 data 必须是形状为 (N, 1) 的数组")
    
    N = data.shape[0]
    if M >= N:
        raise ValueError("时间窗口长度 M 必须小于数据长度 N")

    entropies = []

    # 预计算 log(M)，避免重复运算
    logM = np.log(M)

    for t in range(M, N):
        # 构造两个相邻的窗口 W_t 和 W_{t+1}
        W_t = data[t - M:t, 0]
        W_tp1 = data[t - M + 1:t + 1, 0]

        # 计算各自的标准差
        SD_W_t = np.std(W_t, ddof=1)  # 使用无偏估计
        SD_W_tp1 = np.std(W_tp1, ddof=1)

        # 计算变化率项
        delta_SD = np.abs(SD_W_tp1 - SD_W_t) / logM

        # 计算 W_t 中每个元素对应的概率 P_i
        sum_W_t = np.sum(W_t)
        if sum_W_t == 0:
            P = np.full(M, 1.0 / M)  # 如果和为0，避免除零，均匀分布
        else:
            P = W_t / sum_W_t
        
        # 计算 -sum(P_i log P_i)
        epsilon = 1e-12
        entropy_term = -np.sum(P * np.log(P + epsilon))

        # 计算最终 H(t)
        H_t = delta_SD * entropy_term
        entropies.append(H_t)

    entropies = np.array(entropies)

    # --------- 核心修改部分 ---------
    # 将所有 NaN 替换成 1
    entropies = np.where(np.isnan(entropies), 0, entropies)

    return entropies.tolist()

# -------------------------------
# 测试部分
# -------------------------------

T=40

# 读取 CSV 文件中的 Cases 列作为模拟数据
file_path = f'./Simulated_Data/processed_ts_T{T}.csv'
# 使用 pandas 读取数据，只取 'Cases' 列
data = pd.read_csv(file_path, usecols=['Cases'])
R0_data = pd.read_csv(file_path, usecols=['R0']).values  # 直接作为一维数组使用
# 转换为 numpy 数组，并 reshape 成 (N, 1) 形式
simulated_data = data.values.reshape(-1, 1)
cases_series = simulated_data.flatten()
# 加载 pkl 文件
with open("./Simulated_Data/delta_by_T.pkl", "rb") as f:
    delta_by_T = pickle.load(f)



# 设定窗口长度
M = 100
# 获取数据长度 N
N = len(simulated_data)


#print(len(H_values))#N-M


#其它值计算
def compute_shannon_entropy_vec(windows):
    """
    输入形状: (N, M)，N个长度为M的窗口
    输出: 每个窗口的香农熵
    """
    entropies = []
    for window in windows:
        vals, counts = np.unique(window, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-12))  # 加1e-12防止 log(0)
        entropies.append(entropy)
    return np.array(entropies)

def compute_kolmogorov_complexity_vec(windows):
    """
    输入: N个窗口组成的二维数组
    输出: 每个窗口对应的Kolmogorov复杂度（LZ复杂度）
    """
    complexities = []
    for window in windows:
        # 去趋势处理（线性残差）
        x = np.arange(len(window))
        slope, intercept, *_ = linregress(x, window)
        detrended = window - (slope * x + intercept)

        # 转换为二进制序列
        bit_seq = ''.join(['1' if v > 0 else '0' for v in detrended])

        # LZ复杂度估算
        complexities.append(lempel_ziv_complexity(bit_seq))
    return np.array(complexities)

def lempel_ziv_complexity(s):
    """Lempel-Ziv复杂度估算，用于Kolmogorov复杂度近似"""
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
    使用stride tricks生成滑动窗口，形状为(N, M)
    """
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(arr, window_shape=M)

def compute_entropy_and_complexity_fast(time_series, M):
    windows = sliding_windows(time_series, M)  # 形状 (N, M)
    H_vals = compute_shannon_entropy_vec(windows)
    KC_vals = compute_kolmogorov_complexity_vec(windows)
    return H_vals, KC_vals

def compute_mean(window):
    """
    计算滑动窗口内的均值
    """
    return np.mean(window)

def compute_variance(window, mean):
    """
    计算滑动窗口内的方差
    """
    return np.var(window)

def compute_dispersion_index(variance, mean):
    """
    计算离散指数（方差 / 均值）
    """
    return variance / mean if mean != 0 else np.inf

def compute_autocorrelation(window, mean):
    """
    计算滑动窗口内的自相关（滞后1期）
    """
    n = len(window)
    numerator = np.sum((window[:-1] - mean) * (window[1:] - mean))
    denominator = np.sum((window - mean) ** 2)
    return numerator / denominator if denominator != 0 else 0

def compute_correlation_time(window, mean):
    """
    计算滑动窗口内的相关时间（通过自相关函数拟合）
    """
    n = len(window)
    ac = np.array([compute_autocorrelation(window[:i+1], mean) for i in range(1, n)])
    
    # 拟合 AC(t) = exp(-t / tau)
    log_ac = np.log(ac[ac > 0])  # 只保留大于0的自相关值
    time_lags = np.arange(1, len(log_ac) + 1)
    
    # 只解包 slope 和 intercept
    slope, intercept, _, _, _ = linregress(time_lags, log_ac)
    
    # 相关时间是时间常数 tau
    tau = -1 / slope if slope < 0 else np.inf
    return tau


def sliding_windows(arr, M):
    """
    使用stride tricks生成滑动窗口，形状为(N, M)
    """
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(arr, window_shape=M)

def compute_statistics(time_series, M):
    windows = sliding_windows(time_series, M)  # 形状 (N, M)
    
    # 初始化结果数组
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

#------------------值计算-----------------------------
# 计算窗口熵
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




