import math
import pandas as pd
import xlrd
import os
import numpy as np
import pickle
from scipy.stats import pearsonr,spearmanr,kendalltau
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")
# 检查目录是否存在，如果不存在则创建
directory = 'son_img_FLU_fangzhen_52'
if not os.path.exists(directory):
    os.makedirs(directory)
data_test = np.load('./data/testdata_52.pkl', allow_pickle=True)
print("data_test shape:", data_test.shape)  # 应为 (208, 23)

    # 设置中文字体
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



for gap_year in range(25,26):
    time_window = 6
    file_start_week = gap_year
    # 标准化权重
    weights = [9, 25, 5, 10, 11, 4, 7, 14, 8, 21, 7, 11, 17, 16, 19, 12, 17, 7, 8, 21, 7, 13, 21]

    # 加载 data_test 数据


    # 遍历年份 (从第2年开始，f=16对应2000年，f=17对应2001年，f=18对应2002年)
    for f in range(1,4 ):  # 仅处理 2000-2002 年的数据
        # 提取当前年的 52 行数据
        start_row_current_year = 52 * f
        end_row_current_year = start_row_current_year + 52
        current_year_data = data_test[start_row_current_year:end_row_current_year, :]  # (52, 23)
        week=58
        # 按列求和，得到 infection_number
        infection_number = current_year_data.sum(axis=1).tolist()  # 每行的 23 列相加，得到长度为 52 的列表
        max_num = max(infection_number)  # 最大值
        index = infection_number.index(max_num) + 1  # 爆发期的位置（从 1 开始计数）

        print(f"Year {f}: Max Infection Number = {max_num}, Index = {index}")

        # 提取前一年的最后 6 行数据
        start_row_previous_year = 52 * (f - 1) + 46  # 前一年的最后 6 行
        end_row_previous_year = 52 * (f - 1) + 52
        previous_year_data = data_test[start_row_previous_year:end_row_previous_year, :]  # (6, 23)

        # 合并前一年的 6 行和当前年的 52 行，形成 (58, 23) 的数组
        combined_data = np.vstack((previous_year_data, current_year_data))  # (58, 23)

        # 标准化 combined_data
        data = combined_data / weights  # 按列标准化
        # network
        network = [[0, 2, 5, 11, 17, 7],
                   [1, 9, 18, 17, 12],
                   [2, 0, 5, 6, 8, 15],
                   [3, 19, 16, 11, 17, 12],
                   [4, 13, 16, 10, 20, 22],
                   [5, 0, 11, 10, 6, 2],
                   [6, 2, 5, 10, 20, 8],
                   [7, 0, 17, 18, 9],
                   [8, 2, 6, 20, 21, 14, 15],
                   [9, 1, 18, 7],
                   [10, 5, 11, 16, 4, 20, 6],
                   [12, 19, 3, 17, 1],
                   [13, 19, 16, 4],
                   [11, 0, 17, 3, 16, 10, 5],
                   [14, 15, 8, 21],
                   [15, 2, 8, 14],
                   [16, 19, 3, 11, 10, 4, 13],
                   [17, 1, 12, 3, 11, 0, 7, 18],
                   [18, 1, 9, 7, 17],
                   [19, 12, 3, 16, 13],
                   [20, 22, 4, 21, 8, 6, 10],
                   [21, 22, 20, 8, 14],
                   [22, 4, 20, 21]]

        # 从网络中提取边的信息,为了顺利绘图，无向图
        edge = []
        for item in network:
            for i in range(1, item.__len__()):
                if [item[0], item[i]] in edge or [item[i], item[0]] in edge:
                    continue
                else:
                    edge.append([item[0], item[i]])

        delta_PCC = np.zeros((network.__len__(), week - time_window))  # (23 , 52)
        numTemp = 0
        for i in range(0, week - time_window):  # 52  week=57，时间窗口多5
            # 时间for
            b = 0
            for node in network:  # 23
                # 地图for
                # 两者之间的皮尔逊相关系数
                pcc_temp3 = 0
                pcc_ij = []
                pcct_ij = []
                for j in range(1, node.__len__()):  # 遍历邻接点
                    # 邻接点for
                    # node[0]为当前中心结点
                    pcc_temp1 = abs(kendalltau(data[i: i + time_window, node[0]], data[i: i + time_window, node[j]])[
                                        0])  # 中心节点t-1时刻与其它节点的皮尔森相关系数

                    pcc_temp2 = abs(kendalltau(data[i + 1: i + time_window + 1, node[0]],
                                               data[i + 1: i + time_window + 1, node[j]])[0])  # 中心节点t时刻与其它节点的皮尔森相关系数
                    weight = abs(pcc_temp2 - pcc_temp1)  # dieta_PCCt
                    weight_zi = []
                    weight_zi.append(i)
                    weight_zi.append(node[0])
                    weight_zi.append(node[j])
                    weight_zi.append(weight)
                    # 中心节点与一阶邻点的权重
                    pcc_ij.append(weight)
                    pcc_temp3 += weight  # 求概率

                if pcc_temp3 == 0:  # 总和为0，概率当然为0
                    delta_PCC[b][i] == 0
                    b += 1
                    continue  # 跳出该中心节点的范围
                for k in range(0, pcc_ij.__len__()):  # 为什么不用len(pcc_ij)
                    pcct_ij.append(pcc_ij[k] / pcc_temp3)  # 求概率
                # 计算零的个数
                # zero_count = sum(1 for prob in pcct_ij if prob == 0)
                entropy = 0
                for h in range(0, pcct_ij.__len__()):
                    if pcct_ij[h] > 0:  # 避免log(0)的情况
                        entropy -= (pcct_ij[h]) * math.log(pcct_ij[h], 2)

                sd_temp1 = np.std(data[i: i + time_window, node[0]], ddof=1)
                sd_temp2 = np.std(data[i + 1: i + time_window + 1, node[0]], ddof=1)
                delta_PCC[b][i] = math.sqrt(abs(sd_temp2 - sd_temp1)) * entropy  # 单个节点的It指标
                b += 1
                
        # 求和
        It = []  # 所有节点的It值
        for i in range(0, week - time_window):
            # delta_PCC (23 , 52)
            It.append(np.sum(delta_PCC[:, i]) / network.__len__())  # 计算了所选列中所有数值的总和均






