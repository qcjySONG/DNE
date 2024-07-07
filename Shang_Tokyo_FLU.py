#QBS  1601311287@qq.com
import math
import pandas as pd
import xlrd
import os
import numpy as np
from scipy.stats import pearsonr,spearmanr,kendalltau
import xlwt
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.stats import t, ttest_1samp
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


def ttest(x, m=0, alpha=0.05, tail='both'):
    # Sample size
    n = len(x)

    # Sample mean and standard deviation
    xmean = np.mean(x)
    sd = np.std(x, ddof=1)  # ddof=1 for unbiased estimate of population variance

    # t-statistic
    tval = (xmean - m) / (sd / np.sqrt(n))

    # Degrees of freedom
    df = n - 1

    # Compute p-value
    if tail == 'both':
        p = 2 * t.cdf(-abs(tval), df)
        ci = (xmean - t.ppf(1 - alpha / 2, df) * sd / np.sqrt(n),
              xmean + t.ppf(1 - alpha / 2, df) * sd / np.sqrt(n))
    elif tail == 'right':
        p = t.cdf(-tval, df)
        ci = (xmean - t.ppf(1 - alpha, df) * sd / np.sqrt(n), np.inf)
    elif tail == 'left':
        p = t.cdf(tval, df)
        ci = (-np.inf, xmean + t.ppf(1 - alpha, df) * sd / np.sqrt(n))

    # Determine if the actual significance exceeds the desired significance
    h = p <= alpha

    # Create stats dictionary
    stats = {'tstat': tval, 'df': df, 'sd': sd}

    return h, p




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


#需要调整参数
'''
HFMD:
time_window=6
file_start_week=6
(10, 20)


flu 25


'''
for gap_year in range(25,26):
    time_window = 6
    file_start_week = gap_year

    book1 = xlwt.Workbook()
    book2 = xlwt.Workbook()
    book3 = xlwt.Workbook()
    book4 = xlwt.Workbook()

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    # 设置英文字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 14})

    np.set_printoptions(threshold=10000, linewidth=10000, suppress=True)  # 设置使np输出时更美观

    # 读取Excel文件
    excel_file = "data/东京都流感人数.xls"  # "data/东京都手足口病人数.xls" #"data/东京都流感人数.xls" "data/东京都肠胃炎人数.xls"
    df = pd.read_excel(excel_file)
    # 获取做图时需要的行数，列数固定每行两列
    df_len = len(df) - file_start_week  # 减去22是什么意思
    year = df_len // 52
    if year % 2 == 1:
        rows = (year // 2) + 1
    else:
        rows = year // 2

    point_num = df.shape[1] - 2  # 假设每个文件的前两列均为 时间 和 总和

    # 作图的起始年份
    start_year = round(df.iloc[1][0])

    # 创建图层
    fig = plt.figure(figsize=(30, 7 * rows))
    fig1 = plt.figure(figsize=(18, 4 * rows))
    gap = []  # 提前多少周预测出来？
    It_5 = []  # 记录后5个It值
    for f in range(8, 19):  # 由于程序编写的问题 1-21为2000-2024的
        '''
        #年：{
            地区：{It分数，每周}
            预警点
            爆发点
        }
        '''

        weight_str = []  # 权重文件
        # 感染人数数据，作图用
        infection_number = df.iloc[file_start_week + 52 * f: 52 + file_start_week + 52 * f,
                           1].tolist()  # iloc是根据行号进行切片，行号从0开始
        # 起始行 22 + 52 * f  终止行 74 + 52 * f  74=22+52  选取第2列（总感染人数列）
        max_num = max(infection_number)
        index = infection_number.index(max_num) + 1  # 应该是爆发期绘图的位置

        startweek = file_start_week - time_window + 1 + 52 * f  # 18=22-4
        endweek = 52 + file_start_week + 52 * f  # 74=52+22

        week = endweek - startweek + 1

        book = xlrd.open_workbook(
            'data/东京都流感定点报告数.xls')  # 应该是点的内容  'data/东京都手足口病定点报告数.xls' 'data/东京都流感定点报告数.xls' 'data/东京都肠胃炎定点报告数.xls'
        sheet = book.sheet_by_index(0)
        data = np.zeros((week, point_num))  # 23是 东京都 有23个地区
        for col in range(startweek, endweek + 1):
            for row in range(0, point_num):
                data[col - startweek][row] = sheet.cell(col, row + 1).value  # read data

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

        sheet1 = book1.add_sheet(str(startweek) + '-' + str(endweek))
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
                    # 为什么要注释掉？
                    # pcc_temp1 = np.nan_to_num(pcc_temp1)
                    # pcc_temp2 = np.nan_to_num(pcc_temp2)
                    weight = abs(pcc_temp2 - pcc_temp1)  # dieta_PCCt
                    weight_zi = []
                    weight_zi.append(i)
                    weight_zi.append(node[0])
                    weight_zi.append(node[j])
                    weight_zi.append(weight)
                    weight_str.append(weight_zi)
                    # 中心节点与一阶邻点的权重
                    pcc_ij.append(weight)
                    # print('节点', node[0], '与节点', node[j], '的权重', weight)
                    # 总和
                    pcc_temp3 += weight  # 求概率
                # print('节点', node[0], pcc_temp3)
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
                # entropy = (1/(math.log(node.__len__(), 2))) * entropy
                # entropy = (1 / 100) * entropy
                # 标准差
                sd_temp1 = np.std(data[i: i + time_window, node[0]], ddof=1)
                sd_temp2 = np.std(data[i + 1: i + time_window + 1, node[0]], ddof=1)
                delta_PCC[b][i] = math.sqrt(abs(sd_temp2 - sd_temp1)) * entropy  # 单个节点的It指标
                b += 1


        for i in range(0, week - time_window):  #
            for j in range(0, network.__len__()):  # 23
                sheet1.write(j + 1, i + 1, delta_PCC[j][i])
                if i == 0:
                    # 纵坐标
                    sheet1.write(j + 1, 0, str(network[j][0] + 1))
            sheet1.write(0, i + 1, startweek + time_window + i)

        # print("delta_PCC:",delta_PCC.shape)#(23, 52)

        # 求和
        It = []  # 所有节点的It值
        for i in range(0, week - time_window):
            # delta_PCC (23 , 52)
            It.append(np.sum(delta_PCC[:, i]) / network.__len__())  # 计算了所选列中所有数值的总和均值
            sheet1.write(network.__len__() + 1, i + 1, It[i])

        #len(It)=52,The DNE score represents the 52 weeks of each year, and you can make critical judgments based on it.




