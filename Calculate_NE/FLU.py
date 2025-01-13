import math
import pandas as pd
import xlrd
import os
import numpy as np
from scipy.stats import kendalltau
import pickle

# 检查目录是否存在，如果不存在则创建
directory = 'son_img_FLU'
if not os.path.exists(directory):
    os.makedirs(directory)

all_nums = []
for gap_year in range(25, 26):
    time_window = 4
    file_start_week = gap_year

    # 读取Excel文件
    excel_file = "data/东京都流感人数.xls"
    df = pd.read_excel(excel_file)
    df_len = len(df) - file_start_week
    year = df_len // 52
    if year % 2 == 1:
        rows = (year // 2) + 1
    else:
        rows = year // 2

    point_num = df.shape[1] - 2

    start_year = round(df.iloc[1][0])

    for f in range(8, 19):
        infection_number = df.iloc[file_start_week + 52 * f: 52 + file_start_week + 52 * f, 1].tolist()
        startweek = file_start_week - time_window + 1 + 52 * f
        endweek = 52 + file_start_week + 52 * f
        week = endweek - startweek + 1

        book = xlrd.open_workbook('data/东京都流感定点报告数.xls')
        sheet = book.sheet_by_index(0)
        data = np.zeros((week, point_num))
        for col in range(startweek, endweek + 1):
            for row in range(0, point_num):
                data[col - startweek][row] = sheet.cell(col, row + 1).value

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

        delta_PCC = np.zeros((len(network), week - time_window))
        for i in range(0, week - time_window):
            b = 0
            for node in network:
                pcc_temp3 = 0
                pcc_ij = []
                pcct_ij = []

                for j in range(1, len(node)):
                    pcc_temp1 = abs(kendalltau(data[i: i + time_window, node[0]], data[i: i + time_window, node[j]])[0])
                    pcc_temp2 = abs(kendalltau(data[i + 1: i + time_window + 1, node[0]], data[i + 1: i + time_window + 1, node[j]])[0])
                    weight = abs(pcc_temp2 - pcc_temp1)
                    pcc_ij.append(weight)
                    pcc_temp3 += weight

                if pcc_temp3 == 0:
                    delta_PCC[b][i] = 0
                    b += 1
                    continue

                for k in range(0, len(pcc_ij)):
                    pcct_ij.append(pcc_ij[k] / pcc_temp3)

                entropy = 0
                for h in range(0, len(pcct_ij)):
                    if pcct_ij[h] > 0:
                        entropy -= (pcct_ij[h]) * math.log(pcct_ij[h], 2)

                sd_temp1 = np.std(data[i: i + time_window, node[0]], ddof=1)
                sd_temp2 = np.std(data[i + 1: i + time_window + 1, node[0]], ddof=1)
                delta_PCC[b][i] = (math.sqrt(abs(sd_temp2 - sd_temp1)) * entropy) / np.log(len(node))
                b += 1

        It = []
        for i in range(0, week - time_window):
            It.append(np.sum(delta_PCC[:, i]))
        all_nums.append(It)

with open('./FLU_re/4_all_nums.pkl', 'wb') as f:
    pickle.dump(all_nums, f)