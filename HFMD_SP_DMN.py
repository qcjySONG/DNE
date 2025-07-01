import pandas as pd
import xlrd
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import norm
import copy,os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=10000, linewidth=10000, suppress=True)
# Check if the directory exists, if not, create it
directory = 'SP_DNM_HMDL'
if not os.path.exists(directory):
    os.makedirs(directory)

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei']  # Specify default font
# Set English font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of the negative sign '-' showing as a square when saving images
# Set global font size
plt.rcParams.update({'font.size': 14})

#ttest
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

def Dijkstra(G, v0, INF=999):  # Dijkstra's algorithm
    book = set()
    minv = v0
    way = dict((k, [v0]) for k in G.keys())
    dis = dict((k, INF) for k in G.keys())
    dis[v0] = 0
    while len(book) < len(G):
        book.add(minv)
        for w in G[minv]:
            if dis[minv] + G[minv][w] < dis[w]:
                dis[w] = dis[minv] + G[minv][w]
                way[w] = copy.deepcopy(way.get(minv))
                way[w].append(w)
        new = INF
        for v in dis.keys():
            if v in book:
                continue
            if dis[v] < new:
                new = dis[v]
                minv = v
    return dis, way

# Read Excel file
excel_file = "data/东京都手足口病人数.xls"  # Un-standardized data
df = pd.read_excel(excel_file)
# Get the number of rows needed for plotting, columns are fixed at two per row
df_len = len(df) - 22
year = df_len // 52
if year % 2 == 1:
    rows = (year // 2) + 1
else:
    rows = year // 2
# Starting year for plotting
start_year = round(df.iloc[1][0])


time_window = 6
file_start_week=8
for f in range(8, 20):
    # Infection number data, for plotting
    infection_number = df.iloc[file_start_week + 52 * f: 52 + file_start_week + 52 * f, 1].tolist()  # iloc slices based on row number, which starts from 0
    max_num = max(infection_number)
    index = infection_number.index(max_num) + 1
    startweek = file_start_week - time_window + 1 + 52 * f  # 18=22-4
    endweek = 52 + file_start_week + 52 * f  # 74=52+22
    week = endweek - startweek + 1
    book = xlrd.open_workbook('data/东京都手足口病定点报告数.xls') # Standardized data
    sheet = book.sheet_by_index(0)
    data = np.zeros((week, 23))
    for col in range(startweek, endweek + 1):
        for row in range(0, 23):
            data[col - startweek][row] = sheet.cell(col, row + 1).value  # read data
    #print(data.shape)#(58, 23)
    #weekly_data = df.iloc[file_start_week + 52 * f: 52 + file_start_week + 52 * f]
    #print(weekly_data)
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

    # get edge set
    edge = []
    for item in network:
        for i in range(1, item.__len__()):
            if [item[0], item[i]] in edge or [item[i], item[0]] in edge:
                continue
            else:
                edge.append([item[0], item[i]])

    delta_PCC = np.zeros((edge.__len__(), week - 1))
    for i in range(0, edge.__len__()):
        for j in range(0, week - 1):
            # Pearson correlation coefficient between the two
            if len(data[0: j + 1, edge[i][0]]) < 2 or len(data[0: j + 1, edge[i][1]]) < 2:
                pcc_temp1 = 0
            else:
                pcc_temp1 = abs(pearsonr(data[0: j + 1, edge[i][0]], data[0: j + 1, edge[i][1]])[0])

            pcc_temp2 = abs(pearsonr(data[0: j + 2, edge[i][0]], data[0: j + 2, edge[i][1]])[0])
            pcc_temp1 = np.nan_to_num(pcc_temp1)
            pcc_temp2 = np.nan_to_num(pcc_temp2)
            sd_temp1 = np.std(data[0: j + 1, edge[i]])
            sd_temp2 = np.std(data[0: j + 2, edge[i]])
            delta_PCC[i][j] = abs(pcc_temp2 - pcc_temp1) * abs(sd_temp2 - sd_temp1)

    delta_PCC = np.transpose(delta_PCC)
    delta_PCC = delta_PCC[5:, :]


    distance_NS = np.zeros(delta_PCC.shape[0])
    distance_WE = np.zeros(delta_PCC.shape[0])
    distance_NWSE = np.zeros(delta_PCC.shape[0])
    distance_NESW = np.zeros(delta_PCC.shape[0])
    num_week = np.zeros((delta_PCC.shape[0]))

    # delta_PCC.shape[0] = 52
    for k in range(delta_PCC.shape[0]):
        distance_all = {}
        for i in range(23):
            distance = {}
            for j in range(edge.__len__()):  # 53
                if edge[j][0] == i:
                    distance[edge[j][1] + 1] = delta_PCC[k, j]
                elif edge[j][1] == i:
                    distance[edge[j][0] + 1] = delta_PCC[k, j]
            distance[i + 1] = 0
            distance_all[i + 1] = distance
        disNS, wayNS = Dijkstra(distance_all, 5)
        distance_NS[k] = disNS[1]
        disWE, wayWE = Dijkstra(distance_all, 13)
        distance_WE[k] = disWE[22]
        disNWSE, wayNWSE = Dijkstra(distance_all, 20)
        distance_NWSE[k] = disNWSE[15]
        disNESW, wayNESW = Dijkstra(distance_all, 23)
        distance_NESW[k] = disNESW[10]

        num_week[k] = np.sum(data[k + 5, :])
        all_rode = distance_NS[k] + distance_WE[k] + distance_NESW[k] + distance_NWSE[k]

    
