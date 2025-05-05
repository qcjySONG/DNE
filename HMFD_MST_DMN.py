import xlrd
from myTools import *
import networkx as nx
import numpy as np
from scipy.stats import pearsonr,spearmanr,kendalltau
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd
import warnings,os
# 忽略包含特定文本的警告
warnings.filterwarnings("ignore", message="An input array is constant")

directory = 'MST_HFMD'
if not os.path.exists(directory):
    os.makedirs(directory)
    
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# 设置英文字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
# 设置全局字体大小
plt.rcParams.update({'font.size': 14})


# 方法1中的network列表
network = [
    [0, 2, 5, 11, 17, 7],
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
    [11, 0, 17, 3, 16, 10, 5],
    [12, 19, 3, 17, 1],
    [13, 19, 16, 4],
    [14, 15, 8, 21],
    [15, 2, 8, 14],
    [16, 19, 3, 11, 10, 4, 13],
    [17, 1, 12, 3, 11, 0, 7, 18],
    [18, 1, 9, 7, 17],
    [19, 12, 3, 16, 13],
    [20, 22, 4, 21, 8, 6, 10],
    [21, 22, 20, 8, 14],
    [22, 4, 20, 21]
]

# 生成边列表，确保与方法1一致
edge = []
for item in network:
    for i in range(1, len(item)):
        if [item[0], item[i]] in edge or [item[i], item[0]] in edge:
            continue
        else:
            edge.append([item[0], item[i]])
# 确保边的顺序一致
edge = [sorted(e) for e in edge]
edge.sort(key=lambda x: (x[0], x[1]))

# 参数设置
file_start_week = 8
time_window = 6

# 读取未标准化数据
df = pd.read_excel("data/东京都手足口病人数.xls")
# 读取标准化数据
book = xlrd.open_workbook('data/东京都手足口病定点报告数.xls')
sheet = book.sheet_by_index(0)
start_year = round(df.iloc[1][0])

for f in range(8, 20):  # 调整循环范围以匹配方法1
    startweek = file_start_week - time_window + 1 + 52 * f
    endweek = 52 + file_start_week + 52 * f
    week_count = endweek - startweek + 1
    data = np.zeros((week_count, 23))
    for col in range(startweek, endweek + 1):
        for row in range(23):
            data[col - startweek][row] = sheet.cell(col, row + 1).value

    # 提取感染人数
    infection_number = df.iloc[file_start_week + 52 * f: 52 + file_start_week + 52 * f, 1].tolist()
    max_num = max(infection_number)
    index = infection_number.index(max_num) + 1

    # 计算delta_PCC
    delta_PCC = np.zeros((len(edge), week_count - 1))
    for i in range(len(edge)):
        for j in range(week_count - 1):
            # 提取前j+1周和前j+2周的数据
            data_week1 = data[0:j+1, [edge[i][0], edge[i][1]]]
            data_week2 = data[0:j+2, [edge[i][0], edge[i][1]]]
            
            # 计算皮尔逊相关系数并处理NaN
            if len(data_week1) < 2:
                pcc1 = 0
            else:
                pcc1 = abs(np.nan_to_num(pearsonr(data_week1[:, 0], data_week1[:, 1])[0]))
            
            pcc2 = abs(np.nan_to_num(pearsonr(data_week2[:, 0], data_week2[:, 1])[0]))
            
            # 计算标准差
            std1 = np.std(data_week1)
            std2 = np.std(data_week2)
            
            # 计算delta并处理NaN
            delta = abs(pcc2 - pcc1) * abs(std2 - std1)
            delta = np.nan_to_num(delta)  # 关键：将NaN替换为0
            
            delta_PCC[i][j] = delta

    delta_PCC = np.transpose(delta_PCC)
    delta_PCC = delta_PCC[5:, :]

    # 生成图并计算MST，确保权重无NaN
    graphL = []
    for week_data in delta_PCC:
        # 替换NaN为0
        week_data = np.nan_to_num(week_data)
        graphL.append(week_data.tolist())

    mstList = []
    for week in range(len(graphL)):
        edges = []
        for idx, e in enumerate(edge):
            weight = graphL[week][idx]
            # 确保权重为数值类型
            if np.isnan(weight):
                weight = 0.0
            edges.append((e[0], e[1], weight))
        
        # 创建图并计算MST
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        mst = nx.minimum_spanning_tree(G)
        mst_edges = [(u, v, d['weight']) for u, v, d in mst.edges(data=True)]
        mstList.append(mst_edges)

    # 归一化MST权重和
    It = [sum(weight for _, _, weight in mst) for mst in mstList]
    It = (It - np.min(It)) / (np.max(It) - np.min(It)) if np.max(It) != np.min(It) else np.zeros_like(It)

    It_window = 5




