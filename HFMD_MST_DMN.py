import xlrd
from myTools import *
import networkx as nx
import numpy as np
from scipy.stats import pearsonr,spearmanr,kendalltau
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd
import warnings,os
# Ignore warnings containing specific text
warnings.filterwarnings("ignore", message="An input array is constant")

directory = 'MST_HFMD'
if not os.path.exists(directory):
    os.makedirs(directory)
    
# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei']  # Specify default font
# Set English font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of the negative sign '-' showing as a square when saving images
# Set global font size
plt.rcParams.update({'font.size': 14})


# The 'network' list from method 1
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

# Generate edge list, ensure consistency with method 1
edge = []
for item in network:
    for i in range(1, len(item)):
        if [item[0], item[i]] in edge or [item[i], item[0]] in edge:
            continue
        else:
            edge.append([item[0], item[i]])
# Ensure the order of edges is consistent
edge = [sorted(e) for e in edge]
edge.sort(key=lambda x: (x[0], x[1]))

# Parameter settings
file_start_week = 8
time_window = 6

# Read un-standardized data
df = pd.read_excel("data/东京都手足口病人数.xls")
# Read standardized data
book = xlrd.open_workbook('data/东京都手足口病定点报告数.xls')
sheet = book.sheet_by_index(0)
start_year = round(df.iloc[1][0])

for f in range(8, 20):  # Adjust the loop range to match method 1
    startweek = file_start_week - time_window + 1 + 52 * f
    endweek = 52 + file_start_week + 52 * f
    week_count = endweek - startweek + 1
    data = np.zeros((week_count, 23))
    for col in range(startweek, endweek + 1):
        for row in range(23):
            data[col - startweek][row] = sheet.cell(col, row + 1).value

    # Extract infection numbers
    infection_number = df.iloc[file_start_week + 52 * f: 52 + file_start_week + 52 * f, 1].tolist()
    max_num = max(infection_number)
    index = infection_number.index(max_num) + 1

    # Calculate delta_PCC
    delta_PCC = np.zeros((len(edge), week_count - 1))
    for i in range(len(edge)):
        for j in range(week_count - 1):
            # Extract data for the first j+1 weeks and the first j+2 weeks
            data_week1 = data[0:j+1, [edge[i][0], edge[i][1]]]
            data_week2 = data[0:j+2, [edge[i][0], edge[i][1]]]
            
            # Calculate Pearson correlation coefficient and handle NaN
            if len(data_week1) < 2:
                pcc1 = 0
            else:
                pcc1 = abs(np.nan_to_num(pearsonr(data_week1[:, 0], data_week1[:, 1])[0]))
            
            pcc2 = abs(np.nan_to_num(pearsonr(data_week2[:, 0], data_week2[:, 1])[0]))
            
            # Calculate standard deviation
            std1 = np.std(data_week1)
            std2 = np.std(data_week2)
            
            # Calculate delta and handle NaN
            delta = abs(pcc2 - pcc1) * abs(std2 - std1)
            delta = np.nan_to_num(delta)  # Key: replace NaN with 0
            
            delta_PCC[i][j] = delta

    delta_PCC = np.transpose(delta_PCC)
    delta_PCC = delta_PCC[5:, :]

    # Generate graph and calculate MST, ensure weights have no NaN
    graphL = []
    for week_data in delta_PCC:
        # Replace NaN with 0
        week_data = np.nan_to_num(week_data)
        graphL.append(week_data.tolist())

    mstList = []
    for week in range(len(graphL)):
        edges = []
        for idx, e in enumerate(edge):
            weight = graphL[week][idx]
            # Ensure weight is a numeric type
            if np.isnan(weight):
                weight = 0.0
            edges.append((e[0], e[1], weight))
        
        # Create graph and calculate MST
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        mst = nx.minimum_spanning_tree(G)
        mst_edges = [(u, v, d['weight']) for u, v, d in mst.edges(data=True)]
        mstList.append(mst_edges)

    # Normalize the sum of MST weights
    It = [sum(weight for _, _, weight in mst) for mst in mstList]
    It = (It - np.min(It)) / (np.max(It) - np.min(It)) if np.max(It) != np.min(It) else np.zeros_like(It)

    It_window = 5




