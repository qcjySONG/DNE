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


# Parameters to be adjusted
'''
HFMD:
time_window=6
file_start_week=6
flu 25


'''
for gap_year in range(8,9):
    time_window = 6
    file_start_week = gap_year

    book1 = xlwt.Workbook()
    book2 = xlwt.Workbook()
    book3 = xlwt.Workbook()
    book4 = xlwt.Workbook()

    # Set Chinese font
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Specify default font
    # Set English font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of the negative sign '-' showing as a square when saving images
    # Set global font size
    plt.rcParams.update({'font.size': 14})

    np.set_printoptions(threshold=10000, linewidth=10000, suppress=True)  # Set np output to be more visually appealing

    # Read Excel file
    excel_file = "data/东京都手足口病人数.xls"  # "data/东京都手足口病人数.xls" #"data/东京都流感人数.xls" "data/东京都肠胃炎人数.xls"
    df = pd.read_excel(excel_file)
    # Get the number of rows needed for plotting, columns are fixed at two per row
    df_len = len(df) - file_start_week  # What does subtracting 22 mean?
    year = df_len // 52
    if year % 2 == 1:
        rows = (year // 2) + 1
    else:
        rows = year // 2

    point_num = df.shape[1] - 2  # Assume the first two columns of each file are Time and Total

    # Starting year for plotting
    start_year = round(df.iloc[1][0])

    # Create figure layers
    fig = plt.figure(figsize=(30, 7 * rows))
    fig1 = plt.figure(figsize=(18, 4 * rows))
    gap = []  # How many weeks in advance to predict?
    It_5 = []  # Record the last 5 It values
    for f in range(8, 20):  # Due to a programming issue, 1-21 represents the years 2000-2024
        '''
        #Year: {
        #    Region: {It score, weekly}
        #    Alert point
        #    Outbreak point
        #}
        '''

        weight_str = []  # Weight file
        # Infection number data, for plotting
        infection_number = df.iloc[file_start_week + 52 * f: 52 + file_start_week + 52 * f,
                           1].tolist()  # iloc slices based on row number, which starts from 0
        # Start row 22 + 52 * f, End row 74 + 52 * f, where 74=22+52. Select the 2nd column (total infection count column)
        max_num = max(infection_number)
        index = infection_number.index(max_num) + 1  # This should be the position for plotting the outbreak period

        startweek = file_start_week - time_window + 1 + 52 * f  # 18=22-4
        endweek = 52 + file_start_week + 52 * f  # 74=52+22

        week = endweek - startweek + 1
        print("week:",week)#58

        book = xlrd.open_workbook(
            'data/东京都手足口病定点报告数.xls')  # This should be the content of the points/stations. 'data/东京都手足口病定点报告数.xls' 'data/东京都流感定点报告数.xls' 'data/东京都肠胃炎定点报告数.xls'
        sheet = book.sheet_by_index(0)
        data = np.zeros((week, point_num))  # 23 is for Tokyo, which has 23 wards/regions
        for col in range(startweek, endweek + 1):
            for row in range(0, point_num):
                data[col - startweek][row] = sheet.cell(col, row + 1).value  # read data
        # print("data.shape",data.shape)
        # Extract the last 52 rows of data
        subset_data = data[6:, :]
        # Transpose rows to columns
        subset_data_transposed = subset_data.T
        subsetdf = pd.DataFrame(subset_data_transposed)
        # Specify the save file path and format the filename using an f-string
        file_path = rf'C:\Users\qiqi\Desktop\dynamical network entropy (DNE)\raw_data_and_code\It_ward_excel_HMDL\HMDL_point_num\Year {start_year + f}-{start_year + f + 1}.xlsx'

        # Write the DataFrame to an Excel file
        subsetdf.to_excel(file_path, index=False)

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

        # Extract edge information from the network, for smooth plotting, it's an undirected graph
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
        for i in range(0, week - time_window):  # 52  week=57, time window adds 5
            # Time loop

            b = 0
            for node in network:  # 23
                # Node/Region loop

                # Pearson correlation coefficient between the two
                pcc_temp3 = 0
                pcc_ij = []
                pcct_ij = []

                for j in range(1, node.__len__()):  # Iterate through adjacent nodes
                    # Adjacent node loop
                    # node[0] is the current central node
                    pcc_temp1 = abs(kendalltau(data[i: i + time_window, node[0]], data[i: i + time_window, node[j]])[
                                        0])  # Kendall's tau of the central node with other nodes at time t-1

                    pcc_temp2 = abs(kendalltau(data[i + 1: i + time_window + 1, node[0]],
                                               data[i + 1: i + time_window + 1, node[j]])[0])  # Kendall's tau of the central node with other nodes at time t
                    # Why is this commented out?
                    # pcc_temp1 = np.nan_to_num(pcc_temp1)
                    # pcc_temp2 = np.nan_to_num(pcc_temp2)
                    weight = abs(pcc_temp2 - pcc_temp1)  # dieta_PCCt
                    weight_zi = []
                    weight_zi.append(i)
                    weight_zi.append(node[0])
                    weight_zi.append(node[j])
                    weight_zi.append(weight)
                    weight_str.append(weight_zi)
                    # Weight between the central node and its first-order neighbors
                    pcc_ij.append(weight)
                    # print('节点', node[0], '与节点', node[j], '的权重', weight)
                    # Sum
                    pcc_temp3 += weight  # For calculating probability
                # print('节点', node[0], pcc_temp3)
                if pcc_temp3 == 0:  # If the sum is 0, the probability is of course 0
                    delta_PCC[b][i] == 0
                    b += 1
                    continue  # Break out of the scope of this central node
                for k in range(0, pcc_ij.__len__()):  # Why not use len(pcc_ij)
                    pcct_ij.append(pcc_ij[k] / pcc_temp3)  # Calculate probability
                # Count the number of zeros
                # zero_count = sum(1 for prob in pcct_ij if prob == 0)
                entropy = 0
                for h in range(0, pcct_ij.__len__()):
                    if pcct_ij[h] > 0:  # Avoid log(0) situation
                        entropy -= (pcct_ij[h]) * math.log(pcct_ij[h], 2)
                # entropy = (1/(math.log(node.__len__(), 2))) * entropy
                # entropy = (1 / 100) * entropy
                # Standard deviation
                sd_temp1 = np.std(data[i: i + time_window, node[0]], ddof=1)
                sd_temp2 = np.std(data[i + 1: i + time_window + 1, node[0]], ddof=1)
                delta_PCC[b][i] = math.sqrt(abs(sd_temp2 - sd_temp1)) * entropy  # It indicator for a single node
                b += 1

        # Convert weight_list to a DataFrame
        df_weight = pd.DataFrame(weight_str, columns=['week', 'in', 'out', 'value'])
        # Replace NaN with 0
        df_weight['value'] = df_weight['value'].fillna(0)
        folder_path = r'C:\Users\qiqi\Desktop\dynamical network entropy (DNE)\raw_data_and_code\weight_HMDL'
        file_path = os.path.join(folder_path, f'Year {start_year + f}-{start_year + f + 1}weight.xlsx')
        # Write the DataFrame to an Excel file
        df_weight.to_excel(file_path, index=False)

        for i in range(0, week - time_window):  #
            for j in range(0, network.__len__()):  # 23
                sheet1.write(j + 1, i + 1, delta_PCC[j][i])
                if i == 0:
                    # y-axis coordinate (label)
                    sheet1.write(j + 1, 0, str(network[j][0] + 1))
            sheet1.write(0, i + 1, startweek + time_window + i)

        # print("delta_PCC:",delta_PCC.shape)#(23, 52)

        # Summation
        It = []  # It values for all nodes
        for i in range(0, week - time_window):
            # delta_PCC (23 , 52)
            It.append(np.sum(delta_PCC[:, i]) / network.__len__())  # Calculate the mean of the sum of all values in the selected column
            sheet1.write(network.__len__() + 1, i + 1, It[i])

        #len(It)=52,The DNE score represents the 52 weeks of each year, and you can make critical judgments based on it.



