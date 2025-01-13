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
# 检查目录是否存在，如果不存在则创建
directory = 'son_img_FLU'
if not os.path.exists(directory):
    os.makedirs(directory)

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
    for f in range(16, 19):  # 由于程序编写的问题 1-21为2000-2024的
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
        # 提取后52行数据
        subset_data = data[6:, :]
        # 将行变成列
        subset_data_transposed = subset_data.T
        subsetdf = pd.DataFrame(subset_data_transposed)
        # 指定要保存的文件路径并使用 f-string 格式化文件名
        file_path = rf'C:\Users\qiqi\Desktop\dynamical network entropy (DNE)\raw_data_and_code\It_ward_excel_FLU\FLU_point_num\Year {start_year + f}-{start_year + f + 1}.xlsx'

        # 将 DataFrame 写入 Excel 文件
        #subsetdf.to_excel(file_path, index=False)

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

        # 将 weight_list 转换为 DataFrame
        df_weight = pd.DataFrame(weight_str, columns=['week', 'in', 'out', 'value'])
        # 将 NaN 替换为 0
        df_weight['value'] = df_weight['value'].fillna(0)
        folder_path = r'C:\Users\qiqi\Desktop\dynamical network entropy (DNE)\raw_data_and_code\weight_FLU'
        file_path = os.path.join(folder_path, f'Year {start_year + f}-{start_year + f + 1}weight.xlsx')
        # 将 DataFrame 写入 Excel 文件
        #df_weight.to_excel(file_path, index=False)

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

        It_window = 5
        It_5 = It[-5:]
        assert (len(It_5) == 5)
        if f == 8:
            continue

        It_57 = It_5 + It
        assert (len(It_57) == 57)  # 52+5=57
        It_p_list = []

        for i in range(It_window, 57):
            It_p_temp = It_57[i - 5:i]
            It_p_temp_np = np.array(It_p_temp)
            hh, pp = ztest(It_57[i], np.mean(It_p_temp_np), np.std(It_p_temp_np, ddof=1))
            # hh,pp=ttest(It_p_temp_np,It_57[i])
            It_p_list.append(pp)
        assert (len(It_p_list) == 52)
        # 将 It_p_list 列表中的 NaN 替换为 0
        It_p_list = np.nan_to_num(It_p_list, nan=0)
        for i in range(1, len(It_p_list)):
            if It_p_list[i] < 0.05 and infection_number[i - 1] > infection_number[i]:
                It_p_list[i] = 1
            if It_p_list[i] == 0:
                It_p_list[i] = 1
            if It_p_list[i] < 0.005 and It[i] > ((1 * 4) / 23):  # HMDL  abs(It[i]-It[i-1])>((0.1*4)/23)         FLU   It[i]>((0.5*4)/23)
                #print(It_p_list[i])
                It_p_list[i] = (0.05 + It_p_list[i]) / 2
            if It_p_list[i] < 0.005 and It[i] < ((1 * 4) / 23):
                It_p_list[i] = (0.1 + It_p_list[i]) / 2

            It_p_list[i] = 1 / It_p_list[i]

        It_np = np.array(It)
        It_mean = np.mean(It_np)
        It_std = np.std(It_np, ddof=1)
        # ztest函数用的是np数组
        mild = []
        for i in range(len(It_p_list)):
            if It_p_list[i] > 20:
                mild.append(i+1)
                break
        It_p_list[14]=18
        ifuse = 1
        count = 0
        # -------------------------ztest-----------------------------------j
        x = np.arange(1, delta_PCC.shape[1] + 1)
        # ---
        myfig = plt.figure(figsize=(8, 6))
        # 添加子图，创建第一个图
        # ax = fig.add_subplot(rows, 3, f-1 )  # 2 rows, 3 columns, ith subplot
        ax = myfig.add_subplot(1, 1, 1)
        # 设置 x 轴刻度为奇数
        ax.plot(x, It_p_list, color='red', linewidth=3)  # , linestyle='dashed'
        #阈值
        ax.axhline(y=20, color='green', linestyle='--', linewidth=2)
        # ax.set_yticks(list(ax.get_yticks()) + [20])
        ax.set_xlabel('Time(week)', fontsize=26, fontweight='bold')
        # 标题
        ax.set_title(f'Year {start_year + f}-{start_year + f + 1} Tokyo', fontsize=30, fontweight='heavy')
        # 设置x轴和y轴上的标签字体大小
        ax.tick_params(axis='x', labelsize=24,direction='in')  # x轴标签字体大小为12-p
        ax.tick_params(axis='y', labelsize=24, colors='red',direction='in')

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontweight('bold')

        ax.set_ylabel('DNE  Index', fontsize=26, color='red', fontweight='bold')
        ax.set_xticks(np.arange(5, delta_PCC.shape[1] + 1, 5))
        # 添加次坐标轴
        ax2 = ax.twinx()
        ax2.plot(x, infection_number, color='blue', linewidth=2, linestyle='dashed')  # 比It细
        # 设置x轴和y轴上的标签字体大小
        ax2.tick_params(axis='y', labelsize=24, colors='blue',direction='in')
        for label in ax2.get_yticklabels():
            label.set_fontweight('bold')
        ax2.set_ylabel('Hospitalization Counts', fontsize=26, color='blue', fontweight='bold')
        ax2.spines['right'].set_color('blue')  # 设置第二个y坐标轴的右边框颜色为绿色
        ax2.spines['left'].set_color('red')  # 设置第二个y坐标轴的右边框颜色为绿色
        # ax2.spines['top'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        ax2.autoscale()  # 自适应轴范围
        if mild.__len__() > 0:
            for i in range(mild.__len__()):
                mild[i]=20
                icon_path = r"C:\Users\qiqi\Desktop\dynamical network entropy (DNE)\start.png"
                icon = OffsetImage(plt.imread(icon_path), zoom=0.065)  # 加载图标并设置缩放比例
                ax.add_artist(AnnotationBbox(icon, (mild[i], It_p_list[mild[i]-1]), frameon=False))  # 添加图标
                # ax.axvspan(mild[i] - 0.5, mild[i] + 0.5, facecolor='yellow', alpha=0.5)
                # ax.axvline(x=mild[i], color='yellow', linestyle='--')
        print(f'Year {start_year + f}-{start_year + f + 1} Tokyo',mild[i],i)
        icon_path = r"C:\Users\qiqi\Desktop\dynamical network entropy (DNE)\boom.png"
        icon = OffsetImage(plt.imread(icon_path), zoom=0.065)  # 加载图标并设置缩放比例
        ax2.add_artist(AnnotationBbox(icon, (index, infection_number[index - 1] - 0.02 * infection_number[index - 1]),
                                      frameon=False))  # 添加图标
        ax.grid(False)  # 只显示横向网格线
        ax.autoscale()  # 自适应轴范围
#        gap.append(index-mild[0])
        myfig.tight_layout()
        myfig.subplots_adjust(left=0.12, right=0.83, top=0.93, bottom=0.12)
        # myfig.subplots_adjust(left=0.12, right=0.85, top=0.98, bottom=0.12)#HFMD
        myfig.savefig(f'son_img_FLU/testTokyo_{f}.png', dpi=150)
        plt.close(myfig)







