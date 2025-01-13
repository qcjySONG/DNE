import os  
import pickle  
import matplotlib.pyplot as plt  

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# 设置英文字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
# 设置全局字体大小
plt.rcParams.update({'font.size': 14})

# 定义文件夹路径  
folder_path = './FLU_re/'  # 替换为你的文件夹路径  
names = ["Year 2019-2020 Tokyo", "Year 2018-2019 Tokyo", "Year 2017-2018 Tokyo", "Year 2016-2017 Tokyo", "Year 2015-2016 Tokyo"]  
nums = [-1, -2, -3, -4, -5]  

# 定义颜色和线宽  
colors = [(56/255, 89/255, 137/255), (210/255, 32/255, 39/255), (127/255, 165/255, 183/255)]  
line_widths = [1, 2, 1]  # 第一条和第三条线宽为1，第二条线宽为2  

for i in range(len(nums)):  
    # 创建一个图形  
    plt.figure(figsize=(8, 6))  
    # 遍历文件夹中的所有文件  
    for j, file_name in enumerate(os.listdir(folder_path)):  
        # 只处理 .pkl 文件  
        if file_name.endswith('.pkl'):  
            # 构建完整的文件路径  
            file_path = os.path.join(folder_path, file_name)  
            # 加载 .pkl 文件  
            with open(file_path, 'rb') as f:  
                data = pickle.load(f)  # 加载数据  
            # 提取最后一个子列表  
            last_list = data[nums[i]]  # 假设 data 是一个二维列表  
            # 绘制折线图  
            plt.plot(range(len(last_list)), last_list, label=file_name, color=colors[j % len(colors)], linewidth=line_widths[j % len(line_widths)])  
    
    # 添加标题和标签  
    plt.title(names[i], fontsize=30, fontweight='heavy')  
    plt.xlabel('Time (week)', fontsize=26, fontweight='bold')  
    plt.ylabel('H(t)', fontsize=26, fontweight='bold')  
    # 保存图形为文件  
    save_path = f'./FLU_NE/FLU_{names[i]}.png'  # 保存的文件名和路径  
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 保存为高分辨率 PNG 文件  
    print(f"图形已保存为 {save_path}")