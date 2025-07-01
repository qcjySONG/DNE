import os  
import pickle  
import matplotlib.pyplot as plt  

plt.rcParams['font.sans-serif'] = ['SimHei']  # Specify default font
# Set English font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of the negative sign '-' showing as a square when saving images
# Set global font size
plt.rcParams.update({'font.size': 14})

# Define folder path  
folder_path = './FLU_re/'  # Replace with your folder path  
names = ["Year 2019-2020 Tokyo", "Year 2018-2019 Tokyo", "Year 2017-2018 Tokyo", "Year 2016-2017 Tokyo", "Year 2015-2016 Tokyo"]  
nums = [-1, -2, -3, -4, -5]  

# Define colors and line widths  
colors = [(56/255, 89/255, 137/255), (210/255, 32/255, 39/255), (127/255, 165/255, 183/255)]  
line_widths = [1, 2, 1]  # The first and third lines have a width of 1, the second has a width of 2  

for i in range(len(nums)):  
    # Create a figure  
    plt.figure(figsize=(8, 6))  
    # Iterate through all files in the folder  
    for j, file_name in enumerate(os.listdir(folder_path)):  
        # Process only .pkl files  
        if file_name.endswith('.pkl'):  
            # Build the full file path  
            file_path = os.path.join(folder_path, file_name)  
            # Load the .pkl file  
            with open(file_path, 'rb') as f:  
                data = pickle.load(f)  # Load data  
            # Extract the last sublist  
            last_list = data[nums[i]]  # Assume data is a 2D list  
            # Plot the line graph  
            plt.plot(range(len(last_list)), last_list, label=file_name, color=colors[j % len(colors)], linewidth=line_widths[j % len(line_widths)])  
    
    # Add title and labels  
    plt.title(names[i], fontsize=30, fontweight='heavy')  
    plt.xlabel('Time (week)', fontsize=26, fontweight='bold')  
    plt.ylabel('H(t)', fontsize=26, fontweight='bold')  
    # Save the figure to a file  
    save_path = f'./FLU_NE/FLU_{names[i]}.png'  # Saved file name and path  
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save as a high-resolution PNG file  
    print(f"图形已保存为 {save_path}")