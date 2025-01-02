import csv
from collections import Counter
import matplotlib.pyplot as plt

def analyze_last_number_distribution(csv_file):
    last_numbers = []
    
    with open(csv_file, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            last_numbers.extend(row)
                
                        
    
    # 统计分布
    distribution = Counter(last_numbers)
    
    # # 打印结果
    # print("Last Number Distribution:")
    # for number, count in sorted(distribution.items()):
    #     print(f"{number}: {count} times")
    
    return distribution

def plot_distribution(distribution):
    numbers = sorted(list(map(int, distribution.keys())))
    counts = [distribution[str(num)] for num in numbers]
    plt.bar(numbers, counts, color='skyblue')
    plt.xlabel('Number', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Number Distribution', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 使用示例
csv_file = 'valid.txt'  # 替换为你的CSV文件路径
distribution = analyze_last_number_distribution(csv_file)
plot_distribution(distribution)
