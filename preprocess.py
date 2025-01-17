import pandas as pd
import numpy as np
import sqlite3
import os
import math


def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    conn = sqlite3.connect(file_path)
    sql = 'SELECT * FROM stock_info ORDER BY code, day asc'
    stock_info = pd.read_sql_query(sql, conn)
    print(f'stock_info.size = {len(stock_info)}')
    stock_info = stock_info[~((stock_info['close'] == stock_info['open']) & (stock_info['high'] == stock_info['low']) & (stock_info['high'] == stock_info['close']))]
    print(f'filter stock_info.size = {len(stock_info)}')
    stock_info['pct_change'] = stock_info['close'].pct_change().fillna(0)
    stock_info['high_pct_change'] = (stock_info['high'] - stock_info['close'].shift(1)) / stock_info['close'].shift(1)
    stock_info['high_pct_change'] = stock_info['high_pct_change'].fillna(0)
    stock_info['low_pct_change'] = (stock_info['low'] - stock_info['close'].shift(1)) / stock_info['close'].shift(1)
    stock_info['low_pct_change'] = stock_info['low_pct_change'].fillna(0)
    stock_info['open_pct_change'] = (stock_info['open'] - stock_info['close'].shift(1)) / stock_info['close'].shift(1)
    stock_info['open_pct_change'] = stock_info['open_pct_change'].fillna(0)
    print(stock_info.head())
    return stock_info

# 假设 df 是你的 DataFrame，包含 'code', 'date', 'pct_change' 列
# df = df.sort_values(by=['code', 'date'])


def generate_sequences(df, seq_length, seq_interval):
# # 定义序列长度和间隔
# seq_length = 31
# seq_interval = 15
    # 保存序列的文件
    train_file = 'train.txt'
    valid_file = 'valid.txt'

    # 打开文件准备写入
    with open(train_file, 'w') as train_f, open(valid_file, 'w') as valid_f:
        for code, group in df.groupby('code'):
            pct_changes = group['bin_index'].tolist()
            # high_pct_changes = group['high_bin_index'].tolist()
            # low_pct_changes = group['low_bin_index'].tolist()
            # open_pct_changes = group['open_bin_index'].tolist()
            n = len(pct_changes)
            
            sequences = []
            
            # 生成序列
            for start in range(0, n - seq_length + 1, seq_interval):
                end = start + seq_length
                sequence = pct_changes[start:end]
                # sequence1 = high_pct_changes[start:end]
                # sequence2 = low_pct_changes[start:end]
                # sequence3 = open_pct_changes[start:end]
                # combined_list = ['-'.join(map(str, map(int, items))) for items in zip(sequence, sequence1, sequence2, sequence3)]
                # sequences.append(combined_list)
                sequences.append(sequence)
                
            
            # 如果至少有一个序列
            if sequences:
                # 将最后一个序列写入 valid.txt
                valid_f.write(' '.join(map(str, sequences[-1])) + '\n')
                
                # 将其余序列写入 train.txt
                for seq in sequences[:-1]:
                    train_f.write(' '.join(map(str, seq)) + '\n')
            
            
def pct_trans(pct):
    pct *= 100
    if pct <= -20:
        pct = -19.9
    elif pct >= 20:
        pct = 19.9
    pct = math.ceil(pct)
    pct += 20
    return pct

def pct_trans2(df):
    num_bins = 10
    df['bin'], bins = pd.qcut(df['pct_change'], q=num_bins, retbins=True, duplicates='drop') 
    print(df['bin'].value_counts().sort_index())
    df['bin_index'] = pd.Categorical(df['bin']).codes + 1
    
    # 可能会遇到区间不匹配问题, 需要对区间两端进行处理
    df['high_pct_change'] = np.where(df['high_pct_change'] < bins[0], bins[0], np.where(df['high_pct_change'] > bins[-1], bins[-1], df['high_pct_change']))
    df['low_pct_change'] = np.where(df['low_pct_change'] < bins[0], bins[0], np.where(df['low_pct_change'] > bins[-1], bins[-1], df['low_pct_change']))
    df['open_pct_change'] = np.where(df['open_pct_change'] < bins[0], bins[0], np.where(df['open_pct_change'] > bins[-1], bins[-1], df['open_pct_change']))
    
    df['high_bin'] = pd.cut(df['high_pct_change'], bins=bins)
    df['high_bin_index'] = pd.cut(df['high_pct_change'], bins=bins, labels=False) + 1
    
    df['low_bin'] = pd.cut(df['low_pct_change'], bins=bins)
    df['low_bin_index'] = pd.cut(df['low_pct_change'], bins=bins, labels=False)+ 1
    
    df['open_bin'] = pd.cut(df['open_pct_change'], bins=bins)
    df['open_bin_index'] = pd.cut(df['open_pct_change'], bins=bins, labels=False) + 1
    print(df.head())
    bin_to_index = {str(interval): index + 1 for index, interval in enumerate(df['bin'].cat.categories)}
    bin_index_df = pd.DataFrame(list(bin_to_index.items()), columns=['Interval', 'Index'])
    print(bin_index_df)
    bin_index_df.to_csv('bin_index_mapping.csv', index=False)
    return df
            
if __name__ == "__main__":
    file_path = "C:\\Users\\huang\\Downloads\\stock.nfa"
    df = load_data(file_path)
    df =pct_trans2(df)
    generate_sequences(df, 31, 1)
    print('end')
    