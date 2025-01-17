import pandas as pd
import numpy as np
import sqlite3
import os
import math
from talib import *

def find_positive_after_negative(df, column_name, n_negative=10):
    """
    找出大于等于0，且之前n行数据小于0的位置
    
    参数:
    df: DataFrame
    column_name: 要检查的列名
    n_negative: 之前需要多少个负数，默认10
    
    返回:
    符合条件的行索引
    """
    def check_conditions(group):
        # 获取目标列
        series = group[column_name]
        
        # 创建一个布尔掩码，标记小于0的值
        negative_mask = series < 0
        
        # 使用rolling计算前n_negative个值是否都为负
        all_negative = negative_mask.rolling(window=n_negative).sum() == n_negative
        
        # 当前值大于等于0的掩码
        current_positive = series >= 0
        
        # 将all_negative向前移动一位，这样它就对应于"之前"的n_negative个值
        prev_all_negative = all_negative.shift(1)
        
        # 同时满足两个条件:当前值>=0且之前n_negative个值<0
        return current_positive & prev_all_negative
    
    # 按code分组处理
    result_mask = df.groupby('code', include_groups=False).apply(check_conditions).reset_index(level=0, drop=True)
    
    return df[result_mask]

file_path = 'C:\\Users\\huang\\Downloads\\stock.nfa'
conn = sqlite3.connect(file_path)
sql = 'SELECT * FROM stock_info where day >= "2010-01-01" ORDER BY code, day asc'
stock_info = pd.read_sql_query(sql, conn)

# 按code分组计算MACD
# stock_info['macdhist'] = stock_info.groupby('code', include_groups=False).apply(
#     lambda x: MACD(x['close'], fastperiod=12, slowperiod=26, signalperiod=9)[2]
# ).reset_index(level=0, drop=True)
stock_info['macdhist'] = stock_info.groupby('code')['close'].apply(
    lambda x: MACD(x, fastperiod=12, slowperiod=26, signalperiod=9)[2]
).reset_index(level=0, drop=True)
# 查找符合条件的行
# 10-23年数据，共有185271行数据符合条件
result = find_positive_after_negative(stock_info, 'macdhist', n_negative=10)

# 查看result当前行后1个礼拜内有多少stock_info的数据是涨的
# 计算每个信号后7个交易日的涨跌情况
def calculate_trading_days_performance(result_df, stock_info_df):
    # 创建一个空的DataFrame来存储结果
    stats = pd.DataFrame(index=result_df.index)
    
    # 将stock_info_df按code分组并转成字典,避免重复过滤
    stock_data_dict = {code: group for code, group in stock_info_df.groupby('code')}
    
    # 批量处理所有数据
    all_stats = []
    for code, group in result_df.groupby('code'):
        stock_data = stock_data_dict[code]
        
        # 为每个信号创建一个DataFrame记录后7天数据
        signal_days = group['day'].values
        signal_closes = group['close'].values
        
        # 创建一个包含所有信号日期的Series
        signal_series = pd.Series(index=stock_data.index, data=False)
        signal_indices = stock_data[stock_data['day'].isin(signal_days)].index
        signal_series[signal_indices] = True
        
        # 使用rolling window找到每个信号后的7天
        future_data = []
        for i in range(7):
            future_data.append(stock_data['close'].shift(-i))
        future_prices = pd.concat(future_data, axis=1)
        
        # 只保留信号日的数据
        future_prices = future_prices[signal_series]
        
        # 计算价格变化百分比
        price_changes = ((future_prices.T - signal_closes) / signal_closes * 100).T
        
        # 计算统计数据
        stats_dict = {
            'total_days': (~price_changes.isna()).sum(axis=1),
            'up_days': (price_changes > 0).sum(axis=1),
            'up_ratio': (price_changes > 0).mean(axis=1),
            'max_gain': price_changes.max(axis=1)
        }
        
        # 将统计数据添加到结果中
        for idx, row in group.iterrows():
            for stat_name, stat_values in stats_dict.items():
                stats.loc[idx, stat_name] = stat_values[idx] if idx in stat_values.index else 0
    
    # 合并结果
    result_df = pd.concat([result_df, stats], axis=1)
    
    # 输出统计信息
    print(f"总信号数: {len(result_df)}")
    print(f"平均上涨天数比例: {result_df['up_ratio'].mean():.2%}")
    print(f"平均最大涨幅: {result_df['max_gain'].mean():.2f}%")
    
    return result_df

# 计算性能统计
result = calculate_trading_days_performance(result, stock_info)

