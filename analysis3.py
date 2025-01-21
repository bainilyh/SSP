import pandas as pd
import numpy as np
import sqlite3
import os
import math
from talib import *
import mplfinance as mpf
import matplotlib.pyplot as plt

def calculate_trading_days_performance2(result_df, stock_info_df, n_days=7):
    # 合并数据集
    merge_df = pd.merge(stock_info_df, result_df, on=['ts_code', 'trade_date'], how='left', indicator=True)
    merge_df = merge_df[['ts_code', 'trade_date', 'close_x', 'open_x', 'high_x', 'low_x', 'macdhist_x', 'macd_x', 'macdsignal_x', '_merge']]
    merge_df.columns = ['ts_code', 'trade_date', 'close', 'open', 'high', 'low', 'macdhist', 'macd', 'macdsignal', '_merge']
    
    # 找到信号行
    both_rows = merge_df[merge_df['_merge'] == 'both']
    both_indices = both_rows.index
    
    # 为每个信号创建未来n_days天的索引
    future_indices = np.array([np.arange(idx, min(idx + n_days + 1, len(merge_df))) for idx in both_indices]).flatten()
    
    # 一次性选择所有需要的行
    result_df = merge_df.iloc[future_indices].copy()
    
    # 为每组数据添加对应的初始收盘价
    result_df['group'] = np.repeat(both_indices, [min(n_days + 1, len(merge_df) - idx) for idx in both_indices])
    initial_closes = merge_df.iloc[both_indices]['close'].values
    result_df['initial_close'] = initial_closes[result_df['group'].map(lambda x: both_indices.get_loc(x))]
    
    # 计算变化率
    result_df['change_ratio'] = (result_df['close'] - result_df['initial_close']) / result_df['initial_close']
    
    # 为每个ts_code的both添加从0开始的行号
    result_df['row'] = result_df.groupby(['ts_code', result_df['group']]).cumcount()
    
    # 删除辅助列
    result_df.drop(['group', 'initial_close'], axis=1, inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    
    return result_df