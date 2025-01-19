import pandas as pd
import numpy as np
import sqlite3
import os
import math
from talib import *
import mplfinance as mpf
import matplotlib.pyplot as plt

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
    result_mask = df.groupby('ts_code').apply(check_conditions).reset_index(level=0, drop=True)
    
    return df[result_mask]

file_path = '/Users/bainilyhuang/Downloads/hshqt/src/main/resources/db/tushare.nfa'
conn = sqlite3.connect(file_path)
sql = 'SELECT * FROM daily_stock_info2 ORDER BY ts_code, trade_date asc'
stock_info = pd.read_sql_query(sql, conn)

# 按code分组计算MACD
# stock_info['macdhist'] = stock_info.groupby('ts_code', include_groups=False).apply(
#     lambda x: MACD(x['close'], fastperiod=12, slowperiod=26, signalperiod=9)[2]
# ).reset_index(level=0, drop=True)
# stock_info['macdhist'] = stock_info.groupby('ts_code')['close'].apply(
#     lambda x: MACD(x, fastperiod=12, slowperiod=26, signalperiod=9)[2]
# ).reset_index(level=0, drop=True)

def calculate_macd(group):
    try:
        macd, signal, hist = MACD(group, fastperiod=12, slowperiod=26, signalperiod=9)
        return pd.DataFrame({
            'macd': macd,
            'macdsignal': signal,
            'macdhist': hist
        })
    except Exception as e:
        # 返回相同长度的NaN值
        length = len(group)
        return pd.DataFrame({
            'macd': [np.nan] * length,
            'macdsignal': [np.nan] * length,
            'macdhist': [np.nan] * length
        })
    
# 按股票代码分组计算MACD
macd_results = stock_info.groupby('ts_code')['close'].apply(calculate_macd)
macd_results = macd_results.reset_index(level=0, drop=True)

# 将结果赋值给原始DataFrame
stock_info['macd'] = macd_results['macd']
stock_info['macdsignal'] = macd_results['macdsignal']
stock_info['macdhist'] = macd_results['macdhist']

# 查找符合条件的行
# 10-23年数据，共有185271行数据符合条件
result = find_positive_after_negative(stock_info, 'macdhist', n_negative=10)

# 查看result当前行后1个礼拜内有多少stock_info的数据是涨的
# 计算每个信号后7个交易日的涨跌情况
def calculate_trading_days_performance(result_df, stock_info_df):
    # 为每个股票创建一个序号列,用于后续join
    stock_info_df = stock_info_df.copy()
    stock_info_df['row_num'] = stock_info_df.groupby('ts_code').cumcount()
    
    # 找到result_df中每个信号在stock_info_df中的位置
    result_with_pos = pd.merge(
        result_df,
        stock_info_df[['ts_code', 'trade_date', 'row_num']],
        on=['ts_code', 'trade_date']
    )
    
    # 创建未来7天的数据
    future_prices = []
    for i in range(7):
        # 计算未来第i天的位置
        future_pos = stock_info_df.copy()
        future_pos['row_num'] = future_pos['row_num'] - i
        
        # 重命名close列以区分不同天数
        price_data = future_pos[['ts_code', 'row_num', 'close']]
        price_data = price_data.rename(columns={'close': f'close_{i}'})
        
        future_prices.append(price_data)
    
    # 合并所有未来价格数据
    all_future_prices = future_prices[0]
    for df in future_prices[1:]:
        all_future_prices = pd.merge(
            all_future_prices, 
            df,
            on=['ts_code', 'row_num'],
            how='outer'
        )
    
    # 将未来价格数据与信号数据合并
    final_data = pd.merge(
        result_with_pos,
        all_future_prices,
        on=['ts_code', 'row_num'],
        how='left'
    )
    
    # 计算价格变化百分比
    price_cols = [f'close_{i}' for i in range(7)]
    for col in price_cols:
        final_data[f'{col}_pct'] = (final_data[col] - final_data['close']) / final_data['close'] * 100
    
    pct_cols = [f'close_{i}_pct' for i in range(7)]
    
    # 计算统计数据
    final_data['total_days'] = final_data[price_cols].notna().sum(axis=1)
    final_data['up_days'] = (final_data[pct_cols] > 0).sum(axis=1)
    final_data['up_ratio'] = final_data['up_days'] / final_data['total_days']
    final_data['max_gain'] = final_data[pct_cols].max(axis=1)
    
    # 选择需要的列
    result_columns = list(result_df.columns) + ['total_days', 'up_days', 'up_ratio', 'max_gain']
    result_df = final_data[result_columns]
    
    # 输出统计信息
    print(f"总信号数: {len(result_df)}")
    print(f"平均上涨天数比例: {result_df['up_ratio'].mean():.2%}")
    print(f"平均最大涨幅: {result_df['max_gain'].mean():.2f}%")
    
    return result_df

# 计算性能统计
result = calculate_trading_days_performance(result, stock_info)

def plot_candlestick(df, ts_code, trade_date, n_before=20, n_after=10):
    """
    绘制指定股票在指定日期前后的K线图
    
    参数:
    df: DataFrame, 包含股票数据的DataFrame
    ts_code: str, 股票代码
    trade_date: str, 交易日期
    n_before: int, 向前查看的交易日数量
    n_after: int, 向后查看的交易日数量
    """
    # 筛选指定股票的数据
    stock_df = df[df['ts_code'] == ts_code].copy()
    
    # 将trade_date转换为datetime类型
    target_date = pd.to_datetime(trade_date)
    stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])
    
    # 找到目标日期的索引
    target_idx = stock_df[stock_df['trade_date'] == target_date].index[0]
    
    # 获取目标日期前后的数据
    start_idx = max(0, target_idx - n_before)
    end_idx = min(len(stock_df), target_idx + n_after + 1)
    plot_df = stock_df.iloc[start_idx:end_idx].copy()
    
    # 设置索引为日期
    plot_df.set_index('trade_date', inplace=True)
    
    # 准备绘图数据
    plot_data = plot_df[['open', 'high', 'low', 'close', 'vol']]
    plot_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # 设置标记点
    dates = set([target_date])
    signal = [plot_data.loc[date]['High'] if date in dates else np.nan for date in plot_data.index]
    
    # 设置绘图参数
    kwargs = dict(
        type='candle',
        volume=True,
        figscale=1.5,
        figratio=(28, 16),
        datetime_format='%Y-%m-%d',
        volume_panel=1,
        title=f'\n{ts_code} K线图',
        # 删除这些参数，因为它们将通过style对象设置
        # style='charles',
        # gridstyle='--',
        # gridcolor='gray',
        # grid=True
    )
    
    # 设置风格
    mc = mpf.make_marketcolors(up='red', down='green', edge='black', inherit=True)
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='--',
        gridcolor='gray',
        gridaxis='both'
    )
    
    # 添加收盘价线、MACD指标和标记点
    close_plot = mpf.make_addplot(plot_data['Close'], color='black', width=0.8)
    
    # 分别处理MACD柱状图的正负值
    hist_pos = plot_df['macdhist'].copy()
    hist_neg = plot_df['macdhist'].copy()
    hist_pos[hist_pos <= 0] = np.nan
    hist_neg[hist_neg > 0] = np.nan
    
    # 添加MACD指标
    macd_hist_pos = mpf.make_addplot(hist_pos, type='bar', width=0.7, panel=2, color='red')
    macd_hist_neg = mpf.make_addplot(hist_neg, type='bar', width=0.7, panel=2, color='green')
    macd_line = mpf.make_addplot(plot_df['macd'], panel=2, color='blue', width=0.8)
    signal_line = mpf.make_addplot(plot_df['macdsignal'], panel=2, color='orange', width=0.8)
    
    signal_plot = mpf.make_addplot(signal, type='scatter', markersize=200, marker='v')
    plots = [close_plot, macd_hist_pos, macd_hist_neg, macd_line, signal_line, signal_plot]
    
    # 绘制图表
    mpf.plot(plot_data, **kwargs, style=style, addplot=plots)

# 使用示例:
plot_candlestick(stock_info, '000001.SZ', '20220321', 30, 30)

