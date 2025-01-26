import pandas as pd
import numpy as np
import sqlite3
import os
import math
from talib import *
from talib import MACD
import mplfinance as mpf
import matplotlib.pyplot as plt


# 1. 成交量的均值比n_negative都大？
# 2. n_negative前的close小于当前close？
def find_positive_after_negative(df, column_name, n_negative=10, n_days=7):
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

        if group['ts_code'].iloc[0] == '688019.SZ':
            print()
        
        match_close = group['pre_close'] == group['close'].shift(1)
        pre_match_close = match_close.rolling(window=n_negative).sum() == n_negative
        after_match_close = pre_match_close.shift(-n_days)
        
        # 创建一个布尔掩码，标记小于0的值
        negative_mask = series < 0
        
        # 使用rolling计算前n_negative个值是否都为负
        all_negative = negative_mask.rolling(window=n_negative).sum() == n_negative
        
        # 当前值大于等于0的掩码
        current_positive = series >= 0
        
        # 将all_negative向前移动一位，这样它就对应于"之前"的n_negative个值
        prev_all_negative = all_negative.shift(1)
        
        # 检查当前vol_ma是否大于前n_days内的所有vol_ma
        vol_ma_series = group['vol_ma']
        vol_ma_max = vol_ma_series.rolling(window=n_days).max().shift(1)
        vol_ma_condition = vol_ma_series > vol_ma_max
        
        # 同时满足所有条件
        return current_positive & prev_all_negative & pre_match_close & after_match_close & vol_ma_condition
    
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

def calculate_macd_and_vol_ma(group):
    """同时计算MACD和成交量MA以减少分组操作次数"""
    try:
        # 计算MACD
        macd, signal, hist = MACD(group['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        # 计算成交量MA 
        vol_ma = EMA(group['vol'], timeperiod=7)
        
        return pd.DataFrame({
            'macd': macd,
            'macdsignal': signal, 
            'macdhist': hist,
            'vol_ma': vol_ma
        }, index=group.index)
    except Exception as e:
        # 返回相同长度的NaN值
        length = len(group)
        return pd.DataFrame({
            'macd': [np.nan] * length,
            'macdsignal': [np.nan] * length,
            'macdhist': [np.nan] * length,
            'vol_ma': [np.nan] * length
        }, index=group.index)

# 只需要一次分组操作计算所有指标
results = stock_info.groupby('ts_code', group_keys=False).apply(calculate_macd_and_vol_ma)
results = results.reset_index(level=0, drop=True)

# 一次性将所有结果赋值给原始DataFrame
stock_info[['macd', 'macdsignal', 'macdhist', 'vol_ma']] = results[['macd', 'macdsignal', 'macdhist', 'vol_ma']]

# 查找符合条件的行
# 10-23年数据，共有185271行数据符合条件
result = find_positive_after_negative(stock_info, 'macdhist', n_negative=20)

# def calculate_trading_days_performance2(result_df, stock_info_df):

# 查看result当前行后1个礼拜内有多少stock_info的数据是涨的
# 计算每个信号后7个交易日的涨跌情况
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

# 计算性能统计
result = calculate_trading_days_performance2(result, stock_info)

def plot_candlestick(df, ts_code, trade_date, n_before=20, n_after=10, n_rows=7, pic_path=None, dpi=300):
    """
    绘制指定股票在指定日期前后的K线图
    
    参数:
    df: DataFrame, 包含股票数据的DataFrame
    ts_code: str, 股票代码
    trade_date: str, 交易日期
    n_before: int, 向前查看的交易日数量
    n_after: int, 向后查看的交易日数量
    pic_path: str, 图片保存路径，默认为None不保存
    dpi: int, 图片分辨率，默认300
    """
    # 使用布尔索引一次性筛选数据
    stock_df = df[df['ts_code'] == ts_code].copy()
    
    # 将trade_date转换为datetime类型
    target_date = pd.to_datetime(trade_date)
    stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])
    
    # 使用布尔索引找到目标日期位置
    target_mask = stock_df['trade_date'] == target_date
    target_idx = stock_df.index[target_mask][0]
    
    # 获取目标日期前后的数据
    plot_df = df.iloc[max(0, target_idx - n_before):min(len(df), target_idx + n_after + 1)].copy()
    plot_df['trade_date'] = pd.to_datetime(plot_df['trade_date'])
    plot_df.set_index('trade_date', inplace=True)
    
    # 一次性准备所有绘图数据
    plot_data = plot_df[['open', 'high', 'low', 'close', 'vol']].rename(
        columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'vol': 'Volume'}
    )
    
    # 设置标记点
    mark_date = plot_df.index[plot_df.index.get_loc(target_date) - n_rows]
    signal = pd.Series(np.nan, index=plot_data.index)
    signal[mark_date] = plot_data.loc[mark_date, 'High']
    
    # 预先设置好所有参数
    kwargs = dict(
        type='candle',
        volume=True,
        figscale=1.5,
        figratio=(28, 16),
        datetime_format='%Y-%m-%d',
        volume_panel=1,
        title=f'\n{ts_code}',
        savefig=dict(fname=pic_path, dpi=dpi) if pic_path else None
    )
    
    style = mpf.make_mpf_style(
        marketcolors=mpf.make_marketcolors(up='red', down='green', edge='black', inherit=True),
        gridstyle='--',
        gridcolor='gray',
        gridaxis='both'
    )
    
    # 一次性处理MACD数据
    hist = plot_df['macdhist']
    hist_pos = hist.where(hist > 0)
    hist_neg = hist.where(hist <= 0)
    
    # 构建所有addplot对象
    plots = [
        mpf.make_addplot(plot_data['Close'], color='black', width=0.8),
        mpf.make_addplot(hist_pos, type='bar', width=0.7, panel=2, color='red'),
        mpf.make_addplot(hist_neg, type='bar', width=0.7, panel=2, color='green'),
        mpf.make_addplot(plot_df['macd'], panel=2, color='blue', width=0.8),
        mpf.make_addplot(plot_df['macdsignal'], panel=2, color='orange', width=0.8),
        mpf.make_addplot(plot_df['vol_ma'], panel=1, color='blue', width=0.8),
        mpf.make_addplot(signal, type='scatter', markersize=200, marker='v')
    ]
    
    # 绘制图表
    mpf.plot(plot_data, **kwargs, style=style, addplot=plots)

# 预先计算条件和结果
condition1 = result['row'] == 7
result_df = result[condition1].sort_values(by='change_ratio', ascending=True)

# 使用tqdm显示进度
from tqdm import tqdm
pic_path = '/Users/bainilyhuang/Downloads/pic_path/'

# 并行处理图表生成
from concurrent.futures import ThreadPoolExecutor
import os

def process_plot(row):
    ts_code = row['ts_code']
    trade_date = row['trade_date']
    filename = f'{pic_path}/{ts_code}-{trade_date}.png'
    if not os.path.exists(filename):  # 避免重复生成
        plot_candlestick(stock_info, ts_code, trade_date, 60, 30, pic_path=filename, dpi=300)

# 使用线程池并行处理
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    list(tqdm(executor.map(process_plot, [result_df.iloc[i] for i in range(len(result_df))]), 
             total=len(result_df), 
             desc="Generating plots"))

print('end')

# # 使用示例:
# ts_code = '601872.SH'
# trade_date = '20220921'
# plot_candlestick(stock_info, '601872.SH', '20220921', 60, 30, pic_path=f'/Users/bainilyhuang/Downloads/pic_path/{ts_code}-{trade_date}.png', dpi=300)




# condition2 = (result['row'] == 7) & (result['change_ratio'] > 0)

# len(result[condition2]) / len(result[condition1])


# result_df.head()


# condition3 = (stock_info['ts_code'] == '601872.SH') & (stock_info['trade_date'] >= '20220901') & (stock_info['trade_date'] <= '20220930')
# # i = 0
# # plot_candlestick(stock_info, result_df.iloc[i]['ts_code'], result_df.iloc[i]['trade_date'], 60, 30)



# pd.set_option('display.max_rows', None)        # 显示所有行
# pd.set_option('display.max_columns', None)     # 显示所有列
# pd.set_option('display.width', None)           # 显示的宽度，None表示自动
# pd.set_option('display.max_colwidth', None)    # 列宽度，None表示完整显示
# pd.set_option('display.float_format', lambda x: '%.3f' % x)  # 设置浮点数显示格式，保留3位小数



# 584880 
# 300988.SZ


# mask = ~((old_df['ts_code'].isin(result_df['ts_code'])) & (old_df['trade_date'].isin(result_df['trade_date'])))
