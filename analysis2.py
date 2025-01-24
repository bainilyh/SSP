import pandas as pd
import numpy as np
import sqlite3
import os
import math
from talib import *
import mplfinance as mpf
import matplotlib.pyplot as plt

def find_positive_after_negative(df, macdhist, macdsignal_, n_negative=10, n_days=7):
    """
    找出大于等于0，且之前n行数据小于0的位置，同时要求第1行macdsignal大于0且第n行macdsignal小于0
    
    参数:
    df: DataFrame
    macdhist: 要检查的列名
    n_negative: 之前需要多少个负数，默认10
    
    返回:
    符合条件的行索引
    """
    def check_conditions(group):
        if group['ts_code'].unique()[0] == '300988.SZ':
            print()
        # 获取目标列
        series = group[macdhist]
        macdsignal = group[macdsignal_]
        
        # 检查pre_close和前一天close是否相等
        close_match = (group['pre_close'] == group['close'].shift(1))
        
        # 创建一个布尔掩码，标记小于0的值
        negative_mask = series < 0
        
        # 使用rolling计算前n_negative个值是否都为负
        all_negative = negative_mask.rolling(window=n_negative).sum() == n_negative
        
        # 当前值大于等于0的掩码
        current_positive = series >= 0
        
        # 将all_negative向前移动一位，这样它就对应于"之前"的n_negative个值
        prev_all_negative = all_negative.shift(1)
        
        # 检查macdsignal的条件:第1行>0且第n行<0
        signal_conditions = (macdsignal.shift(n_negative-1) > 0) & (macdsignal < 0)
        
        close_match = close_match.rolling(window=n_negative).sum() == n_negative
        pre_close_match = close_match.shift(1)
        after_close_match = close_match.shift(-n_days)
        
        # 检查当前close是否大于n_negative天前的close
        close_price_condition = group['close'] < group['close'].shift(n_negative)
        
        # 同时满足所有条件:
        # 1. 当前值>=0
        # 2. 之前n_negative个值<0
        # 3. macdsignal的条件
        # 4. pre_close与前一日close相等且在整个周期内都匹配
        # 5. 当前close大于n_negative天前的close
        return current_positive & prev_all_negative & pre_close_match & after_close_match & close_price_condition
    
    # 按code分组处理
    result_mask = df.groupby('ts_code').apply(check_conditions).reset_index(level=0, drop=True)
    
    return df[result_mask]

# file_path = '/Users/bainilyhuang/Downloads/hshqt/src/main/resources/db/tushare.nfa'
file_path = "C:\\Users\\huang\\Downloads\\stock.nfa"
conn = sqlite3.connect(file_path)
sql = 'SELECT * FROM stock_info_daily ORDER BY ts_code, trade_date asc'
# sql = 'SELECT code as ts_code, day as trade_date, close, open, high, low FROM stock_info where day >= "20200101" and day <= "20221231" ORDER BY code, day asc'
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
result = find_positive_after_negative(stock_info, 'macdhist', 'macdsignal', n_negative=20)

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


result_df = calculate_trading_days_performance2(result, stock_info)

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
# result = calculate_trading_days_performance(result, stock_info)

# condition = ''
# df_sorted_desc = result_df[condition].sort_values(by='change_ratio', ascending=False)

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
    end_idx = min(len(df), target_idx + n_after + 1)
    plot_df = df.iloc[start_idx:end_idx].copy()
    
    # 设置索引为日期
    plot_df.set_index('trade_date', inplace=True)
    
    # 确保索引是DatetimeIndex
    plot_df.index = pd.to_datetime(plot_df.index)
    
    # 准备绘图数据
    plot_data = plot_df[['open', 'high', 'low', 'close', 'vol']]
    plot_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # 设置标记点
    dates = set([target_date])
    signal = [plot_data.loc[date]['High'] if date in dates else np.nan for date in plot_data.index]
    signal = np.array(signal)
    
    # 设置绘图参数
    kwargs = dict(
        type='candle',
        volume=True,
        figscale=1.5,
        figratio=(28, 16),
        datetime_format='%Y-%m-%d',
        volume_panel=1,
        title=f'\n{ts_code}',
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
    close_plot = mpf.make_addplot(plot_data['Close'].to_numpy(), color='black', width=0.8)
    
    # 分别处理MACD柱状图的正负值
    hist_pos = plot_df['macdhist'].to_numpy().copy()
    hist_neg = plot_df['macdhist'].to_numpy().copy()
    hist_pos[hist_pos <= 0] = np.nan
    hist_neg[hist_neg > 0] = np.nan
    
    # 添加MACD指标
    macd_hist_pos = mpf.make_addplot(hist_pos, type='bar', width=0.7, panel=2, color='red')
    macd_hist_neg = mpf.make_addplot(hist_neg, type='bar', width=0.7, panel=2, color='green')
    macd_line = mpf.make_addplot(plot_df['macd'].to_numpy(), panel=2, color='blue', width=0.8)
    signal_line = mpf.make_addplot(plot_df['macdsignal'].to_numpy(), panel=2, color='orange', width=0.8)
    
    signal_plot = mpf.make_addplot(signal, type='scatter', markersize=200, marker='v')
    plots = [close_plot, macd_hist_pos, macd_hist_neg, macd_line, signal_line, signal_plot]
    
    # 检查 DataFrame 是否为空
    if plot_df.empty:
        print(f"没有可用的数据用于 {ts_code} 在 {trade_date}。")
        return
    
    # 绘制图表
    mpf.plot(plot_data, **kwargs, style=style, addplot=plots)

# 使用示例:
# plot_candlestick(stock_info, '000619.SZ', '20220721', 30, 30)
print('end')

# 测试
condition = (result_df['row'] == 7)
condition2 = (result_df['row'] == 7) & ((result_df['change_ratio'] > 0))

len(result_df[condition2]) / len(result_df[condition])

df_sorted_desc = result_df[condition].sort_values(by='change_ratio', ascending=True)

a = (stock_info['ts_code'] == '300988.SZ') & (stock_info['trade_date'] > '20220501') & (stock_info['trade_date'] < '20220525')


# pd.io.sql.to_sql(stock_info, name='tqa_info', con=conn, if_exists='append', index=False)
# sql = "select * from condition1"
# con1 = pd.read_sql_query(sql, conn)

def find_diff_merge(a, b):
    # 首先确保两个DataFrame都没有'_merge'列
    if '_merge' in a.columns:
        a = a.drop('_merge', axis=1)
    if '_merge' in b.columns:
        b = b.drop('_merge', axis=1)
    
    result = pd.merge(a, b, 
                     on=['ts_code', 'trade_date'], 
                     how='left', 
                     indicator=True)
    return result[result['_merge'] == 'left_only'].drop('_merge', axis=1)



# cols_to_drop = [col for col in a.columns if col.endswith('_y')]
# a = a.drop(columns=cols_to_drop)
# a.columns = [col[:-2] if col.endswith('_x') else col for col in a.columns]


# # 设置显示所有列
# pd.set_option('display.max_columns', None)
# # 设置显示所有行
# pd.set_option('display.max_rows', None)
# # 设置value的显示长度为100，默认为50
# pd.set_option('display.max_colwidth', 100)
# # 设置显示宽度为None，即不限制宽度
# pd.set_option('display.width', None)
