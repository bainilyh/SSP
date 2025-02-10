import pandas as pd
import numpy as np
import sqlite3
import os
import math
from talib import *
from talib import MACD
import mplfinance as mpf
import matplotlib.pyplot as plt
import seaborn as sns


# 1. 成交量的均值比n_negative都大？
# 2. n_negative前的close小于当前close？
def find_positive_after_negative(df, column_name, n_negative=10, n_days=7, window=10):
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


        above_ma = group['close'] >= group['ma20']
        touch_points = above_ma.shift(1).rolling(window=window).sum()
        price_rate = touch_points.fillna(0)/window
        price_oscillation = (0.4 <= price_rate) & (price_rate <= 0.6)
        
        # # 计算价格波动条件
        # window = 20
        # threshold_ratio = 0.05
        # min_touch_points = 5
        
        # # 计算价格与均线的距离比例
        # distance_ratio = (group['low'] - group['ma20']) / group['ma20']
        
        # # 条件1: 最低价在均线上方
        # above_ma = distance_ratio > -threshold_ratio
        
        # # 条件2: 最高价与均线的距离不能太远
        # not_too_far = (group['high'] - group['ma20']) / group['ma20'] < threshold_ratio * 2
        
        # # 条件3: 至少有min_touch_points个点接近均线
        # touch_points = ((distance_ratio > -threshold_ratio) & 
        #                (distance_ratio < threshold_ratio)).rolling(window=window).sum()
        
        # # 价格波动条件:同时满足3个条件
        # price_oscillation = above_ma & not_too_far & (touch_points >= min_touch_points)
        
        # 同时满足所有条件
        return current_positive & prev_all_negative & pre_match_close & after_match_close & ~price_oscillation
    
    # 按code分组处理
    result_mask = df.groupby('ts_code').apply(check_conditions).reset_index(level=0, drop=True)
    
    return df[result_mask]

# def check_price_oscillation_above_ma20(df, window=20, threshold_ratio=0.05, min_touch_points=5):
#     """
#     判断股价是否在20日均线上方波动
    
#     参数:
#     df: DataFrame, 包含 'close', 'high', 'low' 列的数据框
#     threshold_ratio: float, 波动阈值，默认0.05（5%）
#     min_touch_points: int, 最小接触点数量，默认5
    
#     返回:
#     Series: 每行表示是否满足条件的布尔值
#     """
#     def check_conditions(group):
#         # 计算价格与均线的距离比例
#         distance_ratio = (group['low'] - group['ma20']) / group['ma20']
        
#         # 条件1: 最低价在均线上方
#         above_ma = distance_ratio > -threshold_ratio
        
#         # 条件2: 最高价与均线的距离不能太远
#         not_too_far = (group['high'] - group['ma20']) / group['ma20'] < threshold_ratio * 2
        
#         # 条件3: 至少有min_touch_points个点接近均线
#         touch_points = ((distance_ratio > -threshold_ratio) & 
#                        (distance_ratio < threshold_ratio)).rolling(window=window).sum()
        
#         # 返回每行是否同时满足3个条件
#         return above_ma & not_too_far & (touch_points >= min_touch_points)
    
#     return df.groupby('ts_code').apply(check_conditions).reset_index(level=0, drop=True)

# file_path = '/Users/bainilyhuang/Downloads/hshqt/src/main/resources/db/tushare.nfa'
# conn = sqlite3.connect(file_path)
# sql = 'SELECT * FROM daily_stock_info2 ORDER BY ts_code, trade_date asc'
# stock_info = pd.read_sql_query(sql, conn)

# 按code分组计算MACD
# stock_info['macdhist'] = stock_info.groupby('ts_code', include_groups=False).apply(
#     lambda x: MACD(x['close'], fastperiod=12, slowperiod=26, signalperiod=9)[2]
# ).reset_index(level=0, drop=True)
# stock_info['macdhist'] = stock_info.groupby('ts_code')['close'].apply(
#     lambda x: MACD(x, fastperiod=12, slowperiod=26, signalperiod=9)[2]
# ).reset_index(level=0, drop=True)

def calculate_macd_and_vol_ma(group):
    """同时计算MACD、成交量MA、布林线、ADX、HT_DCPERIOD和ATR指标以减少分组操作次数"""
    try:
        # 计算MACD
        macd, signal, hist = MACD(group['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        # 计算成交量MA 
        vol_ma = EMA(group['vol'], timeperiod=7)
        # 计算布林线
        upperband, middleband, lowerband = BBANDS(group['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        # 计算ADX
        adx = ADX(group['high'], group['low'], group['close'], timeperiod=14)
        # 计算HT_DCPERIOD
        dcperiod = HT_DCPERIOD(group['close'])
        # 计算HT_TRENDLINE
        trendline = HT_TRENDLINE(group['close'])
        # 计算HT_DCPHASE
        dcphase = HT_DCPHASE(group['close'])
        # 计算HT_PHASOR
        inphase, quadrature = HT_PHASOR(group['close'])
        # 计算HT_TRENDMODE
        trendmode = HT_TRENDMODE(group['close'])
        # 计算ATR
        atr = ATR(group['high'], group['low'], group['close'], timeperiod=14)
        # 计算ATR
        natr = NATR(group['high'], group['low'], group['close'], timeperiod=14)
        # 计算MA20
        ma20 = MA(group['close'], timeperiod=20)
        # 计算KD指标
        slowk, slowd = STOCH(group['high'], group['low'], group['close'], 
                            fastk_period=9, slowk_period=3, slowk_matype=0,
                            slowd_period=3, slowd_matype=0)
        
        # 计算修改后的ATR
        # 计算TR值
        high_low = group['high'] - group['low']
        high_close = abs(group['high'] - group['close'].shift(1))
        low_close = abs(group['low'] - group['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        # TR值除以前一日收盘价
        tr_ratio = tr / group['close'].shift(1)
        # 先计算14日移动平均
        modified_atr = tr_ratio.rolling(window=14).mean()
        # 最后转换为百分比并保留4位小数
        modified_atr = (modified_atr * 100).round(4)
        
        return pd.DataFrame({
            'macd': macd,
            'macdsignal': signal, 
            'macdhist': hist,
            'vol_ma': vol_ma,
            'upperband': upperband,
            'middleband': middleband,
            'lowerband': lowerband,
            'adx': adx,
            'dcperiod': dcperiod,
            'trendline': trendline,
            'dcphase': dcphase,
            'inphase': inphase,
            'quadrature': quadrature,
            'trendmode': trendmode,
            'atr': atr,
            'natr': natr,
            'ma20': ma20,
            'modified_atr': modified_atr,
            'slowk': slowk,
            'slowd': slowd
        }, index=group.index)
    except Exception as e:
        # 返回相同长度的NaN值
        length = len(group)
        return pd.DataFrame({
            'macd': [np.nan] * length,
            'macdsignal': [np.nan] * length,
            'macdhist': [np.nan] * length,
            'vol_ma': [np.nan] * length,
            'upperband': [np.nan] * length,
            'middleband': [np.nan] * length,
            'lowerband': [np.nan] * length,
            'adx': [np.nan] * length,
            'dcperiod': [np.nan] * length,
            'trendline': [np.nan] * length,
            'dcphase': [np.nan] * length,
            'inphase': [np.nan] * length,
            'quadrature': [np.nan] * length,
            'trendmode': [np.nan] * length,
            'atr': [np.nan] * length,
            'natr': [np.nan] * length,
            'ma20': [np.nan] * length,
            'modified_atr': [np.nan] * length,
            'slowk': [np.nan] * length,
            'slowd': [np.nan] * length
        }, index=group.index)



# # 只需要一次分组操作计算所有指标
# results = stock_info.groupby('ts_code', group_keys=False).apply(calculate_macd_and_vol_ma)
# results = results.reset_index(level=0, drop=True)

# # 一次性将所有结果赋值给原始DataFrame
# stock_info[['macd', 'macdsignal', 'macdhist', 'vol_ma', 'upperband', 'middleband', 'lowerband', 'adx', 'dcperiod', 'trendline', 'dcphase', 'inphase', 'quadrature', 'trendmode', 'atr', 'ma20', 'natr', 'modified_atr', 'slowk', 'slowd']] = results[['macd', 'macdsignal', 'macdhist', 'vol_ma', 'upperband', 'middleband', 'lowerband', 'adx', 'dcperiod', 'trendline', 'dcphase', 'inphase', 'quadrature', 'trendmode', 'atr', 'ma20', 'natr', 'modified_atr', 'slowk', 'slowd']]

# 查找符合条件的行
# 10-23年数据，共有185271行数据符合条件
# result = find_positive_after_negative(stock_info, 'macdhist', n_negative=20)

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

# # 计算性能统计
# result_df = calculate_trading_days_performance2(result, stock_info)

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
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Mac OS的中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
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
        panel_ratios=(6,2,2,2,2,2,2,2),  # 调整子图比例，增加KD子图
        returnfig=True,  # 返回fig对象以便添加子图
        vlines=dict(vlines=[mark_date], linewidths=1, linestyle='--', colors='gray', alpha=0.3)  # 添加垂直虚线
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
        mpf.make_addplot(plot_df['middleband'], color='blue', width=1),
        mpf.make_addplot(plot_df['upperband'], color='gray', width=1),
        mpf.make_addplot(plot_df['lowerband'], color='gray', width=1),
        mpf.make_addplot(plot_df['trendline'], color='red', width=1.5),  # 添加HT_TRENDLINE线
        mpf.make_addplot(hist_pos, type='bar', width=0.7, panel=2, color='red'),
        mpf.make_addplot(hist_neg, type='bar', width=0.7, panel=2, color='green'),
        mpf.make_addplot(plot_df['macd'], panel=2, color='blue', width=0.8),
        mpf.make_addplot(plot_df['macdsignal'], panel=2, color='orange', width=0.8),
        mpf.make_addplot(plot_df['vol_ma'], panel=1, color='blue', width=0.8),
        mpf.make_addplot(signal, type='scatter', markersize=200, marker='v'),
        mpf.make_addplot(plot_df['slowk'], panel=3, color='blue', width=1.5, ylabel='KD'),  # 添加KD线
        mpf.make_addplot(plot_df['slowd'], panel=3, color='orange', width=1.5),  # 添加KD线
        mpf.make_addplot(plot_df['adx'], panel=4, color='purple', width=1.5, ylabel='ADX'),  # ADX线
        mpf.make_addplot(plot_df['dcperiod'], panel=5, color='brown', width=1.5, ylabel='DCPeriod'),  # 添加DCPeriod线
        mpf.make_addplot(plot_df['atr'], panel=6, color='blue', width=1.5, ylabel='ATR'),  # 添加ATR线
        mpf.make_addplot(plot_df['modified_atr'], panel=7, color='green', width=1.5, ylabel='Modified ATR')  # 添加Modified ATR线
    ]
    
    # 绘制主图表
    fig, axes = mpf.plot(plot_data, **kwargs, style=style, addplot=plots)
    
    # 获取标记点的指标数据
    mark_data = plot_df.loc[mark_date]
    
    # 在左边居中位置添加文本框显示指标数据
    text = (f'close:         {mark_data["close"]:.3f}\n'
            f'middleband:    {mark_data["middleband"]:.3f}\n'
            f'Hist:          {mark_data["macdhist"]:.3f}\n'
            f'slowK:         {mark_data["slowk"]:.3f}\n'
            f'slowD:         {mark_data["slowd"]:.3f}\n'
            f'ADX:           {mark_data["adx"]:.3f}\n'
            f'DCPeriod:      {mark_data["dcperiod"]:.3f}\n'
            f'Trendline:     {mark_data["trendline"]:.3f}\n'
            f'Trendmode:     {mark_data["trendmode"]:.3f}\n'
            f'ATR:           {mark_data["atr"]:.3f}\n'
            f'Modified ATR:  {mark_data["modified_atr"]:.3f}\n')  # 添加Modified ATR值显示
    
    # 创建一个新的子图用于显示指标数据，调整位置到左边居中
    ax_text = fig.add_axes([0.02, 0.45, 0.15, 0.15])  # [left, bottom, width, height]
    ax_text.text(0, 0, text, fontsize=10, verticalalignment='center', family='monospace')
    ax_text.axis('off')  # 隐藏坐标轴
    
    if pic_path:
        plt.savefig(pic_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def show_data(ts_code, trade_date):
    pd.set_option('display.max_rows', None)
    # 计算日期范围
    start_date = pd.to_datetime(trade_date) - pd.Timedelta(days=60)
    end_date = pd.to_datetime(trade_date) + pd.Timedelta(days=30)
    
    # 转换为字符串格式 'YYYYMMDD'
    start_date = start_date.strftime('%Y%m%d') 
    end_date = end_date.strftime('%Y%m%d')
    
    condition = (stock_info['ts_code'] == ts_code) & \
                (stock_info['trade_date'] >= start_date) & \
                (stock_info['trade_date'] <= end_date)
                
    return stock_info[condition]
# # 预先计算条件和结果
# condition1 = result_df['row'] == 7
# result_7_df = result_df[condition1].sort_values(by='change_ratio', ascending=True)



def plot_change_ratio_distribution(result_df, row=None, pic_path=None, dpi=300):
    """
    绘制指定交易日(row)的涨跌幅分布图
    
    参数:
    result_df: DataFrame, 包含change_ratio和row列的数据框
    row: int, 指定要绘制的交易日序号,默认为None表示绘制所有交易日
    pic_path: str, 图片保存路径(可选)
    dpi: int, 图片分辨率，默认300
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 如果指定了row,只绘制该row的分布图
    if row is not None:
        # 创建单个图表
        plt.figure(figsize=(8, 6))
        data = result_df[result_df['row'] == row]['change_ratio'].dropna()
        
        # 绘制直方图和KDE曲线
        sns.histplot(data, bins=30, kde=True, color='skyblue',
                    edgecolor='white', linewidth=0.5)
        
        # 添加统计信息
        mean = data.mean()
        median = data.median()
        plt.axvline(mean, color='r', linestyle='--', linewidth=1)
        plt.axvline(median, color='g', linestyle='-', linewidth=1)
        
        # 设置标题和标签
        plt.title(f'第 {row+1} 交易日涨跌幅分布 (n={len(data)})', fontsize=14)
        plt.xlabel('涨跌幅', fontsize=12)
        plt.ylabel('频数', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加图例
        plt.legend([f'均值: {mean:.2%}', f'中位数: {median:.2%}'],
                  loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if pic_path:
            plt.savefig(pic_path, dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    else:
        # 获取唯一的row值并排序
        rows = sorted(result_df['row'].unique())
        
        # 创建子图布局
        n_cols = 3  # 每行3个子图
        n_rows = math.ceil(len(rows) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        fig.suptitle('各交易日涨跌幅分布', fontsize=16, y=1.02)
        
        # 确保axes总是二维数组
        if len(rows) <= n_cols:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(n_rows, n_cols)
        
        # 遍历每个row绘制分布
        for idx, row in enumerate(rows):
            ax = axes[idx//n_cols, idx%n_cols]
            data = result_df[result_df['row'] == row]['change_ratio'].dropna()
            
            # 绘制直方图和KDE曲线
            sns.histplot(data, bins=30, kde=True, ax=ax, color='skyblue',
                        edgecolor='white', linewidth=0.5)
            
            # 添加统计信息
            mean = data.mean()
            median = data.median()
            ax.axvline(mean, color='r', linestyle='--', linewidth=1)
            ax.axvline(median, color='g', linestyle='-', linewidth=1)
            
            # 设置标题和标签
            ax.set_title(f'第 {row+1} 交易日 (n={len(data)})', fontsize=12)
            ax.set_xlabel('涨跌幅', fontsize=10)
            ax.set_ylabel('频数', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 添加图例
            ax.legend([f'均值: {mean:.2%}', f'中位数: {median:.2%}'],
                     loc='upper right', fontsize=8)

        # 隐藏多余的子图
        for idx in range(len(rows), n_rows*n_cols):
            axes[idx//n_cols, idx%n_cols].axis('off')
        
        plt.tight_layout()
        
        if pic_path:
            plt.savefig(pic_path, dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# 使用示例：
# result = check_price_oscillation_above_ma20(stock_info)

# # 使用tqdm显示进度
# from tqdm import tqdm
# pic_path = '/Users/bainilyhuang/Downloads/pic_path/'

# # 并行处理图表生成
# from concurrent.futures import ThreadPoolExecutor
# import os

# def process_plot(row):
#     ts_code = row['ts_code']
#     trade_date = row['trade_date']
#     filename = f'{pic_path}/{ts_code}-{trade_date}.png'
#     if not os.path.exists(filename):  # 避免重复生成
#         plot_candlestick(stock_info, ts_code, trade_date, 60, 30, pic_path=filename, dpi=300)

# # 使用线程池并行处理
# with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
#     list(tqdm(executor.map(process_plot, [result_df.iloc[i] for i in range(len(result_df))]), 
#              total=len(result_df), 
#              desc="Generating plots"))

# print('end')

# # 使用示例:
# ts_code = '601872.SH'
# trade_date = '20220921'
# plot_candlestick(stock_info, '601872.SH', '20220921', 60, 30, pic_path=f'/Users/bainilyhuang/Downloads/pic_path/{ts_code}-{trade_date}.png', dpi=300)




# condition2 = (result['row'] == 7) & (result['change_ratio'] > 0)

# len(result[condition2]) / len(result[condition1])


# result_df.head()


# condition3 = (stock_info['ts_code'] == '300780.SZ') & (stock_info['trade_date'] >= '20221001') & (stock_info['trade_date'] <= '20221124')
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


# for i in range(len(result_7_df)):
#     plot_candlestick(stock_info, result_7_df.iloc[i]['ts_code'], result_7_df.iloc[i]['trade_date'], 60, 30, 7)


# sql2 = 'SELECT * FROM result_7_df'
# result_7_df_old = pd.read_sql_query(sql2, conn)
# result_7_df_old = result_7_df_old.drop('_merge', axis=1)



# 使用示例：
# plot_change_ratio_distribution(result_df, pic_path='change_ratio_distribution.png')

# pd.io.sql.to_sql(a, name='filter1', con=conn, if_exists='append', index=False)