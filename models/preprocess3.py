import pandas as pd
import numpy as np
import sqlite3
import os
import math
import talib
from talib import abstract
import mplfinance as mpf
import matplotlib.pyplot as plt
import seaborn as sns
from analysis4 import plot_candlestick
from sklearn.preprocessing import MinMaxScaler
import matplotlib.font_manager as fm
import platform
from scipy.stats import gaussian_kde


def label_price_changes(df, n=5):
    """
    标注股票价格变化的买入卖出点，优化后的选股逻辑：
    买入点(1)需同时满足:
    - 后5天内最大涨幅>5%
    - 后5天内最大跌幅>-3% (控制风险)
    - 5日平均成交量大于20日平均成交量 (确保交易活跃)
    - 收盘价大于5日均线 (确保上升趋势)
    
    卖出点(-1)和数据不连续点(-2)保持不变
    其他标记为非买入点(0)
    
    参数:
    df: DataFrame, 包含股票数据的DataFrame,需要包含必要的价格和成交量数据
    
    返回:
    添加了'label'列的DataFrame
    """
    def process_group(group):
        # 计算未来5天的close和pre_close
        future_close = pd.DataFrame({
            'min': group['close'][::-1].rolling(window=n, min_periods=1).min()[::-1].shift(-1),
            'max': group['close'][::-1].rolling(window=n, min_periods=1).max()[::-1].shift(-1)
        })
        current_close = group['close']
        
        # 计算涨跌幅
        max_changes = (future_close['max'] - current_close) / current_close
        min_changes = (future_close['min'] - current_close) / current_close

        macdhist = group['macdhist']
        
        # 检查pre_close是否与前一日close一致
        close_shifted = group['close'].shift(1)
        pre_close_match = (group['pre_close'] == close_shifted) | close_shifted.isna()
        
        # 创建标签
        labels = pd.Series(0, index=group.index)
        
        # 标记数据不连续点
        labels = np.where(~pre_close_match, -2, labels)
        
        # 优化后的买入点标记条件
        buy_conditions = (
            (labels == 0) &                    # 非不连续点
            (max_changes > 0.1) &             # 未来n天最大涨幅>5%
            (min_changes >= -0.03)           # 未来n天最大跌幅>-3%
        )
        
        # 标记买入卖出点
        labels = np.where(buy_conditions, 1, labels)
        labels = np.where((labels == 0) & (min_changes < -0.05), -1, labels)
        
        # 最后5天标记为0
        labels[-n:] = 0
        
        return pd.Series(labels, index=group.index)
    # 按股票代码分组处理
    df['label'] = df.groupby('ts_code', group_keys=False).apply(process_group)
    
    return df


def analyze_label_distribution(df):
    """
    分析标签分布情况
    
    参数:
    df: DataFrame, 包含label列的DataFrame
    
    返回:
    包含标签分布统计信息的DataFrame

                    count  percentage    description  avg_5d_change%
        label                                                   
        -2       3486    0.299219          数据不连续           94.29
        -1     163565   14.039529  卖出点(5日内跌幅>5%)           18.92
        0     747995   64.203816             观望            4.92
        1     249986   21.457436  买入点(5日内涨幅>5%)           13.57
    """
    # 计算每个标签的数量和占比
    label_counts = df['label'].value_counts()
    label_pcts = df['label'].value_counts(normalize=True)
    
    # 创建结果DataFrame
    stats = pd.DataFrame({
        'count': label_counts,
        'percentage': label_pcts * 100
    })
    
    # 添加标签说明
    label_desc = {
        1: '买入点(5日内涨幅>5%)',
        -1: '卖出点(5日内跌幅>5%)', 
        -2: '数据不连续',
        0: '观望'
    }
    stats['description'] = stats.index.map(label_desc)
    
    # 按标签排序
    stats = stats.sort_index()
    
    # 计算每个标签对应的平均涨跌幅
    avg_changes = df.groupby('label')['close'].apply(
        lambda x: ((x.shift(-5) - x) / x).mean() * 100
    ).round(2)
    stats['avg_5d_change%'] = avg_changes
    
    return stats


def load_stock_data(table_name='daily_info', file_path='./data/train.nfa'):
    """
    从SQLite数据库加载股票数据
    
    参数:
    file_path: str, SQLite数据库文件路径
    
    返回:
    DataFrame: 包含股票数据的DataFrame,按股票代码和交易日期排序
    """
    conn = sqlite3.connect(file_path)
    sql = f'SELECT * FROM {table_name} WHERE trade_date >= "2010-01-01" ORDER BY ts_code, trade_date asc'
    stock_info = pd.read_sql_query(sql, conn)
    conn.close()
    return stock_info


def calculate_technical_indicators(group):
    """
    计算技术指标
    
    参数:
    group: DataFrame, 包含股票数据的一个分组，需要包含 OHLCV 数据
    
    返回:
    DataFrame: 包含计算出的技术指标
    """
    try:
        # ===== 基础移动平均线 =====
        ma7 = talib.MA(group['close'], timeperiod=7)
        ma14 = talib.MA(group['close'], timeperiod=14) 
        ma30 = talib.MA(group['close'], timeperiod=30)
        # 调节时间周期，<20 震荡周期 20-25 萌芽  25-50 趋势加速  >50 趋势过高（反转）
        adx7 = talib.ADX(group['high'], group['low'], group['close'], timeperiod=7)
        adx14 = talib.ADX(group['high'], group['low'], group['close'], timeperiod=14)
        
        
        # 抛物线转向指标
        # 参数调优
        # 震荡市优化：降低acceleration（如0.01）以减少假信号，但会延迟反转提示。
        # 趋势市强化：提高maximum（如0.25）以更快跟踪强势趋势，但可能过早止损。
        # 默认参数：acceleration=0.02, maximum=0.2
        
        # 缺点
        # 震荡市失效
        # 滞后性
        sar = talib.SAR(group['high'], group['low'])
        # ===== 布林带指标 =====
        upper, middle, lower = talib.BBANDS(
            group['close'], 
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        
        # ===== MACD指标 =====
        macd, macdsignal, macdhist = talib.MACD(
            group['close'],
            fastperiod=12,
            slowperiod=26, 
            signalperiod=9
        )
        
        # 70以上（超买）：提示短期价格可能回调或转跌，但需注意强趋势市场可能长期超买（如牛市）。
        # 30以下（超卖）：暗示价格可能反弹或转涨，但在熊市中可能持续超卖。
        # ===== RSI指标 =====
        rsi6 = talib.RSI(group['close'], timeperiod=6)
        rsi12 = talib.RSI(group['close'], timeperiod=12)
        rsi24 = talib.RSI(group['close'], timeperiod=24)
        
        # ===== 涨幅指标 =====
        # 计算相对于前N天的涨幅百分比
        pct_1d = group['close'].pct_change(periods=1) * 100  # 1天涨幅
        pct_6d = group['close'].pct_change(periods=6) * 100  # 6天涨幅
        pct_12d = group['close'].pct_change(periods=12) * 100  # 12天涨幅
        pct_24d = group['close'].pct_change(periods=24) * 100  # 24天涨幅
        
        # ===== 除权指标 =====
        # 判断当日是否除权(前收盘价与上一日收盘价不一致)
        has_xrights = (group['pre_close'] != group['close'].shift(1)).astype(int)
        # 计算30日内是否有除权
        has_xrights_30d = has_xrights.rolling(window=30, min_periods=1).sum() > 0
        
        # 合并所有特征
        result_df = pd.DataFrame({
            # 移动平均线
            'ma7': ma7,
            'ma14': ma14, 
            'ma30': ma30,
            
            # ADX
            'adx7': adx7,
            'adx14': adx14,
            
            # 抛物线转向指标
            'sar': sar,
            
            # MACD
            'macd': macd,
            'macdsignal': macdsignal,
            'macdhist': macdhist,
            
            # 布林带
            'bb_upper': upper,
            'bb_middle': middle, 
            'bb_lower': lower,
            
            # RSI
            'rsi6': rsi6,
            'rsi12': rsi12,
            'rsi24': rsi24,
            
            # 涨幅指标
            'pct_change_1d': pct_1d,
            'pct_change_6d': pct_6d,
            'pct_change_12d': pct_12d,
            'pct_change_24d': pct_24d,
            
            # 除权指标
            'has_xrights': has_xrights,
            'has_xrights_30d': has_xrights_30d
        })
        
        return result_df
        
    except Exception as e:
        print(f"计算技术指标时出错: {str(e)}")
        # 返回包含NaN的DataFrame
        columns = [
            'ma7', 'ma14', 'ma30',
            'adx7', 'adx14',
            'sar',
            'macd', 'macdsignal', 'macdhist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'rsi6', 'rsi12', 'rsi24',
            'pct_change_1d', 'pct_change_6d', 
            'pct_change_12d', 'pct_change_24d',
            'has_xrights', 'has_xrights_30d'
        ]
        return pd.DataFrame({col: [np.nan] * len(group) for col in columns})

def add_technical_indicators(stock_info):
    """
    为股票数据添加技术指标
    
    参数:
    stock_info: DataFrame, 原始股票数据
    
    返回:
    DataFrame: 添加技术指标后的股票数据
    """
    # 创建一个新的DataFrame副本
    stock_info = stock_info.copy()
    
    # 按股票代码分组计算指标
    results = stock_info.groupby('ts_code', group_keys=False).apply(calculate_technical_indicators)
    results = results.reset_index(level=0, drop=True)
    
    # 确保不会覆盖原有列
    existing_cols = set(stock_info.columns)
    new_cols = [col for col in results.columns if col not in existing_cols]
    
    # 使用pd.concat合并新的技术指标列
    if new_cols:
        stock_info = pd.concat([
            stock_info,
            results[new_cols]
        ], axis=1)
    
    return stock_info


def select_stocks_by_indicators(df, date=None):
    """
    根据技术指标筛选股票
    
    条件：
    1. RSI24 < 31 (超卖区间)
    2. RSI12 > RSI24 (短期RSI上穿中期RSI)
    3. RSI6 > RSI12 (超短期RSI上穿短期RSI)
    4. MACD柱 > 0 (MACD柱为正)
    
    参数:
    df: DataFrame, 包含技术指标的股票数据
    date: str, 可选，指定日期筛选，格式：'YYYYMMDD'
    
    返回:
    DataFrame: 符合条件的股票数据
    """
    # 基础条件
    conditions = (
        (df['has_xrights_30d'] == False) &
        (df['label'] == 1)
    )
    
    # 如果指定了日期，添加日期过滤
    if date:
        conditions = conditions & (df['trade_date'] == date)
    
    # 过滤掉北交所股票（以.BJ结尾的股票代码）
    conditions = conditions & (~df['ts_code'].str.endswith('.BJ'))
    
    # 应用筛选条件
    selected_stocks = df[conditions].copy()
    
    # 按日期和股票代码排序
    selected_stocks = selected_stocks.sort_values(['trade_date', 'ts_code'])
    
    return selected_stocks


def plot_histogram(df, column_name, bins=50, title=None, figsize=(12, 6)):
    # """
    # 绘制指定列的直方图
    # """
    # # 设置中文字体
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    if column_name not in df.columns:
        raise ValueError(f"列名 '{column_name}' 不存在于数据中")
    
    plt.figure(figsize=figsize)
    
    # 获取数据并清理
    data = df[column_name].dropna()
    # 移除无穷大值
    data = data[~np.isinf(data)]
    # 移除异常值（可选）
    # q1, q3 = data.quantile([0.01, 0.99])
    # data = data[(data >= q1) & (data <= q3)]
    
    # 计算基本统计量
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    
    try:
        # 使用distplot替代histplot+kde
        sns.histplot(data=data, bins=bins, stat='density')
        sns.kdeplot(data=data, color='red', linewidth=2)
    except Exception as e:
        print(f"绘图出错: {str(e)}")
        # 如果出错，尝试只绘制直方图
        plt.hist(data, bins=bins, density=True)
    
    # 添加统计信息和其他图形元素
    plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=1, label=f'均值: {mean_val:.2f}')
    plt.axvline(median_val, color='g', linestyle='dashed', linewidth=1, label=f'中位数: {median_val:.2f}')
    
    if title is None:
        title = f'{column_name} 分布图'
    plt.title(title)
    plt.xlabel(column_name)
    plt.ylabel('密度')
    
    stats_text = f'样本数: {len(data):,}\n'
    stats_text += f'均值: {mean_val:.2f}\n'
    stats_text += f'中位数: {median_val:.2f}\n'
    stats_text += f'标准差: {std_val:.2f}'
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def analyze_feature(df, feature_name, bins=50, figsize=(15, 10)):
    """
    分析特征的有效性
    
    参数:
    df: DataFrame, 包含特征和标签的数据
    feature_name: str, 要分析的特征名
    bins: int, 直方图的区间数
    figsize: tuple, 图表大小
    """
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    plt.figure(figsize=figsize)
    
    # 创建两个子图
    plt.subplot(2, 1, 1)
    
    # 分别获取买入点和非买入点的数据
    buy_data = df[df['label'].isin([1, 2])][feature_name].dropna()
    non_buy_data = df[~df['label'].isin([1, 2])][feature_name].dropna()
    
    # 绘制买入点和非买入点的分布对比
    sns.histplot(data=buy_data, bins=bins, alpha=0.5, label='买入点', color='green')
    sns.histplot(data=non_buy_data, bins=bins, alpha=0.5, label='非买入点', color='red')
    
    # 计算基本统计量
    buy_mean = buy_data.mean()
    non_buy_mean = non_buy_data.mean()
    
    # 添加均值线
    plt.axvline(buy_mean, color='green', linestyle='dashed', linewidth=1, 
                label=f'买入点均值: {buy_mean:.2f}')
    plt.axvline(non_buy_mean, color='red', linestyle='dashed', linewidth=1, 
                label=f'非买入点均值: {non_buy_mean:.2f}')
    
    plt.title(f'{feature_name} 买入点与非买入点分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加箱线图对比
    plt.subplot(2, 1, 2)
    data_to_plot = [
        df[df['label'].isin([1, 2])][feature_name].dropna(),
        df[~df['label'].isin([1, 2])][feature_name].dropna()
    ]
    plt.boxplot(data_to_plot, labels=['买入点', '非买入点'])
    plt.title(f'{feature_name} 买入点与非买入点箱线图对比')
    plt.grid(True, alpha=0.3)
    
    # 计算统计信息
    stats_text = (
        f"买入点统计:\n"
        f"样本数: {len(buy_data):,}\n"
        f"均值: {buy_data.mean():.2f}\n"
        f"中位数: {buy_data.median():.2f}\n"
        f"标准差: {buy_data.std():.2f}\n\n"
        f"非买入点统计:\n"
        f"样本数: {len(non_buy_data):,}\n"
        f"均值: {non_buy_data.mean():.2f}\n"
        f"中位数: {non_buy_data.median():.2f}\n"
        f"标准差: {non_buy_data.std():.2f}\n"
    )
    
    # 计算区分度
    distinction = abs(buy_mean - non_buy_mean) / ((buy_data.std() + non_buy_data.std()) / 2)
    stats_text += f"\n区分度: {distinction:.2f}"
    
    # 在图表右上角添加统计信息
    plt.text(1.15, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return distinction


def identify_price_box(df, ts_code, period='short', min_touches=3, breakout_pct=0.03):
    """
    识别股票的价格箱体，包括支撑位和阻力位
    
    参数:
    df: DataFrame, 包含股票数据的DataFrame
    ts_code: str, 股票代码
    period: str, 'short'表示短线箱体(20个交易日)，'long'表示中长线箱体(120个交易日)
    min_touches: int, 确认支撑/阻力位的最小触碰次数
    breakout_pct: float, 确认突破箱体的百分比阈值
    
    返回:
    DataFrame: 包含箱体信息的DataFrame
    """
    # 过滤指定股票的数据
    stock_data = df[df['ts_code'] == ts_code].copy()
    
    if len(stock_data) == 0:
        print(f"未找到股票代码 {ts_code} 的数据")
        return None
    
    # 按日期排序
    stock_data = stock_data.sort_values('trade_date')
    
    # 设置分析周期
    if period == 'short':
        window = 20  # 短线箱体：20个交易日
    else:
        window = 120  # 中长线箱体：约6个月(120个交易日)
    
    # 初始化结果列表
    box_data = []
    
    # 滑动窗口分析
    for i in range(window, len(stock_data)):
        # 获取当前窗口的数据
        window_data = stock_data.iloc[i-window:i]
        current_date = stock_data.iloc[i-1]['trade_date']
        
        # 计算窗口内的最高价和最低价
        high_prices = window_data['high'].values
        low_prices = window_data['low'].values
        
        # 使用KDE(核密度估计)识别价格聚集区域
        # 对高价进行KDE分析
        high_kde = gaussian_kde(high_prices)
        high_x = np.linspace(min(high_prices), max(high_prices), 1000)
        high_density = high_kde(high_x)
        
        # 对低价进行KDE分析
        low_kde = gaussian_kde(low_prices)
        low_x = np.linspace(min(low_prices), max(low_prices), 1000)
        low_density = low_kde(low_x)
        
        # 找到密度最高的点作为潜在的阻力位和支撑位
        resistance_candidates = high_x[np.argsort(high_density)[-5:]]  # 取密度最高的5个点
        support_candidates = low_x[np.argsort(low_density)[-5:]]  # 取密度最高的5个点
        
        # 计算每个候选点被触及的次数
        def count_touches(prices, level, tolerance=0.01):
            # 计算价格在level附近(±tolerance%)的次数
            lower_bound = level * (1 - tolerance)
            upper_bound = level * (1 + tolerance)
            return sum((prices >= lower_bound) & (prices <= upper_bound))
        
        # 计算阻力位触及次数
        resistance_touches = [count_touches(high_prices, level) for level in resistance_candidates]
        # 计算支撑位触及次数
        support_touches = [count_touches(low_prices, level) for level in support_candidates]
        
        # 选择触及次数最多且达到最小触及次数要求的位置
        valid_resistances = [resistance_candidates[i] for i in range(len(resistance_candidates)) 
                            if resistance_touches[i] >= min_touches]
        valid_supports = [support_candidates[i] for i in range(len(support_candidates)) 
                         if support_touches[i] >= min_touches]
        
        # 如果没有有效的支撑位或阻力位，使用窗口内的最高价和最低价
        if not valid_resistances:
            resistance = max(high_prices)
        else:
            resistance = max(valid_resistances)  # 使用最高的有效阻力位
            
        if not valid_supports:
            support = min(low_prices)
        else:
            support = min(valid_supports)  # 使用最低的有效支撑位
        
        # 计算箱体高度占支撑位的百分比
        box_height_pct = (resistance - support) / support * 100
        
        # 验证箱体的有效性
        # 1. 成交量分析：箱体底部是否缩量
        bottom_volume = window_data[window_data['low'] <= support * 1.02]['vol'].mean()
        avg_volume = window_data['vol'].mean()
        volume_ratio = bottom_volume / avg_volume if avg_volume > 0 else 1
        
        # 2. 计算当前价格相对于箱体的位置
        current_price = stock_data.iloc[i-1]['close']
        relative_position = (current_price - support) / (resistance - support) if resistance > support else 0.5
        
        # 3. 检测是否突破箱体
        breakout_up = current_price > resistance * (1 + breakout_pct)
        breakout_down = current_price < support * (1 - breakout_pct)
        
        # 添加到结果列表
        box_data.append({
            'trade_date': current_date,
            'support': support,
            'resistance': resistance,
            'box_height_pct': box_height_pct,
            'volume_ratio': volume_ratio,
            'relative_position': relative_position,
            'breakout_up': breakout_up,
            'breakout_down': breakout_down,
            'resistance_touches': max(resistance_touches) if resistance_touches else 0,
            'support_touches': max(support_touches) if support_touches else 0
        })
    
    # 转换为DataFrame
    box_df = pd.DataFrame(box_data)
    
    return box_df

def plot_price_box(df, ts_code, start_date=None, end_date=None, period='short'):
    """
    绘制股票价格箱体图
    
    参数:
    df: DataFrame, 包含股票数据的DataFrame
    ts_code: str, 股票代码
    start_date: str, 开始日期，格式：'YYYYMMDD'
    end_date: str, 结束日期，格式：'YYYYMMDD'
    period: str, 'short'表示短线箱体，'long'表示中长线箱体
    """
    # 过滤指定股票的数据
    stock_data = df[df['ts_code'] == ts_code].copy()
    
    if len(stock_data) == 0:
        print(f"未找到股票代码 {ts_code} 的数据")
        return
    
    # 按日期排序
    stock_data = stock_data.sort_values('trade_date')
    
    # 日期过滤
    if start_date:
        stock_data = stock_data[stock_data['trade_date'] >= start_date]
    if end_date:
        stock_data = stock_data[stock_data['trade_date'] <= end_date]
    
    if len(stock_data) == 0:
        print(f"指定日期范围内没有数据")
        return
    
    # 计算箱体
    box_df = identify_price_box(df, ts_code, period=period)
    
    if box_df is None or len(box_df) == 0:
        print("无法计算箱体")
        return
    
    # 合并数据
    merged_data = pd.merge(stock_data, box_df, on='trade_date', how='left')
    merged_data = merged_data.fillna(method='ffill')  # 向前填充缺失值
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 获取数据并清理
    trade_date = merged_data['trade_date'].dropna()
    close = merged_data['close'].dropna()
    # 移除无穷大值
    # trade_date = trade_date[~np.isinf(trade_date)]
    # close = close[~np.isinf(close)]
    
    # 绘制K线图
    ax.plot(trade_date, close, label='收盘价', color='blue')
    
    # 绘制箱体
    for i in range(1, len(merged_data)):
        # 如果支撑位或阻力位发生变化，绘制新的箱体
        if (merged_data.iloc[i]['support'] != merged_data.iloc[i-1]['support'] or 
            merged_data.iloc[i]['resistance'] != merged_data.iloc[i-1]['resistance']):
            
            # 获取箱体的开始和结束日期
            start_idx = i-1
            end_idx = i
            while end_idx < len(merged_data)-1 and merged_data.iloc[end_idx+1]['support'] == merged_data.iloc[i]['support']:
                end_idx += 1
            
            # 绘制箱体
            date_range = merged_data.iloc[start_idx:end_idx+1]['trade_date']
            support_level = merged_data.iloc[i]['support']
            resistance_level = merged_data.iloc[i]['resistance']
            
            # 绘制支撑位和阻力位
            ax.fill_between(date_range, support_level, resistance_level, alpha=0.2, color='gray')
            ax.plot(date_range, [support_level] * len(date_range), 'g--', linewidth=1)
            ax.plot(date_range, [resistance_level] * len(date_range), 'r--', linewidth=1)
    
    # 标记突破点
    breakout_up = merged_data[merged_data['breakout_up'] == True]
    breakout_down = merged_data[merged_data['breakout_down'] == True]
    
    ax.scatter(breakout_up['trade_date'], breakout_up['close'], color='red', marker='^', s=100, label='向上突破')
    ax.scatter(breakout_down['trade_date'], breakout_down['close'], color='green', marker='v', s=100, label='向下突破')
    
    # 设置图表标题和标签
    period_text = "短线" if period == 'short' else "中长线"
    ax.set_title(f'{ts_code} {period_text}箱体分析', fontsize=15)
    ax.set_xlabel('日期')
    ax.set_ylabel('价格')
    ax.legend()
    
    # 旋转x轴日期标签
    plt.xticks(rotation=45)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    plt.show()


def advanced_price_box(df, ts_code, period='short', atr_multiplier=2, fib_levels=[0.382, 0.5, 0.618]):
    """
    使用进阶方法识别股票的价格箱体，包括ATR、斐波那契回撤和筹码分布
    
    参数:
    df: DataFrame, 包含股票数据的DataFrame
    ts_code: str, 股票代码
    period: str, 'short'表示短线箱体(20个交易日)，'long'表示中长线箱体(120个交易日)
    atr_multiplier: float, ATR乘数，用于设定箱体宽度
    fib_levels: list, 斐波那契回撤水平
    
    返回:
    DataFrame: 包含进阶箱体信息的DataFrame
    """
    # 过滤指定股票的数据
    stock_data = df[df['ts_code'] == ts_code].copy()
    
    if len(stock_data) == 0:
        print(f"未找到股票代码 {ts_code} 的数据")
        return None
    
    # 按日期排序
    stock_data = stock_data.sort_values('trade_date')
    
    # 设置分析周期
    if period == 'short':
        window = 20  # 短线箱体：20个交易日
    else:
        window = 120  # 中长线箱体：约6个月(120个交易日)
    
    # 计算ATR (平均真实波幅)
    def calculate_atr(data, period=14):
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # 计算真实波幅 (True Range)
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
        tr = np.insert(tr, 0, tr1[0])  # 插入第一个值
        
        # 计算ATR
        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    # 计算ATR
    stock_data['atr'] = calculate_atr(stock_data)
    
    # 初始化结果列表
    advanced_box_data = []
    
    # 滑动窗口分析
    for i in range(window, len(stock_data)):
        # 获取当前窗口的数据
        window_data = stock_data.iloc[i-window:i]
        current_date = stock_data.iloc[i-1]['trade_date']
        current_price = stock_data.iloc[i-1]['close']
        current_atr = stock_data.iloc[i-1]['atr']
        
        # 1. 基于ATR的箱体边界
        atr_resistance = current_price + atr_multiplier * current_atr
        atr_support = current_price - atr_multiplier * current_atr
        
        # 2. 斐波那契回撤水平
        # 找到窗口内的最高点和最低点
        window_high = window_data['high'].max()
        window_low = window_data['low'].min()
        price_range = window_high - window_low
        
        # 计算斐波那契回撤水平
        fib_retracements = {}
        if window_data['close'].iloc[-1] > window_data['close'].iloc[0]:  # 上升趋势
            for level in fib_levels:
                fib_retracements[level] = window_high - price_range * level
        else:  # 下降趋势
            for level in fib_levels:
                fib_retracements[level] = window_low + price_range * level
        
        # 3. 筹码分布分析
        # 使用成交量加权的价格分布来近似筹码分布
        price_bins = np.linspace(window_low * 0.95, window_high * 1.05, 50)
        volume_distribution = np.zeros_like(price_bins)
        
        for j in range(len(window_data)):
            # 找到价格所在的bin
            price = window_data['close'].iloc[j]
            vol = window_data['vol'].iloc[j]
            
            # 简单地将成交量分配到最接近的价格bin
            bin_idx = np.abs(price_bins - price).argmin()
            volume_distribution[bin_idx] += vol
        
        # 找到成交量最大的几个价格区间作为筹码密集区
        top_volume_indices = np.argsort(volume_distribution)[-5:]  # 取前5个成交量最大的价格区间
        chip_concentration_prices = price_bins[top_volume_indices]
        
        # 筹码密集区中低于当前价格的最高价作为支撑位
        chip_supports = chip_concentration_prices[chip_concentration_prices < current_price]
        chip_support = max(chip_supports) if len(chip_supports) > 0 else atr_support
        
        # 筹码密集区中高于当前价格的最低价作为阻力位
        chip_resistances = chip_concentration_prices[chip_concentration_prices > current_price]
        chip_resistance = min(chip_resistances) if len(chip_resistances) > 0 else atr_resistance
        
        # 综合三种方法确定最终的支撑位和阻力位
        # 1. 如果斐波那契回撤水平在ATR范围内，优先使用斐波那契水平
        fib_support = max([level for level in fib_retracements.values() if level < current_price], default=atr_support)
        fib_resistance = min([level for level in fib_retracements.values() if level > current_price], default=atr_resistance)
        
        # 2. 如果筹码密集区在斐波那契水平附近(±5%)，则使用筹码密集区
        if 0.95 * fib_support <= chip_support <= 1.05 * fib_support:
            final_support = chip_support
        else:
            final_support = fib_support
            
        if 0.95 * fib_resistance <= chip_resistance <= 1.05 * fib_resistance:
            final_resistance = chip_resistance
        else:
            final_resistance = fib_resistance
        
        # 3. 确保支撑位不高于当前价格，阻力位不低于当前价格
        final_support = min(final_support, current_price * 0.99)
        final_resistance = max(final_resistance, current_price * 1.01)
        
        # 计算箱体高度占支撑位的百分比
        box_height_pct = (final_resistance - final_support) / final_support * 100
        
        # 计算当前价格相对于箱体的位置
        relative_position = (current_price - final_support) / (final_resistance - final_support) if final_resistance > final_support else 0.5
        
        # 检测是否突破箱体
        breakout_pct = 0.03  # 3%的突破阈值
        breakout_up = current_price > final_resistance * (1 + breakout_pct)
        breakout_down = current_price < final_support * (1 - breakout_pct)
        
        # 添加到结果列表
        advanced_box_data.append({
            'trade_date': current_date,
            'close': current_price,
            'atr': current_atr,
            'atr_support': atr_support,
            'atr_resistance': atr_resistance,
            'fib_support': fib_support,
            'fib_resistance': fib_resistance,
            'chip_support': chip_support,
            'chip_resistance': chip_resistance,
            'final_support': final_support,
            'final_resistance': final_resistance,
            'box_height_pct': box_height_pct,
            'relative_position': relative_position,
            'breakout_up': breakout_up,
            'breakout_down': breakout_down
        })
    
    # 转换为DataFrame
    advanced_box_df = pd.DataFrame(advanced_box_data)
    
    return advanced_box_df

def plot_advanced_price_box(df, ts_code, start_date=None, end_date=None, period='short'):
    """
    绘制进阶股票价格箱体图，包括ATR、斐波那契回撤和筹码分布
    
    参数:
    df: DataFrame, 包含股票数据的DataFrame
    ts_code: str, 股票代码
    start_date: str, 开始日期，格式：'YYYYMMDD'
    end_date: str, 结束日期，格式：'YYYYMMDD'
    period: str, 'short'表示短线箱体，'long'表示中长线箱体
    """
    # 过滤指定股票的数据
    stock_data = df[df['ts_code'] == ts_code].copy()
    
    if len(stock_data) == 0:
        print(f"未找到股票代码 {ts_code} 的数据")
        return
    
    # 按日期排序
    stock_data = stock_data.sort_values('trade_date')
    
    # 日期过滤
    if start_date:
        stock_data = stock_data[stock_data['trade_date'] >= start_date]
    if end_date:
        stock_data = stock_data[stock_data['trade_date'] <= end_date]
    
    if len(stock_data) == 0:
        print(f"指定日期范围内没有数据")
        return
    
    # 计算进阶箱体
    advanced_box_df = advanced_price_box(df, ts_code, period=period)
    
    if advanced_box_df is None or len(advanced_box_df) == 0:
        print("无法计算进阶箱体")
        return
    
    # 合并数据
    merged_data = pd.merge(stock_data, advanced_box_df, on='trade_date', how='left')
    merged_data = merged_data.fillna(method='ffill')  # 向前填充缺失值
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # 绘制K线图
    ax.plot(merged_data['trade_date'], merged_data['close_x'], label='收盘价', color='blue')
    
    # 绘制箱体
    for i in range(1, len(merged_data)):
        # 如果支撑位或阻力位发生变化，绘制新的箱体
        if (merged_data.iloc[i]['final_support'] != merged_data.iloc[i-1]['final_support'] or 
            merged_data.iloc[i]['final_resistance'] != merged_data.iloc[i-1]['final_resistance']):
            
            # 获取箱体的开始和结束日期
            start_idx = i-1
            end_idx = i
            while end_idx < len(merged_data)-1 and merged_data.iloc[end_idx+1]['final_support'] == merged_data.iloc[i]['final_support']:
                end_idx += 1
            
            # 绘制箱体
            date_range = merged_data.iloc[start_idx:end_idx+1]['trade_date']
            support_level = merged_data.iloc[i]['final_support']
            resistance_level = merged_data.iloc[i]['final_resistance']
            
            # 绘制支撑位和阻力位
            ax.fill_between(date_range, support_level, resistance_level, alpha=0.2, color='gray')
            ax.plot(date_range, [support_level] * len(date_range), 'g--', linewidth=1, label='支撑位' if i==1 else "")
            ax.plot(date_range, [resistance_level] * len(date_range), 'r--', linewidth=1, label='阻力位' if i==1 else "")
            
            # 绘制ATR边界（虚线）
            ax.plot(date_range, [merged_data.iloc[i]['atr_support']] * len(date_range), 'g:', linewidth=0.5, alpha=0.5, label='ATR支撑位' if i==1 else "")
            ax.plot(date_range, [merged_data.iloc[i]['atr_resistance']] * len(date_range), 'r:', linewidth=0.5, alpha=0.5, label='ATR阻力位' if i==1 else "")
            
            # 绘制斐波那契水平（点线）
            ax.plot(date_range, [merged_data.iloc[i]['fib_support']] * len(date_range), 'g-.', linewidth=0.5, alpha=0.5, label='斐波那契支撑位' if i==1 else "")
            ax.plot(date_range, [merged_data.iloc[i]['fib_resistance']] * len(date_range), 'r-.', linewidth=0.5, alpha=0.5, label='斐波那契阻力位' if i==1 else "")
    
    # 标记突破点
    breakout_up = merged_data[merged_data['breakout_up'] == True]
    breakout_down = merged_data[merged_data['breakout_down'] == True]
    
    ax.scatter(breakout_up['trade_date'], breakout_up['close_x'], color='red', marker='^', s=100, label='向上突破')
    ax.scatter(breakout_down['trade_date'], breakout_down['close_x'], color='green', marker='v', s=100, label='向下突破')
    
    # 设置图表标题和标签
    period_text = "短线" if period == 'short' else "中长线"
    ax.set_title(f'{ts_code} {period_text}进阶箱体分析', fontsize=15)
    ax.set_xlabel('日期')
    ax.set_ylabel('价格')
    
    # 创建自定义图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    
    # 旋转x轴日期标签
    plt.xticks(rotation=45)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    plt.show()
    
    # 绘制筹码分布热图
    plt.figure(figsize=(15, 4))
    
    # 创建价格区间
    price_min = merged_data['low_x'].min() * 0.95
    price_max = merged_data['high_x'].max() * 1.05
    price_bins = np.linspace(price_min, price_max, 100)
    
    # 创建日期网格
    dates = merged_data['trade_date'].unique()
    
    # 初始化热图数据
    heatmap_data = np.zeros((len(dates), len(price_bins)-1))
    
    # 填充热图数据
    for i, date in enumerate(dates):
        day_data = merged_data[merged_data['trade_date'] == date]
        if len(day_data) > 0:
            price = day_data['close_x'].values[0]
            volume = day_data['vol'].values[0]
            
            # 使用正态分布模拟筹码分布
            sigma = day_data['atr'].values[0] / 2  # 使用ATR/2作为分布宽度
            for j in range(len(price_bins)-1):
                bin_center = (price_bins[j] + price_bins[j+1]) / 2
                # 计算在该价格区间的筹码密度
                density = np.exp(-((bin_center - price) ** 2) / (2 * sigma ** 2))
                heatmap_data[i, j] = density * volume
    
    # 绘制热图
    plt.imshow(heatmap_data, aspect='auto', cmap='hot_r', 
               extent=[price_min, price_max, len(dates)-1, 0],
               interpolation='nearest')
    
    # 添加颜色条
    cbar = plt.colorbar(label='筹码密度')
    
    # 设置y轴刻度为日期
    plt.yticks(np.arange(0, len(dates), max(1, len(dates)//10)), 
               [dates[i] for i in range(0, len(dates), max(1, len(dates)//10))],
               rotation=45)
    
    # 添加当前价格线
    last_price = merged_data['close_x'].iloc[-1]
    plt.axvline(x=last_price, color='blue', linestyle='-', linewidth=1, label='当前价格')
    
    # 添加支撑位和阻力位
    last_support = merged_data['final_support'].iloc[-1]
    last_resistance = merged_data['final_resistance'].iloc[-1]
    plt.axvline(x=last_support, color='green', linestyle='--', linewidth=1, label='支撑位')
    plt.axvline(x=last_resistance, color='red', linestyle='--', linewidth=1, label='阻力位')
    
    plt.title(f'{ts_code} 筹码分布热图')
    plt.xlabel('价格')
    plt.ylabel('日期')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print()
    print('加载数据')
    stock_info = load_stock_data(table_name='stock_info_daily', file_path='C:\\Users\\huang\\Downloads\\stock.nfa')
    print('添加技术指标')
    stock_info = add_technical_indicators(stock_info)
    print('设置标签')
    stock_info = label_price_changes(stock_info, n=7)
    # 测试选股函数
    # print('选股测试')
    # # # latest_date = stock_info['trade_date'].max()
    # stock_info = select_stocks_by_indicators(stock_info)
    print('分析标签分布')
    stats = analyze_label_distribution(stock_info)
    print(stats)
    # 将数值列中的空值填充为0
    # selected = selected.fillna(0)
    identify_price_box(stock_info, '000001.SZ')
    print('完成')
    # stock_info.to_csv('selected_stocks.csv', index=False)
    
    
    # print(f'最新交易日期: {latest_date}')
    # print(f'符合条件的股票数量: {len(selected)}')
    # if len(selected) > 0:
    #     print('\n符合条件的股票:')
    #     print(selected[['ts_code', 'trade_date', 'close', 'rsi6', 'rsi12', 'rsi24', 'macdhist']].head())
   
    # stock_info2 = stock_info2.drop(['open', 'close', 'low', 'high', 'ma7', 'ma14', 'ma30', 'macd', 'macdsignal', 'macdhist', 'midpoint', 'midprice', 'vol_ma3', 'vol_ma7', 'vol_ma14', 'vol_ma20', 'vol_std_20', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'], axis=1) 
    # condition=(stock_info['label']==2) & (~stock_info['ts_code'].str.endswith('.BJ'))
    
    # 测试直方图绘制函数
    # 绘制RSI6的分布
    # plot_histogram(stock_info, 'pct_change_24d', bins=100, title='RSI6分布图')
    # # 绘制MACD柱状值的分布
    # plot_histogram(stock_info, 'macdhist', bins=100, title='MACD柱状值分布图')
    
    # # 分析特征
    # features_to_analyze = ['rsi6', 'rsi24', 'macdhist', 'pct_change_24d']
    # distinctions = {}
    # for feature in features_to_analyze:
    #     print(f"\n分析特征: {feature}")
    #     distinction = analyze_feature(stock_info, feature)
    #     distinctions[feature] = distinction
    
    # # 打印特征区分度排名
    # print("\n特征区分度排名:")
    # for feature, dist in sorted(distinctions.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{feature}: {dist:.3f}")
    
    
    
