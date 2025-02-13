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


def label_price_changes(df, n=5):
    """
    标注股票价格变化的买入卖出点:
    - 后5天内涨幅>5%标记为买入点(1)
    - 后5天内跌幅>5%标记为卖出点(-1) 
    - 后5天内任意一天pre_close与前一日close不一致时标记为(-2)
    - 其他标记为非买入点(0)
    
    参数:
    df: DataFrame, 包含股票数据的DataFrame,需要包含['ts_code', 'trade_date', 'close', 'pre_close']列
    
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
        
        # 检查pre_close是否与前一日close一致
        close_shifted = group['close'].shift(1)
        pre_close_match = (group['pre_close'] == close_shifted) | close_shifted.isna()
        
        # 创建标签
        labels = pd.Series(0, index=group.index)
        
        # 标记数据不连续点
        labels = np.where(~pre_close_match, -2, labels)
        
        # 标记买入卖出点
        labels = np.where((labels == 0) & (max_changes > 0.05), 1, labels)
        labels = np.where((labels == 0) & (min_changes < -0.05), -1, labels)
        
        # 最后5天标记为0
        labels[-5:] = 0
        
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
    sql = f'SELECT * FROM {table_name} ORDER BY ts_code, trade_date asc'
    stock_info = pd.read_sql_query(sql, conn)
    conn.close()
    return stock_info



def save_stock_data(stock_info, table_name, file_path='./data/train.nfa', mode='append'):
    """
    将股票数据保存到SQLite数据库中。
    此函数将stock_info数据切分为20份，分批插入到指定表中。
    
    参数:
    stock_info: DataFrame, 要保存的股票数据
    table_name: str, 要保存的表名
    file_path: str, SQLite数据库文件路径
    mode: str, 第一个批次的插入模式（例如 'replace' 或 'append'）
    """
    import math
    conn = sqlite3.connect(file_path)
    num_records = len(stock_info)
    num_batches = 20
    batch_size = math.ceil(num_records / num_batches)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = stock_info.iloc[start_idx:end_idx]
        if not batch.empty:
            current_mode = mode if i == 0 else 'append'
            pd.io.sql.to_sql(batch, name=table_name, con=conn, if_exists=current_mode, index=False)
    conn.close()


def detect_ma_crossovers(group, ma_ratio):
    """
    检测均线穿越1的反转点并计算距离上一个反转点的天数
    
    参数:
    group: DataFrame, 包含股票数据的一个分组
    ma_ratio: Series, 均线比率序列(如ratio_ma7_close)
    
    返回:
    reversal_points: Series, 反转点标记（1表示向上穿越，-1表示向下穿越，0表示非反转点）
    days_since_last_reversal: Series, 距离上一个反转点的天数
    """
    # 初始化反转点序列
    reversal_points = pd.Series(0, index=group.index)
    
    # 获取前一日的ratio值
    prev_ratio = ma_ratio.shift(1)
    
    # 检测向上穿越1的点和向下穿越1的点
    up_cross = (prev_ratio < 1) & (ma_ratio >= 1)
    down_cross = (prev_ratio > 1) & (ma_ratio <= 1)
    
    # 分别标记向上和向下穿越点
    reversal_points[up_cross] = 1
    reversal_points[down_cross] = -1
    
    # 使用abs(reversal_points)来创建分组标识，这样向上和向下穿越点都会开始新的分组
    group_id = (abs(reversal_points) > 0).cumsum()
    
    # 计算距离天数（向量化操作）
    # 1. 创建一个布尔掩码，标识非零的group_id
    valid_groups = group_id > 0
    
    # 2. 为每个有效组创建一个累计计数
    days_since_last_reversal = pd.Series(0, index=group.index)
    if valid_groups.any():
        # 对每个组内进行累计计数
        days_since_last_reversal[valid_groups] = (
            group_id[valid_groups]
            .map(group_id[valid_groups].groupby(group_id[valid_groups]).cumcount())
        )
    
    return reversal_points, days_since_last_reversal


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
        
        # ===== 布林带指标 =====
        upper, middle, lower = talib.BBANDS(group['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        # ===== MACD指标 =====
        macd, macdsignal, macdhist = talib.MACD(group['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        
        # ===== 中点和中间价格 =====
        midpoint = talib.MIDPOINT(group['close'], timeperiod=14)
        midprice = talib.MIDPRICE(group['high'], group['low'], timeperiod=14)
        
        # ===== 动量指标 =====
        adx = talib.ADX(group['high'], group['low'], group['close'], timeperiod=14)
        adxr = talib.ADXR(group['high'], group['low'], group['close'], timeperiod=14)
        
        # ===== 成交量特征 =====
        ma_periods = [3, 7, 14, 20]
        vol_features = {}
        for period in ma_periods:
            vol_ma = talib.MA(group['vol'], timeperiod=period)
            vol_features[f'vol_ma{period}'] = vol_ma
            vol_features[f'vol_ma{period}_ratio'] = group['vol'] / vol_ma
        
        # 成交量趋势特征
        vol_features['vol_change'] = group['vol'].pct_change()
        vol_features['vol_acc'] = vol_features['vol_change'].pct_change()
        
        # 成交量波动性特征
        vol_features['vol_std_20'] = talib.STDDEV(group['vol'], timeperiod=20)
        vol_features['vol_std_ratio_20'] = vol_features['vol_std_20'] / vol_features['vol_ma20']
        
        # 价格成交量相关性
        price_change = group['close'].pct_change()
        vol_features['price_vol_corr'] = talib.CORREL(price_change, vol_features['vol_change'], timeperiod=20)
        
        # ===== 相对特征 =====
        # 计算斜率
        close_slope = talib.LINEARREG_SLOPE(group['close'], timeperiod=7)
        ma7_slope = talib.LINEARREG_SLOPE(ma7, timeperiod=7)
        
        # 均线比率
        ratio_features = {
            'ratio_ma7_close': ma7 / group['close'],
            'ratio_ma14_close': ma14 / group['close'],
            'ratio_ma30_close': ma30 / group['close'],
            'ratio_ma7_ma14': ma7 / ma14,
            'ratio_ma7_ma30': ma7 / ma30,
            'ratio_ma14_ma30': ma14 / ma30
        }
        
        # 计算反转点和距离天数
        reversal_features = {}
        for period, ratio in [('7', ratio_features['ratio_ma7_close']), 
                            ('14', ratio_features['ratio_ma14_close']), 
                            ('30', ratio_features['ratio_ma30_close'])]:
            rev_points, days_since = detect_ma_crossovers(group, ratio)
            reversal_features[f'ma{period}_reversal'] = rev_points
            reversal_features[f'ma{period}_days_since_reversal'] = days_since
        
        # 布林带相对特征
        bb_features = {
            'ratio_upper_close': upper / group['close'],
            'ratio_middle_close': middle / group['close'],
            'ratio_lower_close': lower / group['close'],
            'bb_width': (upper - lower) / middle,
            'bb_position': (group['close'] - lower) / (upper - lower)
        }
        
        # 合并所有特征
        result_df = pd.DataFrame({
            # 基础指标
            'macd': macd, 'macdsignal': macdsignal, 'macdhist': macdhist,
            'ma7': ma7, 'ma14': ma14, 'ma30': ma30,
            'midpoint': midpoint, 'midprice': midprice,
            'adx': adx, 'adxr': adxr,
            'close_slope': close_slope,
            'ma7_slope': ma7_slope,
            **vol_features,
            **ratio_features,
            **reversal_features,
            **bb_features
        })
        
        return result_df
        
    except Exception as e:
        print(f"Error calculating technical indicators: {str(e)}")
        # 返回包含NaN的DataFrame，列名与正常计算结果相同
        columns = ['macd', 'macdsignal', 'macdhist', 'ma7', 'ma14', 'ma30',
                  'midpoint', 'midprice', 'adx', 'adxr', 'close_slope', 'ma7_slope']
        columns.extend([f'vol_ma{p}' for p in ma_periods])
        columns.extend([f'vol_ma{p}_ratio' for p in ma_periods])
        columns.extend(['vol_change', 'vol_acc', 'vol_std_20', 'vol_std_ratio_20', 'price_vol_corr'])
        columns.extend(ratio_features.keys())
        columns.extend(reversal_features.keys())
        columns.extend(bb_features.keys())
        
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


def plot_random_positive_samples(stock_info, n_samples=100, n_before=60, n_after=30):
    """
    随机抽取n_samples个label为1的样本并绘制K线图
    
    参数:
    stock_info: DataFrame, 包含股票数据
    n_samples: int, 要抽取的样本数量
    n_before: int, 向前查看的交易日数量
    n_after: int, 向后查看的交易日数量
    """
    # 获取label为1的样本
    positive_samples = stock_info[stock_info['label'] == 1]
    
    # 随机抽取n_samples个样本
    if len(positive_samples) > n_samples:
        samples = positive_samples.sample(n=n_samples, random_state=42)
    else:
        samples = positive_samples
        print(f"Warning: Only {len(positive_samples)} positive samples available")
    
    # 遍历绘制每个样本的K线图
    for idx, row in enumerate(samples.itertuples(), 1):
        print(f"Processing sample {idx}/{len(samples)}: {row.ts_code} - {row.trade_date}")
        try:
            plot_candlestick(stock_info, row.ts_code, row.trade_date, 
                           n_before=n_before, n_after=n_after)
            plt.close()  # 关闭图形，释放内存
        except Exception as e:
            print(f"Error plotting {row.ts_code} - {row.trade_date}: {str(e)}")
            continue


def get_latest_records(stock_info, n=5):
    """
    获取每个股票最后第n天的记录
    
    参数:
    stock_info: DataFrame, 包含股票数据的DataFrame
    n: int, 获取最后第n天的记录，默认为5天
    
    返回:
    DataFrame: 每个股票最后第n天的记录
    """
    # 按股票代码分组，获取每组最后第n条记录
    latest_records = stock_info.sort_values('trade_date').groupby('ts_code').nth(-n)
    
    # 按交易日期降序排序
    latest_records = latest_records.sort_values(['ts_code', 'trade_date'], ascending=[True, False])
    
    print(f"\n共有 {len(latest_records)} 只股票的最后第{n}天记录")
    print(f"最新交易日期: {latest_records['trade_date'].max()}")
    print(f"最早交易日期: {latest_records['trade_date'].min()}")
    
    return latest_records


def add_volume_features(df):
    """
    构建全面的成交量特征
    
    参数:
    df: DataFrame, 包含基础数据的DataFrame，需要包含['vol', 'amount', 'close']等列
    
    返回:
    DataFrame: 添加成交量特征后的DataFrame
    """
    def process_group(group):
        # 1. 基础成交量移动平均，使用talib中ma接口计算移动平均
        ma_periods = [3, 7, 14, 30]
        for period in ma_periods:
            group[f'vol_ma{period}'] = talib.MA(group['vol'], timeperiod=period)
            # 相对于移动平均的成交量比
            group[f'vol_ma{period}_ratio'] = group['vol'] / group[f'vol_ma{period}']
        
        # 2. 成交量趋势特征
        # 计算成交量变化率
        group['vol_change'] = group['vol'].pct_change()
        # 成交量加速度（二阶变化）
        group['vol_acc'] = group['vol_change'].pct_change()
        
        # 3. 成交量波动性特征
        # 使用talib的STDDEV接口计算成交量标准差
        group['vol_std_20'] = talib.STDDEV(group['vol'], timeperiod=20)
        # 使用talib计算的成交量标准差除以MA20得到成交量波动率
        group['vol_std_ratio_20'] = group['vol_std_20'] / group['vol_ma20']
        
        # 4. 价格成交量相关性特征
        # 计算价格变化
        group['price_change'] = group['close'].pct_change()
        # 价格成交量相关系数，使用talib中的皮尔逊相关系数计算
        group['price_vol_corr'] = talib.CORREL(group['price_change'], group['vol_change'], timeperiod=20)
        
        # 5. 成交量分布特征
        # 计算相对于过去N日的成交量分位数
        for period in [10, 20, 60]:
            group[f'vol_quantile_{period}'] = group['vol'].rolling(window=period).apply(
                lambda x: pd.Series(x).rank().iloc[-1] / len(x)
            )
        
        # 6. 成交量累积特征
        # 计算N日累积成交量
        for period in [3, 5, 10]:
            group[f'vol_sum_{period}'] = group['vol'].rolling(window=period).sum()
            # 当日成交量占累积成交量的比例
            group[f'vol_sum_{period}_ratio'] = group['vol'] / group[f'vol_sum_{period}']
        
        # 7. 成交金额特征
        # 计算平均成交价
        group['avg_price'] = group['amount'] / group['vol']
        # 计算成交均价与收盘价的差异
        group['avg_price_diff'] = (group['avg_price'] - group['close']) / group['close']
        
        # 8. 大单特征（如果有tick数据）
        if 'amount' in group.columns:
            # 计算单笔平均成交金额
            group['avg_trade_amount'] = group['amount'] / group['vol']
            # 相对于历史的单笔成交金额水平
            group['avg_trade_amount_ma20'] = group['avg_trade_amount'].rolling(20).mean()
            group['avg_trade_amount_ratio'] = group['avg_trade_amount'] / group['avg_trade_amount_ma20']
        
        # 9. 成交量支撑/压力特征
        # 计算高成交量位置的价格水平
        def get_volume_price_levels(x):
            if len(x) < 20:
                return np.nan
            # 获取成交量前20%的价格水平
            threshold = np.percentile(x['vol'], 80)
            high_vol_prices = x[x['vol'] >= threshold]['close']
            return high_vol_prices.mean()
        
        group['vol_price_level'] = group.rolling(20).apply(get_volume_price_levels)
        group['price_to_vol_level'] = group['close'] / group['vol_price_level']
        
        # 10. 组合指标
        # 价量背离指标
        group['price_vol_divergence'] = (
            (group['close'] > group['close'].shift(1)) & 
            (group['vol'] < group['vol'].shift(1))
        ).astype(int)
        
        # 成交量冲击指标
        group['vol_impact'] = (group['high'] - group['low']) * group['vol'] / group['amount']
        
        # 11. 成交量均线多空指标
        # 短期均线与长期均线的比较
        group['vol_ma_trend'] = group['vol_ma5'] / group['vol_ma20']
        
        # 12. 成交量异常检测
        # 计算成交量Z-score
        group['vol_zscore'] = (group['vol'] - group['vol_ma20']) / group['vol_std_20']
        # 标记异常成交量
        group['vol_spike'] = (abs(group['vol_zscore']) > 2).astype(int)
        
        return group
    
    # 按股票代码分组处理
    df = df.groupby('ts_code', group_keys=False).apply(process_group)
    
    # 处理无穷值和空值
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 打印特征统计信息
    vol_features = [col for col in df.columns if 'vol' in col]
    print("\n成交量特征统计:")
    print(f"总计添加成交量特征数: {len(vol_features)}")
    
    # 检查异常值
    for feature in vol_features:
        outliers = df[abs(df[feature]) > 10].shape[0]
        if outliers > 0:
            print(f"警告: {feature} 中有 {outliers} 条异常值(>10)")
    
    return df


def normalize_stock_prices(df, price_cols=['open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']):
    """
    对股票价格数据进行归一化处理：
    1. 检测并处理股票拆分
    2. 对每个拆分段进行归一化
    
    参数:
    df: DataFrame, 包含股票数据的DataFrame
    price_cols: list, 需要归一化的价格列名列表
    
    返回:
    DataFrame: 包含归一化后价格的DataFrame，新增ts_code_split列标识不同的拆分段
    """
    def process_group(group):
        # 检查pre_close是否与前一日close一致
        close_shifted = group['close'].shift(1)
        pre_close_match = (group['pre_close'] == close_shifted) | close_shifted.isna()
        
        # 标记不同的拆分段
        split_group = (~pre_close_match).cumsum()
        group['ts_code_split'] = group['ts_code'] + '_' + split_group.astype(str)
        
        # 对每个拆分段使用sklearn的MinMaxScaler进行min max归一化
        for split_id in split_group.unique():
            split_mask = split_group == split_id
            if split_mask.sum() > 0:  # 确保拆分段有数据
                scaler = MinMaxScaler()
                data = group.loc[split_mask, price_cols]
                scaled_data = scaler.fit_transform(data)
                for idx, col in enumerate(price_cols):
                    group.loc[split_mask, f'{col}_norm'] = scaled_data[:, idx]
        return group
    
    # 创建副本避免修改原始数据
    # df_normalized = df.copy()
    df_normalized = df
    
    # 确保数据按股票代码和交易日期排序
    df_normalized = df_normalized.sort_values(['ts_code', 'trade_date'])
    
    # 按股票代码分组处理
    df_normalized = df_normalized.groupby('ts_code', group_keys=False).apply(process_group)
    
    # 打印统计信息
    split_counts = df_normalized.groupby('ts_code')['ts_code_split'].nunique()
    total_splits = split_counts.sum()
    stocks_with_splits = (split_counts > 1).sum()
    
    print("\n归一化处理统计:")
    print(f"总股票数: {len(split_counts)}")
    print(f"发生拆分的股票数: {stocks_with_splits}")
    print(f"总拆分段数: {total_splits}")
    print(f"平均每只股票拆分段数: {total_splits/len(split_counts):.2f}")
    
    # 检查异常值
    for col in price_cols:
        norm_col = f'{col}_norm'
        if norm_col in df_normalized.columns:
            abnormal = df_normalized[df_normalized[norm_col] > 10].shape[0]
            if abnormal > 0:
                print(f"\n警告: {norm_col} 中有 {abnormal} 条归一化后值大于10的记录")
    
    # 新增逻辑: 删除原始价格列，并将归一化后的列名恢复为原始名称
    price_cols.append('ts_code')
    df_normalized.drop(columns=price_cols, inplace=True)
    rename_dict = {f"{col}_norm": col for col in price_cols if f"{col}_norm" in df_normalized.columns}
    rename_dict['ts_code_split'] = 'ts_code'
    df_normalized.rename(columns=rename_dict, inplace=True)
    
    # 计算pct_chg：前一日的close减去当日的close；如果没有前一日数据则设为0
    df_normalized['pct_chg'] = df_normalized.groupby('ts_code')['close'].shift(1) - df_normalized['close']
    df_normalized['pct_chg'].fillna(0, inplace=True)
    
    return df_normalized


if __name__ == '__main__':
    
