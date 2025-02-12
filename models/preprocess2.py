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


def calculate_technical_indicators(group):
    """
    天数间隔：[3, 5, 10, 20, 30, 60]
    """
    try:
        # 成交量指标
        vol_ma = talib.EMA(group['vol'], timeperiod=7)
        
        # ===== 重叠研究指标 =====
        # 布林带
        upper, middle, lower = talib.BBANDS(group['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        # 双指数移动平均线
        dema = talib.DEMA(group['close'], timeperiod=30)
        # 指数移动平均线
        ema = talib.EMA(group['close'], timeperiod=30)
        # 希尔伯特变换瞬时趋势
        trendline = talib.HT_TRENDLINE(group['close'])
        # 卡玛考夫曼自适应移动平均
        kama = talib.KAMA(group['close'], timeperiod=30)
        # 移动平均线
        ma7 = talib.MA(group['close'], timeperiod=7)
        ma13 = talib.MA(group['close'], timeperiod=13)
        ma26 = talib.MA(group['close'], timeperiod=26)
        # MESA自适应移动平均
        mama, fama = talib.MAMA(group['close'])
        # 中点价格
        midpoint = talib.MIDPOINT(group['close'], timeperiod=14)
        # 中间价格
        midprice = talib.MIDPRICE(group['high'], group['low'], timeperiod=14)
        # 抛物线转向指标
        sar = talib.SAR(group['high'], group['low'])
        # 抛物线转向指标扩展
        sarext = talib.SAREXT(group['high'], group['low'])
        # 简单移动平均线
        sma = talib.SMA(group['close'], timeperiod=30)
        # 三重指数移动平均线
        t3 = talib.T3(group['close'], timeperiod=5)
        # 三重指数移动平均
        tema = talib.TEMA(group['close'], timeperiod=30)
        # 三角移动平均
        trima = talib.TRIMA(group['close'], timeperiod=30)
        # 加权移动平均
        wma = talib.WMA(group['close'], timeperiod=30)
        
        # ===== 动量指标 =====
        adx = talib.ADX(group['high'], group['low'], group['close'], timeperiod=14)
        adxr = talib.ADXR(group['high'], group['low'], group['close'], timeperiod=14)
        apo = talib.APO(group['close'], fastperiod=12, slowperiod=26, matype=0)
        aroondown, aroonup = talib.AROON(group['high'], group['low'], timeperiod=14)
        aroonosc = talib.AROONOSC(group['high'], group['low'], timeperiod=14)
        bop = talib.BOP(group['open'], group['high'], group['low'], group['close'])
        cci = talib.CCI(group['high'], group['low'], group['close'], timeperiod=14)
        cmo = talib.CMO(group['close'], timeperiod=14)
        dx = talib.DX(group['high'], group['low'], group['close'], timeperiod=14)
        macd, macdsignal, macdhist = talib.MACD(group['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        macdext, macdextsignal, macdexthist = talib.MACDEXT(group['close'], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
        macdfix, macdfixsignal, macdfixhist = talib.MACDFIX(group['close'], signalperiod=9)
        mfi = talib.MFI(group['high'], group['low'], group['close'], group['vol'], timeperiod=14)
        minus_di = talib.MINUS_DI(group['high'], group['low'], group['close'], timeperiod=14)
        minus_dm = talib.MINUS_DM(group['high'], group['low'], timeperiod=14)
        mom = talib.MOM(group['close'], timeperiod=10)
        plus_di = talib.PLUS_DI(group['high'], group['low'], group['close'], timeperiod=14)
        plus_dm = talib.PLUS_DM(group['high'], group['low'], timeperiod=14)
        ppo = talib.PPO(group['close'], fastperiod=12, slowperiod=26, matype=0)
        roc = talib.ROC(group['close'], timeperiod=10)
        rocp = talib.ROCP(group['close'], timeperiod=10)
        rocr = talib.ROCR(group['close'], timeperiod=10)
        rocr100 = talib.ROCR100(group['close'], timeperiod=10)
        rsi = talib.RSI(group['close'], timeperiod=14)
        slowk, slowd = talib.STOCH(group['high'], group['low'], group['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        fastk, fastd = talib.STOCHF(group['high'], group['low'], group['close'], fastk_period=5, fastd_period=3, fastd_matype=0)
        fastk_rsi, fastd_rsi = talib.STOCHRSI(group['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        trix = talib.TRIX(group['close'], timeperiod=30)
        ultosc = talib.ULTOSC(group['high'], group['low'], group['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
        willr = talib.WILLR(group['high'], group['low'], group['close'], timeperiod=14)
        
        # 波动率指标
        atr = talib.ATR(group['high'], group['low'], group['close'], timeperiod=14)
        natr = talib.NATR(group['high'], group['low'], group['close'], timeperiod=14)
        
        # ===== 成交量指标 =====
        # Chaikin A/D Line - 累积派发线
        ad = talib.AD(group['high'], group['low'], group['close'], group['vol'])
        
        # Chaikin A/D Oscillator - 佳庆振荡器
        adosc = talib.ADOSC(group['high'], group['low'], group['close'], group['vol'], 
                           fastperiod=3, slowperiod=10)
        
        # On Balance Volume - 能量潮
        obv = talib.OBV(group['close'], group['vol'])
        
        # ===== 周期指标 =====
        # 希尔伯特变换 - 主导周期
        dcperiod = talib.HT_DCPERIOD(group['close'])
        
        # 希尔伯特变换 - 主导周期相位
        dcphase = talib.HT_DCPHASE(group['close'])
        
        # 希尔伯特变换 - 相量分量
        inphase, quadrature = talib.HT_PHASOR(group['close'])
        
        # 希尔伯特变换 - 正弦波
        sine, leadsine = talib.HT_SINE(group['close'])
        
        # 希尔伯特变换 - 趋势/周期模式
        trendmode = talib.HT_TRENDMODE(group['close'])
        
        return pd.DataFrame({
            'macd': macd, 'macdsignal': macdsignal, 'macdhist': macdhist,
            'vol_ma': vol_ma, 'upperband': upper, 'middleband': middle, 
            'lowerband': lower, 'dema': dema, 'ema': ema, 'trendline': trendline,
            'kama': kama, 'ma7': ma7, 'ma13': ma13, 'ma26': ma26,
            'mama': mama, 'fama': fama, 'midpoint': midpoint, 'midprice': midprice,
            'sar': sar, 'sarext': sarext, 'sma': sma, 't3': t3, 'tema': tema,
            'trima': trima, 'wma': wma,
            # 动量指标
            'adx': adx, 'adxr': adxr, 'apo': apo, 'aroondown': aroondown,
            'aroonup': aroonup, 'aroonosc': aroonosc, 'bop': bop, 'cci': cci,
            'cmo': cmo, 'dx': dx, 'macdext': macdext, 'macdextsignal': macdextsignal,
            'macdexthist': macdexthist, 'macdfix': macdfix, 'macdfixsignal': macdfixsignal,
            'macdfixhist': macdfixhist, 'mfi': mfi, 'minus_di': minus_di,
            'minus_dm': minus_dm, 'mom': mom, 'plus_di': plus_di, 'plus_dm': plus_dm,
            'ppo': ppo, 'roc': roc, 'rocp': rocp, 'rocr': rocr, 'rocr100': rocr100,
            'rsi': rsi, 'slowk': slowk, 'slowd': slowd, 'fastk': fastk, 'fastd': fastd,
            'fastk_rsi': fastk_rsi, 'fastd_rsi': fastd_rsi, 'trix': trix,
            'ultosc': ultosc, 'willr': willr,
            # 波动率指标
            'atr': atr, 'natr': natr,
            # 新增成交量指标
            'ad': ad, 'adosc': adosc, 'obv': obv,
            # 新增周期指标
            'dcperiod': dcperiod,
            'dcphase': dcphase,
            'inphase': inphase,
            'quadrature': quadrature,
            'sine': sine,
            'leadsine': leadsine,
            'trendmode': trendmode,
            
        })
        
    except Exception as e:
        # 异常时返回NaN值
        length = len(group)
        columns = ['macd', 'macdsignal', 'macdhist', 'vol_ma', 'upperband',
                  'middleband', 'lowerband', 'dema', 'ema', 'trendline', 'kama',
                  'ma7', 'ma13', 'ma26', 'mama', 'fama', 'midpoint', 'midprice',
                  'sar', 'sarext', 'sma', 't3', 'tema', 'trima', 'wma',
                  'adx', 'adxr', 'apo', 'aroondown', 'aroonup', 'aroonosc', 'bop',
                  'cci', 'cmo', 'dx', 'macdext', 'macdextsignal', 'macdexthist',
                  'macdfix', 'macdfixsignal', 'macdfixhist', 'mfi', 'minus_di',
                  'minus_dm', 'mom', 'plus_di', 'plus_dm', 'ppo', 'roc', 'rocp',
                  'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk', 'fastd',
                  'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr', 'atr', 'natr',
                  # 新增成交量指标到异常处理列表
                  'ad', 'adosc', 'obv',
                  # 新增周期指标到异常处理列表
                  'dcperiod', 'dcphase', 'inphase', 'quadrature',
                  'sine', 'leadsine', 'trendmode',
                  # 新增形态识别指标到异常处理列表
                  'cdl2crows', 'cdl3blackcrows', 'cdl3inside', 'cdl3linestrike',
                  'cdl3outside', 'cdl3starsinsouth', 'cdl3whitesoldiers',
                  'cdlabandonedbaby', 'cdladvanceblock', 'cdlbelthold',
                  'cdlbreakaway', 'cdlclosingmarubozu', 'cdlconcealbabyswall',
                  'cdlcounterattack', 'cdldarkcloudcover', 'cdldoji',
                  'cdldojistar', 'cdldragonflydoji', 'cdlengulfing',
                  'cdleveningdojistar', 'cdleveningstar', 'cdlgapsidesidewhite',
                  'cdlgravestonedoji', 'cdlhammer', 'cdlhangingman',
                  'cdlharami', 'cdlharamicross', 'cdlhighwave',
                  'cdlhikkake', 'cdlhikkakemod', 'cdlhomingpigeon',
                  'cdlidentical3crows', 'cdlinneck', 'cdlinvertedhammer',
                  'cdlkicking', 'cdlkickingbylength', 'cdlladderbottom',
                  'cdllongleggeddoji', 'cdllongline', 'cdlmarubozu',
                  'cdlmatchinglow', 'cdlmathold', 'cdlmorningdojistar',
                  'cdlmorningstar', 'cdlonneck', 'cdlpiercing',
                  'cdlrickshawman', 'cdlrisefall3methods', 'cdlseparatinglines',
                  'cdlshootingstar', 'cdlshortline', 'cdlspinningtop',
                  'cdlstalledpattern', 'cdlsticksandwich', 'cdltakuri',
                  'cdltasukigap', 'cdlthrusting', 'cdltristar',
                  'cdlunique3river', 'cdlupsidegap2crows', 'cdlxsidegap3methods'
        ]
        return pd.DataFrame({col: [np.nan] * length for col in columns})

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
        ma_periods = [3, 5, 10, 20, 30]
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
    
