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
        future_close = group['close'].shift(-1).rolling(window=n, min_periods=1).agg(['min', 'max'])
        future_pre_close = group['pre_close'].shift(-1).rolling(window=n, min_periods=1)
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
    计算一组技术指标,包括MACD、成交量MA、布林线、ADX等
    
    参数:
    group: DataFrame, 包含单个股票的行情数据,需要包含close/high/low/vol等列
    
    返回:
    DataFrame: 包含计算出的各项技术指标
    """
    try:
        # MACD指标
        macd, signal, hist = talib.MACD(group['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        
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
        
        # ===== 形态识别指标 =====
        # 两只乌鸦
        cdl2crows = talib.CDL2CROWS(group['open'], group['high'], group['low'], group['close'])
        # 三只乌鸦
        cdl3blackcrows = talib.CDL3BLACKCROWS(group['open'], group['high'], group['low'], group['close'])
        # 三内部上涨和下跌
        cdl3inside = talib.CDL3INSIDE(group['open'], group['high'], group['low'], group['close'])
        # 三线打击
        cdl3linestrike = talib.CDL3LINESTRIKE(group['open'], group['high'], group['low'], group['close'])
        # 三外部上涨和下跌
        cdl3outside = talib.CDL3OUTSIDE(group['open'], group['high'], group['low'], group['close'])
        # 南方三星
        cdl3starsinsouth = talib.CDL3STARSINSOUTH(group['open'], group['high'], group['low'], group['close'])
        # 三白兵
        cdl3whitesoldiers = talib.CDL3WHITESOLDIERS(group['open'], group['high'], group['low'], group['close'])
        # 弃婴
        cdlabandonedbaby = talib.CDLABANDONEDBABY(group['open'], group['high'], group['low'], group['close'])
        # 大敌当前
        cdladvanceblock = talib.CDLADVANCEBLOCK(group['open'], group['high'], group['low'], group['close'])
        # 捉腰带线
        cdlbelthold = talib.CDLBELTHOLD(group['open'], group['high'], group['low'], group['close'])
        # 脱离
        cdlbreakaway = talib.CDLBREAKAWAY(group['open'], group['high'], group['low'], group['close'])
        # 收盘缺影线
        cdlclosingmarubozu = talib.CDLCLOSINGMARUBOZU(group['open'], group['high'], group['low'], group['close'])
        # 藏婴吞没
        cdlconcealbabyswall = talib.CDLCONCEALBABYSWALL(group['open'], group['high'], group['low'], group['close'])
        # 反击线
        cdlcounterattack = talib.CDLCOUNTERATTACK(group['open'], group['high'], group['low'], group['close'])
        # 乌云压顶
        cdldarkcloudcover = talib.CDLDARKCLOUDCOVER(group['open'], group['high'], group['low'], group['close'])
        # 十字
        cdldoji = talib.CDLDOJI(group['open'], group['high'], group['low'], group['close'])
        # 十字星
        cdldojistar = talib.CDLDOJISTAR(group['open'], group['high'], group['low'], group['close'])
        # 蜻蜓十字
        cdldragonflydoji = talib.CDLDRAGONFLYDOJI(group['open'], group['high'], group['low'], group['close'])
        # 吞噬模式
        cdlengulfing = talib.CDLENGULFING(group['open'], group['high'], group['low'], group['close'])
        # 十字暮星
        cdleveningdojistar = talib.CDLEVENINGDOJISTAR(group['open'], group['high'], group['low'], group['close'])
        # 暮星
        cdleveningstar = talib.CDLEVENINGSTAR(group['open'], group['high'], group['low'], group['close'])
        # 向上/下跳空并列阳线
        cdlgapsidesidewhite = talib.CDLGAPSIDESIDEWHITE(group['open'], group['high'], group['low'], group['close'])
        # 墓碑十字
        cdlgravestonedoji = talib.CDLGRAVESTONEDOJI(group['open'], group['high'], group['low'], group['close'])
        # 锤子线
        cdlhammer = talib.CDLHAMMER(group['open'], group['high'], group['low'], group['close'])
        # 上吊线
        cdlhangingman = talib.CDLHANGINGMAN(group['open'], group['high'], group['low'], group['close'])
        # 母子线
        cdlharami = talib.CDLHARAMI(group['open'], group['high'], group['low'], group['close'])
        # 十字孕线
        cdlharamicross = talib.CDLHARAMICROSS(group['open'], group['high'], group['low'], group['close'])
        # 风高浪大线
        cdlhighwave = talib.CDLHIGHWAVE(group['open'], group['high'], group['low'], group['close'])
        # 陷阱
        cdlhikkake = talib.CDLHIKKAKE(group['open'], group['high'], group['low'], group['close'])
        # 修正陷阱
        cdlhikkakemod = talib.CDLHIKKAKEMOD(group['open'], group['high'], group['low'], group['close'])
        # 家鸽
        cdlhomingpigeon = talib.CDLHOMINGPIGEON(group['open'], group['high'], group['low'], group['close'])
        # 三胞胎乌鸦
        cdlidentical3crows = talib.CDLIDENTICAL3CROWS(group['open'], group['high'], group['low'], group['close'])
        # 颈内线
        cdlinneck = talib.CDLINNECK(group['open'], group['high'], group['low'], group['close'])
        # 倒锤子线
        cdlinvertedhammer = talib.CDLINVERTEDHAMMER(group['open'], group['high'], group['low'], group['close'])
        # 反冲形态
        cdlkicking = talib.CDLKICKING(group['open'], group['high'], group['low'], group['close'])
        # 由较长缺影线决定的反冲形态
        cdlkickingbylength = talib.CDLKICKINGBYLENGTH(group['open'], group['high'], group['low'], group['close'])
        # 梯底
        cdlladderbottom = talib.CDLLADDERBOTTOM(group['open'], group['high'], group['low'], group['close'])
        # 长脚十字
        cdllongleggeddoji = talib.CDLLONGLEGGEDDOJI(group['open'], group['high'], group['low'], group['close'])
        # 长蜡烛线
        cdllongline = talib.CDLLONGLINE(group['open'], group['high'], group['low'], group['close'])
        # 光头光脚/缺影线
        cdlmarubozu = talib.CDLMARUBOZU(group['open'], group['high'], group['low'], group['close'])
        # 相同低价
        cdlmatchinglow = talib.CDLMATCHINGLOW(group['open'], group['high'], group['low'], group['close'])
        # 铺垫
        cdlmathold = talib.CDLMATHOLD(group['open'], group['high'], group['low'], group['close'])
        # 十字晨星
        cdlmorningdojistar = talib.CDLMORNINGDOJISTAR(group['open'], group['high'], group['low'], group['close'])
        # 晨星
        cdlmorningstar = talib.CDLMORNINGSTAR(group['open'], group['high'], group['low'], group['close'])
        # 颈上线
        cdlonneck = talib.CDLONNECK(group['open'], group['high'], group['low'], group['close'])
        # 刺透形态
        cdlpiercing = talib.CDLPIERCING(group['open'], group['high'], group['low'], group['close'])
        # 黄包车夫
        cdlrickshawman = talib.CDLRICKSHAWMAN(group['open'], group['high'], group['low'], group['close'])
        # 上升/下降三法
        cdlrisefall3methods = talib.CDLRISEFALL3METHODS(group['open'], group['high'], group['low'], group['close'])
        # 分离线
        cdlseparatinglines = talib.CDLSEPARATINGLINES(group['open'], group['high'], group['low'], group['close'])
        # 射击之星
        cdlshootingstar = talib.CDLSHOOTINGSTAR(group['open'], group['high'], group['low'], group['close'])
        # 短蜡烛线
        cdlshortline = talib.CDLSHORTLINE(group['open'], group['high'], group['low'], group['close'])
        # 纺锤
        cdlspinningtop = talib.CDLSPINNINGTOP(group['open'], group['high'], group['low'], group['close'])
        # 停顿形态
        cdlstalledpattern = talib.CDLSTALLEDPATTERN(group['open'], group['high'], group['low'], group['close'])
        # 条形三明治
        cdlsticksandwich = talib.CDLSTICKSANDWICH(group['open'], group['high'], group['low'], group['close'])
        # 探水竿
        cdltakuri = talib.CDLTAKURI(group['open'], group['high'], group['low'], group['close'])
        # 跳空并列阴阳线
        cdltasukigap = talib.CDLTASUKIGAP(group['open'], group['high'], group['low'], group['close'])
        # 插入
        cdlthrusting = talib.CDLTHRUSTING(group['open'], group['high'], group['low'], group['close'])
        # 三星
        cdltristar = talib.CDLTRISTAR(group['open'], group['high'], group['low'], group['close'])
        # 奇特三河床
        cdlunique3river = talib.CDLUNIQUE3RIVER(group['open'], group['high'], group['low'], group['close'])
        # 向上跳空的两只乌鸦
        cdlupsidegap2crows = talib.CDLUPSIDEGAP2CROWS(group['open'], group['high'], group['low'], group['close'])
        # 上升/下降跳空三法
        cdlxsidegap3methods = talib.CDLXSIDEGAP3METHODS(group['open'], group['high'], group['low'], group['close'])
        
        return pd.DataFrame({
            'macd': macd, 'macdsignal': signal, 'macdhist': hist,
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
            # 新增形态识别指标
            'cdl2crows': cdl2crows,
            'cdl3blackcrows': cdl3blackcrows,
            'cdl3inside': cdl3inside,
            'cdl3linestrike': cdl3linestrike,
            'cdl3outside': cdl3outside,
            'cdl3starsinsouth': cdl3starsinsouth,
            'cdl3whitesoldiers': cdl3whitesoldiers,
            'cdlabandonedbaby': cdlabandonedbaby,
            'cdladvanceblock': cdladvanceblock,
            'cdlbelthold': cdlbelthold,
            'cdlbreakaway': cdlbreakaway,
            'cdlclosingmarubozu': cdlclosingmarubozu,
            'cdlconcealbabyswall': cdlconcealbabyswall,
            'cdlcounterattack': cdlcounterattack,
            'cdldarkcloudcover': cdldarkcloudcover,
            'cdldoji': cdldoji,
            'cdldojistar': cdldojistar,
            'cdldragonflydoji': cdldragonflydoji,
            'cdlengulfing': cdlengulfing,
            'cdleveningdojistar': cdleveningdojistar,
            'cdleveningstar': cdleveningstar,
            'cdlgapsidesidewhite': cdlgapsidesidewhite,
            'cdlgravestonedoji': cdlgravestonedoji,
            'cdlhammer': cdlhammer,
            'cdlhangingman': cdlhangingman,
            'cdlharami': cdlharami,
            'cdlharamicross': cdlharamicross,
            'cdlhighwave': cdlhighwave,
            'cdlhikkake': cdlhikkake,
            'cdlhikkakemod': cdlhikkakemod,
            'cdlhomingpigeon': cdlhomingpigeon,
            'cdlidentical3crows': cdlidentical3crows,
            'cdlinneck': cdlinneck,
            'cdlinvertedhammer': cdlinvertedhammer,
            'cdlkicking': cdlkicking,
            'cdlkickingbylength': cdlkickingbylength,
            'cdlladderbottom': cdlladderbottom,
            'cdllongleggeddoji': cdllongleggeddoji,
            'cdllongline': cdllongline,
            'cdlmarubozu': cdlmarubozu,
            'cdlmatchinglow': cdlmatchinglow,
            'cdlmathold': cdlmathold,
            'cdlmorningdojistar': cdlmorningdojistar,
            'cdlmorningstar': cdlmorningstar,
            'cdlonneck': cdlonneck,
            'cdlpiercing': cdlpiercing,
            'cdlrickshawman': cdlrickshawman,
            'cdlrisefall3methods': cdlrisefall3methods,
            'cdlseparatinglines': cdlseparatinglines,
            'cdlshootingstar': cdlshootingstar,
            'cdlshortline': cdlshortline,
            'cdlspinningtop': cdlspinningtop,
            'cdlstalledpattern': cdlstalledpattern,
            'cdlsticksandwich': cdlsticksandwich,
            'cdltakuri': cdltakuri,
            'cdltasukigap': cdltasukigap,
            'cdlthrusting': cdlthrusting,
            'cdltristar': cdltristar,
            'cdlunique3river': cdlunique3river,
            'cdlupsidegap2crows': cdlupsidegap2crows,
            'cdlxsidegap3methods': cdlxsidegap3methods
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


def preprocess_data(stock_info, label_col='label', test_size=0.1):
    """
    数据预处理:
    1. 处理缺失值
    2. 删除每个股票最后n个交易日的数据
    3. 划分训练测试集
    4. 转换数据格式并保存到SQLite
    
    参数:
    stock_info: DataFrame, 原始股票数据
    label_col: str, 标签列名
    test_size: float, 测试集比例
    
    返回:
    tuple: (X_train, y_train, X_test, y_test)
    """
    # 删除包含无效标签的行
    df = stock_info[stock_info['label'].isin([-1, 0, 1])].copy()
    
    # 将-1和0的标签合并为0
    df['label'] = df['label'].map({-1: 0, 0: 0, 1: 1})
    
    # 删除每个股票最后n个交易日的数据
    n = 5  # 最后5个交易日
    df = df.sort_values(['ts_code', 'trade_date'])
    df = df.groupby('ts_code').apply(
        lambda x: x.iloc[:-n] if len(x) > n else x
    ).reset_index(drop=True)
    
    # 丢弃缺失值
    df = df.fillna(0)
    
    # 定义特征列（排除非特征列）
    exclude_cols = ['ts_code', 'trade_date', label_col]
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    
    # 划分数据集（按时间顺序）
    df.sort_values('trade_date', inplace=True)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # 准备数据
    X_train = train_df[feature_columns]
    y_train = train_df[label_col]
    X_test = test_df[feature_columns]
    y_test = test_df[label_col]
    
    # 打印数据集信息
    print("\nDataset Info:")
    print(f"Total samples: {len(df):,}")
    print(f"Training samples: {len(train_df):,}")
    print(f"Testing samples: {len(test_df):,}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Label distribution:")
    print(df[label_col].value_counts(normalize=True).round(4) * 100)
    
    # 保存数据到SQLite
    try:
        # 保存训练集
        save_stock_data(train_df, 'train_data', file_path='./data/train.nfa', mode='replace')
        print("Saved training dataset")
        
        # 保存测试集
        save_stock_data(test_df, 'test_data', file_path='./data/train.nfa', mode='replace')
        print("Saved testing dataset")
        
        # 保存特征列信息
        feature_df = pd.DataFrame({'feature_name': feature_columns})
        save_stock_data(feature_df, 'feature_columns', file_path='./data/train.nfa', mode='replace')
        print("Saved feature columns information")
        
    except Exception as e:
        print(f"Error saving data to SQLite: {str(e)}")
    
    return X_train, y_train, X_test, y_test, feature_columns


if __name__ == '__main__':
    print('加载 数据')
    stock_info = load_stock_data(table_name='daily_info', file_path='./data/train.nfa')
    print('设置标签')
    stock_info = label_price_changes(stock_info)
    # save_stock_data(stock_info, 'stock_info_with_lable')
    print('分析标签分布')
    stats = analyze_label_distribution(stock_info)
    print(stats)
    print('添加技术指标')
    stock_info = add_technical_indicators(stock_info)
    print('存储技术指标')
    save_stock_data(stock_info, 'technical_indicators_info', file_path='./data/train.nfa', mode='append')
    print('预处理数据')
    preprocess_data(stock_info)
    print('完成')
    # print(stock_info.head())
    # stock_info = load_stock_data(table_name='stock_info_with_technical_indicators')
    # plot_random_positive_samples(stock_info, n_samples=100)
    
    # # 获取最新记录
    # latest_records = get_latest_records(stock_info, n=6)
    # print("\n最新记录示例:")
    # print(latest_records[['ts_code', 'trade_date', 'close', 'label']].head())
    
    # # 分析标签分布
    # label_stats = analyze_label_distribution(latest_records)
    # print("\n标签分布统计:")
    # print(label_stats)

    # # 过滤正样本
    # filtered_stock_info = filter_positive_samples(latest_records)
    # print("\n过滤后的标签分布统计:")
    # print(analyze_label_distribution(filtered_stock_info))

    # # 增强数据质量
    # train_df, eval_df = enhance_data_quality(latest_records)
    # print("\n增强后的标签分布统计:")
    # print(analyze_label_distribution(train_df))
    # print(analyze_label_distribution(eval_df))


    # pd.set_option('display.max_info_columns', 200) 
    
    # key = Fernet.generate_key()
    # b'FT248o1y-EYWal2Uy0m4ulH8uBNhS3ETmibNR3RVsew='