"""
沪深300 ETF 趋势跟踪 + 波动率风控策略
三个策略：ADX Override / Momentum Override / Absolute 15%
"""

import pandas as pd
import numpy as np
import os


def compute_atr(df, period=14):
    """计算 Average True Range (Wilder平滑)"""
    high = df['high_etf']
    low = df['low_etf']
    close = df['close_etf']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def compute_adx(df, period=14):
    """计算 ADX (Average Directional Index, Wilder标准算法)"""
    high = df['high_etf']
    low = df['low_etf']
    close = df['close_etf']

    atr = compute_atr(df, period)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def compute_volatility(df, period=20):
    """计算年化波动率"""
    close = df['close_etf']
    vol = close.pct_change().rolling(period).std() * np.sqrt(252) * 100
    return vol


def compute_ma(df, period=50):
    """计算移动均线"""
    return df['close_etf'].rolling(period).mean()


def compute_momentum(df, period=20):
    """计算N日动量"""
    close = df['close_etf']
    return close / close.shift(period) - 1


# ========== 策略1: +ADX>25 Override ==========
def adx_override_strategy(df):
    """
    买入信号: 价格 > MA50 AND (波动率 < 15% OR ADX > 25)
    卖出信号: 价格 < MA50 OR 波动率 >= 15%
    """
    close = df['close_etf']
    vol = compute_volatility(df)
    ma50 = compute_ma(df)
    adx = compute_adx(df)

    above_ma = close > ma50
    low_vol = vol < 15
    strong_trend = adx > 25

    signal = (above_ma & (low_vol | strong_trend)).astype(int)
    return signal, {'vol': vol, 'ma50': ma50, 'adx': adx, 'vol_threshold': 15}


# ========== 策略2: +Momentum>10% Override ==========
def momentum_override_strategy(df):
    """
    买入信号: 价格 > MA50 AND (波动率 < 15% OR 近20日涨幅 > 10%)
    卖出信号: 价格 < MA50 OR 波动率 >= 15%
    """
    close = df['close_etf']
    vol = compute_volatility(df)
    ma50 = compute_ma(df)
    momentum = compute_momentum(df)

    above_ma = close > ma50
    low_vol = vol < 15
    strong_momentum = momentum > 0.10

    signal = (above_ma & (low_vol | strong_momentum)).astype(int)
    return signal, {'vol': vol, 'ma50': ma50, 'momentum': momentum, 'vol_threshold': 15}


# ========== 策略3: Base Absolute 15% ==========
def absolute_15_strategy(df):
    """
    买入信号: 价格 > MA50 AND 波动率 < 15%
    卖出信号: 价格 < MA50 OR 波动率 >= 15%
    """
    close = df['close_etf']
    vol = compute_volatility(df)
    ma50 = compute_ma(df)

    above_ma = close > ma50
    low_vol = vol < 15

    signal = (above_ma & low_vol).astype(int)
    return signal, {'vol': vol, 'ma50': ma50, 'vol_threshold': 15}


# ========== 策略注册表 ==========
STRATEGIES = {
    'adx_override': {
        'name': '+ADX>25 Override',
        'func': adx_override_strategy,
        'description': '价格>MA50 AND (波动率<15% OR ADX>25)',
        'params': {}
    },
    'momentum_override': {
        'name': '+Momentum>10% Override',
        'func': momentum_override_strategy,
        'description': '价格>MA50 AND (波动率<15% OR 近20日涨>10%)',
        'params': {}
    },
    'absolute_15': {
        'name': 'Base Absolute 15%',
        'func': absolute_15_strategy,
        'description': '价格>MA50 AND 波动率<15%',
        'params': {}
    }
}


def get_strategy(name):
    """获取指定策略"""
    if name not in STRATEGIES:
        raise ValueError(f'Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}')
    return STRATEGIES[name]


def calculate_signals(df, strategy_name):
    """计算策略信号"""
    strategy = get_strategy(strategy_name)
    signal, indicators = strategy['func'](df)
    return signal, indicators


def generate_trade_signal(df, strategy_name, position='cash'):
    """
    生成交易信号

    Args:
        df: 包含日期、OHLC的ETF历史数据DataFrame (需有high_etf/low_etf/close_etf列)
        strategy_name: 策略名称
        position: 当前持仓 ('csi300' 或 'bond')

    Returns:
        dict: 包含 signal, action, indicators 等信息
    """
    signal, indicators = calculate_signals(df, strategy_name)
    strategy = get_strategy(strategy_name)

    last_signal = signal.iloc[-1] if len(signal) > 0 else 0
    last_vol = indicators['vol'].iloc[-1] if len(indicators['vol']) > 0 else 0
    last_ma50 = indicators['ma50'].iloc[-1] if len(indicators['ma50']) > 0 else 0
    last_close = df['close_etf'].iloc[-1] if len(df) > 0 else 0

    # 确定行动
    if last_signal == 1:
        if position == 'bond':
            action = 'BUY_CSI300'
        else:
            action = 'HOLD_CSI300'
    else:
        if position == 'csi300':
            action = 'SELL_CSI300'
        else:
            action = 'HOLD_BOND'

    return {
        'signal': 'CSI300' if last_signal == 1 else 'BOND',
        'action': action,
        'position': 'csi300' if last_signal == 1 else 'bond',
        'indicators': {
            'close': last_close,
            'ma50': last_ma50,
            'vol_20': last_vol,
            'vol_threshold': indicators.get('vol_threshold', 15),
            'above_ma50': last_close > last_ma50 if not np.isnan(last_ma50) else False,
            'low_vol': last_vol < indicators.get('vol_threshold', 15) if not np.isnan(last_vol) else False
        },
        'strategy_name': strategy['name'],
        'strategy_desc': strategy['description']
    }
