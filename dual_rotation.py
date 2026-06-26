"""
双品种轮动信号系统: 510310 + 159995
对每个品种独立运行 ADX Override，择优持仓
"""
import pandas as pd
import numpy as np
import os
import sys
import duckdb
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, 'csi300_data.duckdb')
RF = 0.025

def ensure_chip_data():
    """确保159995历史数据在库中"""
    conn = duckdb.connect(DB_PATH)
    try:
        conn.execute('SELECT 1 FROM etf_159995_daily LIMIT 1').fetchone()
        conn.close()
        return True
    except Exception:
        pass
    conn.close()

    print('下载 159995 芯片ETF 历史数据...')
    try:
        import akshare as ak
        df = ak.fund_etf_hist_sina(symbol='sz159995')
        df['date'] = pd.to_datetime(df['date'])
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=5)
        df = df[df['date'] >= cutoff].sort_values('date')
        conn = duckdb.connect(DB_PATH)
        conn.execute('DROP TABLE IF EXISTS etf_159995_daily')
        conn.execute('CREATE TABLE etf_159995_daily (date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, amount DOUBLE)')
        conn.execute('INSERT INTO etf_159995_daily SELECT * FROM df')
        conn.close()
        print(f'  159995: {len(df)}条  {df["date"].min().date()}~{df["date"].max().date()}')
        return True
    except Exception as e:
        print(f'  下载失败: {e}')
        return False


def compute_adx_signal(df):
    """对单个ETF计算ADX Override信号"""
    close = df['close']
    high = df['high']
    low = df['low']

    ma50 = close.rolling(50).mean()
    vol = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    up = high.diff(); dn = -low.diff()
    p_dm = pd.Series(0.0, index=df.index); n_dm = pd.Series(0.0, index=df.index)
    p_dm.loc[(up > dn) & (up > 0)] = up
    n_dm.loc[(dn > up) & (dn > 0)] = dn
    pdi = 100 * p_dm.ewm(alpha=1/14, adjust=False).mean() / atr14
    ndi = 100 * n_dm.ewm(alpha=1/14, adjust=False).mean() / atr14
    adx = (100 * abs(pdi - ndi) / (pdi + ndi + 1e-10)).ewm(alpha=1/14, adjust=False).mean()

    above = close > ma50
    low_vol = vol < 15
    trend = adx > 25
    signal = (above & (low_vol | trend)).astype(int)

    return {
        'close': close, 'ma50': ma50, 'vol': vol,
        'adx': adx, 'signal': signal,
        'last_close': close.iloc[-1],
        'last_ma50': ma50.iloc[-1],
        'last_vol': vol.iloc[-1],
        'last_adx': adx.iloc[-1],
    }


def get_dual_signals():
    """获取双品种信号"""
    conn = duckdb.connect(DB_PATH)

    df310 = conn.execute('SELECT date, open, high, low, close FROM etf_510310_daily ORDER BY date').fetchdf()
    df310['date'] = pd.to_datetime(df310['date'])

    try:
        df995 = conn.execute('SELECT date, open, high, low, close FROM etf_159995_daily ORDER BY date').fetchdf()
        df995['date'] = pd.to_datetime(df995['date'])
        has_995 = True
    except Exception:
        has_995 = False

    conn.close()

    r310 = compute_adx_signal(df310)
    r995 = compute_adx_signal(df995) if has_995 else None

    return {
        '510310': {
            'name': '沪深300ETF',
            'price': r310['last_close'],
            'ma50': r310['last_ma50'],
            'vol': r310['last_vol'],
            'adx': r310['last_adx'],
            'signal': int(r310['signal'].iloc[-1]),
        },
        '159995': {
            'name': '芯片ETF',
            'price': r995['last_close'] if r995 else 0,
            'ma50': r995['last_ma50'] if r995 else 0,
            'vol': r995['last_vol'] if r995 else 0,
            'adx': r995['last_adx'] if r995 else 0,
            'signal': int(r995['signal'].iloc[-1]) if r995 else 0,
        } if has_995 else None,
        'date': str(df310['date'].max().date()),
    }


def backtest_dual_rotation(warmup=300):
    """回测双品种轮动"""
    conn = duckdb.connect(DB_PATH)
    df310 = conn.execute('SELECT date, open, high, low, close FROM etf_510310_daily ORDER BY date').fetchdf()
    df310['date'] = pd.to_datetime(df310['date'])

    try:
        df995 = conn.execute('SELECT date, open, high, low, close FROM etf_159995_daily ORDER BY date').fetchdf()
        df995['date'] = pd.to_datetime(df995['date'])
    except Exception:
        conn.close()
        return None
    conn.close()

    r310 = compute_adx_signal(df310)
    r995 = compute_adx_signal(df995)

    # 合并日期对齐
    merged = df310[['date']].merge(df995[['date']], on='date', how='inner')
    common_dates = merged['date']

    n = len(common_dates)
    daily_ret = pd.Series(0.0, index=range(n))
    bond_daily = (1 + RF) ** (1/252) - 1

    in_position = None  # '310' or '995' or None

    for i in range(warmup, n):
        d = common_dates.iloc[i]
        d_prev = common_dates.iloc[i - 1]

        # 获取前一天收盘信号
        idx310 = df310[df310['date'] == d_prev].index
        idx995 = df995[df995['date'] == d_prev].index

        if len(idx310) == 0 or len(idx995) == 0:
            continue

        sig310 = int(r310['signal'].iloc[idx310[0]])
        sig995 = int(r995['signal'].iloc[idx995[0]])

        idx310_today = df310[df310['date'] == d].index
        idx995_today = df995[df995['date'] == d].index
        if len(idx310_today) == 0 or len(idx995_today) == 0:
            continue

        # 轮动逻辑：两个信号都有 → 选ADX更高的；只有一个 → 选有信号的；都没有 → 债券
        if sig310 == 1 and sig995 == 1:
            adx310 = r310['adx'].iloc[idx310[0]]
            adx995 = r995['adx'].iloc[idx995[0]]
            chosen = '310' if adx310 >= adx995 else '995'
        elif sig310 == 1:
            chosen = '310'
        elif sig995 == 1:
            chosen = '995'
        else:
            chosen = None

        # 计算当日收益
        if chosen == '310':
            o = df310['open'].iloc[idx310_today[0]]
            c = df310['close'].iloc[idx310_today[0]]
            daily_ret.iloc[i] = c / o - 1 if o > 0 else 0
        elif chosen == '995':
            o = df995['open'].iloc[idx995_today[0]]
            c = df995['close'].iloc[idx995_today[0]]
            daily_ret.iloc[i] = c / o - 1 if o > 0 else 0
        else:
            daily_ret.iloc[i] = bond_daily

    ret = daily_ret.iloc[warmup:]
    cum = (1 + ret).cumprod()
    total = cum.iloc[-1] - 1
    ny = len(ret) / 252
    ann = (1 + total) ** (1 / ny) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ann - RF) / vol if vol > 0 else -99
    peak = cum.expanding().max()
    dd = (cum / peak - 1).min()

    # 单品种对照
    def single_backtest(signal_series, close_series, open_series):
        s = signal_series.iloc[warmup:]
        c = close_series.iloc[warmup:]
        o = open_series.iloc[warmup:]
        aligned = min(len(s), len(c), len(o))
        s, c, o = s.iloc[:aligned], c.iloc[:aligned], o.iloc[:aligned]
        r = s.shift(1).fillna(0) * (c / o - 1) + (1 - s.shift(1).fillna(0)) * bond_daily
        cum_s = (1 + r).cumprod()
        tot = cum_s.iloc[-1] - 1
        ann_s = (1 + tot) ** (1 / (len(r)/252)) - 1
        dd_s = (cum_s / cum_s.expanding().max() - 1).min()
        return ann_s, dd_s

    ann310, dd310 = single_backtest(r310['signal'], df310['close'], df310['open'])
    ann995, dd995 = single_backtest(r995['signal'], df995['close'], df995['open'])

    return {
        'dual_ann': ann, 'dual_sharpe': sharpe, 'dual_dd': dd,
        'ann_310': ann310, 'dd_310': dd310,
        'ann_995': ann995, 'dd_995': dd995,
        'start': common_dates.iloc[warmup].date(),
        'end': common_dates.iloc[-1].date(),
        'ny': ny,
    }


if __name__ == '__main__':
    ensure_chip_data()
    signals = get_dual_signals()

    print('=' * 60)
    print('  双品种 ADX Override 信号')
    print(f'  数据日期: {signals["date"]}')
    print('=' * 60)

    for code in ['510310', '159995']:
        if signals[code] is None:
            continue
        s = signals[code]
        sig_text = '持有' if s['signal'] == 1 else '空仓'
        print(f'\n  【{s["name"]} ({code})】')
        print(f'  价格: {s["price"]:.4f}  MA50: {s["ma50"]:.4f}')
        print(f'  波动率: {s["vol"]:.2f}%  ADX: {s["adx"]:.2f}')
        print(f'  信号: {sig_text}')

    print()

    # 回测
    bt = backtest_dual_rotation()
    if bt:
        print('=' * 60)
        print(f'  双品种轮动回测 ({bt["start"]} ~ {bt["end"]}, {bt["ny"]:.1f}年)')
        print('=' * 60)
        print(f'  {"":<20} {"年化":>8} {"最大回撤":>10}')
        print(f'  {"双品种轮动":<20} {bt["dual_ann"]:>+7.2%}  {bt["dual_dd"]:>9.2%}')
        print(f'  {"仅510310":<20} {bt["ann_310"]:>+7.2%}  {bt["dd_310"]:>9.2%}')
        print(f'  {"仅159995":<20} {bt["ann_995"]:>+7.2%}  {bt["dd_995"]:>9.2%}')
