"""
开盘价执行回测 vs 收盘价回测对比
信号在T日收盘生成 → T+1日以开盘价买入/卖出
"""
import pandas as pd
import numpy as np
import os
import sys
import duckdb
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, 'csi300_data.duckdb')
RF = 0.025

# ============ 确保数据库有 open 列 ============
def ensure_open_data():
    conn = duckdb.connect(DB_PATH)
    cols = conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name='csi300_daily'").fetchdf()
    conn.close()

    has_open = 'open' in cols['column_name'].values
    if has_open:
        df = load_data()
        if df['open'].notna().sum() > 100:
            return df

    print('重新下载含开盘价的数据...')
    try:
        import akshare as ak
        df = ak.stock_zh_index_daily(symbol='sh000300')
        df['date'] = pd.to_datetime(df['date'])
        need_cols = ['date', 'open', 'high', 'low', 'close']
        df = df[need_cols].dropna().sort_values('date').reset_index(drop=True)

        # 过滤5年
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=5)
        df = df[df['date'] >= cutoff].copy()

        conn = duckdb.connect(DB_PATH)
        conn.execute('DROP TABLE IF EXISTS csi300_daily')
        conn.execute('CREATE TABLE csi300_daily (date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE)')
        conn.execute('INSERT INTO csi300_daily SELECT * FROM df')
        conn.close()
        print(f'已更新 duckdb: {len(df)} 条, 含开盘价')
        return df
    except Exception as e:
        print(f'下载失败: {e}')
        return None

def load_data():
    conn = duckdb.connect(DB_PATH)
    df = conn.execute('SELECT date, open, high, low, close FROM csi300_daily ORDER BY date').fetchdf()
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df

# ============ 计算指标 ============
def compute_indicators(df):
    close = df['close']
    high = df['high']
    low = df['low']

    ma50 = close.rolling(50).mean()
    vol_20 = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

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

    return ma50, vol_20, adx

# ============ 回测: 收盘价执行 ============
def backtest_close(df, warmup=300):
    close = df['close']
    ma50, vol_20, adx = compute_indicators(df)
    bond_daily = (1 + RF) ** (1/252) - 1

    above = close > ma50
    low_vol = vol_20 < 15
    trend = adx > 25
    signal = (above & (low_vol | trend)).astype(int).shift(1).fillna(0)

    daily_ret = close.pct_change()
    start = warmup + 1
    strat_ret = signal.iloc[start:] * daily_ret.iloc[start:] + (1 - signal.iloc[start:]) * bond_daily
    return compute_stats(strat_ret)

# ============ 回测: 开盘价执行 ============
def backtest_open(df, warmup=300):
    """
    真实执行模拟:
    - T日收盘生成信号 → T+1日开盘执行
    - 仅用T日收盘信号决定T+1日操作，不偷看T+1日收盘信号
    """
    close = df['close']
    open_p = df['open']
    ma50, vol_20, adx = compute_indicators(df)
    bond_daily = (1 + RF) ** (1/252) - 1

    above = close > ma50
    low_vol = vol_20 < 15
    trend = adx > 25
    signal_raw = (above & (low_vol | trend)).astype(int)

    n = len(df)
    in_position = False
    entry_opens = []
    exit_opens = []

    for i in range(warmup + 1, n):
        sig = signal_raw.iloc[i - 1]  # T-1日收盘信号 → 决定T日操作

        if sig == 1:
            if not in_position:
                entry_opens.append(open_p.iloc[i])
            in_position = True
        else:
            if in_position:
                exit_opens.append(open_p.iloc[i])
            in_position = False

    # 构建每日收益序列
    daily_returns = pd.Series(0.0, index=df.index)
    in_position = False

    for i in range(warmup + 1, n):
        sig = signal_raw.iloc[i - 1]
        was_holding = in_position

        if sig == 1:
            in_position = True
            daily_returns.iloc[i] = close.iloc[i] / open_p.iloc[i] - 1
        else:
            in_position = False
            if was_holding:
                daily_returns.iloc[i] = 0
            else:
                daily_returns.iloc[i] = bond_daily

    # 如回测结束时仍持仓，按最后收盘价平仓
    if in_position:
        pass  # 最后一天已用close计算收益

    strat_ret = daily_returns.iloc[warmup + 1:]
    stats = compute_stats(strat_ret)

    # 额外统计
    if len(entry_opens) > 0:
        holding_periods = []
        for e, x in zip(entry_opens, exit_opens[:len(entry_opens)]):
            if e > 0:
                holding_periods.append(x / e - 1)
        stats['n_entries'] = len(entry_opens)
        stats['avg_holding_ret'] = np.mean(holding_periods) if holding_periods else 0

    return stats


def compute_stats(ret):
    cum = (1 + ret).cumprod()
    total = cum.iloc[-1] - 1
    ny = len(ret) / 252
    ann = (1 + total) ** (1 / ny) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ann - RF) / vol if vol > 0 else -99
    peak = cum.expanding().max()
    dd = (cum / peak - 1).min()
    return {
        'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
        'ann_vol': vol, 'total_ret': total, 'n_days': len(ret)
    }


# ============ 运行 ============
if __name__ == '__main__':
    df = ensure_open_data()
    if df is None:
        print('数据获取失败')
        sys.exit(1)

    print(f'\n数据: {len(df)}条, {df["date"].min().date()} ~ {df["date"].max().date()}')
    print(f'最新: 开{df["open"].iloc[-1]:.2f} 高{df["high"].iloc[-1]:.2f} 低{df["low"].iloc[-1]:.2f} 收{df["close"].iloc[-1]:.2f}')

    # 测试不同warmup
    for warmup in [60, 300]:
        r_close = backtest_close(df, warmup)
        r_open = backtest_open(df, warmup)

        start = df['date'].iloc[warmup].date()
        end = df['date'].iloc[-1].date()
        ny = (r_open['n_days']) / 252

        print(f'\n{"="*80}')
        print(f'回测区间: {start} ~ {end} (约{ny:.1f}年)  warmup={warmup}')
        print(f'{"="*80}')
        print(f'{"方法":<20} {"年化收益":>10} {"Sharpe":>8} {"最大回撤":>10} {"年化波动":>10}')
        print(f'{"-"*80}')
        print(f'{"收盘价执行 (理论)":<20} {r_close["ann_ret"]:>+9.2%}  {r_close["sharpe"]:>7.2f}  {r_close["max_dd"]:>9.2%}  {r_close["ann_vol"]:>9.2%}')
        print(f'{"开盘价执行 (真实)":<20} {r_open["ann_ret"]:>+9.2%}  {r_open["sharpe"]:>7.2f}  {r_open["max_dd"]:>9.2%}  {r_open["ann_vol"]:>9.2%}')
        diff = r_close['ann_ret'] - r_open['ann_ret']
        print(f'{"差额 (滑点影响)":<20} {diff:>+9.2%}')
        if 'n_entries' in r_open:
            print(f'  开盘执行入场次数: {r_open["n_entries"]}  平均每笔收益: {r_open.get("avg_holding_ret", 0):+.2%}')

    print(f'\n说明:')
    print(f'  收盘价执行 = 信号在T日收盘后生成,假设能以T+1日收盘价成交 (有前视偏差)')
    print(f'  开盘价执行 = 信号在T日收盘后生成,以T+1日开盘价成交 (真实可执行)')
