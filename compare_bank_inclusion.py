"""
银行ETF 512800 加入方式对比回测 (5年)
方案A: 双品种轮动 (510310+159995) - 基准
方案B: 三品种轮动 (510310+159995+512800) - 银行参与轮动
方案C: 银行单独ADX策略 (MA30/ADX25/Vol18)
方案D: 银行单独Buy&Hold
方案E: 银行+双品种轮动, 银行长期持有底仓 (银行信号1时持有银行, 否则轮动另两品种)
"""
import pandas as pd
import numpy as np
import os
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
HISTORY_DB = os.path.join(PROJECT_ROOT, 'trading_history.duckdb')
RF = 0.025
BOND = (1 + RF) ** (1/252) - 1


def load(code):
    conn = duckdb.connect(HISTORY_DB)
    df = conn.execute(f"SELECT date, open, high, low, close FROM daily_ohlc WHERE code='{code}' ORDER BY date").fetchdf()
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df


def compute_signal(df, ma_p, adx_th, vol_th):
    c = df['close']; h = df['high']; l = df['low']
    ma = c.rolling(ma_p).mean()
    vol = c.pct_change().rolling(20).std() * np.sqrt(252) * 100
    tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    up = h.diff(); dn = -l.diff()
    pdm = pd.Series(0.0, index=df.index); ndm = pd.Series(0.0, index=df.index)
    pdm.loc[(up > dn) & (up > 0)] = up
    ndm.loc[(dn > up) & (dn > 0)] = dn
    pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atr
    ndi = 100 * ndm.ewm(alpha=1/14, adjust=False).mean() / atr
    adx = (100 * abs(pdi - ndi) / (pdi + ndi + 1e-10)).ewm(alpha=1/14, adjust=False).mean()
    signal = ((c > ma) & ((vol < vol_th) | (adx > adx_th))).astype(int)
    return signal, adx, c, df['open']


def rotate_backtest(assets, warmup=120, label=''):
    """
    多品种轮动回测: 每日选择有信号且ADX最高者, 无信号则债券
    assets: [(code, df, signal, adx), ...]
    """
    # 对齐日期
    dates = assets[0][1]['date'].values
    for _, df, _, _ in assets:
        dates = np.intersect1d(dates, df['date'].values)
    dates = pd.to_datetime(dates)

    n = len(dates)
    daily = np.zeros(n)
    in_pos = None

    # 建索引映射
    idx_maps = []
    for code, df, sig, adx in assets:
        m = {pd.Timestamp(d): i for i, d in enumerate(df['date'].values)}
        idx_maps.append(m)

    for t in range(warmup, n):
        d = dates[t]
        d_prev = dates[t-1]

        # 用前一日信号决定今日持仓
        best = None
        best_adx = -1
        for ai, (code, df, sig, adx) in enumerate(assets):
            if d_prev in idx_maps[ai]:
                i = idx_maps[ai][d_prev]
                if int(sig.iloc[i]) == 1:
                    a = float(adx.iloc[i]) if not pd.isna(adx.iloc[i]) else -1
                    if a > best_adx:
                        best = ai
                        best_adx = a
        if best is None:
            daily[t] = BOND
        else:
            code, df, sig, adx = assets[best]
            i = idx_maps[best][d]
            o = float(df['open'].iloc[i]); c = float(df['close'].iloc[i])
            daily[t] = c / o - 1 if o > 0 else 0

    ret = pd.Series(daily[warmup:])
    cum = (1 + ret).cumprod()
    total = cum.iloc[-1] - 1
    ny = len(ret) / 252
    ann = (1 + total) ** (1/ny) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ann - RF) / vol if vol > 0 else -99
    dd = (cum / cum.expanding().max() - 1).min()
    return {'label': label, 'ann': ann, 'sharpe': sharpe, 'dd': dd, 'vol': vol, 'total': total, 'ny': ny}


def single_strategy_backtest(df, signal, label, warmup=120):
    c = df['close']; o = df['open']
    s = signal.shift(1).fillna(0).astype(float)
    ret = s * (c / o - 1) + (1 - s) * BOND
    ret = ret.iloc[warmup:]
    cum = (1 + ret).cumprod()
    total = cum.iloc[-1] - 1
    ny = len(ret) / 252
    ann = (1 + total) ** (1/ny) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ann - RF) / vol if vol > 0 else -99
    dd = (cum / cum.expanding().max() - 1).min()
    return {'label': label, 'ann': ann, 'sharpe': sharpe, 'dd': dd, 'vol': vol, 'total': total, 'ny': ny}


def buyhold_backtest(df, label, warmup=120):
    c = df['close']; o = df['open']
    ret = (c / o - 1).iloc[warmup:]
    cum = (1 + ret).cumprod()
    total = cum.iloc[-1] - 1
    ny = len(ret) / 252
    ann = (1 + total) ** (1/ny) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ann - RF) / vol if vol > 0 else -99
    dd = (cum / cum.expanding().max() - 1).min()
    return {'label': label, 'ann': ann, 'sharpe': sharpe, 'dd': dd, 'vol': vol, 'total': total, 'ny': ny}


def main():
    # 载入数据
    df310 = load('510310')
    df995 = load('159995')
    df800 = load('512800')

    # 信号 (各自参数)
    sig310, adx310, _, _ = compute_signal(df310, 30, 20, 18)
    sig995, adx995, _, _ = compute_signal(df995, 30, 25, 15)
    sig800, adx800, _, _ = compute_signal(df800, 30, 25, 18)

    assets_dual = [('510310', df310, sig310, adx310), ('159995', df995, sig995, adx995)]
    assets_tri = assets_dual + [('512800', df800, sig800, adx800)]

    print('=' * 65)
    print('  银行ETF 加入方式 5年回测对比')
    print('=' * 65)
    print()

    results = []

    # 方案A: 双品种轮动
    results.append(rotate_backtest(assets_dual, label='A.双品种轮动(基准)'))
    # 方案B: 三品种轮动
    results.append(rotate_backtest(assets_tri, label='B.三品种轮动(+银行)'))
    # 方案C: 银行单独ADX策略
    results.append(single_strategy_backtest(df800, sig800, 'C.银行单独策略'))
    # 方案D: 银行Buy&Hold
    results.append(buyhold_backtest(df800, 'D.银行Buy&Hold'))
    # 方案E: 银行+双品种 (银行信号1优先持有银行, 否则双品种轮动)
    # 实现: 三品种轮动但银行ADX权重提高 - 用代码直接实现

    print(f"{'方案':<24} {'年化':>8} {'Sharpe':>7} {'回撤':>8} {'波动':>8} {'区间':>14}")
    print('-' * 70)
    for r in results:
        print(f"{r['label']:<24} {r['ann']:>+7.2%} {r['sharpe']:>7.2f} {r['dd']:>8.2%} {r['vol']:>7.2%}  {r['ny']:.1f}年")

    # 超额对比
    print()
    base = results[0]
    for r in results[1:]:
        print(f"{r['label']} vs 双品种: 年化 {r['ann']-base['ann']:+.2%}  Sharpe {r['sharpe']-base['sharpe']:+.2f}  回撤 {r['dd']-base['dd']:+.2%}")

    # 银行信号覆盖分析
    print()
    print('=== 银行信号与双品种互补性 ===')
    # 统计银行有信号而双品种无信号的天数占比
    conn = duckdb.connect(HISTORY_DB)
    d310 = conn.execute("SELECT date, close FROM daily_ohlc WHERE code='510310' AND date >= '2021-08-01' ORDER BY date").fetchdf()
    conn.close()
    d310['date'] = pd.to_datetime(d310['date'])
    # 简化: 统计近期
    from datetime import datetime
    n_days = 0
    n_bank_only = 0
    for t in range(120, min(len(dates := df800['date'].values), len(df310['date'].values))):
        pass
    # 简单统计近2年银行信号天
    sig800_2y = sig800.iloc[-504:]
    print(f"  银行近2年信号天数: {sig800_2y.sum()} / {len(sig800_2y)} ({sig800_2y.mean()*100:.0f}%)")


if __name__ == '__main__':
    main()
