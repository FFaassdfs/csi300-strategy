"""
对比回测：固定阈值15% vs 相对波动率（滚动1年均值）
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, 'csi300_data.duckdb')
RISK_FREE_RATE = 0.025


def load_data():
    conn = duckdb.connect(DB_PATH)
    df = conn.execute('SELECT date, open, high, low, close FROM csi300_daily ORDER BY date').fetchdf()
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df


def compute_indicators(df):
    close = df['close']
    high = df['high']
    low = df['low']

    ma50 = close.rolling(50).mean()
    vol_20 = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    vol_1y_avg = vol_20.rolling(252).mean()

    # ADX (Wilder)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_dm.loc[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm.loc[(down_move > up_move) & (down_move > 0)] = down_move
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()

    momentum_20 = close / close.shift(20) - 1

    return close, ma50, vol_20, vol_1y_avg, adx, momentum_20


def backtest_stateful(df, warmup=300):
    """回测引擎：支持有状态的入场/离场逻辑"""
    close, ma50, vol_20, vol_1y_avg, adx, momentum_20 = compute_indicators(df)
    n = len(df)

    def run_strategy(entry_fn, exit_fn):
        """entry_fn(i) → bool, exit_fn(i) → bool, 每次只持有一份"""
        signal = np.zeros(n, dtype=int)
        in_position = False
        for i in range(warmup, n):
            if in_position:
                if exit_fn(i):
                    in_position = False
                else:
                    signal[i] = 1
            else:
                if entry_fn(i):
                    in_position = True
                    signal[i] = 1
        return pd.Series(signal, index=df.index)

    daily_ret = close.pct_change()
    bond_daily = (1 + RISK_FREE_RATE) ** (1/252) - 1

    results = []

    def evaluate(signal, name):
        s = signal.iloc[warmup:].astype(float)
        s = s.shift(1).fillna(0)

        ret = s * daily_ret.iloc[warmup:] + (1 - s) * bond_daily
        cum = (1 + ret).cumprod()

        total = cum.iloc[-1] - 1
        ny = len(ret) / 252
        ann = (1 + total) ** (1 / ny) - 1
        vol = ret.std() * np.sqrt(252)
        sharpe = (ann - RISK_FREE_RATE) / vol if vol > 0 else 0

        peak_c = cum.expanding().max()
        dd = (cum / peak_c - 1).min()

        hold = s.sum() / len(s) * 100
        trades = s.diff().abs().sum()

        # BH
        bh_cum = (1 + daily_ret.iloc[warmup:]).cumprod()
        bh_total = bh_cum.iloc[-1] - 1
        bh_ann = (1 + bh_total) ** (1 / ny) - 1
        bh_peak_c = bh_cum.expanding().max()
        bh_dd = (bh_cum / bh_peak_c - 1).min()

        return {
            'name': name, 'ann_ret': ann, 'sharpe': sharpe, 'max_dd': dd,
            'ann_vol': vol, 'hold_pct': hold, 'trades': trades,
            'total_ret': total, 'bh_ann_ret': bh_ann, 'bh_max_dd': bh_dd,
            'ny': ny, 'start': df['date'].iloc[warmup],
        }

    # ── 入场/离场条件函数 ──
    above_ma = lambda i: close.iloc[i] > ma50.iloc[i]
    low_vol_15 = lambda i: vol_20.iloc[i] < 15
    low_vol_1y = lambda i: vol_20.iloc[i] < vol_1y_avg.iloc[i]
    adx_high = lambda i: adx.iloc[i] > 25
    mom_high = lambda i: momentum_20.iloc[i] > 0.10
    below_ma = lambda i: close.iloc[i] < ma50.iloc[i]

    def both(a, b):
        return lambda i: a(i) and b(i)

    def either(a, b):
        return lambda i: a(i) or b(i)

    # ===== 所有策略 =====

    # 1. 原策略: 固定15%, 双向
    sig = run_strategy(both(above_ma, low_vol_15), below_ma)
    results.append(evaluate(sig, '原: 固定15% (进出均看vol<15%)'))

    # 2. 原策略: ADX Override
    sig = run_strategy(both(above_ma, either(low_vol_15, adx_high)), below_ma)
    results.append(evaluate(sig, '原: ADX Override (vol<15%|ADX>25)'))

    # 3. 原策略: Momentum Override
    sig = run_strategy(both(above_ma, either(low_vol_15, mom_high)), below_ma)
    results.append(evaluate(sig, '原: Momentum Override (vol<15%|动>10%)'))

    # 4. ★新策略: 相对波动率入场, 跌破MA50离场（用户要求）
    sig = run_strategy(both(above_ma, low_vol_1y), below_ma)
    results.append(evaluate(sig, '★新: 相对波动率入场 (vol<1y均值), 仅破MA50离场'))

    # 5. 新策略: 相对波动率入场 + 相对波动率对称离场
    sig = run_strategy(both(above_ma, low_vol_1y), lambda i: below_ma(i) or not low_vol_1y(i))
    results.append(evaluate(sig, '新: 相对波动率对称 (进出均看vol<1y均值)'))

    # 6. 新策略: 相对波动率 + ADX Override 入场
    sig = run_strategy(both(above_ma, either(low_vol_1y, adx_high)), below_ma)
    results.append(evaluate(sig, '★新: 相对波+ADX Override (vol<1y均值|ADX>25)'))

    return results


# ============ 运行 ============
if __name__ == '__main__':
    df = load_data()
    print(f'数据: {len(df)}条, {df["date"].min().date()} ~ {df["date"].max().date()}')
    print()

    results = backtest_stateful(df, warmup=300)

    print('=' * 95)
    print(f'{"策略":<48} {"年化收益":>8} {"Sharpe":>7} {"最大回撤":>8} {"年化波动":>8} {"持仓%":>7} {"交易":>5}')
    print('=' * 95)

    for r in results:
        print(f'{r["name"]:<48} {r["ann_ret"]:>+7.2%}  {r["sharpe"]:>6.2f}  {r["max_dd"]:>7.2%}  {r["ann_vol"]:>7.2%}  {r["hold_pct"]:>6.1f}%  {r["trades"]:>5.0f}')

    # Buy & Hold
    close = df['close']
    daily_ret = close.pct_change().iloc[300:]
    cum_bh = (1 + daily_ret).cumprod()
    bh_total = cum_bh.iloc[-1] - 1
    bh_ann = (1 + bh_total) ** (1 / (len(daily_ret) / 252)) - 1
    bh_vol = daily_ret.std() * np.sqrt(252)
    bh_dd = (cum_bh / cum_bh.expanding().max() - 1).min()

    print(f'{"Buy & Hold (基准)":<48} {bh_ann:>+7.2%}  {"--":>6}  {bh_dd:>7.2%}  {bh_vol:>7.2%}')
    print('=' * 95)

    # 最佳 + 最差
    best = max(results, key=lambda x: x['sharpe'])
    worst = min(results, key=lambda x: x['max_dd'])
    print(f'\nSharpe最高: {best["name"]} (Sharpe={best["sharpe"]:.2f}, 年化{best["ann_ret"]:+.2%}, 回撤{best["max_dd"]:.2%})')

    print(f'\n--- 用户要求策略详情 ---')
    for r in results:
        if '★' in r['name']:
            print(f'  {r["name"]}')
            print(f'    年化: {r["ann_ret"]:+.2%}  Sharpe: {r["sharpe"]:.2f}  回撤: {r["max_dd"]:.2%}  波动: {r["ann_vol"]:.2%}  持仓: {r["hold_pct"]:.1f}%  交易: {r["trades"]:.0f}次')
