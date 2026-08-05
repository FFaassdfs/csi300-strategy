"""
信号有效性验证 + 策略重新评估
基于 trading_history.duckdb 完整历史数据
1. 买入信号后未来 5/10/20 天收益统计
2. 触发条件分组对比 (ADX>25 vs 低波动)
3. 全历史回测 (开盘价执行)
4. 参数敏感性分析
"""
import pandas as pd
import numpy as np
import os
import duckdb
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
HISTORY_DB = os.path.join(PROJECT_ROOT, 'trading_history.duckdb')
RF = 0.025
BOND_DAILY = (1 + RF) ** (1/252) - 1

ASSETS = {
    '510310': '沪深300ETF',
    '159995': '芯片ETF',
    '512660': '军工ETF',
}


def load_ohlc(conn, code):
    df = conn.execute(
        f"SELECT date, open, high, low, close FROM daily_ohlc WHERE code='{code}' ORDER BY date"
    ).fetchdf()
    df['date'] = pd.to_datetime(df['date'])
    return df


def compute_signal(df, ma_p=50, adx_p=14, vol_p=20, vol_th=15, adx_th=25):
    """标准 ADX Override 信号"""
    c = df['close']; h = df['high']; l = df['low']
    ma = c.rolling(ma_p).mean()
    vol = c.pct_change().rolling(vol_p).std() * np.sqrt(252) * 100
    tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/adx_p, adjust=False).mean()
    up = h.diff(); dn = -l.diff()
    pdm = pd.Series(0.0, index=df.index); ndm = pd.Series(0.0, index=df.index)
    pdm.loc[(up > dn) & (up > 0)] = up
    ndm.loc[(dn > up) & (dn > 0)] = dn
    pdi = 100 * pdm.ewm(alpha=1/adx_p, adjust=False).mean() / atr
    ndi = 100 * ndm.ewm(alpha=1/adx_p, adjust=False).mean() / atr
    adx = (100 * abs(pdi - ndi) / (pdi + ndi + 1e-10)).ewm(alpha=1/adx_p, adjust=False).mean()
    above = c > ma
    low_vol = vol < vol_th
    trend = adx > adx_th
    signal = (above & (low_vol | trend)).astype(int)
    return signal, {'signal': signal, 'above': above, 'low_vol': low_vol, 'trend': trend, 'adx': adx, 'vol': vol, 'ma': ma}


def signal_effectiveness(df, signal, label):
    """信号有效性: 买入信号后 N 天收益"""
    c = df['close']
    ret = c.pct_change()
    sig_idx = np.where(signal.values == 1)[0]
    sig_idx = sig_idx[sig_idx >= 50]  # 跳过预热

    # 未来收益 = 次日开盘买入, N 天后收盘卖出
    o = df['open'].values
    cl = c.values

    print(f'\n{"="*70}')
    print(f'  {label} 信号有效性验证 ({len(sig_idx)} 个买入信号)')
    print(f'{"="*70}')

    for horizon in [3, 5, 10, 20, 30]:
        futs = []
        for i in sig_idx:
            j = i + horizon
            if j < len(df):
                # 次日开盘买入 (i+1 开盘), 第 j 天收盘卖出
                entry = o[i+1] if i+1 < len(df) else cl[i]
                exit_p = cl[j]
                futs.append(exit_p / entry - 1)
        if futs:
            futs = np.array(futs)
            win = (futs > 0).mean() * 100
            avg = futs.mean() * 100
            med = np.median(futs) * 100
            bh = (cl[np.array(sig_idx) + horizon if np.array(sig_idx).max() + horizon < len(df) else -1][:len(futs)] / o[np.array(sig_idx[:len(futs)]) + 1] - 1) * 100 if False else 0
            print(f'  T+{horizon:>2}d: 胜率 {win:>5.1f}%  平均 {avg:>+6.2f}%  中位 {med:>+6.2f}%')
    return


def trigger_comparison(df, sig_info, label):
    """按触发条件分组"""
    c = df['close']; o = df['open']
    signal = sig_info['signal']
    above = sig_info['above'].values
    low_vol = sig_info['low_vol'].values
    trend = sig_info['trend'].values
    sig = signal.values

    groups = {
        'ADX>25 触发': (sig == 1) & trend,
        '低波动触发': (sig == 1) & low_vol & ~trend,
    }

    print(f'\n  {label} 触发条件对比 (T+20收益):')
    cl = c.values; op = o.values
    for gname, mask in groups.items():
        idx = np.where(mask)[0]
        idx = idx[idx >= 50]
        futs = []
        for i in idx:
            j = i + 20
            if j < len(df):
                futs.append(cl[j] / op[i+1] - 1)
        if futs:
            futs = np.array(futs)
            print(f'    {gname:<12} {len(futs):>4}次  胜率{(futs>0).mean()*100:>5.1f}%  平均{futs.mean()*100:>+6.2f}%')
    return


def full_backtest(df, signal, label, warmup=60):
    """全历史回测 (开盘价执行)"""
    c = df['close']; o = df['open']
    s = signal.shift(1).fillna(0).astype(float)
    ret = s * (c / o - 1) + (1 - s) * BOND_DAILY
    ret = ret.iloc[warmup:]
    cum = (1 + ret).cumprod()

    total = cum.iloc[-1] - 1
    ny = len(ret) / 252
    ann = (1 + total) ** (1 / ny) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ann - RF) / vol if vol > 0 else -99
    dd = (cum / cum.expanding().max() - 1).min()
    trades = s.diff().abs().sum()
    hold = s.sum() / len(s) * 100

    # Buy & Hold
    bh_ret = (c / o - 1).iloc[warmup:]
    bh_cum = (1 + bh_ret).cumprod()
    bh_ann = (1 + (bh_cum.iloc[-1] - 1)) ** (1 / ny) - 1
    bh_dd = (bh_cum / bh_cum.expanding().max() - 1).min()

    print(f'  {label}:')
    print(f'    {df["date"].iloc[warmup].date()} ~ {df["date"].iloc[-1].date()}  ({ny:.1f}年)')
    print(f'    年化 {ann:+.2%}  Sharpe {sharpe:.2f}  回撤 {dd:.2%}  波动 {vol:.2%}')
    print(f'    交易 {trades:.0f} 次  持仓 {hold:.1f}%')
    print(f'    Buy&Hold: 年化 {bh_ann:+.2%}  回撤 {bh_dd:.2%}')
    print(f'    超额: {ann - bh_ann:+.2%}')
    return ann, sharpe, dd


def param_sensitivity(df, label):
    """参数敏感性: MA周期 x ADX阈值 x vol阈值"""
    print(f'\n  {label} 参数敏感性 (年化收益 %):')
    results = []
    for ma_p in [30, 50, 100]:
        for adx_th in [20, 25, 30]:
            for vol_th in [12, 15, 18]:
                signal, _ = compute_signal(df, ma_p=ma_p, adx_th=adx_th, vol_th=vol_th)
                c = df['close']; o = df['open']
                s = signal.shift(1).fillna(0).astype(float)
                ret = s * (c / o - 1) + (1 - s) * BOND_DAILY
                ret = ret.iloc[110:]  # 足够预热
                cum = (1 + ret).cumprod()
                total = cum.iloc[-1] - 1
                ny = len(ret) / 252
                ann = (1 + total) ** (1 / ny) - 1
                vol = ret.std() * np.sqrt(252)
                sharpe = (ann - RF) / vol if vol > 0 else -99
                dd = (cum / cum.expanding().max() - 1).min()
                results.append((ma_p, adx_th, vol_th, ann, sharpe, dd))

    results.sort(key=lambda x: x[4], reverse=True)
    print(f'    {"MA":>4} {"ADXth":>6} {"Volth":>6} {"年化":>8} {"Sharpe":>7} {"回撤":>8}')
    for r in results[:10]:
        print(f'    {r[0]:>4} {r[1]:>6} {r[2]:>6} {r[3]:>+7.2%} {r[4]:>7.2f} {r[5]:>8.2%}')
    print(f'    ... (共 {len(results)} 组合, 显示 Top10)')
    return


def main():
    print('=' * 70)
    print('  信号有效性验证 + 策略重新评估')
    print(f'  数据源: {HISTORY_DB}')
    print('=' * 70)

    conn = duckdb.connect(HISTORY_DB)

    for code, name in ASSETS.items():
        df = load_ohlc(conn, code)
        if len(df) < 200:
            print(f'\n{name} 数据不足: {len(df)}')
            continue
        print(f'\n{"#"*70}')
        print(f'  # {name} ({code})  {df["date"].min().date()} ~ {df["date"].max().date()}  {len(df)}条')
        print(f'{"#"*70}')

        signal, sig_info = compute_signal(df)
        signal_effectiveness(df, signal, name)
        trigger_comparison(df, sig_info, name)
        full_backtest(df, signal, name)
        param_sensitivity(df, name)

    conn.close()
    print('\n' + '=' * 70)
    print('  完成')
    print('=' * 70)


if __name__ == '__main__':
    main()
