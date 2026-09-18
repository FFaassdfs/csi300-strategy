# -*- coding: utf-8 -*-
"""
长历史压力测试 (#1) — 指数代理, 检验策略在 2015股灾/2018熊市/2020疫情/2024急涨 等极端段的表现
代理: 510310→sh000300(2002+) | 512800→sz399986中证银行(2015-05-19+) | 159995→H30184中证全指半导体(2014+)
口径: 生产参数(config.ASSETS) + 入场连续2日确认 + 实盘轮动逻辑(每日重选ADX最高, 无需间隔)
      T收盘定目标 → T+1开盘调仓 | 单边10bp | 空仓按国债2.5% | 可选波动刹车(vol>35%→0.6)
局限: 指数为价格指数(不含分红), 银行指数 2015-05-19 起(预热后 2015-08-18 起有效);
      2015年6月首轮股灾因预热不足无法覆盖, 但覆盖其后的第二轮(8月)及2016年初熔断段
"""
import os, sys
import numpy as np
import pandas as pd
import socket
socket.setdefaulttimeout(25)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from config import RF

COST = 0.001
BRAKE_TH, BRAKE_W = 35.0, 0.6
BOND = (1 + RF) ** (1 / 252) - 1
OUT = []


def log(m):
    print(m, flush=True)
    OUT.append(m)


# 品种: 代码 -> (代理指数, 数据源, (ma_p, adx_th, vol_th))
PROXY = {
    '510310': ('sh000300', 'sina', (30, 20, 18)),
    '159995': ('H30184', 'csindex', (30, 25, 15)),
    '512800': ('sz399986', 'sina', (30, 25, 18)),
}
ORDER = ['510310', '159995', '512800']


def fetch_index(sym, src):
    import akshare as ak
    if src == 'sina':
        df = ak.stock_zh_index_daily(symbol=sym)
        df = df[['date', 'open', 'high', 'low', 'close']].copy()
    else:
        df = ak.stock_zh_index_hist_csindex(symbol=sym, start_date='20130101', end_date='20261231')
        df = df.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close'})
        df = df[['date', 'open', 'high', 'low', 'close']]
    df['date'] = pd.to_datetime(df['date'])
    for c in ['open', 'high', 'low', 'close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna().sort_values('date').reset_index(drop=True)


def compute_ind(df, ma_p, adx_th, vol_th):
    from signal_core import compute_signal_core
    c, h, l = df['close'], df['high'], df['low']
    core = compute_signal_core(c, h, l, ma_p=ma_p, adx_th=adx_th, vol_th=vol_th)
    return {'open': df['open'].values, 'close': c.values,
            'ma': core['ma'].values, 'vol': core['vol'].values,
            'adx': core['adx'].values, 'signal': core['signal'].values,
            'ma_p': ma_p, 'adx_th': adx_th, 'vol_th': vol_th}


def perf(eq):
    rets = np.diff(eq) / eq[:-1]
    if len(rets) < 20:
        return 0., 0., 0., 0.
    years = len(rets) / 252
    cagr = (eq[-1] / eq[0]) ** (1 / years) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252 - RF) / vol if vol > 1e-9 else 0.
    peak = np.maximum.accumulate(eq)
    return cagr, vol, sharpe, ((eq - peak) / peak).min()


def simulate(aligned, cal, i0, i1, brake):
    """aligned: {code: dict(数组, 已对齐到 cal, 缺失为 nan)}; 实盘轮动逻辑(无间隔切换)"""
    O = {cd: aligned[cd]['open'] for cd in ORDER}
    C = {cd: aligned[cd]['close'] for cd in ORDER}
    ADX = {cd: aligned[cd]['adx'] for cd in ORDER}
    VOL = {cd: aligned[cd]['vol'] for cd in ORDER}
    SIG = {cd: aligned[cd]['signal'] for cd in ORDER}

    def avail(cd, t):
        return np.isfinite(C[cd][t]) and np.isfinite(ADX[cd][t])

    def conf(cd, t):
        return avail(cd, t) and SIG[cd][t] == 1 and t > 0 and SIG[cd][t - 1] == 1

    n = i1 - i0
    equity = np.ones(n + 1)
    eq = 1.0
    w = {cd: 0.0 for cd in ORDER}
    pending = None
    held = None
    for k in range(n):
        t = i0 + k
        # 执行上一收盘目标 (今日开盘); 无目标=维持
        if k > 0 and pending is not None:
            tgt = pending
        else:
            tgt = w
        # 权重变动 (仅对可用品种)
        changed = 0.0
        for cd in ORDER:
            if abs(tgt.get(cd, 0.0) - w[cd]) > 1e-9:
                changed += abs(tgt.get(cd, 0.0) - w[cd])
        if changed > 1e-9:
            eq *= (1 - COST * changed)
        old = dict(w)
        w = {cd: tgt.get(cd, 0.0) for cd in ORDER}
        # 当日收益
        day_ret = 0.0
        for cd in ORDER:
            wo, wn = old[cd], w[cd]
            keep = min(wo, wn); sell = max(0.0, wo - wn); buy = max(0.0, wn - wo)
            if keep > 1e-9:
                day_ret += keep * (C[cd][t] / C[cd][t - 1] - 1)
            if sell > 1e-9:
                day_ret += sell * (O[cd][t] / C[cd][t - 1] - 1)
            if buy > 1e-9:
                day_ret += buy * (C[cd][t] / O[cd][t] - 1)
        cash = 1.0 - sum(w.values())
        if cash > 1e-9:
            day_ret += cash * BOND
        eq *= (1 + day_ret)
        equity[k + 1] = eq
        held = next((cd for cd in ORDER if w[cd] > 1e-9), None)
        # 收盘决策 (实盘口径: 持仓 signal=1 免确认; 新入场需确认; 选ADX最高)
        cands = [cd for cd in ORDER if conf(cd, t)]
        if held is not None and avail(held, t) and SIG[held][t] == 1 and held not in cands:
            cands.append(held)
        pick = max(cands, key=lambda cd: ADX[cd][t]) if cands else None
        nt = {}
        if pick is not None:
            nt[pick] = BRAKE_W if (brake and VOL[pick][t] > BRAKE_TH) else 1.0
        pending = nt
    return equity


def window_stats(eq, cal_dates, i0, i1, a, b):
    """返回窗口内 策略收益/最大回撤 (窗口局部峰)"""
    ia = int(np.searchsorted(cal_dates, np.datetime64(a)))
    ib = int(np.searchsorted(cal_dates, np.datetime64(b), side='right')) - 1
    ia = max(ia, i0); ib = min(ib, i1)
    if ib <= ia:
        return None
    seg = eq[ia - i0: ib - i0 + 1]
    ret = seg[-1] / seg[0] - 1
    peak = np.maximum.accumulate(seg)
    dd = ((seg - peak) / peak).min()
    return ret, dd


def bh_stats(close, cal_dates, a, b):
    ia = int(np.searchsorted(cal_dates, np.datetime64(a)))
    ib = int(np.searchsorted(cal_dates, np.datetime64(b), side='right')) - 1
    seg = close[ia: ib + 1]
    ret = seg[-1] / seg[0] - 1
    peak = np.maximum.accumulate(seg)
    return ret, ((seg - peak) / peak).min()


def main():
    log('# 长历史压力测试 (指数代理)')
    log('')
    indices = {}
    for cd, (sym, src, params) in PROXY.items():
        df = fetch_index(sym, src)
        indices[cd] = (df, params)
        log(f'- {cd} 代理 {sym}: {df["date"].iloc[0].date()} ~ {df["date"].iloc[-1].date()} ({len(df)}行)')

    # 联合日历
    all_dates = sorted(set().union(*[set(indices[cd][0]['date']) for cd in ORDER]))
    cal = pd.DatetimeIndex(all_dates)
    aligned = {}
    for cd in ORDER:
        df, params = indices[cd]
        ind = compute_ind(df, *params)
        idx = {d: i for i, d in enumerate(df['date'])}
        m = np.array([idx.get(d, -1) for d in cal])
        def take(key):
            arr = np.full(len(cal), np.nan)
            src = ind[key]
            ok = m >= 0
            arr[ok] = src[m[ok]]
            return arr
        aligned[cd] = {k: take(k) for k in ['open', 'close', 'ma', 'vol', 'adx', 'signal']}

    # 起点: 全部品种预热后 (银行 2015-05-19 起 + ~60 交易日)
    i0 = int(np.searchsorted(cal.values, np.datetime64('2015-08-18')))
    i1 = len(cal)
    log('')
    log(f'有效期间: {cal[i0].date()} ~ {cal[i1-1].date()} ({i1-i0} 个交易日)')
    log(f'口径: T收盘定目标→T+1开盘调仓 | 单边10bp | 空仓国债2.5% | 2日确认 | 实盘轮动逻辑')
    log('')

    eq_b = simulate(aligned, cal, i0, i1, brake=True)
    eq_n = simulate(aligned, cal, i0, i1, brake=False)
    c300 = aligned['510310']['close']

    log('## 全期表现')
    log('')
    log('| 方案 | 年化 | 波动 | Sharpe | 最大回撤 | 期末净值 |')
    log('|---|---|---|---|---|---|')
    for name, e in [('策略(含波动刹车)', eq_b), ('策略(无刹车)', eq_n)]:
        c, v, s, d = perf(e)
        log(f'| {name} | {c:+.1%} | {v:.1%} | {s:.2f} | {d:.1%} | {e[-1]:.2f}x |')
    sub = c300[i0:i1]
    bh_eq = sub / sub[0]
    c, v, s, d = perf(bh_eq)
    log(f'| Buy&Hold 沪深300 | {c:+.1%} | {v:.1%} | {s:.2f} | {d:.1%} | {bh_eq[-1]:.2f}x |')
    log('')

    WINDOWS = [
        ('2015-08-18', '2016-01-28', '2015第二轮股灾+2016熔断'),
        ('2018-01-24', '2019-01-04', '2018单边熊市'),
        ('2020-01-20', '2020-03-23', '2020疫情暴跌'),
        ('2022-01-05', '2024-08-30', '2022-2024长熊'),
        ('2024-09-18', '2024-10-08', '2024-09暴力反弹'),
        ('2020-04-30', '2026-09-17', '可比段(与ETF回测重叠)'),
    ]
    log('## 危机窗口对照')
    log('')
    log('| 窗口 | 区间 | 策略(刹车) 收益/回撤 | 策略(无刹车) 收益/回撤 | 沪深300 收益/回撤 |')
    log('|---|---|---|---|---|')
    for a, b, label in WINDOWS:
        sb = window_stats(eq_b, cal.values, i0, i1, a, b)
        sn = window_stats(eq_n, cal.values, i0, i1, a, b)
        bb = bh_stats(c300, cal.values, a, b)
        if sb is None:
            continue
        log(f'| {label} | {a}~{b} | {sb[0]:+.1%} / {sb[1]:.1%} | {sn[0]:+.1%} / {sn[1]:.1%} | {bb[0]:+.1%} / {bb[1]:.1%} |')
    log('')

    os.makedirs(os.path.join(PROJECT_ROOT, 'reports'), exist_ok=True)
    out = os.path.join(PROJECT_ROOT, 'reports', 'longhistory_validation_20260917.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(OUT) + '\n')
    log(f'报告已保存: {out}')


if __name__ == '__main__':
    main()
