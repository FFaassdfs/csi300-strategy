# -*- coding: utf-8 -*-
"""
ERP 股债利差估值择时叠加测试 (2026-09-18)
ERP = 沪深300盈利收益率(1/PE_TTM) - 10年国债收益率; ERP 分位低 = 股票相对债券贵 → 降仓
实现: 滚动5年分位(无前视) → 敞口系数 exposure∈[0,1] 乘在目标权重上, 其余计现金(国债2.5%)
检验: (a) ETF池 2020-04~今  (b) 指数代理池 2015-08~今 (含2015股灾/2018熊/2021顶)
"""
import os, sys, socket
socket.setdefaulttimeout(25)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import pandas as pd
import duckdb

import validate_tier1 as T
import validate_rotation_rule as RR
import validate_longhistory as LH
from config import HISTORY_DB

RF = 0.025
COST = 0.001
BRAKE_TH, BRAKE_W = 35.0, 0.6
BOND = (1 + RF) ** (1 / 252) - 1


# ---------- ERP 数据 ----------
def build_erp():
    import akshare as ak
    pe = ak.stock_index_pe_lg(symbol='沪深300')
    pcol = '滚动市盈率' if '滚动市盈率' in pe.columns else [c for c in pe.columns if '市盈率' in c][0]
    pe = pe[['日期', pcol]].rename(columns={pcol: 'pe'})
    pe['日期'] = pd.to_datetime(pe['日期'])
    pe = pe.dropna().sort_values('日期').set_index('日期')
    y = ak.bond_zh_us_rate()
    y = y[['日期', '中国国债收益率10年']].rename(columns={'中国国债收益率10年': 'y10'})
    y['日期'] = pd.to_datetime(y['日期'])
    y = y.dropna().sort_values('日期').set_index('日期')
    print(f'PE数据: {pe.index[0].date()}~{pe.index[-1].date()} ({len(pe)}行) | 10Y国债: {y.index[0].date()}~{y.index[-1].date()}')
    df = pe.join(y, how='outer').sort_index().ffill()          # 前向填充(因果)
    df = df.dropna()
    df['erp'] = 100.0 / df['pe'] - df['y10']                    # 盈利收益率(%) - 10Y(%)
    W = 1260   # 滚动5年
    df['pct'] = df['erp'].rolling(W, min_periods=250).apply(lambda s: (s.iloc[-1] > s.iloc[:-1]).mean() * 100, raw=False)
    return df[['erp', 'pct']].dropna()


# ---------- 通用回测(带 ERP 敞口) ----------
def simulate(inds, dates, codes, exposure, brake=True):
    O = {c: inds[c]['open'] for c in codes}
    C = {c: inds[c]['close'] for c in codes}
    MA = {c: inds[c]['ma'] for c in codes}
    VOL = {c: inds[c]['vol'] for c in codes}
    ADX = {c: inds[c]['adx'] for c in codes}
    SIG = {c: inds[c]['signal'] for c in codes}
    VT = {c: inds[c]['vol_th'] for c in codes}
    AT = {c: inds[c]['adx_th'] for c in codes}
    i1 = len(dates); n = i1
    equity = np.ones(n + 1); eq = 1.0
    w_cur = {c: 0.0 for c in codes}; w_tgt = {c: 0.0 for c in codes}
    held = None; trades = 0

    def conf(cd, t):
        return SIG[cd][t] == 1 and t > 0 and SIG[cd][t - 1] == 1

    for k in range(n):
        t = k
        if k > 0:
            for cd in codes:
                if abs(w_cur[cd] - w_tgt[cd]) > 1e-9:
                    trades += 1
                    eq *= (1 - COST * abs(w_cur[cd] - w_tgt[cd]))
            day_ret = 0.0
            for cd in codes:
                wo, wn = w_cur[cd], w_tgt[cd]
                keep, sell, buy = min(wo, wn), max(0.0, wo - wn), max(0.0, wn - wo)
                if keep > 1e-9:
                    day_ret += keep * (C[cd][t] / C[cd][t - 1] - 1)
                if sell > 1e-9:
                    day_ret += sell * (O[cd][t] / C[cd][t - 1] - 1)
                if buy > 1e-9:
                    day_ret += buy * (C[cd][t] / O[cd][t] - 1)
            cash = 1.0 - sum(w_tgt.values())
            if cash > 1e-9:
                day_ret += cash * BOND
            w_cur = dict(w_tgt)
            eq *= (1 + day_ret)
            equity[k + 1] = eq
            held = next((c for c in codes if w_cur[c] > 1e-9), None)
        cands = [c for c in codes if conf(c, t)]
        pick = max(cands, key=lambda c: ADX[c][t]) if cands else None
        if held is not None:
            if not ((C[held][t] >= MA[held][t]) and (VOL[held][t] < VT[held] or ADX[held][t] > AT[held])):
                pick = None
        nw = {c: 0.0 for c in codes}
        if pick is not None:
            bw = BRAKE_W if (brake and VOL[pick][t] > BRAKE_TH) else 1.0
            nw[pick] = bw * exposure[t]
        w_tgt = nw
    return equity


def report(eq, dates, label, i_start):
    rets = np.diff(eq) / eq[:-1]
    years = len(rets) / 252
    cagr = eq[-1] ** (1 / years) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252 - RF) / vol if vol > 1e-9 else 0
    peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min()
    return f"| {label} | {cagr:+.1%} | {vol:.1%} | {sharpe:.2f} | {dd:.1%} | {eq[-1]:.2f}x |"


def main():
    erp = build_erp()
    print(erp.tail(3).to_string())
    print(f"ERP 分位分布: min={erp['pct'].min():.0f} 中位={erp['pct'].median():.0f} max={erp['pct'].max():.0f}")
    for d in ['2015-06-12', '2018-01-24', '2021-02-10', '2024-09-30', '2026-09-17']:
        dt = pd.Timestamp(d)
        sub = erp[erp.index <= dt]
        if len(sub):
            print(f"  {d}: ERP={sub['erp'].iloc[-1]:.2f} 分位={sub['pct'].iloc[-1]:.0f}")
    print()

    VARIANTS = [
        ('V0 无择时', lambda p: 1.0),
        ('V1 极贵(<20分位)→0.5仓', lambda p: 0.5 if p < 20 else 1.0),
        ('V2 极贵(<10分位)→0.4仓', lambda p: 0.4 if p < 10 else 1.0),
        ('V3 偏贵(<40)→0.7, 极贵(<20)→0.4', lambda p: 0.4 if p < 20 else (0.7 if p < 40 else 1.0)),
    ]

    # ===== (a) ETF 池 2020-04-30 ~ 今 =====
    conn = duckdb.connect(HISTORY_DB, read_only=True)
    inds, dates = RR.load_aligned(conn)
    conn.close()
    dts = pd.DatetimeIndex(dates)
    i0 = int(np.searchsorted(dts.values, np.datetime64(RR.START)))
    sub_dates = dts[i0:]
    erp_d = erp['pct'].reindex(dts, method='ffill').values
    print(f"## (a) ETF池 {dates[i0].date()} ~ {dates[-1].date()}")
    print('| 方案 | 年化 | 波动 | Sharpe | 最大回撤 | 期末净值 |')
    print('|---|---|---|---|---|---|')
    for name, fn in VARIANTS:
        expo = np.where(np.isnan(erp_d), 1.0, [fn(p) for p in np.nan_to_num(erp_d, nan=50.0)])
        eq = simulate(inds, dates, RR.ORDER, expo)[i0:]
        print(report(eq, sub_dates, name, i0))
    print()

    # ===== (b) 指数代理池 2015-08-18 ~ 今 =====
    idx = {}
    for cd, (sym, src, params) in LH.PROXY.items():
        idx[cd] = (LH.fetch_index(sym, src), params)
    all_dates = sorted(set().union(*[set(idx[cd][0]['date']) for cd in LH.ORDER]))
    cal = pd.DatetimeIndex(all_dates)
    aligned = {}
    for cd in LH.ORDER:
        df, params = idx[cd]
        ind = LH.compute_ind(df, *params)
        mp = {d: i for i, d in enumerate(df['date'])}
        m = np.array([mp.get(d, -1) for d in cal])
        arrs = {}
        for key in ['open', 'close', 'ma', 'vol', 'adx', 'signal']:
            a = np.full(len(cal), np.nan)
            ok = m >= 0
            a[ok] = ind[key][m[ok]]
            arrs[key] = pd.Series(a).ffill().values   # 因果前向填充(补个别缺失交易日)
        arrs['vol_th'] = params[2]
        arrs['adx_th'] = params[1]
        aligned[cd] = arrs
    j0 = int(np.searchsorted(cal.values, np.datetime64('2015-08-18')))
    erp_c = erp['pct'].reindex(cal, method='ffill').values
    print(f"## (b) 指数代理池 {cal[j0].date()} ~ {cal[-1].date()} (含2015股灾/2018熊/2021顶)")
    print('| 方案 | 年化 | 波动 | Sharpe | 最大回撤 | 期末净值 |')
    print('|---|---|---|---|---|---|')
    for name, fn in VARIANTS:
        expo = np.where(np.isnan(erp_c), 1.0, [fn(p) for p in np.nan_to_num(erp_c, nan=50.0)])
        eq = simulate(aligned, cal, LH.ORDER, expo)[j0:]
        print(report(eq, cal[j0:], name, j0))
    print()

    # 窗口对照
    print('## 关键窗口 (指数代理池, V0 vs V3)')
    expo0 = np.ones(len(cal)); expo3 = np.where(np.isnan(erp_c), 1.0, [VARIANTS[3][1](p) for p in np.nan_to_num(erp_c, nan=50.0)])
    eq0 = simulate(aligned, cal, LH.ORDER, expo0); eq3 = simulate(aligned, cal, LH.ORDER, expo3)
    print('| 窗口 | V0收益/回撤 | V3收益/回撤 |')
    print('|---|---|---|')
    for a, b, lab in [('2015-06-12', '2016-01-28', '2015股灾'), ('2018-01-24', '2019-01-04', '2018熊'),
                      ('2021-02-10', '2022-04-30', '2021顶后下跌'), ('2022-01-05', '2024-08-30', '2022-24长熊')]:
        ia = int(np.searchsorted(cal.values, np.datetime64(a))); ib = int(np.searchsorted(cal.values, np.datetime64(b), side='right')) - 1
        for nm, e in [('V0', eq0), ('V3', eq3)]:
            seg = e[ia:ib + 1]
            r = seg[-1] / seg[0] - 1
            pk = np.maximum.accumulate(seg)
            d = ((seg - pk) / pk).min()
            if nm == 'V0':
                line = f"| {lab} | {r:+.1%} / {d:.1%} |"
            else:
                line += f" {r:+.1%} / {d:.1%} |"
        print(line)


if __name__ == '__main__':
    main()
