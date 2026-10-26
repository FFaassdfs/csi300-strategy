# -*- coding: utf-8 -*-
"""
执行时序对比回测 (2026-10-26) — 只读 trading_history.duckdb, 不写库
A(现行基准): T收盘信号 → T+1开盘成交   B(当日执行): T收盘信号 → T收盘成交
B 是"盘中触发+当日操作"的回测近似: 真实世界盘中触发用当时已知实时价(无未来函数),
成交在触发价与收盘价之间, T收盘成交为中性近似; 收盘翻空的日子盘中几乎必然已在
某个30分钟检查点触发(仅尾盘15分钟翻转例外)。
决策门槛: B 年化 ≥ A 年化 - 1pp → 落地盘中预警链路; 否则停止重新讨论。
口径: R0轮动 + 2日确认 + 波动刹车(35%/60%) | 单边10bp | 空仓国债2.5%
用法: python validate_execution_timing.py
输出: 控制台 + reports/execution_timing_validation_<date>.md
"""
import os
import sys
import datetime as _dt

import numpy as np
import pandas as pd
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import validate_tier1 as T
import validate_param_plateau as P
from config import ORDER

COST = 0.001


def simulate_sameday(inds, params, i0, i1, cost=COST, brake=True):
    """B口径: T收盘决策 → T收盘立即成交 (C[t]), 持仓整天收益 C[t]/C[t-1]"""
    A = {cd: {k: inds[cd][k].tolist() for k in ('open', 'close', 'ma', 'vol', 'adx', 'signal')}
         for cd in ORDER}
    C = {cd: A[cd]['close'] for cd in ORDER}
    MA = {cd: A[cd]['ma'] for cd in ORDER}
    VOL = {cd: A[cd]['vol'] for cd in ORDER}
    ADX = {cd: A[cd]['adx'] for cd in ORDER}
    SIG = {cd: A[cd]['signal'] for cd in ORDER}

    n = i1 - i0
    equity = np.ones(n + 1)
    eq = 1.0
    w_cur = {cd: 0.0 for cd in ORDER}
    held = None
    trades = 0
    for k in range(n):
        t = i0 + k
        if k > 0:
            # 当日收益 (持仓为昨日收盘执行所定, 整天 C[t]/C[t-1])
            day_ret = 0.0
            for cd in ORDER:
                if w_cur[cd] > 1e-9:
                    day_ret += w_cur[cd] * (C[cd][t] / C[cd][t - 1] - 1)
            cash_w = 1.0 - sum(w_cur.values())
            if cash_w > 1e-9:
                day_ret += cash_w * T.BOND_DAILY
            eq *= (1 + day_ret)
            equity[k + 1] = eq
            held = next((cd for cd in ORDER if w_cur[cd] > 1e-9), None)
        # T收盘: 决策 + 立即成交
        cands = [cd for cd in ORDER if t > 0 and SIG[cd][t] == 1 and SIG[cd][t - 1] == 1]
        nw = {cd: 0.0 for cd in ORDER}
        pick = max(cands, key=lambda cd: ADX[cd][t]) if cands else None
        if held is not None:
            vt, at = params[held]['vol_th'], params[held]['adx_th']
            sig_ok = (C[held][t] >= MA[held][t]) and (VOL[held][t] < vt or ADX[held][t] > at)
            if not sig_ok:
                pick = None
        if pick is not None:
            nw[pick] = P.BRAKE_W if (brake and VOL[pick][t] > P.BRAKE_TH) else 1.0
        for cd in ORDER:
            if abs(w_cur[cd] - nw[cd]) > 1e-9:
                trades += 1
                eq *= (1 - cost * abs(w_cur[cd] - nw[cd]))
        w_cur = dict(nw)
    return equity, trades


def gap_stats(aligned, i0, i1):
    """隔夜跳空统计: 普通隔夜 vs 周末(周五收盘→下一交易日开盘)"""
    rows = []
    for cd in ORDER:
        o = aligned[cd]['open'].values
        c = aligned[cd]['close'].values
        d = pd.DatetimeIndex(aligned[cd]['date'])
        gaps, wk = [], []
        for t in range(max(i0, 1), i1):
            g = (o[t] / c[t - 1] - 1) * 100
            gaps.append(g)
            if d[t].weekday() == 0:  # 周一开盘 = 跨周末
                wk.append(g)
        rows.append((cd, np.mean(gaps), np.std(gaps), min(gaps),
                     np.mean(wk), np.std(wk), min(wk), len(wk)))
    return rows


def friday_flip_stats(inds, i0, i1):
    """周五收盘信号翻转的成交差异: B=周五收盘卖/买 vs A=下一交易日开盘卖/买
    sell_diff: A卖价相对B的偏离(负=A卖得更差, 即周五尾盘卖出更优)
    buy_diff:  B买价相对A的偏离(负=B买得更贵)"""
    rows = []
    for cd in ORDER:
        o = inds[cd]['open']
        c = inds[cd]['close']
        d = pd.DatetimeIndex(inds[cd]['date'])
        sig = inds[cd]['signal']
        sell_diff, buy_diff = [], []
        for t in range(max(i0, 2), i1 - 1):
            if d[t].weekday() != 4:
                continue
            if sig[t] == 0 and sig[t - 1] == 1:      # 周五翻空 → 卖出
                sell_diff.append((o[t + 1] / c[t] - 1) * 100)
            elif sig[t] == 1 and sig[t - 1] == 1 and sig[t - 2] == 0:  # 周五确认成立 → 买入
                buy_diff.append((c[t] / o[t + 1] - 1) * 100)
        rows.append((cd, len(sell_diff), np.mean(sell_diff) if sell_diff else np.nan,
                     len(buy_diff), np.mean(buy_diff) if buy_diff else np.nan))
    return rows


def main():
    conn = duckdb.connect(P.HISTORY_DB, read_only=True)
    aligned, dates = P.load_aligned(conn)
    conn.close()
    params = P.base_params()
    inds = P.make_inds(aligned, params)
    i0 = int(np.searchsorted(pd.DatetimeIndex(dates).values, np.datetime64(P.START)))
    i1 = len(dates)

    print('=' * 100)
    print('  执行时序对比回测: A=次日开盘(现行) vs B=当日执行(盘中触发近似)')
    print('=' * 100)
    print(f"期间: {dates[i0].date()} ~ {dates[i1-1].date()} ({i1-i0} 个交易日) | 单边10bp | 2日确认+刹车")

    eqA, trA = P.simulate(inds, params, i0, i1)
    eqB, trB = simulate_sameday(inds, params, i0, i1)
    pA, pB = T.perf(eqA, i1 - i0), T.perf(eqB, i1 - i0)
    print(f"\nA 次日开盘(现行): {T.fmt_perf(pA)}  调仓{trA}次")
    print(f"B 当日执行(新):   {T.fmt_perf(pB)}  调仓{trB}次")
    diff_pp = (pB[0] - pA[0]) * 100
    verdict = '✅ 落地' if diff_pp >= -1.0 else '❌ 停止,重新讨论'
    print(f"\n决策门槛: B-A 年化差 {diff_pp:+.1f}pp (≥-1pp 落地) → {verdict}")

    print('\n--- 隔夜跳空统计 (%) ---')
    print(f"{'品种':<16}{'普通均值':>8}{'普通σ':>7}{'最差':>7}{'周末均值':>9}{'周末σ':>7}{'周末最差':>8}{'样本':>5}")
    for cd, m, s, w, wm, ws, wmin, n_wk in gap_stats(aligned, i0, i1):
        print(f"{cd} {P.ASSETS[cd]['name']:<9}{m:>8.2f}{s:>7.2f}{w:>7.2f}{wm:>9.2f}{ws:>7.2f}{wmin:>8.2f}{n_wk:>5}")

    print('\n--- 周五信号翻转的执行价差异 (A=下一交易日开盘 vs B=周五收盘, %) ---')
    for cd, ns, ms, nb, mb in friday_flip_stats(inds, i0, i1):
        print(f"{cd}: 卖出翻转{ns}次, A相对B平均{ms:+.2f}% (负=A卖得更差→周五尾盘卖出更优); "
              f"买入翻转{nb}次, B相对A平均{mb:+.2f}% (负=B买得更贵)")

    print(f"\n引擎自检: A口径应=validate_friction F0 (+17.8%/0.83/-16.2%, 235笔) → {T.fmt_perf(pA)} {trA}笔")

    out = os.path.join(PROJECT_ROOT, 'reports',
                       f"execution_timing_validation_{_dt.datetime.now().strftime('%Y%m%d')}.md")
    with open(out, 'w', encoding='utf-8') as f:
        f.write('# 执行时序对比回测 (validate_execution_timing.py)\n\n')
        f.write(f'- A 次日开盘(现行): {T.fmt_perf(pA)} / {trA}笔\n')
        f.write(f'- B 当日执行(盘中触发近似): {T.fmt_perf(pB)} / {trB}笔\n')
        f.write(f'- 决策门槛: B-A 年化差 {diff_pp:+.1f}pp → {verdict}\n')
    print(f'\n已保存: {out}')


if __name__ == '__main__':
    main()
