# -*- coding: utf-8 -*-
"""
轮动选择规则对照实验 (#3) — 只读 trading_history.duckdb, 不写库
R0 = 现行回测引擎口径 (tier1 run_rotation): 每日在"已确认候选"中选 ADX 最高者全额持有;
     持仓品种信号失效 → 清仓; 若另一候选 ADX 更高 → 切换
R1 = 等权: 已确认候选 1/N 等权 (集合变化时才调仓)
R2 = 持仓优先 sticky (与 send_advice 邮件实盘逻辑一致): 持仓品种原始 signal=1 就保留,
     信号消失才切到候选中 ADX 最高者 (新入场仍需连续2日确认)
口径: T收盘定目标权重 → T+1开盘调仓 | 单边成本10bp | 空仓按国债2.5% | 入场2日确认
"""
import os, sys
import numpy as np
import pandas as pd
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import validate_tier1 as T
from config import ASSETS, HISTORY_DB

ORDER = ['510310', '159995', '512800']
COST = 0.001
BRAKE_TH = 35.0
BRAKE_W = 0.6
CONFIRM = 2
START = '2020-04-30'


def load_aligned(conn):
    dfs = {cd: T.load_asset(conn, cd).set_index('date') for cd in ORDER}
    dates = sorted(set(dfs[ORDER[0]].index) & set(dfs[ORDER[1]].index) & set(dfs[ORDER[2]].index))
    inds = {}
    for cd in ORDER:
        df = dfs[cd].loc[dates].reset_index()
        inds[cd] = T.compute_ind(df, ASSETS[cd]['ma_p'], ASSETS[cd]['adx_th'], ASSETS[cd]['vol_th'])
    return inds, list(dates)


def simulate(inds, i0, i1, rule, brake, cost=COST, caps=None, min_w=0.0):
    O = {cd: inds[cd]['open'] for cd in ORDER}
    C = {cd: inds[cd]['close'] for cd in ORDER}
    MA = {cd: inds[cd]['ma'] for cd in ORDER}
    VOL = {cd: inds[cd]['vol'] for cd in ORDER}
    ADX = {cd: inds[cd]['adx'] for cd in ORDER}
    SIG = {cd: inds[cd]['signal'] for cd in ORDER}
    TH = {cd: (inds[cd]['vol_th'], inds[cd]['adx_th']) for cd in ORDER}

    n = i1 - i0
    equity = np.ones(n + 1)
    eq = 1.0
    w_cur = {cd: 0.0 for cd in ORDER}   # 当前实际持有权重
    w_tgt = {cd: 0.0 for cd in ORDER}   # 上一收盘决定的目标权重
    held = None
    trades, turnover = 0, 0.0
    caps = caps or {}

    def cap_of(cd):
        """品种权重上限; min_w>0 时启用下限(择时品种空仓时也保留底仓)"""
        return float(caps.get(cd, 1.0))

    def conf(cd, t):
        return SIG[cd][t] == 1 and t > 0 and SIG[cd][t - 1] == 1

    for k in range(n):
        t = i0 + k
        # ---- 执行上一收盘目标 (今日开盘) ----
        if k > 0:
            for cd in ORDER:
                wo, wn = w_cur[cd], w_tgt[cd]
                if abs(wo - wn) < 1e-9:
                    continue
                trades += 1
                turnover += abs(wo - wn)
                eq *= (1 - cost * abs(wo - wn))
            # ---- 今日收益 (保留/卖出/买入三段 + 现金) ----
            day_ret = 0.0
            for cd in ORDER:
                wo, wn = w_cur[cd], w_tgt[cd]
                keep = min(wo, wn); sell = max(0.0, wo - wn); buy = max(0.0, wn - wo)
                if keep > 1e-9:
                    day_ret += keep * (C[cd][t] / C[cd][t - 1] - 1)
                if sell > 1e-9:
                    day_ret += sell * (O[cd][t] / C[cd][t - 1] - 1)
                if buy > 1e-9:
                    day_ret += buy * (C[cd][t] / O[cd][t] - 1)
            cash_w = 1.0 - sum(w_tgt.values())
            if cash_w > 1e-9:
                day_ret += cash_w * T.BOND_DAILY
            w_cur = dict(w_tgt)
            eq *= (1 + day_ret)
            equity[k + 1] = eq
            held = next((cd for cd in ORDER if w_cur[cd] > 1e-9), None)

        # ---- 收盘决策 → 次日开盘执行 ----
        cands = [cd for cd in ORDER if conf(cd, t)]
        nw = {cd: 0.0 for cd in ORDER}
        if rule == 'R1':
            for cd in cands:
                bw = BRAKE_W if (brake and VOL[cd][t] > BRAKE_TH) else 1.0
                nw[cd] = bw / len(cands)
        else:
            pick = None
            if rule == 'R2' and held is not None and SIG[held][t] == 1:
                pick = held
            elif cands:
                pick = max(cands, key=lambda cd: ADX[cd][t])
            if rule == 'R0' and held is not None:
                vt, at = TH[held]
                sig_ok = (C[held][t] >= MA[held][t]) and (VOL[held][t] < vt or ADX[held][t] > at)
                if not sig_ok:
                    pick = None
            if pick is not None:
                bw = BRAKE_W if (brake and VOL[pick][t] > BRAKE_TH) else 1.0
                nw[pick] = min(cap_of(pick), bw)
        w_tgt = nw
    return equity, trades, turnover


def main():
    conn = duckdb.connect(HISTORY_DB, read_only=True)
    inds, dates = load_aligned(conn)
    conn.close()

    i0 = int(np.searchsorted(pd.DatetimeIndex(dates).values, np.datetime64(START)))
    i1 = len(dates)
    print(f"期间: {dates[i0].date()} ~ {dates[i1-1].date()} ({i1-i0} 个交易日)")
    print("口径: T收盘定目标 → T+1开盘调仓 | 单边10bp | 空仓国债2.5% | 入场连续2日确认")
    print("个股 brake = 20日年化波动>35% 时该品种权重上限 0.6\n")

    rows = []
    for rule, name in [('R0', 'R0 回测引擎(每日重选ADX最高)'),
                       ('R1', 'R1 等权(候选1/N)'),
                       ('R2', 'R2 持仓优先sticky(=邮件逻辑)')]:
        for brake in (False, True):
            eq, trades, turnover = simulate(inds, i0, i1, rule, brake)
            p = T.perf(eq, i1 - i0)
            tag = 'brake' if brake else '  无刹车'
            rows.append((name, tag, T.fmt_perf(p), trades, f"{turnover:.0f}", f"{eq[-1]:.2f}x"))
            print(f"{name:<30} [{tag}] {T.fmt_perf(p)}  调仓{trades}次 换手{turnover:.0f} 净值{eq[-1]:.2f}x")

    print("\n引擎校验参照 (tier1 E4 连续2日确认, 无刹车): +20.4% / 24.2% / 0.79 / -21.6%, 215笔")


if __name__ == '__main__':
    main()
