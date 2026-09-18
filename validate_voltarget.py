# -*- coding: utf-8 -*-
"""
波动率目标仓位实验 (Tier-2) — 只读历史库
思想: 二值0/100仓位 → 仓位 = min(100%, 目标波动 / 品种实际20日波动)
     高波动品种(芯片~45%)自动降权, 低波动品种(银行~15%)接近满仓
机制: T收盘定权重 → T+1开盘按权重差调仓(带5%再平衡带, 差额才计成本)
     切换品种仍按全额进出; 入场仍需2日确认; 空仓部分按国债计息
"""
import os
import sys
import numpy as np
import pandas as pd
import duckdb
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from config import ASSETS, HISTORY_DB, RF
from validate_tier1 import compute_ind, perf, fmt_perf, md_table, BOND_DAILY
from validate_expansion import align, COST

OUT = []


def log(msg):
    print(msg, flush=True)
    OUT.append(msg)


def run_rotation_vol(data, codes, i0, i1, cost=COST, target_vol=None, band=0.05, brake=None):
    """
    target_vol=None: 二值仓位基线(权重0/1)
    否则: 权重 = min(1, target_vol / 20日年化波动)  (vol为百分数, 如14.9=14.9%)
    brake=(vol阈值, 上限): 平时满仓, 仅当品种波动超阈值时降到上限 (极端波动刹车)
    返回 (equity, switches, rebalances, avg_w dict, avg_exposure)
    """
    O = {cd: data[cd]['open'] for cd in codes}
    C = {cd: data[cd]['close'] for cd in codes}
    MA = {cd: data[cd]['ma'] for cd in codes}
    VOL = {cd: data[cd]['vol'] for cd in codes}
    ADX = {cd: data[cd]['adx'] for cd in codes}
    SIG = {cd: data[cd]['signal'] for cd in codes}
    VT = {cd: data[cd]['vol_th'] for cd in codes}

    def target_weight(cd, t):
        v = VOL[cd][t]
        if not np.isfinite(v) or v <= 0:
            return 1.0
        if brake is not None:
            return brake[1] if v > brake[0] else 1.0
        if target_vol is None:
            return 1.0
        return min(1.0, target_vol * 100.0 / v)

    n = i1 - i0
    equity = np.ones(n + 1)
    eq, held, pending = 1.0, None, None
    w, just_in = 0.0, False
    w_planned = 0.0      # 昨日收盘定的今日权重
    w_target_new = 0.0   # 最近一次决策的入场权重
    switches = rebalances = 0
    w_sum, w_days = 0.0, 0
    w_asset_sum = {cd: 0.0 for cd in codes}

    for k in range(n):
        t = i0 + k
        day_ret = 0.0
        # ---- 开盘: 品种切换 ----
        if pending is not None and pending != held:
            if held is not None:
                day_ret += w * (O[held][t] / C[held][t - 1] - 1)
                eq *= (1 - cost * w)
                switches += 1
                held, w, just_in = None, 0.0, False
            if pending != 'FLAT':
                eq *= (1 - cost * w_target_new)
                held, w, just_in = pending, w_target_new, True
                switches += 1
            pending = None
        # ---- 开盘: 同品种再平衡 (超带才调, 差额计成本) ----
        elif held is not None and abs(w_planned - w) > band:
            eq *= (1 - cost * abs(w_planned - w))
            w = w_planned
            rebalances += 1
        # ---- 当日收益 ----
        if held is not None:
            core = (C[held][t] / O[held][t] - 1) if just_in else (C[held][t] / C[held][t - 1] - 1)
            day_ret += w * core + (1 - w) * BOND_DAILY
            w_sum += w
            w_asset_sum[held] += w
            w_days += 1
        else:
            day_ret += BOND_DAILY
        just_in = False
        eq *= (1 + day_ret)
        equity[k + 1] = eq
        # ---- T收盘决策 ----
        cands = [cd for cd in codes if SIG[cd][t] == 1 and SIG[cd][t - 1] == 1]
        pick = max(cands, key=lambda cd: ADX[cd][t]) if cands else None
        if held is not None:
            sig_ok = SIG[held][t] == 1 and (VOL[held][t] < VT[held] or ADX[held][t] > data[held]['adx_th'])
            if not sig_ok:
                pick = None
        w_target_new = target_weight(pick, t) if pick is not None else 0.0
        w_planned = w_target_new
        if pick != held:
            pending = pick if pick is not None else 'FLAT'
    avg_w = {cd: (w_asset_sum[cd] / w_days if w_days else 0) for cd in codes}
    avg_exposure = (w_sum / w_days) if w_days else 0.0
    return equity, switches, rebalances, avg_w, avg_exposure


def main():
    log(f'# 波动率目标仓位实验 ({datetime.now().strftime("%Y-%m-%d %H:%M")})')
    log(f'> 仓位=min(100%, 目标波动/20日实际波动) | 2日确认不变 | 单边成本{COST*10000:.0f}bp | 再平衡带5% | 空仓部分按国债{RF:.1%}计息')

    dfs, params_map = {}, {}
    conn = duckdb.connect(HISTORY_DB, read_only=True)
    for code, info in ASSETS.items():
        df = conn.execute("SELECT date, open, high, low, close FROM daily_ohlc WHERE code=? ORDER BY date", [code]).fetchdf()
        df['date'] = pd.to_datetime(df['date'])
        dfs[code] = df.reset_index(drop=True)
        params_map[code] = (info['ma_p'], info['adx_th'], info['vol_th'])
    conn.close()
    aligned, axis = align(dfs, params_map)
    i0, i1 = 1, len(axis)
    log(f'\n期间: {axis[0].date()} ~ {axis[-1].date()} ({i1}个交易日)')

    variants = [('V0 二值仓位(现行)', None, 0.05, None)] + \
               [(f'V{idx} 目标波动{tv}%', tv / 100, 0.05, None) for idx, tv in enumerate((10, 12, 15, 18, 20), start=1)] + \
               [('V6 目标波动15%+带宽10%', 0.15, 0.10, None),
                ('V7 极端波动刹车 vol>35%→60%', None, 0.05, (35, 0.6)),
                ('V8 极端波动刹车 vol>30%→50%', None, 0.05, (30, 0.5))]
    rows = []
    for name, tv, band, brake in variants:
        eq, sw, rb, avg_w, avg_exp = run_rotation_vol(aligned, list(ASSETS), i0, i1, target_vol=tv, band=band, brake=brake)
        p = perf(eq, i1 - 1)
        wtxt = ' '.join(f'{ASSETS[cd]["name"][:2]}:{avg_w[cd]:.0%}' for cd in ASSETS)
        rows.append([name, fmt_perf(p), sw, rb, f'{avg_exp:.0%}', wtxt, f'{eq[-1]:.2f}x'])
    log('\n'.join(md_table(rows, ['方案', '年化/波动/Sharpe/最大回撤', '切换次数', '再平衡次数', '平均仓位', '分品种平均权重', '期末净值'])))

    # 2026年以来子样本 (当前regime)
    idx26 = int(np.searchsorted(axis.values, np.datetime64('2026-01-01')))
    log(f'\n### 2026年以来子样本 ({axis[idx26].date()} ~ {axis[-1].date()})')
    rows = []
    for name, tv, band, brake in variants:
        eq, sw, rb, avg_w, avg_exp = run_rotation_vol(aligned, list(ASSETS), idx26, i1, target_vol=tv, band=band, brake=brake)
        p = perf(eq, i1 - idx26)
        rows.append([name, fmt_perf(p), sw + rb, f'{eq[-1]:.2f}x'])
    log('\n'.join(md_table(rows, ['方案', '年化/波动/Sharpe/最大回撤', '调仓次数', '期末净值'])))

    out_path = os.path.join(PROJECT_ROOT, 'reports', f'voltarget_validation_{datetime.now().strftime("%Y%m%d")}.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(OUT))
    log(f'\n报告已保存: {out_path}')


if __name__ == '__main__':
    main()
