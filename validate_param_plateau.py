# -*- coding: utf-8 -*-
"""
参数平台敏感度扫描 (2026-10-08) — 只读 trading_history.duckdb, 不写库
问题: 现行参数 (config.py: 510310 MA30/ADX20/Vol18, 159995 MA30/ADX25/Vol15, 512800 MA30/ADX25/Vol18)
      是站在"参数平台"上还是"参数针尖"上?
      (tier1 E1 walk-forward 已验证无过拟合, 本实验补"平台视角": 参数邻域的成绩地形)
方法: 在现行 R0 轮动引擎(入场2日确认 + 极端波动刹车, 单边10bp)上
  P1 单参数扰动 OAT: 每品种逐个拧 MA/ADX阈值/波动率阈值, 其余保持现行值 → 年化/Sharpe/回撤 + 极差
  P2 ADX×Vol 联合热力图 (MA固定30): 检验"波动率主闸门 + ADX开门"这对核心参数的交互面
判读: 邻域成绩平滑变化 = 平台(参数鲁棒, 规律可信); 尖峰暴跌 = 针尖(过拟合嫌疑)
注意: 拧单品种参数会同时改变 其信号 与 三品种间 ADX 竞争(选谁持有), 是系统级敏感度
口径: T收盘定目标 → T+1开盘 | 空仓国债2.5% | 与 validate_friction.py F0 完全一致
      (数据含2026-09下旬, 基准≈+15.3%/0.72/-18.4%, 与§11.3 #16 略低是区间差异)
用法: python validate_param_plateau.py
输出: 控制台 + reports/param_plateau_validation_<date>.md
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import validate_tier1 as T
from config import ASSETS, HISTORY_DB, ORDER

COST = 0.001           # 单边10bp (与 friction F0 一致)
BRAKE_TH, BRAKE_W = 35.0, 0.6
START = '2020-04-30'

MA_LIST = [20, 25, 30, 35, 40, 50]     # MA周期扫描
ADX_LIST = [15, 20, 25, 30]            # ADX阈值扫描
VOL_LIST = [12, 15, 18, 22]            # 波动率阈值扫描
PLATEAU_PP = 3.0                       # 年化极差 ≤3pp 视为平台

REPORT_LINES = []


def log(msg):
    print(msg, flush=True)
    REPORT_LINES.append(str(msg))


def base_params():
    return {cd: {'ma_p': ASSETS[cd]['ma_p'], 'adx_th': ASSETS[cd]['adx_th'],
                 'vol_th': ASSETS[cd]['vol_th']} for cd in ORDER}


def load_aligned(conn):
    dfs = {cd: T.load_asset(conn, cd).set_index('date') for cd in ORDER}
    dates = sorted(set(dfs[ORDER[0]].index) & set(dfs[ORDER[1]].index) & set(dfs[ORDER[2]].index))
    aligned = {cd: dfs[cd].loc[dates].reset_index() for cd in ORDER}
    return aligned, list(dates)


def make_inds(aligned, params):
    return {cd: T.compute_ind(aligned[cd], params[cd]['ma_p'], params[cd]['adx_th'],
                              params[cd]['vol_th']) for cd in ORDER}


def simulate(inds, params, i0, i1, cost=COST, brake=True):
    """R0 轮动引擎 (与 validate_friction 相同, 阈值改为按 params 品种级取用)"""
    A = {cd: {k: inds[cd][k].tolist() for k in ('open', 'close', 'ma', 'vol', 'adx', 'signal')}
         for cd in ORDER}
    O = {cd: A[cd]['open'] for cd in ORDER}
    C = {cd: A[cd]['close'] for cd in ORDER}
    MA = {cd: A[cd]['ma'] for cd in ORDER}
    VOL = {cd: A[cd]['vol'] for cd in ORDER}
    ADX = {cd: A[cd]['adx'] for cd in ORDER}
    SIG = {cd: A[cd]['signal'] for cd in ORDER}

    n = i1 - i0
    equity = np.ones(n + 1)
    eq = 1.0
    w_cur = {cd: 0.0 for cd in ORDER}
    w_tgt = {cd: 0.0 for cd in ORDER}
    held = None
    trades = 0
    for k in range(n):
        t = i0 + k
        if k > 0:
            exec_w = dict(w_tgt)
            day_ret = 0.0
            for cd in ORDER:
                wo, wn = w_cur[cd], exec_w[cd]   # 注意: 与"旧权重"比, 不能先把 w_cur 覆盖成目标
                keep = min(wo, wn); sell = max(0.0, wo - wn); buy = max(0.0, wn - wo)
                if abs(wo - wn) > 1e-9:
                    trades += 1
                    eq *= (1 - cost * abs(wo - wn))
                if keep > 1e-9:
                    day_ret += keep * (C[cd][t] / C[cd][t - 1] - 1)
                if sell > 1e-9:
                    day_ret += sell * (O[cd][t] / C[cd][t - 1] - 1)
                if buy > 1e-9:
                    day_ret += buy * (C[cd][t] / O[cd][t] - 1)
            cash_w = 1.0 - sum(exec_w.values())
            if cash_w > 1e-9:
                day_ret += cash_w * T.BOND_DAILY
            eq *= (1 + day_ret)
            w_cur = exec_w
            equity[k + 1] = eq
            held = next((cd for cd in ORDER if w_cur[cd] > 1e-9), None)

        cands = [cd for cd in ORDER if t > 0 and SIG[cd][t] == 1 and SIG[cd][t - 1] == 1]
        nw = {cd: 0.0 for cd in ORDER}
        pick = max(cands, key=lambda cd: ADX[cd][t]) if cands else None
        if held is not None:
            vt, at = params[held]['vol_th'], params[held]['adx_th']
            sig_ok = (C[held][t] >= MA[held][t]) and (VOL[held][t] < vt or ADX[held][t] > at)
            if not sig_ok:
                pick = None
        if pick is not None:
            nw[pick] = BRAKE_W if (brake and VOL[pick][t] > BRAKE_TH) else 1.0
        w_tgt = nw
    return equity, trades


def run(aligned, params, i0, i1):
    inds = make_inds(aligned, params)
    eq, tr = simulate(inds, params, i0, i1)
    cagr, vol, sharpe, maxdd = T.perf(eq, i1 - i0)
    return cagr, sharpe, maxdd, tr


def fmt(r):
    return f"{r[0]:+.1%} / {r[1]:.2f} / {r[2]:.1%}"


def main():
    conn = duckdb.connect(HISTORY_DB, read_only=True)
    aligned, dates = load_aligned(conn)
    conn.close()
    i0 = int(np.searchsorted(pd.DatetimeIndex(dates).values, np.datetime64(START)))
    i1 = len(dates)

    log('=' * 104)
    log('  参数平台敏感度扫描 — 现行参数邻域地形 (平台 or 针尖?)')
    log('=' * 104)
    log(f"期间: {dates[i0].date()} ~ {dates[i1-1].date()} ({i1-i0} 个交易日) | 单边10bp | 2日确认+波动刹车")

    base = base_params()
    base_res = run(aligned, base, i0, i1)
    log(f"\n基准(现行参数): {fmt(base_res)}  调仓{base_res[3]}次")
    log('(数据含2026-09下旬回撤段, 故低于§11.3 #16 的+17.2%; 引擎口径一致)\n')

    # ---- P1 单参数扰动 ----
    log('--- P1 单参数扰动 OAT (其余参数保持现行值) ---')
    verdicts = []
    for cd in ORDER:
        p = base[cd]
        log(f"\n{cd} {ASSETS[cd]['name']} (现行 MA{p['ma_p']}/ADX{p['adx_th']}/Vol{p['vol_th']})")
        cur = f"MA{p['ma_p']}/ADX{p['adx_th']}/Vol{p['vol_th']}"
        for key, label, values in [('ma_p', 'MA', MA_LIST), ('adx_th', 'ADX', ADX_LIST),
                                   ('vol_th', 'Vol', VOL_LIST)]:
            row = []
            for v in values:
                q = base_params(); q[cd][key] = v
                if key == 'ma_p':
                    tag = f"MA{v}"
                elif key == 'adx_th':
                    tag = f"ADX{v}"
                else:
                    tag = f"Vol{v}"
                r = run(aligned, q, i0, i1)
                star = '★' if v == p[key] else ' '
                row.append((v, r, star))
                log(f"  {label:<4}{tag:<6}{star} {fmt(r)}  调仓{r[3]}")
            cagrs = [x[1][0] for x in row]
            spread = (max(cagrs) - min(cagrs)) * 100
            tag = '平台✓' if spread <= PLATEAU_PP else ('一般~' if spread <= 8 else '敏感✗')
            verdicts.append((cd, label, spread, tag))
            log(f"  → {label} 极差 {spread:.1f}pp [{tag}]")

    # ---- P2 ADX×Vol 热力图 ----
    log('\n--- P2 ADX×Vol 联合热力图 (年化收益, MA固定30, ★=现行) ---')
    for cd in ORDER:
        p = base[cd]
        log(f"\n{cd} {ASSETS[cd]['name']} (现行 ADX{p['adx_th']}/Vol{p['vol_th']})")
        log('       ' + ''.join(f"{'Vol'+str(v):>10}" for v in VOL_LIST))
        for at in ADX_LIST:
            cells = []
            for vt in VOL_LIST:
                q = base_params(); q[cd].update(ma_p=30, adx_th=at, vol_th=vt)
                r = run(aligned, q, i0, i1)
                star = '★' if (at == p['adx_th'] and vt == p['vol_th']) else ' '
                cells.append(f"{r[0]:+8.1%}{star}")
            log(f"  ADX{at:<3}" + ''.join(f"{c:>10}" for c in cells))

    # ---- 汇总判读 ----
    log('\n--- 平台判定汇总 (年化极差, ≤3pp=平台 / ≤8pp=一般 / >8pp=敏感) ---')
    for cd, label, spread, tag in verdicts:
        log(f"{cd} {label:<4} 极差{spread:5.1f}pp  [{tag}]")

    log('\n--- 说明 ---')
    log('1. OAT 只拧单品种单参数, 其余品种保持现行 → 反映系统级敏感度(含三品种ADX竞争的连带效应)')
    log('2. 热力图 MA 固定30: MA扫描见 P1; ADX为横纵轴是本策略的核心机制(波动率闸门+ADX开门)')
    log('3. 与 tier1 E1 互证: E1 验"训练期最优参数盲测是否失效"(防过拟合), 本实验验"现行参数邻域是否平缓"(防针尖)')
    log('4. 平台≠建议改参数: 若现行值已在平台上且邻域无显著更优点, 维持现状(§十三D 勿随意改参数)')

    os.makedirs(os.path.join(PROJECT_ROOT, 'reports'), exist_ok=True)
    out = os.path.join(PROJECT_ROOT, 'reports',
                       f"param_plateau_validation_{datetime.now().strftime('%Y%m%d')}.md")
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write('# 参数平台敏感度扫描 (validate_param_plateau.py)\n\n')
        fh.write('```\n' + '\n'.join(REPORT_LINES) + '\n```\n')
    print(f'\n已保存: {out}')


if __name__ == '__main__':
    main()
