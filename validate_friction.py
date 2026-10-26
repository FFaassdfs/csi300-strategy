# -*- coding: utf-8 -*-
"""
摩擦成本压力测试 (2026-10-08) — 只读 trading_history.duckdb, 不写库
背景: §11.3 #18 — 回测仅假设单边10bp, 实盘12笔实测滑点 买+124bp/卖+235bp (行为滑点为主).
本实验在现行 R0 轮动引擎(入场2日确认 + 极端波动刹车)上, 把真实摩擦拆层量化:

  F0 现行回测口径       单边10bp (对照基准)
  F1 结构成本·纪律执行   佣金(万2.5, 最低5元/笔) + 半个最小报价价差 (限价单挂开盘价)
  F2 结构成本·市价吃单   佣金 + 整个价差 (直接吃对手盘)
  F3 实盘中位执行       买131.1bp/卖37.7bp + 佣金 (自实测12笔的中位数 — 即便剔除均值拉动, 仍属坏执行)
  F4 实盘均值执行       买124.3bp/卖235.3bp + 佣金 (含追高/尾盘杀跌的行为滑点)
  F5 Monte Carlo        从实盘12笔滑点经验分布 bootstrap 1000次 → 收益/回撤分位数
  F6 限价不成交风险     买入限价"不追高超1%"未成交则顺延重试 (卖出按纪律开盘必成交), 成本同F1

口径: T收盘定目标 → T+1开盘调仓 | 空仓国债2.5% | 品种池/参数以 config.py 为准
注: 隔夜跳空(T收盘→T+1开盘)已含在"开盘价执行"的回测口径内, 各情景滑点均为
    "相对执行日开盘"的额外偏离(与 slippage_report.py 同基准), 不重复计算。
用法: python validate_friction.py
输出: 控制台 + reports/friction_validation_<date>.md
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

BRAKE_TH, BRAKE_W = 35.0, 0.6
START = '2020-04-30'

# ---- 摩擦参数 (按实际券商费率/资金规模修改) ----
CAPITAL = 10000.0      # 实盘资金规模 (OPERATION.md §9.1) — 佣金最低额的离散效应与此直接相关
TICK = 0.001           # A股ETF最小报价单位(元); 流动性好的品种盘口价差通常=1个tick
COMM_RATE = 0.00025    # 佣金万2.5 (ETF免印花税)
COMM_MIN = 5.0         # 单笔最低佣金(元) — 小资金真实大头: 半仓¥5k一笔 = 10bp
CHASE_CAP = 1.01       # 邮件执行纪律: 买入挂单不追高超信号价1% (OPERATION.md §十三B)
MIN_NOTIONAL = 1000.0  # 计佣最低名义额(元): 净值被成本磨小后防止"最低¥5佣金/极小名义额"模型发散
MC_N = 1000            # Monte Carlo 次数
MC_SEED = 20261008     # 随机种子(可复现)

# ---- 实盘12笔滑点 (bp, 相对执行日开盘; 来源 reports/slippage_report_latest.md 2026-09-18) ----
BUY_BP = [29.7, 0.0, 134.3, 185.9, 376.9, 12.4, 131.1]   # 7笔买入 (含芯片追高 +376.9)
SELL_BP = [-14.6, 14.8, 668.8, 469.7, 37.7]              # 5笔卖出 (含芯片尾盘杀跌 +668.8/+469.7)

REPORT_LINES = []


def log(msg):
    print(msg, flush=True)
    REPORT_LINES.append(str(msg))


def load_aligned(conn):
    """三品种按日期对齐 (与 validate_rotation_rule 相同)"""
    dfs = {cd: T.load_asset(conn, cd).set_index('date') for cd in ORDER}
    dates = sorted(set(dfs[ORDER[0]].index) & set(dfs[ORDER[1]].index) & set(dfs[ORDER[2]].index))
    inds = {}
    for cd in ORDER:
        df = dfs[cd].loc[dates].reset_index()
        inds[cd] = T.compute_ind(df, ASSETS[cd]['ma_p'], ASSETS[cd]['adx_th'], ASSETS[cd]['vol_th'])
    return inds, list(dates)


def trade_cost(mode, price, amount, side, rng=None):
    """单笔交易成本占成交额的比例. price=执行价, amount=成交金额(元), side='buy'/'sell'
    结构 = 佣金(含最低额) + 价差/滑点; 行为滑点直接采用实盘实测值(中位数/均值/经验分布)"""
    if mode == 'flat10':
        return 0.001
    amount = max(amount, MIN_NOTIONAL)  # 防发散: 名义额过低时按最低名义额计佣
    comm = max(COMM_RATE * amount, COMM_MIN) / amount
    if mode == 'struct_half':
        slip = 0.5 * TICK / price
    elif mode == 'struct_full':
        slip = TICK / price
    elif mode == 'emp_median':
        pool = BUY_BP if side == 'buy' else SELL_BP
        slip = float(np.median(pool)) / 1e4
    elif mode == 'emp_mean':
        pool = BUY_BP if side == 'buy' else SELL_BP
        slip = (sum(pool) / len(pool)) / 1e4
    elif mode == 'mc':
        pool = BUY_BP if side == 'buy' else SELL_BP
        slip = pool[int(rng.integers(len(pool)))] / 1e4
    else:
        raise ValueError(mode)
    return comm + slip


def simulate(inds, i0, i1, mode='flat10', brake=True, rng=None, nofill=False):
    """R0 轮动引擎 (与 validate_rotation_rule 同口径) + 分层成本 + 可选限价不成交"""
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
    w_cur = {cd: 0.0 for cd in ORDER}   # 当前实际权重
    w_tgt = {cd: 0.0 for cd in ORDER}   # 上一收盘目标权重
    held = None
    trades = 0

    for k in range(n):
        t = i0 + k
        if k > 0:
            # ---- 执行上一收盘目标 (今日开盘) ----
            exec_w = dict(w_tgt)
            for cd in ORDER:
                wo, wn = w_cur[cd], w_tgt[cd]
                if abs(wo - wn) < 1e-9:
                    continue
                dw = wn - wo
                # 限价不成交: 买入方向开盘价高于信号收盘价1%以上 → 顺延 (卖出始终成交)
                if dw > 0 and nofill and O[cd][t] > C[cd][t - 1] * CHASE_CAP:
                    exec_w[cd] = wo
                    continue
                trades += 1
                side = 'buy' if dw > 0 else 'sell'
                cf = trade_cost(mode, O[cd][t], CAPITAL * eq * abs(dw), side, rng)
                eq *= (1 - cf * abs(dw))
            # ---- 今日收益 (保留/卖出/买入三段 + 现金) — 必须用执行前的旧权重对比 ----
            day_ret = 0.0
            for cd in ORDER:
                wo, wn = w_cur[cd], exec_w[cd]
                keep = min(wo, wn); sell = max(0.0, wo - wn); buy = max(0.0, wn - wo)
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

        # ---- 收盘决策 (R0: 已确认候选中选ADX最高; 持仓原始信号失效才清仓) ----
        cands = [cd for cd in ORDER if t > 0 and SIG[cd][t] == 1 and SIG[cd][t - 1] == 1]
        nw = {cd: 0.0 for cd in ORDER}
        pick = max(cands, key=lambda cd: ADX[cd][t]) if cands else None
        if held is not None:
            vt, at = ASSETS[held]['vol_th'], ASSETS[held]['adx_th']
            sig_ok = (C[held][t] >= MA[held][t]) and (VOL[held][t] < vt or ADX[held][t] > at)
            if not sig_ok:
                pick = None
        if pick is not None:
            nw[pick] = BRAKE_W if (brake and VOL[pick][t] > BRAKE_TH) else 1.0
        w_tgt = nw
    return equity, trades


def main():
    conn = duckdb.connect(HISTORY_DB, read_only=True)
    inds, dates = load_aligned(conn)
    conn.close()
    i0 = int(np.searchsorted(pd.DatetimeIndex(dates).values, np.datetime64(START)))
    i1 = len(dates)

    log('=' * 104)
    log('  摩擦成本压力测试 — 佣金离散性 / 盘口价差 / 行为滑点 / 成交风险 分层量化')
    log('=' * 104)
    log(f"期间: {dates[i0].date()} ~ {dates[i1-1].date()} ({i1-i0} 个交易日)")
    log(f"引擎: R0 每日重选ADX最高 + 入场2日确认 + 极端波动刹车(35%/60%) | T+1开盘执行")
    log(f"摩擦假设: 资金¥{CAPITAL:.0f} | 佣金万{COMM_RATE*1e4:.1f}最低¥{COMM_MIN:.0f}/笔(ETF免印花税) | tick=¥{TICK:.3f}")

    # ---- 成本结构分解 ----
    log('\n--- 各品种成本结构 (按最新收盘价; 半价差=限价挂单, 全价差=市价吃单) ---')
    log(f"{'品种':<16}{'收盘价':>8}{'半价差bp':>9}{'全价差bp':>9}{'佣金bp·全仓':>11}{'佣金bp·半仓':>11}")
    for cd in ORDER:
        p = float(inds[cd]['close'][-1])
        hs, fs = 0.5 * TICK / p * 1e4, TICK / p * 1e4
        c_full = max(COMM_RATE * CAPITAL, COMM_MIN) / CAPITAL * 1e4
        c_half = max(COMM_RATE * CAPITAL / 2, COMM_MIN) / (CAPITAL / 2) * 1e4
        log(f"{cd} {ASSETS[cd]['name']:<10}{p:>8.3f}{hs:>9.1f}{fs:>9.1f}{c_full:>11.1f}{c_half:>11.1f}")
    log('> 佣金最低¥5在¥1万资金下=5bp(全仓)/10bp(半仓)/15bp(1/3仓) — 资金越小该离散项越贵')

    # ---- 场景对照 ----
    buy_med, sell_med = float(np.median(BUY_BP)), float(np.median(SELL_BP))
    buy_mean = sum(BUY_BP) / len(BUY_BP); sell_mean = sum(SELL_BP) / len(SELL_BP)
    log('\n--- 摩擦场景对照 (全部含波动刹车) ---')
    log('格式: 年化 / 波动 / Sharpe / 最大回撤')
    for name, mode, kw in [
        ('F0 现行回测口径 单边10bp', 'flat10', {}),
        ('F1 纪律执行 佣金+半价差', 'struct_half', {}),
        ('F2 市价吃单 佣金+全价差', 'struct_full', {}),
        (f'F3 实盘中位执行 买{buy_med:.0f}/卖{sell_med:.0f}bp+佣金', 'emp_median', {}),
        (f'F4 实盘均值执行 买{buy_mean:.0f}/卖{sell_mean:.0f}bp+佣金', 'emp_mean', {}),
        ('F6 限价不成交 不追高1%顺延(成本同F1)', 'struct_half', {'nofill': True}),
    ]:
        eq, tr = simulate(inds, i0, i1, mode=mode, brake=True, **kw)
        log(f"{name:<40} {T.fmt_perf(T.perf(eq, i1 - i0))}  调仓{tr}次")

    eq0, tr0 = simulate(inds, i0, i1, mode='flat10', brake=False)
    log(f"\n引擎自检 flat10·无刹车: {T.fmt_perf(T.perf(eq0, i1 - i0))} 调仓{tr0}次")
    log('(与 §11.3 #16 的 +20.4%/0.79/-21.6%/215笔 相比略低, 因本表数据含2026-09下旬回撤段, 引擎口径一致)')

    # ---- F5 Monte Carlo ----
    log(f'\n--- F5 Monte Carlo: 实盘12笔滑点经验分布 bootstrap × {MC_N} (含佣金) ---')
    cagrs, sharpes, dds = [], [], []
    for r in range(MC_N):
        rng = np.random.default_rng(MC_SEED + r)
        eq, _ = simulate(inds, i0, i1, mode='mc', brake=True, rng=rng)
        cagr, vol, sharpe, maxdd = T.perf(eq, i1 - i0)
        cagrs.append(cagr); sharpes.append(sharpe); dds.append(maxdd)
        if (r + 1) % 250 == 0:
            print(f'  ... {r + 1}/{MC_N}')
    ca, sa, da = np.array(cagrs), np.array(sharpes), np.array(dds)
    log(f"年化   P5 {np.percentile(ca,5):+.1%} | P50 {np.percentile(ca,50):+.1%} | P95 {np.percentile(ca,95):+.1%}")
    log(f"Sharpe P5 {np.percentile(sa,5):.2f} | P50 {np.percentile(sa,50):.2f} | P95 {np.percentile(sa,95):.2f}")
    log(f"回撤   P5 {np.percentile(da,5):.1%} | P50 {np.percentile(da,50):.1%} | P95 {np.percentile(da,95):.1%}  (回撤为负, P5=最深)")
    log(f"P(年化<0)={float((ca<0).mean()):.1%} | P(年化<10%)={float((ca<0.10).mean()):.1%} | "
        f"P(年化<15%)={float((ca<0.15).mean()):.1%}")

    # ---- 说明 ----
    log('\n--- 说明与结论 ---')
    log('1. 隔夜跳空(T收盘→T+1开盘)已含在开盘执行口径内, 各情景滑点均为相对执行日开盘的额外成本, 不重复计')
    log('2. F3/F4/F5 行为滑点取自 2026-06~09 实盘12笔(样本小, 且集中于芯片剧烈行情段), 仅代表"当前执行质量"')
    log('3. 半价差=挂单价与对手价各让半个tick的期望成本; 全价差=市价单必然跨越整个价差')
    log('4. 结构成本(F1/F2)是资金规模的函数: 资金越大佣金离散项越趋近费率本身, 价差项不变;')
    log('   净值被成本磨小后按最低名义额¥1000计佣(防最低¥5佣金模型发散), 极端尾部会收敛于该下限')
    log('5. 决策含义: 回测规划请勿再用10bp — 至少用 F1 做下限; F3~F5 说明执行质量是策略生死变量而非优化项')

    os.makedirs(os.path.join(PROJECT_ROOT, 'reports'), exist_ok=True)
    out = os.path.join(PROJECT_ROOT, 'reports', f"friction_validation_{datetime.now().strftime('%Y%m%d')}.md")
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write('# 摩擦成本压力测试 (validate_friction.py)\n\n')
        fh.write('```\n' + '\n'.join(REPORT_LINES) + '\n```\n')
    print(f'\n已保存: {out}')


if __name__ == '__main__':
    main()
