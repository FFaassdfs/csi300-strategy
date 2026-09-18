# -*- coding: utf-8 -*-
"""
品种池扩展 + 策略对比实验 (只读现有库, 候选品种在线拉取, 不写库)
E-A 候选品种准入筛选: 黄金/纳指/国债/创业板/有色/酒, 用统一参数+2日确认跑单品种回测
E-B 轮动池对比: 现行3品种 vs 扩池, 含"空仓持国债ETF"变体
E-C 双动量对比 (60日/20日动量排名 + 绝对动量门槛)
成本: 单边10bp; 执行: T收盘决策/T+1开盘; 新品种统一参数 MA30/ADX25/Vol15 (避免为新品种单独调参过拟合)
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
from signal_core import compute_signal_core
from validate_tier1 import compute_ind, perf, fmt_perf, md_table, BOND_DAILY

COST = 0.001
WARMUP = 80
NEW_PARAMS = (30, 25, 15)

CANDIDATES = {
    '518880': {'name': '黄金ETF',    'code': 'sh518880'},
    '513100': {'name': '纳指ETF',    'code': 'sh513100'},
    '511260': {'name': '国债ETF',    'code': 'sh511260'},
    '159915': {'name': '创业板ETF',  'code': 'sz159915'},
    '512400': {'name': '有色金属ETF', 'code': 'sh512400'},
    '512690': {'name': '酒ETF',      'code': 'sh512690'},
}

OUT = []


def log(msg):
    print(msg, flush=True)
    OUT.append(msg)


def fetch_sina(sym):
    import akshare as ak
    import socket
    socket.setdefaulttimeout(20)
    df = ak.fund_etf_hist_sina(symbol=sym)
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    c = df['close'].values
    for i in range(1, len(c)):
        if c[i] > 0 and c[i-1] > 0 and c[i-1] / c[i] > 1.8:
            ratio = round(c[i-1] / c[i])
            for col in ['open', 'high', 'low', 'close']:
                df.loc[df.index[:i], col] = df.loc[df.index[:i], col] / ratio
            break
    return df


def align(dfs, params_map):
    dates = None
    for code, df in dfs.items():
        s = set(df['date'])
        dates = s if dates is None else (dates & s)
    axis = pd.DatetimeIndex(sorted(dates))
    axis = axis[axis >= axis[0] + pd.Timedelta(days=WARMUP)]
    aligned = {}
    for code, df in dfs.items():
        ind = compute_ind(df, *params_map[code])
        pos = df.set_index('date').index.get_indexer(axis)
        a = dict(ind)
        for key in ('open', 'close', 'ma', 'vol', 'adx', 'atr', 'signal'):
            a[key] = ind[key][pos]
        aligned[code] = a
    return aligned, axis


def run_rotation2(data, codes, i0, i1, cost=COST, confirm_days=2, flat_code=None):
    """ADX轮动 (含2日确认); flat_code不为空时, 无信号品种改为持有该品种(如国债ETF)"""
    look = list(codes) + ([flat_code] if flat_code and flat_code in data else [])
    O = {cd: data[cd]['open'] for cd in look}
    C = {cd: data[cd]['close'] for cd in look}
    MA = {cd: data[cd]['ma'] for cd in look}
    VOL = {cd: data[cd]['vol'] for cd in look}
    ADX = {cd: data[cd]['adx'] for cd in look}
    SIG = {cd: data[cd]['signal'] for cd in look}
    VT = {cd: data[cd]['vol_th'] for cd in look}

    n = i1 - i0
    equity = np.ones(n + 1)
    eq, held, pending, just_in = 1.0, None, None, False
    trades, wins, t0 = 0, 0, 1.0
    for k in range(n):
        t = i0 + k
        day_ret = 0.0
        if pending is not None and pending != held:
            if held is not None:
                day_ret += O[held][t] / C[held][t - 1] - 1
                eq *= (1 - cost)
                if eq > t0:
                    wins += 1
                trades += 1
                held = None
            if pending != 'FLAT':
                eq *= (1 - cost)
                held, just_in = pending, True
                t0 = eq
                trades += 1
        pending = None
        if held is not None:
            day_ret += C[held][t] / O[held][t] - 1 if just_in else C[held][t] / C[held][t - 1] - 1
        else:
            day_ret += BOND_DAILY
        just_in = False
        eq *= (1 + day_ret)
        equity[k + 1] = eq
        cands = []
        for cd in codes:
            if SIG[cd][t] == 1 and SIG[cd][t - 1] == 1:  # 2日确认
                cands.append(cd)
        pick = max(cands, key=lambda cd: ADX[cd][t]) if cands else (flat_code or 'FLAT')
        if held is not None and held != flat_code:
            sig_ok = (VOL[held][t] < VT[held] or ADX[held][t] > data[held]['adx_th']) and SIG[held][t] == 1
            if not sig_ok:
                pick = flat_code or 'FLAT'
        if pick != held:
            pending = pick
    return equity, trades, (wins / trades if trades else 0)


def run_dm(data, codes, i0, i1, mom=60, cost=COST, flat_code=None, margin=0.02, freq=1):
    """双动量: 持有mom日动量最强且跑赢国债的品种, 否则空仓(或flat_code)
    margin: 挑战者动量需超过持有者margin才切换 (降低换手); freq: 每《freq》个交易日才做一次决策"""
    look = list(codes) + ([flat_code] if flat_code and flat_code in data else [])
    C = {cd: data[cd]['close'] for cd in look}
    O = {cd: data[cd]['open'] for cd in look}
    n = i1 - i0
    equity = np.ones(n + 1)
    eq, held, pending, just_in = 1.0, None, None, False
    trades, wins, t0 = 0, 0, 1.0
    bond_s = (1 + RF) ** (mom / 252) - 1
    for k in range(n):
        t = i0 + k
        day_ret = 0.0
        if pending is not None and pending != held:
            if held is not None:
                day_ret += O[held][t] / C[held][t - 1] - 1
                eq *= (1 - cost)
                if eq > t0:
                    wins += 1
                trades += 1
                held = None
            if pending != 'FLAT':
                eq *= (1 - cost)
                held, just_in = pending, True
                t0 = eq
                trades += 1
        pending = None
        if held is not None:
            day_ret += C[held][t] / O[held][t] - 1 if just_in else C[held][t] / C[held][t - 1] - 1
        else:
            day_ret += BOND_DAILY
        just_in = False
        eq *= (1 + day_ret)
        equity[k + 1] = eq
        if k % freq != 0 or t - mom < 0:
            continue
        best, best_s = None, -9e9
        for cd in codes:
            s = C[cd][t] / C[cd][t - mom] - 1
            if s > best_s:
                best_s, best = s, cd
        if held is None or held == flat_code:
            pick = best if best_s > bond_s else (flat_code or 'FLAT')
        else:
            held_s = C[held][t] / C[held][t - mom] - 1
            if best != held and best_s > held_s + margin:
                pick = best
            elif held_s > bond_s:
                pick = held
            else:
                pick = flat_code or 'FLAT'
        if pick != held:
            pending = pick
    return equity, trades, (wins / trades if trades else 0)


def main():
    log(f'# 品种池扩展 + 策略对比实验 ({datetime.now().strftime("%Y-%m-%d %H:%M")})')
    log(f'> 成本单边{COST*10000:.0f}bp | 执行T+1开盘 | 新品种统一参数MA30/ADX25/Vol15 | 现行3品种用各自生产参数')

    # ---- 拉数据 ----
    dfs, params_map = {}, {}
    conn = duckdb.connect(HISTORY_DB, read_only=True)
    for code, info in ASSETS.items():
        df = conn.execute("SELECT date, open, high, low, close FROM daily_ohlc WHERE code=? ORDER BY date", [code]).fetchdf()
        df['date'] = pd.to_datetime(df['date'])
        dfs[code] = df.reset_index(drop=True)
        params_map[code] = (info['ma_p'], info['adx_th'], info['vol_th'])
    conn.close()
    for code, info in CANDIDATES.items():
        try:
            df = fetch_sina(info['code'])
            dfs[code] = df
            params_map[code] = NEW_PARAMS
            log(f'[OK] {info["name"]}({code}): {df["date"].iloc[0].date()} ~ {df["date"].iloc[-1].date()} ({len(df)}行)')
        except Exception as e:
            log(f'[FAIL] {info["name"]}({code}): {e}')
    avail = [c for c in list(ASSETS) + list(CANDIDATES) if c in dfs]

    # ---- E-A 候选品种准入筛选 ----
    log('\n## E-A 候选品种准入筛选 (单品种ADX Override + 2日确认, 全样本)')
    rows = []
    for code in avail:
        name = ASSETS[code]['name'] if code in ASSETS else CANDIDATES[code]['name']
        ind = compute_ind(dfs[code], *params_map[code])
        i0, i1 = WARMUP, len(dfs[code])
        eq, tr, wr = (None, 0, 0)
        try:
            from validate_tier1 import run_single
            eq, tr, wr = run_single(ind, i0, i1, cost=COST, confirm_days=2)
        except Exception as e:
            log(f'[WARN] {code}: {e}')
            continue
        p = perf(eq, i1 - i0)
        bh = ind['close'][i0:i1] / ind['open'][i0]
        p_bh = perf(np.r_[1.0, bh], i1 - i0)
        # 2026年以来子样本
        idx26 = int(np.searchsorted(dfs[code]['date'].values, np.datetime64('2026-01-01')))
        p26, tr26 = None, 0
        if idx26 > WARMUP and idx26 < i1:
            eq26, tr26, _ = (None, 0, 0)
            from validate_tier1 import run_single as rs
            eq26, tr26, _ = rs(ind, idx26, i1, cost=COST, confirm_days=2)
            p26 = perf(eq26, i1 - idx26)
        rows.append([f'{name}({code})', f"{dfs[code]['date'].iloc[0].date()}",
                     fmt_perf(p), tr, f'{wr:.0%}', f'{p_bh[0]:+.1%}',
                     fmt_perf(p26) if p26 else '-', tr26])
    log('\n'.join(md_table(rows, ['品种', '数据起点', '全样本 年化/波动/Sharpe/回撤', '交易数', '胜率', 'B&H年化', '2026年以来', '26交易'])))
    log('> 准入参考线: 全样本Sharpe≥0.6 且 2026年以来不塌方; 高波动品种须同时看回撤')

    # ---- E-B/E-C 轮动池对比 ----
    all_codes = [c for c in avail if c != '511260']
    aligned, axis = align(dfs, params_map)
    i0, i1 = 0, len(axis)
    log(f'\n## E-B/E-C 轮动池与策略对比  期间: {axis[0].date()} ~ {axis[-1].date()} ({i1}个交易日)')
    variants = [
        ('V0 现行3品种(ADX)', run_rotation2, [c for c in ASSETS], None, {}),
        ('V1 3品种+空仓持国债ETF', run_rotation2, [c for c in ASSETS], '511260', {}),
        ('V2 3+黄金(ADX)', run_rotation2, [c for c in ASSETS] + ['518880'], None, {}),
        ('V3 3+黄金+纳指(ADX)', run_rotation2, [c for c in ASSETS] + ['518880', '513100'], None, {}),
        ('V4 全池6+国债垫底(ADX)', run_rotation2, all_codes, '511260' if '511260' in aligned else None, {}),
        ('V5 双动量60日 日频', run_dm, all_codes, None, {}),
        ('V6 双动量60日+2%滞后', run_dm, all_codes, '511260' if '511260' in aligned else None, {'margin': 0.02}),
        ('V7 双动量60日 周频+国债垫底', run_dm, all_codes, '511260' if '511260' in aligned else None, {'freq': 5}),
        ('V8 双动量60日 月频+国债垫底', run_dm, all_codes, '511260' if '511260' in aligned else None, {'freq': 21}),
    ]
    rows = []
    for name, fn, codes, flat, kw in variants:
        codes = [c for c in codes if c in aligned]
        try:
            eq, tr, wr = fn(aligned, codes, 1, i1, flat_code=flat, **kw)
            p = perf(eq, i1 - 1)
            rows.append([name, fmt_perf(p), tr, f'{wr:.0%}', f'{eq[-1]:.2f}x'])
        except Exception as e:
            rows.append([name, f'ERROR: {e}', '-', '-', '-'])
    log('\n'.join(md_table(rows, ['方案', '年化/波动/Sharpe/最大回撤', '交易数', '胜率', '期末净值'])))

    out_path = os.path.join(PROJECT_ROOT, 'reports', f'expansion_validation_{datetime.now().strftime("%Y%m%d")}.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(OUT))
    log(f'\n报告已保存: {out_path}')


if __name__ == '__main__':
    main()
