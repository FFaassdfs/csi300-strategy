# -*- coding: utf-8 -*-
"""
防御腿替换测试 (2026-09-18) — 银行ETF(512800) vs 红利低波ETF(512890) / 中证红利ETF(515080)
口径: 生产规则 R0(每日重选ADX最高) + 2日确认 + 波动刹车(35%/60%) + T+1开盘 + 单边10bp + 空仓国债2.5%
数据: 3只原品种读历史库; 红利类从新浪补齐并入库(增量)
"""
import os, sys, socket
socket.setdefaulttimeout(25)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import pandas as pd
import duckdb
from config import ASSETS, HISTORY_DB
from signal_core import compute_signal_core, adjust_splits

RF = 0.025
COST = 0.001
BRAKE_TH, BRAKE_W = 35.0, 0.6
BOND = (1 + RF) ** (1 / 252) - 1
START = '2020-04-30'

# 新增候选: 用防御品种模板参数 (与银行一致)
NEW = {
    '512890': {'name': '红利低波ETF', 'code': 'sh512890', 'ma_p': 30, 'adx_th': 25, 'vol_th': 18},
    '515080': {'name': '中证红利ETF', 'code': 'sh515080', 'ma_p': 30, 'adx_th': 25, 'vol_th': 18},
}


def ensure_data():
    """把候选品种历史补入库"""
    for cid, info in NEW.items():
        conn = duckdb.connect(HISTORY_DB)
        n = conn.execute('SELECT COUNT(*) FROM daily_ohlc WHERE code=?', [cid]).fetchone()[0]
        conn.close()
        if n > 0:
            continue
        import akshare as ak
        df = ak.fund_etf_hist_sina(symbol=info['code'])
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].dropna(subset=['close']).sort_values('date')
        adjust_splits(df)
        conn = duckdb.connect(HISTORY_DB)
        rows = [(r['date'].date(), cid, info['name'], float(r['open']), float(r['high']), float(r['low']),
                 float(r['close']), float(r.get('volume', 0)), float(r.get('amount', 0))) for _, r in df.iterrows()]
        conn.executemany('INSERT OR REPLACE INTO daily_ohlc VALUES (?,?,?,?,?,?,?,?,?)', rows)
        conn.close()
        print(f'  入库 {cid} {info["name"]}: {len(rows)} 条')


def load_pool(codes):
    conn = duckdb.connect(HISTORY_DB, read_only=True)
    dfs = {}
    for cd in codes:
        df = conn.execute("SELECT date, open, high, low, close FROM daily_ohlc WHERE code=? ORDER BY date", [cd]).fetchdf()
        df['date'] = pd.to_datetime(df['date'])
        dfs[cd] = df.set_index('date')
    conn.close()
    dates = sorted(set.intersection(*[set(dfs[c].index) for c in codes]))
    out = {}
    for cd in codes:
        info = ASSETS.get(cd) or NEW[cd]
        df = dfs[cd].loc[dates].reset_index()
        c, h, l = df['close'], df['high'], df['low']
        core = compute_signal_core(c, h, l, ma_p=info['ma_p'], adx_th=info['adx_th'], vol_th=info['vol_th'])
        out[cd] = {'open': df['open'].values, 'close': c.values, 'ma': core['ma'].values,
                   'vol': core['vol'].values, 'adx': core['adx'].values, 'signal': core['signal'].values,
                   'vol_th': info['vol_th'], 'adx_th': info['adx_th']}
    return out, dates


def simulate(inds, dates, codes, brake=True):
    O = {c: inds[c]['open'] for c in codes}
    C = {c: inds[c]['close'] for c in codes}
    MA = {c: inds[c]['ma'] for c in codes}
    VOL = {c: inds[c]['vol'] for c in codes}
    ADX = {c: inds[c]['adx'] for c in codes}
    SIG = {c: inds[c]['signal'] for c in codes}
    TH = {c: (inds[c]['vol_th'], inds[c]['adx_th']) for c in codes}
    i0 = int(np.searchsorted(pd.DatetimeIndex(dates).values, np.datetime64(START)))
    i1 = len(dates)
    n = i1 - i0
    equity = np.ones(n + 1); eq = 1.0
    w_cur = {c: 0.0 for c in codes}; w_tgt = {c: 0.0 for c in codes}
    held = None; trades = 0

    def conf(cd, t):
        return SIG[cd][t] == 1 and t > 0 and SIG[cd][t - 1] == 1

    for k in range(n):
        t = i0 + k
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
            vt, at = TH[held]
            if not ((C[held][t] >= MA[held][t]) and (VOL[held][t] < vt or ADX[held][t] > at)):
                pick = None
        nw = {c: 0.0 for c in codes}
        if pick is not None:
            bw = BRAKE_W if (brake and VOL[pick][t] > BRAKE_TH) else 1.0
            nw[pick] = bw
        w_tgt = nw

    rets = np.diff(equity) / equity[:-1]
    years = len(rets) / 252
    cagr = (equity[-1]) ** (1 / years) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252 - RF) / vol if vol > 1e-9 else 0
    peak = np.maximum.accumulate(equity)
    dd = ((equity - peak) / peak).min()
    return cagr, vol, sharpe, dd, trades, equity[-1]


print('补齐候选数据...')
ensure_data()
print()

POOLS = [
    ('V0 现行: 300+芯片+银行', ['510310', '159995', '512800']),
    ('V1 银行→红利低波(512890)', ['510310', '159995', '512890']),
    ('V2 银行→中证红利(515080)', ['510310', '159995', '515080']),
    ('V3 银行+红利低波(4品种)', ['510310', '159995', '512800', '512890']),
]

for brake in (True, False):
    print(f"### 波动刹车 {'开' if brake else '关'}   期间 {START}~最新")
    print('| 组合 | 年化 | 波动 | Sharpe | 最大回撤 | 调仓 | 期末净值 |')
    print('|---|---|---|---|---|---|---|')
    for name, codes in POOLS:
        inds, dates = load_pool(codes)
        cagr, vol, sharpe, dd, tr, nav = simulate(inds, dates, codes, brake)
        print(f"| {name} | {cagr:+.1%} | {vol:.1%} | {sharpe:.2f} | {dd:.1%} | {tr} | {nav:.2f}x |")
    print()
