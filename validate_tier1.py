# -*- coding: utf-8 -*-
"""
第一档验证实验 (Tier-1 Validation) — 只读历史库, 不动实盘
E1 Walk-forward 样本外参数验证 (检验过拟合)
E2 置信度分桶有效性 (强/中/弱是否真能区分未来收益)
E3 成本敏感性 (单边 0/5/10/20bp)
E4 防打脸过滤器 A/B (连续2日确认 / MA±1%缓冲带 / ADX上行)
E5 ATR 灾难止损 (2x / 2.5x / 3x)

回测规则与实盘一致: T日收盘决策 → T+1开盘执行, 空仓期按无风险利率计息
数据: trading_history.duckdb (只读)   信号: signal_core (与生产一致)
输出: 控制台 + reports/tier1_validation_<date>.md
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
from signal_core import compute_signal_core, compute_confidence

BOND_DAILY = (1 + RF) ** (1 / 252) - 1
WARMUP = 80
REPORT_LINES = []


def log(msg):
    print(msg, flush=True)
    REPORT_LINES.append(msg)


def md_table(rows, headers):
    out = ['| ' + ' | '.join(headers) + ' |',
           '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(x) for x in r) + ' |')
    return out


def load_asset(conn, code):
    df = conn.execute(
        "SELECT date, open, high, low, close FROM daily_ohlc WHERE code=? ORDER BY date", [code]
    ).fetchdf()
    df['date'] = pd.to_datetime(df['date'])
    return df.reset_index(drop=True)


def compute_ind(df, ma_p, adx_th, vol_th):
    """指标计算走 signal_core, 另补 ATR/vol 数组"""
    c, h, l = df['close'], df['high'], df['low']
    core = compute_signal_core(c, h, l, ma_p=ma_p, adx_th=adx_th, vol_th=vol_th)
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return {
        'open': df['open'].values.astype(float),
        'close': c.values.astype(float),
        'date': df['date'],
        'ma': core['ma'].values,
        'vol': core['vol'].values,
        'adx': core['adx'].values,
        'atr': atr.values,
        'signal': core['signal'].values,
        'ma_p': ma_p, 'adx_th': adx_th, 'vol_th': vol_th,
    }


def perf(equity, n_days):
    """equity: 净值数组(含期初1.0); 返回 (cagr, vol, sharpe, maxdd)"""
    rets = np.diff(equity) / equity[:-1]
    if len(rets) < 20 or np.all(rets == 0):
        return 0.0, 0.0, 0.0, 0.0
    years = len(rets) / 252
    cagr = (equity[-1] / equity[0]) ** (1 / years) - 1
    years = rets.std() * np.sqrt(252)
    vol = years
    sharpe = (rets.mean() * 252 - RF) / vol if vol > 1e-9 else 0.0
    peak = np.maximum.accumulate(equity)
    maxdd = ((equity - peak) / peak).min()
    return cagr, vol, sharpe, maxdd


def run_single(d, i0, i1, cost=0.0, confirm_days=1, ma_buf=0.0, adx_rising=False, atr_stop=None):
    """单品种状态机回测 (T收盘决策/T+1开盘执行). 返回 (equity, trades, win_rate)"""
    n = i1 - i0
    o, c, ma, vol, adx, atr, sig = (d['open'], d['close'], d['ma'], d['vol'],
                                    d['adx'], d['atr'], d['signal'])
    equity = np.ones(n + 1)
    eq, held, pending = 1.0, False, None
    entry_px = entry_atr = None
    just_in = False
    trades, wins = 0, 0
    trade_eq0 = 1.0
    for k in range(n):
        t = i0 + k
        day_ret = 0.0
        if pending == 'sell' and held:
            day_ret += o[t] / c[t - 1] - 1
            eq *= (1 - cost)
            if eq > trade_eq0:
                wins += 1
            held = False
            trades += 1
        elif pending == 'buy' and not held:
            eq *= (1 - cost)
            held, just_in = True, True
            entry_px, entry_atr = o[t], atr[t - 1]
            trade_eq0 = eq
        pending = None
        if held:
            day_ret += c[t] / o[t] - 1 if just_in else c[t] / c[t - 1] - 1
        else:
            day_ret += BOND_DAILY
        just_in = False
        eq *= (1 + day_ret)
        equity[k + 1] = eq
        # T日收盘决策
        if held:
            exit_cond = (c[t] < ma[t] * (1 - ma_buf)) or (vol[t] >= d['vol_th'] and adx[t] <= d['adx_th'])
            stop_cond = atr_stop is not None and c[t] < entry_px - atr_stop * entry_atr
            if exit_cond or stop_cond:
                pending = 'sell'
        else:
            if (c[t] > ma[t] * (1 + ma_buf)) and (vol[t] < d['vol_th'] or adx[t] > d['adx_th']):
                if confirm_days <= 1 or sig[t - confirm_days + 1:t + 1].sum() == confirm_days:
                    if (not adx_rising) or adx[t] > adx[t - 1]:
                        pending = 'buy'
    return equity, trades, (wins / trades if trades else 0.0)


def run_rotation(data, codes, i0, i1, cost=0.0, confirm_days=1, ma_buf=0.0,
                 adx_rising=False, atr_stop=None):
    """三品种轮动回测: 每日收盘在'有信号品种'中选ADX最高者为目标持仓, T+1开盘切换"""
    n = i1 - i0
    O = {cd: data[cd]['open'] for cd in codes}
    C = {cd: data[cd]['close'] for cd in codes}
    MA = {cd: data[cd]['ma'] for cd in codes}
    VOL = {cd: data[cd]['vol'] for cd in codes}
    ADX = {cd: data[cd]['adx'] for cd in codes}
    ATR = {cd: data[cd]['atr'] for cd in codes}
    SIG = {cd: data[cd]['signal'] for cd in codes}
    VT = {cd: data[cd]['vol_th'] for cd in codes}

    equity = np.ones(n + 1)
    eq, held, pending, just_in = 1.0, None, None, False
    entry = {}
    trades, wins, trade_eq0 = 0, 0, 1.0
    for k in range(n):
        t = i0 + k
        day_ret = 0.0
        if pending is not None and pending != held:
            if held is not None:
                day_ret += O[held][t] / C[held][t - 1] - 1
                eq *= (1 - cost)
                if eq > trade_eq0:
                    wins += 1
                trades += 1
                held = None
            if pending != 'FLAT':
                eq *= (1 - cost)
                held, just_in = pending, True
                entry[held] = (O[held][t], ATR[held][t - 1])
                trade_eq0 = eq
                trades += 1
        pending = None
        if held is not None:
            day_ret += C[held][t] / O[held][t] - 1 if just_in else C[held][t] / C[held][t - 1] - 1
        else:
            day_ret += BOND_DAILY
        just_in = False
        eq *= (1 + day_ret)
        equity[k + 1] = eq
        # 收盘决策: 目标持仓 = 候选中ADX最高
        cands = []
        for cd in codes:
            if (C[cd][t] > MA[cd][t] * (1 + ma_buf)) and (VOL[cd][t] < VT[cd] or ADX[cd][t] > data[cd]['adx_th']):
                if confirm_days <= 1 or SIG[cd][t - confirm_days + 1:t + 1].sum() == confirm_days:
                    if (not adx_rising) or ADX[cd][t] > ADX[cd][t - 1]:
                        cands.append(cd)
        pick = max(cands, key=lambda cd: ADX[cd][t]) if cands else 'FLAT'
        if held is not None:
            px, e_atr = entry[held]
            stop = atr_stop is not None and C[held][t] < px - atr_stop * e_atr
            sig_ok = (C[held][t] >= MA[held][t] * (1 - ma_buf)) and \
                     (VOL[held][t] < VT[held] or ADX[held][t] > data[held]['adx_th'])
            if stop or not sig_ok:
                pick = 'FLAT'
        if pick != held:
            pending = pick
    return equity, trades, (wins / trades if trades else 0.0)


def fmt_perf(p):
    cagr, vol, sharpe, maxdd = p
    return f"{cagr:+.1%} / {vol:.1%} / {sharpe:.2f} / {maxdd:.1%}"


# ============ E1: Walk-forward ============
def e1_walkforward(conn):
    log('\n## E1 Walk-forward 样本外验证 (训练期选参 → 测试期检验)')
    grid = [(mp, at, vt) for mp in (20, 30, 50) for at in (15, 20, 25, 30) for vt in (12, 15, 18, 22)]
    years = [(2024, 2025), (2025, 2026), (2026, 2027)]
    all_rows = []
    for code, info in ASSETS.items():
        df = load_asset(conn, code)
        inds = {g: compute_ind(df, *g) for g in grid}
        prod = (info['ma_p'], info['adx_th'], info['vol_th'])
        n = len(df)
        first_valid = WARMUP
        log(f"\n### {info['name']} ({code})  数据 {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
        for ty, ny in years:
            if ty == 2026 and code == '159995':
                pass
            t_start = int(np.searchsorted(df['date'].values, np.datetime64(f'{ty}-01-01')))
            t_end = int(np.searchsorted(df['date'].values, np.datetime64(f'{ny}-01-01')))
            if t_start >= n or t_start <= first_valid:
                continue
            t_end = min(t_end, n)
            # 训练期选参
            best, best_sh, best_tr = None, -9e9, 0
            for g in grid:
                eq, tr, _ = run_single(inds[g], first_valid, t_start)
                if tr < 8:
                    continue
                sh = perf(eq, t_start - first_valid)[2]
                if sh > best_sh:
                    best, best_sh, best_tr = g, sh, tr
            use = best if best else prod
            # 测试期评估
            eq_best, tr_b, wr_b = run_single(inds[use], t_start, t_end)
            eq_prod, tr_p, _ = run_single(inds[prod], t_start, t_end)
            p_best = perf(eq_best, t_end - t_start)
            p_prod = perf(eq_prod, t_end - t_start)
            # 买入持有基准
            bh = inds[use]['close'][t_start:t_end] / inds[use]['open'][t_start]
            p_bh = perf(np.r_[1.0, bh], t_end - t_start)
            all_rows.append([code, f'{ty}', f'{use[0]}/{use[1]}/{use[2]}', f'{best_sh:.2f}',
                             f'{p_best[2]:.2f}', f'{p_best[0]:+.1%}', f'{p_prod[2]:.2f}',
                             f'{p_prod[0]:+.1%}', f'{p_bh[0]:+.1%}', tr_b])
    log('\n' + '\n'.join(md_table(all_rows, ['品种', '测试年', '选中参数(MA/ADX/Vol)', '训练Sharpe',
                                             '样本外Sharpe', '样本外年化', '现行参数Sharpe',
                                             '现行参数年化', '买入持有年化', '交易数'])))
    return all_rows


# ============ E2: 置信度分桶 ============
def e2_confidence(conn):
    log('\n## E2 置信度分桶有效性 (信号日后5/10/20个交易日, 次日开盘入场, 超额=相对国债)')
    rows, pooled = [], []
    for code, info in ASSETS.items():
        df = load_asset(conn, code)
        d = compute_ind(df, info['ma_p'], info['adx_th'], info['vol_th'])
        c, o, ma, adx, vol = d['close'], d['open'], d['ma'], d['adx'], d['vol']
        sig = d['signal']
        buckets = {'<40(弱)': [], '40-59(中)': [], '60-74(中)': [], '75+(强)': []}
        trig = {'趋势触发': [], '低波触发': []}
        for t in range(WARMUP, len(c) - 21):
            if sig[t] != 1:
                continue
            score, _, _ = compute_confidence(float(c[t]), float(ma[t]), float(adx[t]),
                                             float(vol[t]), info)
            entry = o[t + 1]
            fw = {N: c[t + N] / entry - 1 - ((1 + RF) ** (N / 252) - 1) for N in (5, 10, 20)}
            key = '<40(弱)' if score < 40 else '40-59(中)' if score < 60 else '60-74(中)' if score < 75 else '75+(强)'
            buckets[key].append(fw)
            tk = '趋势触发' if adx[t] > info['adx_th'] else '低波触发'
            trig[tk].append(fw)
            pooled.append((code, score, fw))
        log(f"\n### {info['name']} ({code})")
        b_rows = []
        for bk, lst in buckets.items():
            if not lst:
                b_rows.append([bk, 0, '-', '-', '-', '-', '-', '-'])
                continue
            arr5 = np.array([x[5] for x in lst]); arr10 = np.array([x[10] for x in lst]); arr20 = np.array([x[20] for x in lst])
            b_rows.append([bk, len(lst), f'{arr5.mean():+.2%}', f'{(arr5 > 0).mean():.0%}',
                           f'{arr10.mean():+.2%}', f'{arr20.mean():+.2%}', f'{(arr20 > 0).mean():.0%}',
                           f'{arr20.max():+.1%}'])
        log('\n'.join(md_table(b_rows, ['置信桶', '样本数', '5日超额', '5日胜率', '10日超额',
                                        '20日超额', '20日胜率', '20日最好'])))
        t_rows = []
        for tk, lst in trig.items():
            if not lst:
                continue
            arr20 = np.array([x[20] for x in lst])
            t_rows.append([tk, len(lst), f'{arr20.mean():+.2%}', f'{(arr20 > 0).mean():.0%}'])
        log('\n'.join(md_table(t_rows, ['触发方式', '样本数', '20日超额', '20日胜率'])))
    # 汇总
    p_rows = []
    for lo, hi, label in [(0, 40, '<40'), (40, 60, '40-59'), (60, 75, '60-74'), (75, 101, '75+')]:
        lst = [(s, fw) for _, s, fw in pooled if lo <= s < hi]
        if not lst:
            continue
        arr20 = np.array([fw[20] for _, fw in lst])
        p_rows.append([label, len(lst), f'{arr20.mean():+.2%}', f'{(arr20 > 0).mean():.0%}',
                       f'{np.median(arr20):+.2%}'])
    log('\n### 三品种汇总 (20日超额)')
    log('\n'.join(md_table(p_rows, ['置信桶', '样本数', '平均超额', '胜率', '中位数'])))
    return pooled


# ============ E3/E4/E5: 轮动模拟 ============
def build_rotation_data(conn, align='intersection'):
    data = {}
    for code, info in ASSETS.items():
        df = load_asset(conn, code)
        data[code] = (df, compute_ind(df, info['ma_p'], info['adx_th'], info['vol_th']))
    # 日期轴 = 三品种交集
    dates = sorted(set(data['510310'][0]['date']).intersection(
        set(data['159995'][0]['date']), set(data['512800'][0]['date'])))
    axis = pd.DatetimeIndex(dates)
    axis = axis[axis >= axis[0] + pd.Timedelta(days=WARMUP)]
    aligned = {}
    for code, (df, ind) in data.items():
        pos = df.set_index('date').index.get_indexer(axis)
        a = dict(ind)
        for key in ('open', 'close', 'ma', 'vol', 'adx', 'atr', 'signal'):
            a[key] = ind[key][pos]
        aligned[code] = a
    return aligned, axis


def e3_e4_e5(conn):
    codes = list(ASSETS.keys())
    data, axis = build_rotation_data(conn)
    i0, i1 = 0, len(axis)
    period = f"{axis[0].date()} ~ {axis[-1].date()} ({i1} 个交易日)"
    log(f'\n## E3/E4/E5 三品种轮动模拟  期间: {period}  空仓计国债({RF:.1%})')
    log('\n### E3 成本敏感性 (现行参数, 无过滤器)')
    rows = []
    for bp in (0, 5, 10, 20):
        eq, tr, wr = run_rotation(data, codes, i0, i1, cost=bp / 10000)
        p = perf(eq, i1)
        rows.append([f'{bp}bp', fmt_perf(p), tr, f'{wr:.0%}', f'{eq[-1]:.2f}x'])
    log('\n'.join(md_table(rows, ['单边成本', '年化/波动/Sharpe/最大回撤', '交易数', '胜率', '期末净值'])))

    log('\n### E4 防打脸过滤器 A/B (单边成本10bp)')
    variants = [
        ('基线(现行)', dict()),
        ('连续2日确认', dict(confirm_days=2)),
        ('MA±1%缓冲带', dict(ma_buf=0.01)),
        ('ADX需上行', dict(adx_rising=True)),
        ('2日确认+缓冲带', dict(confirm_days=2, ma_buf=0.01)),
    ]
    rows = []
    for name, kw in variants:
        eq, tr, wr = run_rotation(data, codes, i0, i1, cost=0.001, **kw)
        p = perf(eq, i1)
        rows.append([name, fmt_perf(p), tr, f'{wr:.0%}', f'{eq[-1]:.2f}x'])
    log('\n'.join(md_table(rows, ['变体', '年化/波动/Sharpe/最大回撤', '交易数', '胜率', '期末净值'])))

    log('\n### E5 ATR灾难止损 (单边成本10bp, 止损后等信号再入场)')
    rows = []
    for name, kw in [('基线(无止损)', dict())] + [(f'{m}xATR止损', dict(atr_stop=m)) for m in (2.0, 2.5, 3.0)]:
        eq, tr, wr = run_rotation(data, codes, i0, i1, cost=0.001, **kw)
        p = perf(eq, i1)
        rows.append([name, fmt_perf(p), tr, f'{wr:.0%}', f'{eq[-1]:.2f}x'])
    log('\n'.join(md_table(rows, ['变体', '年化/波动/Sharpe/最大回撤', '交易数', '胜率', '期末净值'])))


def main():
    log(f'# Tier-1 策略验证报告  ({datetime.now().strftime("%Y-%m-%d %H:%M")})')
    log(f'> 数据: trading_history.duckdb 只读 | 参数: config.py (510310: MA30/ADX20/Vol18, 159995: MA30/ADX25/Vol15, 512800: MA30/ADX25/Vol18)')
    log(f'> 执行假设: T收盘信号→T+1开盘成交, 单边成本默认10bp, 空仓按国债{RF:.1%}计息')
    conn = duckdb.connect(HISTORY_DB, read_only=True)
    try:
        e1_walkforward(conn)
        e2_confidence(conn)
        e3_e4_e5(conn)
    finally:
        conn.close()
    out = os.path.join(PROJECT_ROOT, 'reports', f'tier1_validation_{datetime.now().strftime("%Y%m%d")}.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(REPORT_LINES))
    log(f'\n报告已保存: {out}')


if __name__ == '__main__':
    main()
