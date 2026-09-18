# -*- coding: utf-8 -*-
"""
实盘执行质量报告 (2026-09-18)
逐笔对比「邮件信号价 → 执行日开盘 → 实际成交价」, 量化滑点与隔夜跳空成本。
- 信号价 = 成交日前一交易日收盘 (收盘邮件所给)
- 基准执行价 = 成交日开盘 (策略规定 T+1 开盘执行)
- 成交价 = trades CSV 记录的实际成交价 (含拆分复权)
用法: python slippage_report.py
"""
import os
import sys
import socket
socket.setdefaulttimeout(20)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import pandas as pd
import duckdb
from config import ASSETS, HISTORY_DB
from signal_core import adjust_splits


def split_factors(code):
    """从新浪原始数据推导复权因子: 成交日 d 的因子 = 所有晚于 d 的复权事件乘积累计
    (库内价格已复权, 故历史成交价需同乘该因子后再比价)"""
    import akshare as ak
    try:
        raw = ak.fund_etf_hist_sina(symbol=ASSETS[code]['code'])
        raw['date'] = pd.to_datetime(raw['date'])
        raw = raw[['date', 'open', 'high', 'low', 'close']].dropna().sort_values('date').reset_index(drop=True)
        events = adjust_splits(raw)  # 原地复权, 返回事件列表
        fac = []
        for d, ratio_str, typ in events:
            num, den = ratio_str.split(':')
            r = float(num) / float(den)   # 拆分 1:2 -> 0.5 (库内把此前价格除以2) ; 份额合并 2:1 -> 2.0 (乘2)
            fac.append((pd.Timestamp(d), r))
        return fac
    except Exception as e:
        print(f'[WARN] {code} 复权因子获取失败({e}); 按无复权处理')
        return []


def main():
    conn = duckdb.connect(HISTORY_DB, read_only=True)
    rows = []
    for code in ASSETS:
        f = os.path.join(PROJECT_ROOT, 'trades', f'{code}_trades.csv')
        if not os.path.exists(f):
            continue
        tr = pd.read_csv(f)
        fac = split_factors(code)
        last_ref = None  # 同一品种已匹配的成交日, 约束后续笔不得早于它 (保持时序)
        for _, r in tr.iterrows():
            action = str(r.get('action', '')).strip().upper()
            if action not in ('BUY', 'SELL'):
                continue
            d = pd.Timestamp(str(r['date']))
            px = float(r['price'])
            sh = int(float(r['shares']))
            # 复权因子 (库内已复权)
            factor = 1.0
            for ev_date, fv in fac:
                if ev_date > d:
                    factor *= fv
            px_adj = px * factor
            # 基准: 成交日开盘 + 前一交易日收盘 (成交日非交易日时就近取 ±3 自然日内交易日并标注)
            cur = conn.execute("SELECT date, open, high, low FROM daily_ohlc WHERE code=? AND date=?", [code, d.date()]).fetchone()
            note = ''
            if cur is None:
                alt = conn.execute(
                    "SELECT date, open, high, low FROM daily_ohlc WHERE code=? AND date BETWEEN ? AND ? ORDER BY ABS(DATEDIFF('day', date, ?)) LIMIT 1",
                    [code, (d - pd.Timedelta(days=3)).date(), (d + pd.Timedelta(days=3)).date(), d.date()]).fetchone()
                if alt is None:
                    print(f'[WARN] {code} {d.date()} 附近无行情, 跳过')
                    continue
                cur = alt
                note = f'(非交易日,就近取{alt[0]})'
            # 校验: 成交价须落在参考日 [最低,最高] 内 (ETF 涨跌停限制内), 否则就近匹配真实成交日
            ref_lo, ref_hi = float(cur[3]), float(cur[2])
            if not (ref_lo - 1e-9 <= px_adj <= ref_hi + 1e-9):
                floor = last_ref if last_ref is not None else (d - pd.Timedelta(days=6)).date()
                near = conn.execute(
                    "SELECT date, open, high, low FROM daily_ohlc WHERE code=? AND date BETWEEN ? AND ? AND low<=? AND high>=? ORDER BY date LIMIT 1",
                    [code, floor, (d + pd.Timedelta(days=6)).date(), px_adj, px_adj]).fetchone()
                if near is None:
                    near = conn.execute(
                        "SELECT date, open, high, low FROM daily_ohlc WHERE code=? AND date BETWEEN ? AND ? AND low<=? AND high>=? ORDER BY ABS(DATEDIFF('day', date, ?)) LIMIT 1",
                        [code, (d - pd.Timedelta(days=6)).date(), (d + pd.Timedelta(days=6)).date(), px_adj, px_adj, d.date()]).fetchone()
                if near is not None:
                    note += f' ⚠成交价不在{cur[0]}区间内, 按疑似实际成交日 {near[0]} 计'
                    cur = near
            last_ref = pd.Timestamp(cur[0])
            prev = conn.execute("SELECT date, close FROM daily_ohlc WHERE code=? AND date < ? ORDER BY date DESC LIMIT 1", [code, cur[0]]).fetchone()
            if prev is None:
                print(f'[WARN] {code} {d.date()} 缺少信号日行情, 跳过')
                continue
            open_px = float(cur[1])
            sig_close = float(prev[1])
            gap_bp = (open_px / sig_close - 1) * 1e4
            slip_bp = (px_adj / open_px - 1) * 1e4
            if action == 'SELL':
                slip_bp = -slip_bp
            cost_yuan = sh * (px_adj - open_px) * (1 if action == 'BUY' else -1)
            rows.append({
                'date': d.strftime('%Y-%m-%d') + note, 'code': code, 'action': action,
                '信号日收盘': round(sig_close, 4), '执行日开盘': round(open_px, 4),
                '实际成交': round(px, 4), '复权后成交': round(px_adj, 4),
                '隔夜跳空bp': round(gap_bp, 1), '滑点bp': round(slip_bp, 1), '成本¥': round(cost_yuan, 2),
            })
    conn.close()

    if not rows:
        print('无有效交易记录')
        return
    df = pd.DataFrame(rows)
    print('=' * 96)
    print('  实盘执行质量报告 (信号价 → 执行日开盘 → 实际成交)')
    print('=' * 96)
    print(df.to_string(index=False))
    print()
    print('--- 汇总 ---')
    for act in ['BUY', 'SELL']:
        sub = df[df['action'] == act]
        if len(sub):
            print(f"{act}: {len(sub)}笔 | 平均滑点 {sub['滑点bp'].mean():+.1f}bp | "
                  f"平均隔夜跳空 {sub['隔夜跳空bp'].mean():+.1f}bp | 滑点成本合计 ¥{sub['成本¥'].sum():+.2f}")
    print(f"全部: 滑点成本合计 ¥{df['成本¥'].sum():+.2f} (正=对策略不利)")
    print()
    print('说明: 滑点bp = 相对执行日开盘的偏离(已按方向换算为"成本", 正=更差); '
          '隔夜跳空 = 开盘/信号日收盘-1, 属策略设计(T+1开盘执行)的固有成本, 非执行失误')
    os.makedirs(os.path.join(PROJECT_ROOT, 'reports'), exist_ok=True)
    out = os.path.join(PROJECT_ROOT, 'reports', 'slippage_report_latest.md')
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write('# 实盘执行质量报告\n\n')
        fh.write(df.to_markdown(index=False) + '\n')
    print(f'已保存: {out}')


if __name__ == '__main__':
    main()
