# -*- coding: utf-8 -*-
"""
买入追高上限敏感性扫描 (2026-10-28) — 只读 trading_history.duckdb, 不写库
问题: 买入挂单"不追高超+1%"的上限定多少最优? (卖出无条件执行不在本实验范围)
方法: 复用 validate_friction 的 nofill 引擎 (F6: 买入开盘价高于信号收盘价×上限 → 顺延),
      扫描上限 0.5%/1%/1.5%/2%/3%/无上限, 其余=F1 成本(佣金+半价差)
      模块级常量 CHASE_CAP 通过 monkeypatch 逐档设置 (脚本内注明, 仅本实验)
口径: R0轮动 + 2日确认 + 波动刹车 | T+1开盘执行 | 单边10bp
用法: python validate_chase_cap.py
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
import validate_friction as V
from config import ORDER

START = V.START


def main():
    conn = duckdb.connect(V.HISTORY_DB, read_only=True)
    inds, dates = V.load_aligned(conn)
    conn.close()
    i0 = int(np.searchsorted(pd.DatetimeIndex(dates).values, np.datetime64(START)))
    i1 = len(dates)

    print('=' * 100)
    print('  买入追高上限敏感性扫描 (nofill顺延引擎, F1成本=佣金+半价差)')
    print('=' * 100)
    print(f"期间: {dates[i0].date()} ~ {dates[i1-1].date()} ({i1-i0} 个交易日) | 单边10bp | 2日确认+刹车")
    print('上限含义: 次日开盘价 > 信号日收盘价×(1+上限) → 当日不买入, 顺延下一检查点\n')

    rows = []
    for cap in (0.005, 0.01, 0.015, 0.02, 0.03, None):
        # CHASE_CAP 在引擎中是乘数(1.01=允许+1%), 此处把超额百分比转为乘数
        V.CHASE_CAP = (1 + cap) if cap is not None else 1e9   # monkeypatch 模块常量(仅本实验)
        label = f'{cap:.1%}' if cap is not None else '无上限'
        eq, tr = V.simulate(inds, i0, i1, mode='struct_half', brake=True, nofill=True)
        p = T.perf(eq, i1 - i0)
        rows.append((label, p, tr))
        print(f"追高上限 {label:<8} {T.fmt_perf(p)}  调仓{tr}次")

    # 基准参照: 无顺延(上限=0, 市价必成交, F1成本)
    V.CHASE_CAP = 1e9
    eq0, tr0 = V.simulate(inds, i0, i1, mode='struct_half', brake=True, nofill=False)
    print(f"\n基准(无顺延机制, 必成交): {T.fmt_perf(T.perf(eq0, i1 - i0))}  调仓{tr0}次")
    print('参照(现行回测口径 F0=10bp 必成交): +17.8% / 0.83 / -16.2%, 235笔')

    best = max(rows, key=lambda r: r[1][2])  # 按 Sharpe
    print(f"\nSharpe 最优: 追高上限 {best[0]} ({T.fmt_perf(best[1])})")

    out = os.path.join(PROJECT_ROOT, 'reports',
                       f"chase_cap_validation_{_dt.datetime.now().strftime('%Y%m%d')}.md")
    with open(out, 'w', encoding='utf-8') as f:
        f.write('# 买入追高上限敏感性扫描 (validate_chase_cap.py)\n\n')
        for label, p, tr in rows:
            f.write(f'- 追高上限 {label}: {T.fmt_perf(p)} / {tr}笔\n')
        f.write(f'- 基准(无顺延): {T.fmt_perf(T.perf(eq0, i1 - i0))} / {tr0}笔\n')
    print(f'\n已保存: {out}')


if __name__ == '__main__':
    main()
