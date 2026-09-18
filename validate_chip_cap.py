# -*- coding: utf-8 -*-
"""
芯片ETF 仓位上限回测 (2026-09-18) — 只读历史库
背景: 芯片 20日年化波动常达 40%+, 单品种满仓时组合波动大; volatarget 已证明"连续按波动率缩放"是负优化,
      但"固定上限"未测过 (E1 walk-forward 显示芯片样本外退化也最剧烈)
口径: 与 validate_rotation_rule 完全一致 (R0 每日重选ADX最高, 2日确认, T+1开盘, 10bp, 空仓国债2.5%)
"""
import os, sys
import numpy as np
import pandas as pd
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import validate_tier1 as T
import validate_rotation_rule as RR
from config import HISTORY_DB

conn = duckdb.connect(HISTORY_DB, read_only=True)
inds, dates = RR.load_aligned(conn)
conn.close()
i0 = int(np.searchsorted(pd.DatetimeIndex(dates).values, np.datetime64(RR.START)))
i1 = len(dates)

print(f"期间: {dates[i0].date()} ~ {dates[i1-1].date()} ({i1-i0} 个交易日)")
print("口径: R0(每日重选ADX最高) | 2日确认 | T+1开盘 | 单边10bp | 空仓国债2.5% | 未用仓位计入现金\n")

VARIANTS = [
    ('现行(无上限)', {}),
    ('芯片≤70%', {'159995': 0.7}),
    ('芯片≤60%', {'159995': 0.6}),
    ('芯片≤50%', {'159995': 0.5}),
    ('芯片≤40%', {'159995': 0.4}),
    ('芯片≤50% + 300≤70%', {'159995': 0.5, '510310': 0.7}),
]

for brake in (True, False):
    print(f"### 波动刹车 {'开启' if brake else '关闭'}")
    print('| 方案 | 年化/波动/Sharpe/最大回撤 | 调仓次数 | 期末净值 |')
    print('|---|---|---|---|')
    base = None
    for name, caps in VARIANTS:
        eq, trades, turnover = RR.simulate(inds, i0, i1, 'R0', brake, caps=caps)
        p = T.perf(eq, i1 - i0)
        if base is None:
            base = p
        delta = p[0] - base[0]
        print(f"| {name} | {T.fmt_perf(p)} | {trades} | {eq[-1]:.2f}x |")
    print()

print("参照: 现行生产 = 波动刹车开启 + 无上限")
