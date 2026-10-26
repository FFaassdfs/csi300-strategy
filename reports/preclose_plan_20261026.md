# 尾盘前瞻执行链 实施计划

> **执行方式**：本会话内逐步执行（inline），Task 1 为决策门槛，通过后继续。
> 设计文档：`reports/preclose_design_20261026.md`

**Goal**：盘中条件预警（翻空/确认成立时尽早邮件提醒）+ 14:30 尾窗执行 + 16:00 收盘汇报兜底，消除隔夜/周末缺口导致的"无法按既定方针执行"。

**Architecture**：信号本身零改动（`signal_core.py` 不动），只改执行时序。新增 `preclose_signal.py`（复用 `send_advice` 邮件基建与 `compute_signals` 合成K线口径）+ `validate_execution_timing.py`（A/B 执行时序回测）+ `send_advice.py` 小改（补登记提示）+ 计划任务变更。

**Tech Stack**：Python 3.12（`C:\Program Files\Python312\python.exe`）、duckdb（只读）、smtplib、Windows Task Scheduler。

## Global Constraints

- 信号口径唯一权威 `signal_core.py` — **任何信号逻辑不改**
- 交易日历内运行：脚本内时间窗校验（9:45-11:30 / 13:00-15:00），窗外退出
- 计划任务**纯 Python 调用**（`python.exe + 脚本`，无 shell 包装，OPERATION §6.4；手动命令用 pwsh 勿用 powershell.exe）
- 中文脚本必须写成 `.py` 文件执行（禁止 `python -c` 查中文数据）
- SMTP 授权码在 `local_secrets.py`（不入 git）
- 验证脚本只读 `trading_history.duckdb`，不写库
- 引擎收益三段式语义（入场日 O→C、出场日 C_prev→O）——今日已修过 `w_cur` 提前覆盖 bug，勿再引入
- balance 必填规则不变；合成K线门控：库内无当日K线才追加

---

### Task 1: `validate_execution_timing.py` — A/B 执行时序回测（决策门槛）

**Files:**
- Create: `hs300/validate_execution_timing.py`
- Output: `reports/execution_timing_validation_<date>.md`

**Interfaces:**
- Consumes: `validate_param_plateau` 的 `load_aligned/make_inds/base_params/simulate`（A 口径引擎，已与 rotation_rule 交叉验证一致）+ `validate_tier1` 的 `perf/fmt_perf/BOND_DAILY`
- Produces: 决策结论（B 年化 ≥ A-1pp → 落地）写入报告

**B 口径引擎语义**：T 收盘决策 + T 收盘立即成交（`C[t]`），无隔夜分段；持仓整天收益 = `C[t]/C[t-1]`。

- [ ] **Step 1.1 写脚本**（完整代码）：

```python
# -*- coding: utf-8 -*-
"""
执行时序对比回测 (2026-10-26) — 只读 trading_history.duckdb, 不写库
A(现行基准): T收盘信号 → T+1开盘成交   B(当日执行): T收盘信号 → T收盘成交
B 是"盘中触发+当日操作"的回测近似: 真实世界盘中触发用当时已知实时价(无未来函数),
成交在触发价与收盘价之间, T收盘成交为中性近似; 收盘翻空的日子盘中几乎必然已在
某个30分钟检查点触发(仅尾盘15分钟翻转例外)。
决策门槛: B 年化 ≥ A 年化 - 1pp → 落地盘中预警链路; 否则停止重新讨论。
口径: R0轮动 + 2日确认 + 波动刹车(35%/60%) | 单边10bp | 空仓国债2.5%
用法: python validate_execution_timing.py
"""
import os, sys
import numpy as np
import pandas as pd
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import validate_tier1 as T
import validate_param_plateau as P
from config import ORDER

COST = 0.001


def simulate_sameday(inds, params, i0, i1, cost=COST, brake=True):
    """B口径: T收盘决策 → T收盘立即成交 (C[t]), 持仓整天收益 C[t]/C[t-1]"""
    A = {cd: {k: inds[cd][k].tolist() for k in ('open', 'close', 'ma', 'vol', 'adx', 'signal')}
         for cd in ORDER}
    C = {cd: A[cd]['close'] for cd in ORDER}
    MA = {cd: A[cd]['ma'] for cd in ORDER}
    VOL = {cd: A[cd]['vol'] for cd in ORDER}
    ADX = {cd: A[cd]['adx'] for cd in ORDER}
    SIG = {cd: A[cd]['signal'] for cd in ORDER}

    n = i1 - i0
    equity = np.ones(n + 1)
    eq = 1.0
    w_cur = {cd: 0.0 for cd in ORDER}
    held = None
    trades = 0
    for k in range(n):
        t = i0 + k
        if k > 0:
            # 当日收益 (持仓为昨日收盘执行所定, 整天 C[t]/C[t-1])
            day_ret = 0.0
            for cd in ORDER:
                if w_cur[cd] > 1e-9:
                    day_ret += w_cur[cd] * (C[cd][t] / C[cd][t - 1] - 1)
            cash_w = 1.0 - sum(w_cur.values())
            if cash_w > 1e-9:
                day_ret += cash_w * T.BOND_DAILY
            eq *= (1 + day_ret)
            equity[k + 1] = eq
            held = next((cd for cd in ORDER if w_cur[cd] > 1e-9), None)
        # T收盘: 决策 + 立即成交
        cands = [cd for cd in ORDER if t > 0 and SIG[cd][t] == 1 and SIG[cd][t - 1] == 1]
        nw = {cd: 0.0 for cd in ORDER}
        pick = max(cands, key=lambda cd: ADX[cd][t]) if cands else None
        if held is not None:
            vt, at = params[held]['vol_th'], params[held]['adx_th']
            sig_ok = (C[held][t] >= MA[held][t]) and (VOL[held][t] < vt or ADX[held][t] > at)
            if not sig_ok:
                pick = None
        if pick is not None:
            nw[pick] = P.BRAKE_W if (brake and VOL[pick][t] > P.BRAKE_TH) else 1.0
        for cd in ORDER:
            if abs(w_cur[cd] - nw[cd]) > 1e-9:
                trades += 1
                eq *= (1 - cost * abs(w_cur[cd] - nw[cd]))
        w_cur = dict(nw)
    return equity, trades


def gap_stats(aligned, i0, i1):
    """隔夜跳空统计: 普通隔夜 vs 周末(周五收盘→下一交易日开盘)"""
    rows = []
    for cd in ORDER:
        o = aligned[cd]['open'].values
        c = aligned[cd]['close'].values
        d = pd.DatetimeIndex(aligned[cd]['date'])
        gaps, wk = [], []
        for t in range(max(i0, 1), i1):
            g = (o[t] / c[t - 1] - 1) * 100
            gaps.append(g)
            if d[t].weekday() == 0:  # 周一开盘 = 跨周末
                wk.append(g)
        rows.append((cd, np.mean(gaps), np.std(gaps), min(gaps), np.mean(wk), np.std(wk), min(wk), len(wk)))
    return rows


def friday_flip_stats(inds, i0, i1):
    """周五收盘信号翻转的成交差异: B=周五收盘卖/买 vs A=下一交易日开盘卖/买"""
    rows = []
    for cd in ORDER:
        o = inds[cd]['open']
        c = inds[cd]['close']
        d = pd.DatetimeIndex(inds[cd]['date'])
        sig = inds[cd]['signal']
        sell_diff, buy_diff = [], []
        for t in range(max(i0, 2), i1 - 1):
            if d[t].weekday() != 4:
                continue
            if sig[t] == 0 and sig[t - 1] == 1:      # 周五翻空 → 卖出
                sell_diff.append((o[t + 1] / c[t] - 1) * 100)  # A卖价比B低多少(负=A更差)
            elif sig[t] == 1 and sig[t - 1] == 1 and sig[t - 2] == 0:  # 周五确认成立 → 买入
                buy_diff.append((c[t] / o[t + 1] - 1) * 100)   # B买价比A低多少(负=B更贵)
        rows.append((cd, len(sell_diff), np.mean(sell_diff) if sell_diff else np.nan,
                     len(buy_diff), np.mean(buy_diff) if buy_diff else np.nan))
    return rows


def main():
    conn = duckdb.connect(P.HISTORY_DB, read_only=True)
    aligned, dates = P.load_aligned(conn)
    conn.close()
    params = P.base_params()
    inds = P.make_inds(aligned, params)
    i0 = int(np.searchsorted(pd.DatetimeIndex(dates).values, np.datetime64(P.START)))
    i1 = len(dates)

    print('=' * 100)
    print('  执行时序对比回测: A=次日开盘(现行) vs B=当日执行(盘中触发近似)')
    print('=' * 100)
    print(f"期间: {dates[i0].date()} ~ {dates[i1-1].date()} ({i1-i0} 个交易日) | 单边10bp | 2日确认+刹车")

    eqA, trA = P.simulate(inds, params, i0, i1)
    eqB, trB = simulate_sameday(inds, params, i0, i1)
    pA, pB = T.perf(eqA, i1 - i0), T.perf(eqB, i1 - i0)
    print(f"\nA 次日开盘(现行): {T.fmt_perf(pA)}  调仓{trA}次")
    print(f"B 当日执行(新):   {T.fmt_perf(pB)}  调仓{trB}次")
    diff_pp = (pB[0] - pA[0]) * 100
    verdict = '✅ 落地' if diff_pp >= -1.0 else '❌ 停止,重新讨论'
    print(f"\n决策门槛: B-A 年化差 {diff_pp:+.1f}pp (≥-1pp 落地) → {verdict}")

    print('\n--- 隔夜跳空统计 (%) ---')
    print(f"{'品种':<16}{'普通均值':>8}{'普通σ':>7}{'最差':>7}{'周末均值':>9}{'周末σ':>7}{'周末最差':>8}{'样本':>5}")
    for cd, m, s, w, wm, ws, wmin, n_wk in gap_stats(aligned, i0, i1):
        print(f"{cd} {P.ASSETS[cd]['name']:<9}{m:>8.2f}{s:>7.2f}{w:>7.2f}{wm:>9.2f}{ws:>7.2f}{wmin:>8.2f}{n_wk:>5}")

    print('\n--- 周五信号翻转的执行价差异 (A=下周一开盘 vs B=周五收盘, %) ---')
    for cd, ns, ms, nb, mb in friday_flip_stats(inds, i0, i1):
        print(f"{cd}: 卖出翻转{ns}次, A相对B平均{ms:+.2f}% (负=A卖得更差); "
              f"买入翻转{nb}次, B相对A平均{mb:+.2f}% (负=B买得更贵)")

    # 引擎自检
    print(f"\n引擎自检: A口径应=validate_friction F0 (+17.8%/0.83/-16.2%, 235笔) → {T.fmt_perf(pA)} {trA}笔")

    import datetime as _dt
    out = os.path.join(PROJECT_ROOT, 'reports', f"execution_timing_validation_{_dt.datetime.now().strftime('%Y%m%d')}.md")
    with open(out, 'w', encoding='utf-8') as f:
        f.write('# 执行时序对比回测 (validate_execution_timing.py)\n\n')
        f.write(f'- A 次日开盘: {T.fmt_perf(pA)} / {trA}笔\n- B 当日执行: {T.fmt_perf(pB)} / {trB}笔\n')
        f.write(f'- 决策: B-A 年化差 {diff_pp:+.1f}pp → {verdict}\n')
    print(f'\n已保存: {out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 1.2 运行并核对**：`& "C:\Program Files\Python312\python.exe" validate_execution_timing.py`
  - 引擎自检行必须是 +17.8%/0.83/-16.2%/235笔（与 friction F0 一致）
  - 读完整输出，确认决策门槛结论
- [ ] **Step 1.3 决策**：B ≥ A-1pp → 继续 Task 2；否则停下向用户汇报

---

### Task 2: `preclose_signal.py` — 盘中条件预警

**Files:**
- Create: `hs300/preclose_signal.py`
- State: `reports/alert_state.json`（运行时生成）

**Interfaces:**
- Consumes: `send_advice.send_email/get_cash/compute_signals/get_prices_from_history`、`intraday_signal.get_realtime_quotes`、`signal_core.holdings_from_trades/qvix_position_ratio/fetch_qvix`
- Produces: `reports/alert_state.json`（字段 `date/last_signals/sell_alerts/buy_alerted/open_instructions`，Task 3 的 16:00 邮件读它）

- [ ] **Step 2.1 写脚本**（完整代码见下）：

```python
# -*- coding: utf-8 -*-
"""
盘中条件预警 — 尾盘前瞻执行链核心 (2026-10-26)
设计: reports/preclose_design_20261026.md | 信号零改动, 只改执行时序
调度: 计划任务 'CSI300盘中预警' 工作日 9:45/10:15/10:45/11:15/13:15/13:45/14:15/14:30
行为:
  时间窗外退出 → 实时行情(新浪→腾讯兜底) → compute_signals(合成K线) →
  持仓翻空→【卖出提醒】(≤2次/日, 立即) ; 空仓确认成立→【买入提醒】(≤1次/日) ;
  14:30-14:45 且有未执行指令 →【尾窗提醒】重发 ; 无指令不发(安静模式) ;
  状态: reports/alert_state.json
用法: python preclose_signal.py [--dry-run] [--force]
"""
import os
import sys
import json
import glob
import requests
from datetime import datetime
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import socket
socket.setdefaulttimeout(15)

from config import ASSETS, ORDER
from send_advice import send_email, get_cash, compute_signals
from intraday_signal import get_realtime_quotes
from signal_core import holdings_from_trades, qvix_position_ratio, fetch_qvix

ALERT_STATE = os.path.join(PROJECT_ROOT, 'reports', 'alert_state.json')
TRADES_DIR = os.path.join(PROJECT_ROOT, 'trades')
SELL_MAX, BUY_MAX = 2, 1


def log(msg):
    line = f'[{datetime.now().strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line)
    try:
        os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)
        with open(os.path.join(PROJECT_ROOT, 'logs', 'preclose_alert.log'), 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def in_session(now):
    t = now.hour * 60 + now.minute
    return now.weekday() < 5 and ((9*60+45 <= t <= 11*60+30) or (13*60 <= t <= 15*60))


def in_preclose_window(now):
    t = now.hour * 60 + now.minute
    return 14*60+30 <= t < 14*60+45


def get_realtime_quotes_ft():
    """新浪实时 → 失败品种腾讯实时兜底 (盘中可用)"""
    quotes = get_realtime_quotes()
    for code, info in ASSETS.items():
        if code in quotes:
            continue
        try:
            r = requests.get(f"https://qt.gtimg.cn/q={info['code']}", timeout=8)
            r.encoding = 'gbk'
            p = r.text.split('"')[1].split('~')
            if len(p) > 34 and float(p[3]) > 0:
                quotes[code] = {'price': float(p[3]), 'prev': float(p[4]),
                                'open': float(p[5]), 'high': float(p[33]), 'low': float(p[34])}
        except Exception:
            pass
    return quotes


def load_state():
    try:
        with open(ALERT_STATE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st):
    with open(ALERT_STATE, 'w', encoding='utf-8') as f:
        json.dump(st, f, ensure_ascii=False, indent=1)


def reset_if_new_day(st, today):
    if st.get('date') != today:
        st.clear()
        st.update({'date': today, 'last_signals': {}, 'sell_alerts': {},
                   'buy_alerted': [], 'open_instructions': []})


def today_executions(today):
    """今日已登记的 (buy/sell, code) 集合"""
    executed = set()
    for f in glob.glob(os.path.join(TRADES_DIR, '*_trades.csv')):
        try:
            df = pd.read_csv(f)
            rows = df[df['date'] == today] if 'date' in df.columns else df.iloc[0:0]
            for _, r in rows.iterrows():
                a = str(r.get('action', '')).strip().upper()
                if a in ('BUY', 'SELL'):
                    executed.add((a.lower(), str(r.get('code'))))
        except Exception:
            pass
    return executed


def build_alert_html(instructions, signals, holdings, qvix, qratio, missing, tail_window):
    today = datetime.now().strftime('%Y-%m-%d')
    now_txt = datetime.now().strftime('%H:%M')
    L = [f'<h2>盘中策略预警 ({now_txt})</h2>',
         f'<p style="color:#666;">{today} | 基于{now_txt}实时价的盘中信号 | 信号逻辑与主策略完全一致</p>']
    if missing:
        names = '、'.join(ASSETS[c]['name'] if c in ASSETS else c for c in missing)
        L.append(f'<div style="border:2px solid #d00;background:#fff3f3;padding:8px;margin:8px 0;">'
                 f'<b style="color:#d00;">⚠ 数据异常</b>: {names} 信号缺失, 谨慎执行</div>')
    for ins in instructions:
        crit = ('<br><span style="color:#d80;">⚠ 临界状态(贴近MA30/ADX阈值), 信号翻转概率高, 谨慎执行</span>'
                if ins.get('critical') else '')
        if ins['type'] == 'sell':
            L.append(f'<div style="border:2px solid red;border-radius:8px;padding:12px;margin:8px 0;background:#fff8f8;">'
                     f'<b style="font-size:16px;color:red;">▶ 尽快卖出</b> <b>{ins["name"]}</b> 全部 {ins["shares"]} 份<br>'
                     f'<span style="color:#333;">盘中信号已翻空 ({ins["reason"]}), 当日内执行, 不必等收盘</span>'
                     f'{crit}<br><span style="color:#999;font-size:12px;">现价 {ins["price"]:.4f} 附近限价单 | '
                     f'今日第{ins["count"]}次提醒</span></div>')
        else:
            L.append(f'<div style="border:2px solid #c00;border-radius:8px;padding:12px;margin:8px 0;background:#fff;">'
                     f'<b style="font-size:16px;color:#c00;">▶ 可买入</b> <b>{ins["name"]}</b> 约¥{ins["amount"]:,.0f} '
                     f'(≈{ins["shares"]}份)<br>'
                     f'<span style="color:#333;">确认成立(连续2日信号) 置信{ins["conf"]}, '
                     f'金额=可用¥{ins["base"]:,.0f} × QVIX{qratio:.0%} × 刹车{ins["brake_w"]:.0%}</span>'
                     f'{crit}<br><span style="color:#999;font-size:12px;">现价 {ins["price"]:.4f} 附近限价单, 不追高超+1%</span></div>')
    L.append('<h3>盘中信号明细</h3><table border="1" cellpadding="5" cellspacing="0" style="border-collapse:collapse;">')
    L.append('<tr style="background:#f0f0f0;"><th>品种</th><th>现价</th><th>MA30</th><th>ADX</th>'
             '<th>波动</th><th>信号</th><th>确认</th><th>持仓</th></tr>')
    for code in ORDER:
        if code not in signals:
            continue
        s = signals[code]
        sig = '<b style="color:green;">持有</b>' if s['signal'] else '<b style="color:red;">空仓</b>'
        conf_flag = ('✓' if s.get('confirmed') == 1 else '✗首日') if s['signal'] else '-'
        hold = holdings.get(code, 0)
        L.append(f'<tr><td>{s["name"]}</td><td>{s["price"]:.4f}</td><td>{s["ma"]:.4f}</td>'
                 f'<td>{s["adx"]}</td><td>{s["vol"]}%</td><td>{sig}</td><td>{conf_flag}</td>'
                 f'<td>{hold if hold > 0 else "-"}</td></tr>')
    L.append('</table>')
    if qvix:
        L.append(f'<p>📊 QVIX={qvix} → 买入按{qratio:.0%}仓位执行</p>')
    if tail_window:
        L.append('<div style="border:2px solid #d80;background:#fffaf0;padding:8px;margin:8px 0;">'
                 '<b style="color:#d80;">⏰ 尾盘执行窗口</b>: 距收盘约30分钟, 以上指令请在本窗口内'
                 '<b>限价单执行</b> (现价附近, 不追高); 错过则按 16:00 收盘邮件的次日开盘指令兜底</div>')
    L.append('<p style="color:#999;font-size:12px;">⚠ 盘中信号可能抖动(尤其贴近MA30/阈值时), '
             '若收盘维持以 16:00 收盘邮件为准; 执行后请登记 trades CSV (balance 必填)</p>')
    L.append('<p style="color:#999;font-size:12px;">本邮件由量化策略系统自动发送, 仅供参考, 不构成投资建议</p>')
    return '\n'.join(L)


def main():
    dry_run = '--dry-run' in sys.argv
    force = '--force' in sys.argv
    now = datetime.now()
    today = now.strftime('%Y-%m-%d')

    if not force and not in_session(now):
        log('时间窗外, 退出')
        return

    quotes = get_realtime_quotes_ft()
    if not quotes:
        log('实时行情获取失败(新浪+腾讯), 退出(状态不更新, 下个检查点重试)')
        return
    prices = {c: q['price'] for c, q in quotes.items()}

    # 信号 (append=True 合成K线; confirmed 已含"昨日收盘信号=1 且 今日盘中=1")
    signals, missing = compute_signals(prices, append=True)
    if not signals:
        log('信号计算失败, 退出')
        return
    stale = [c for c in signals if signals[c].get('data_date', '') < today]
    if stale:
        log(f'[WARN] 数据滞后品种(合成K线基于旧收盘): {stale}')

    holdings = holdings_from_trades(TRADES_DIR, list(ASSETS))
    holding = {c: s for c, s in holdings.items() if s > 0}
    cash = get_cash()
    qvix = fetch_qvix()
    qratio, _ = qvix_position_ratio(qvix)

    st = load_state()
    reset_if_new_day(st, today)

    instructions = []
    # 卖出提醒: 持仓品种盘中翻空 (立即, ≤2次/日)
    for code, sh in holding.items():
        s = signals.get(code)
        if s is None or s['signal'] != 0:
            continue
        if st['sell_alerts'].get(code, 0) >= SELL_MAX:
            continue
        instructions.append({
            'type': 'sell', 'code': code, 'name': s['name'], 'shares': sh,
            'price': s['price'], 'reason': s['reason'],
            'critical': abs(s['price'] / s['ma'] - 1) * 100 < 0.5,
            'count': st['sell_alerts'].get(code, 0) + 1})
    # 买入提醒: 空仓 + 确认成立 (≤1次/日; 弱信号不提醒)
    sells_value = sum(i['shares'] * i['price'] for i in instructions if i['type'] == 'sell')
    for code in ORDER:
        if holdings.get(code, 0) > 0 or code in st['buy_alerted']:
            continue
        s = signals.get(code)
        if s is None or s['signal'] != 1 or s.get('confirmed') != 1:
            continue
        if s['conf']['level'] == '弱':
            continue
        amt = (cash + sells_value) * qratio * s.get('brake_w', 1.0)
        if s['conf']['level'] == '中':
            amt /= 2
        instructions.append({
            'type': 'buy', 'code': code, 'name': s['name'],
            'amount': amt, 'base': cash + sells_value,
            'shares': int(amt / s['price'] / 100) * 100,
            'price': s['price'], 'conf': f'{s["conf"]["level"]}({s["conf"]["score"]}分)',
            'brake_w': s.get('brake_w', 1.0),
            'critical': (ASSETS[code]['adx_th'] - s['adx']) < 1})

    tail_window = in_preclose_window(now)
    # 尾窗提醒: 无新指令但当日有未执行指令 → 重发(不计入提醒次数)
    if tail_window and not instructions and st['open_instructions']:
        executed = today_executions(today)
        pend = [dict(i, resent=True) for i in st['open_instructions']
                if (i['type'], i['code']) not in executed]
        instructions = pend

    if instructions and not dry_run:
        kinds = '+'.join(i['type'] for i in instructions)
        names = '+'.join(i['name'] for i in instructions)
        tag = '尾窗提醒' if tail_window else '盘中预警'
        subject = f'[{today} {tag}] {kinds}: {names}'
        html = build_alert_html(instructions, signals, holdings, qvix, qratio, missing, tail_window)
        ok = send_email(subject, html)
        log(f'邮件{"已发送" if ok else "发送失败"}: {subject}')
    elif instructions:
        log(f'[dry-run] 将发送 {len(instructions)} 条指令: ' +
            '; '.join(f"{i['type']} {i['name']}" for i in instructions))
    else:
        log('无操作指令, 安静模式(不发邮件)')

    # 更新状态 (重发的尾窗提醒不计数)
    for i in instructions:
        if i.get('resent'):
            continue
        if i['type'] == 'sell':
            st['sell_alerts'][i['code']] = st['sell_alerts'].get(i['code'], 0) + 1
        elif i['code'] not in st['buy_alerted']:
            st['buy_alerted'].append(i['code'])
    st['last_signals'] = {c: signals[c]['signal'] for c in signals}
    if instructions:
        executed = today_executions(today)
        st['open_instructions'] = [i for i in (st['open_instructions'] + instructions)
                                   if (i['type'], i['code']) not in executed]
    save_state(st)
    log(f'检查完成: 指令{len(instructions)}条 (卖出提醒{st["sell_alerts"]}, 买入已提醒{st["buy_alerted"]})')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2.2 冒烟**：`python preclose_signal.py --force --dry-run`（应打印指令/安静模式，不发邮件）；再跑 `python preclose_signal.py`（当前时间若窗外应打印退出）
- [ ] **Step 2.3 核对**：`reports/alert_state.json` 生成且字段正确；`logs/preclose_alert.log` 有记录

---

### Task 3: `send_advice.py` — 16:00 收盘邮件补登记提示（小改）

**Files:**
- Modify: `hs300/send_advice.py`

**Interfaces:**
- Consumes: `reports/alert_state.json`（Task 2 产物）

- [ ] **Step 3.1 新增函数**（放在 `get_cash` 之后）：

```python
def build_registration_reminder_html():
    """close模式: 盘中预警指令 vs 今日trades登记 → 补登记提示 (无当日提醒返回None)"""
    import json
    path = os.path.join(ROOT, 'reports', 'alert_state.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            st = json.load(f)
    except Exception:
        return None
    today = datetime.now().strftime('%Y-%m-%d')
    if st.get('date') != today or not st.get('open_instructions'):
        return None
    executed = set()
    for f2 in glob.glob(os.path.join(ROOT, 'trades', '*_trades.csv')):
        try:
            df = pd.read_csv(f2)
            rows = df[df['date'] == today] if 'date' in df.columns else df.iloc[0:0]
            for _, r in rows.iterrows():
                a = str(r.get('action', '')).strip().upper()
                if a in ('BUY', 'SELL'):
                    executed.add((a.lower(), str(r.get('code'))))
        except Exception:
            pass
    pend = [i for i in st['open_instructions'] if (i['type'], i['code']) not in executed]
    if pend:
        names = '、'.join(f"{'卖出' if i['type'] == 'sell' else '买入'}{ASSETS.get(i['code'], {}).get('name', i['code'])}"
                          for i in pend)
        return (f'<div style="border:2px solid #d80;background:#fffaf0;padding:8px;margin:8px 0;">'
                f'<b style="color:#d80;">📝 补登记提醒</b>: 今日盘中预警 {names} 未见 trades 登记, '
                f'若已执行请补登记 (balance 必填); 若未执行请忽略并按本邮件次日开盘指令执行</div>')
    return '<p style="color:#27ae60;">📝 今日盘中预警指令均已执行登记 ✓</p>'
```

（`import glob, pandas as pd` 已在模块内可用：`get_cash` 内是局部 import — 此函数内同样局部 `import glob, pandas as pd`）

- [ ] **Step 3.2 插入调用**：`build_email_html` 内卖出指令循环之后、`# ===== 二、信号明细 =====` 之前：

```python
    if not is_mid:
        reg = build_registration_reminder_html()
        if reg:
            L.append(reg)
```

- [ ] **Step 3.3 冒烟**：`python -c "import send_advice"`（纯导入验证，无中文数据查询）+ 手动检查 close 模式 HTML 生成不报错：
  `python send_advice.py close` 的替代验证——用 dry 方式：临时跑 `python -c` 不行（中文），写一次性验证脚本或直接看下一交易日 16:00 实发。**验证口径**：语法级 `python -m py_compile send_advice.py` + 导入级 `import` 通过。

---

### Task 4: 计划任务变更（pwsh，§6.4）

**Files:** Windows Task Scheduler（无代码文件）

- [ ] **Step 4.1 查现状**：`Get-ScheduledTask | Where-Object TaskName -like '*CSI300*' | Format-Table TaskName, State -AutoSize`
- [ ] **Step 4.2 新增盘中预警**（多触发时间，动作纯 Python）：

```powershell
$act = New-ScheduledTaskAction -Execute "C:\Program Files\Python312\python.exe" `
    -Argument '"D:\opencode\个人投资相关\hs300\preclose_signal.py"'
$triggers = '9:45','10:15','10:45','11:15','13:15','13:45','14:15','14:30' | ForEach-Object {
    New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $_
}
Register-ScheduledTask -TaskName "CSI300盘中预警" -Action $act -Trigger $triggers -Force
```

- [ ] **Step 4.3 取消旧任务**：`Unregister-ScheduledTask -TaskName "CSI300午间快照" -Confirm:$false`；`Unregister-ScheduledTask -TaskName "CSI300午间邮件" -Confirm:$false`
- [ ] **Step 4.4 核对**：`Get-ScheduledTask | Where-Object TaskName -like '*CSI300*'` 应只剩 盘中预警/收盘邮件/策略每日刷新；Execute 列全为 `python.exe`

---

### Task 5: OPERATION.md 文档同步

- [ ] §2.2 信号时点 → 改"T日盘中实时监测(每30分钟)+收盘确认；有操作当日内执行(限价)，无操作收盘汇报；次日开盘为兜底"
- [ ] §3.1 每日流程表 → 删 11:35/12:00 行，加"9:45~14:30 每30分钟 盘中条件预警(有操作才发邮件) / 14:30 尾窗提醒"
- [ ] §6.1 定时任务清单 → 加 CSI300盘中预警，删午间快照/午间邮件
- [ ] §十三B 检查表 → 加"收到盘中预警→当日内限价执行；执行后登记"
- [ ] §11.3 #27 → 执行时序对比结论（Task 1 数字）
- [ ] 改动记录 → 2026-10-26 条目

---

### Task 6: 端到端验证 + 收尾

- [ ] 下一个交易日盘中（9:45 后）观察 `logs/preclose_alert.log` 与邮箱实收
- [ ] 16:00 收盘邮件确认补登记提示行正常
- [ ] 全部通过后：连同此前未提交文件（OPERATION/README/friction/plateau 脚本与报告）一并 commit（**需用户确认**）
