# -*- coding: utf-8 -*-
"""
盘中条件预警 — "明日指令预告"链路 (2026-10-26, 2026-10-28 增开盘执行提醒)
设计: reports/preclose_design_20261026.md §11 (回测否决当日执行, 语义=预告不改执行时点)
调度: 计划任务 'CSI300盘中预警' 工作日 9:00/9:45/10:15/10:45/11:15/13:15/13:45/14:15/14:30
行为:
  9:00-9:15 → 开盘执行提醒 (昨晚收盘邮件指令重发: 持仓翻空→开盘无条件卖出, 确认成立→开盘买入, 防忘记) ;
  时间窗外退出 → 实时行情(新浪→腾讯兜底) → compute_signals(合成K线) →
  持仓品种盘中翻空 → 【卖出预告】(若收盘维持, 明日开盘卖出; ≤2次/日, 立即) ;
  空仓品种确认成立 → 【买入预告】(若收盘维持, 明日开盘买入; ≤1次/日) ;
  14:30-14:45 且当日有预警 → 【明日指令预览】重发+周末提示 ;
  无预警不发邮件(安静模式) ; 状态: reports/alert_state.json
用法: python preclose_signal.py [--dry-run] [--morning] [--force]
  --morning 强制走开盘提醒分支(测试用); --force 跳过时间窗(测试用)
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

from config import ASSETS, ORDER, HISTORY_DB
from send_advice import send_email, get_cash, compute_signals, get_prices_from_history
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
    """工作日交易时段: 9:45-11:30 / 13:00-15:00"""
    t = now.hour * 60 + now.minute
    return now.weekday() < 5 and ((9*60+45 <= t <= 11*60+30) or (13*60 <= t <= 15*60))


def in_preclose_window(now):
    """14:30-14:45 明日指令预览检查点"""
    t = now.hour * 60 + now.minute
    return 14*60+30 <= t < 14*60+45


def in_preopen_window(now):
    """9:00-9:15 开盘执行提醒窗口 (开盘前)"""
    t = now.hour * 60 + now.minute
    return 9*60 <= t < 9*60+15


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


def build_alert_html(instructions, signals, holdings, qvix, qratio, missing, preview):
    """预警邮件: 明日指令预告 (不改执行时点, 以16:00收盘邮件为准)"""
    today = datetime.now().strftime('%Y-%m-%d')
    now_txt = datetime.now().strftime('%H:%M')
    title = '明日指令预览' if preview else '盘中策略预警'
    L = [f'<h2>{title} ({now_txt})</h2>',
         f'<p style="color:#666;">{today} | 基于{now_txt}实时价的盘中信号 | 信号逻辑与主策略完全一致</p>']
    if missing:
        names = '、'.join(ASSETS[c]['name'] if c in ASSETS else c for c in missing)
        L.append(f'<div style="border:2px solid #d00;background:#fff3f3;padding:8px;margin:8px 0;">'
                 f'<b style="color:#d00;">⚠ 数据异常</b>: {names} 信号缺失, 谨慎参考</div>')
    for ins in instructions:
        crit = ('<span style="color:#d80;">　<b>⚠ 临界状态</b>(贴近MA30/ADX阈值), 信号翻转概率高, 预告可靠性低</span>'
                if ins.get('critical') else '')
        if ins['type'] == 'sell':
            title = _hdr(f'▶ 卖出预告 — {ins["name"]} ({ins["code"]})', '#0a8f3c')
            cells = (_cell('持仓份额', f'{ins["shares"]:,}份', '#0e7a35')
                     + _cell('现价', f'{ins["price"]:.4f}'))
            rules = (f'盘中信号已翻空 ({ins["reason"]}); 若收盘维持 → '
                     f'<b style="color:#0e7a35;">明日开盘卖出 — 无条件执行, 不设任何价格限制, 价格再低也跑</b><br>'
                     f'<span style="color:#333;">市价/对手价挂单确保成交 · 勿拖到下午</span>'
                     f'{crit}　<span style="color:#999;font-size:12px;">今日第{ins["count"]}次预告</span>')
            L.append(_card(title, cells, rules, '#0a8f3c', '#f0fff4'))
        else:
            title = _hdr(f'▶ 买入预告 — {ins["name"]} ({ins["code"]}) '
                         f'<span style="font-size:13px;font-weight:normal;">置信{ins["conf"]}</span>', '#c0392b')
            cells = (_hl_cell('挂单上限 (现价+1%)', f'{ins["price"]*1.01:.4f}')
                     + _cell('现价', f'{ins["price"]:.4f}')
                     + _cell('明日买入金额', f'¥{ins["amount"]:,.0f}')
                     + _cell('约份额', f'{ins["shares"]:,}份'))
            rules = (f'确认成立(连续2日信号); 若收盘维持 → <b style="color:#c0392b;">明日开盘买入</b><br>'
                     f'<b style="color:#c0392b;">⛔ 高开超上限 → 不追</b>等回落/次日　'
                     f'<b style="color:#0e7a35;">⬇ 低开 → 直接买（更便宜）</b><br>'
                     f'<span style="color:#333;">金额 = 可用¥{ins["base"]:,.0f} × QVIX{qratio:.0%} × '
                     f'刹车{ins["brake_w"]:.0%}</span>{crit}')
            L.append(_card(title, cells, rules, '#c0392b', '#fff5f5'))
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
    if preview:
        wk_txt = ('<br><span style="color:#d80;">📅 周末提示: 若为卖出指令, 周一开盘执行(跨周末), '
                  '周末消息面变化可能影响跳空, 请留意</span>' if datetime.now().weekday() == 4 else '')
        L.append('<div style="border:2px solid #d80;background:#fffaf0;padding:8px;margin:8px 0;">'
                 '<b style="color:#d80;">📅 明日指令预览</b>: 距收盘约30分钟, 以上预告若收盘维持'
                 '即成为明日开盘指令(以 16:00 收盘邮件为准确认)' + wk_txt + '</div>')
    L.append('<p style="color:#999;font-size:12px;">⚠ 盘中信号可能抖动(尤其贴近MA30/阈值时), '
             '本邮件仅为预告, <b>以 16:00 收盘邮件为准</b>; 执行时点不变: 次日开盘</p>')
    L.append('<p style="color:#999;font-size:12px;">本邮件由量化策略系统自动发送, 仅供参考, 不构成投资建议</p>')
    return '\n'.join(L)


# ---- 邮件票据卡构造器 (纯表格, 邮箱兼容; 红买绿卖A股习惯) ----
def _card(title_html, cells_html, rules_html, border, rules_bg):
    return (f'<table cellpadding="0" cellspacing="0" style="border-collapse:collapse;width:100%;margin:12px 0;">'
            f'<tr><td style="border:3px solid {border};border-radius:10px;background:#fff;padding:0;">'
            f'<table cellpadding="0" cellspacing="0" style="border-collapse:collapse;width:100%;">'
            f'<tr>{title_html}</tr>'
            f'<tr><td style="padding:14px 16px;">'
            f'<table cellpadding="0" cellspacing="0" style="border-collapse:collapse;"><tr>{cells_html}</tr></table>'
            f'<div style="background:{rules_bg};border-radius:6px;padding:10px 12px;margin-top:10px;font-size:14px;">{rules_html}</div>'
            f'</td></tr></table></td></tr></table>')


def _hdr(text, bg):
    return f'<td style="background:{bg};color:#fff;padding:10px 16px;font-size:18px;font-weight:bold;">{text}</td>'


def _cell(label, value, color='#333'):
    return (f'<td style="padding-right:36px;vertical-align:bottom;">'
            f'<div style="color:#999;font-size:13px;">{label}</div>'
            f'<div style="font-size:30px;font-weight:bold;color:{color};">{value}</div></td>')


def _hl_cell(label, value, bg='#c0392b', fg='#ffffff'):
    """高亮单元: 彩色背景徽章式大数字 (挂单上限等关键值; 嵌套表格实现, 邮箱兼容)"""
    return (f'<td style="padding-right:36px;vertical-align:bottom;">'
            f'<div style="color:#999;font-size:13px;">{label}</div>'
            f'<table cellpadding="0" cellspacing="0" style="border-collapse:collapse;margin-top:2px;">'
            f'<tr><td style="background:{bg};color:{fg};font-size:30px;font-weight:bold;'
            f'padding:4px 14px;border-radius:8px;">{value}</td></tr></table></td>')


def build_morning_html(instructions, signals, holdings, qvix, qratio, missing):
    """开盘执行提醒 (9:00): 昨晚16:00收盘邮件指令的重发, 防忘记"""
    today = datetime.now().strftime('%Y-%m-%d')
    L = [f'<h2>⏰ 今日开盘执行提醒 (9:00)</h2>',
         f'<p style="color:#666;">{today} | 昨晚16:00收盘邮件指令的重发, 开盘执行, 防忘记</p>']
    if missing:
        names = '、'.join(ASSETS[c]['name'] if c in ASSETS else c for c in missing)
        L.append(f'<div style="border:2px solid #d00;background:#fff3f3;padding:8px;margin:8px 0;">'
                 f'<b style="color:#d00;">⚠ 数据异常</b>: {names} 信号缺失, 以昨晚收盘邮件为准</div>')
    for ins in instructions:
        if ins['type'] == 'sell':
            title = _hdr(f'▶ 开盘卖出 — {ins["name"]} ({ins["code"]})', '#0a8f3c')
            cells = (_cell('持仓份额', f'{ins["shares"]:,}份', '#0e7a35')
                     + _cell('昨收参考', f'{ins["price"]:.4f}'))
            rules = (f'<b style="color:#0e7a35;">⚠ 无条件执行 — 不设任何价格限制, 价格再低也跑</b><br>'
                     f'<span style="color:#333;">市价/对手价挂单确保成交 · 开盘尽早执行勿拖到下午'
                     f'（实测尾盘杀跌 -6.7%）<br>原因: {ins["reason"]}</span>')
            L.append(_card(title, cells, rules, '#0a8f3c', '#f0fff4'))
        else:
            title = _hdr(f'▶ 开盘买入 — {ins["name"]} ({ins["code"]}) '
                         f'<span style="font-size:13px;font-weight:normal;">置信{ins["conf"]}</span>', '#c0392b')
            cells = (_hl_cell('挂单上限 (昨收+1%)', f'{ins["price"]*1.01:.4f}')
                     + _cell('昨收参考价', f'{ins["price"]:.4f}')
                     + _cell('买入金额', f'¥{ins["amount"]:,.0f}')
                     + _cell('约份额', f'{ins["shares"]:,}份'))
            rules = (f'<b style="color:#c0392b;">⛔ 高开超上限 → 不追</b><span style="color:#333;">, '
                     f'等盘中回落或次日</span>　　<b style="color:#0e7a35;">⬇ 低开 → 直接买（更便宜）</b><br>'
                     f'<span style="color:#333;">金额 = 可用¥{ins["base"]:,.0f} × QVIX{qratio:.0%} × '
                     f'刹车{ins["brake_w"]:.0%}</span>')
            L.append(_card(title, cells, rules, '#c0392b', '#fff5f5'))
    L.append('<h3>昨日收盘信号 (指令依据)</h3><table border="1" cellpadding="5" cellspacing="0" style="border-collapse:collapse;">')
    L.append('<tr style="background:#f0f0f0;"><th>品种</th><th>收盘</th><th>信号</th><th>确认</th><th>持仓</th></tr>')
    for code in ORDER:
        if code not in signals:
            continue
        s = signals[code]
        sig = '<b style="color:green;">持有</b>' if s['signal'] else '<b style="color:red;">空仓</b>'
        conf_flag = ('✓' if s.get('confirmed') == 1 else '✗') if s['signal'] else '-'
        hold = holdings.get(code, 0)
        L.append(f'<tr><td>{s["name"]}</td><td>{s["price"]:.4f}</td><td>{sig}</td><td>{conf_flag}</td>'
                 f'<td>{hold if hold > 0 else "-"}</td></tr>')
    L.append('</table>')
    if qvix:
        L.append(f'<p>📊 QVIX={qvix} → 买入按{qratio:.0%}仓位执行</p>')
    L.append('<p style="color:#999;font-size:12px;">本提醒为昨晚收盘邮件指令的重发, 口径一致; '
             '若开盘前有重大消息变化, 以 16:00 收盘邮件为最终确认</p>')
    L.append('<p style="color:#999;font-size:12px;">本邮件由量化策略系统自动发送, 仅供参考, 不构成投资建议</p>')
    return '\n'.join(L)


def morning_reminder(dry_run=False):
    """开盘执行提醒 (9:00 触发): 昨日收盘信号+当前持仓 → 今日开盘指令 (买卖同权)
    纯本地数据 (昨日收盘已入库), 不依赖实时行情"""
    today = datetime.now().strftime('%Y-%m-%d')
    prices = get_prices_from_history()
    if not prices:
        log('库内无数据, 跳过开盘提醒')
        return
    signals, missing = compute_signals(prices, append=False)  # 纯库内 = 昨日收盘口径
    if not signals:
        log('信号计算失败, 跳过开盘提醒')
        return
    holdings = holdings_from_trades(TRADES_DIR, list(ASSETS))
    holding = {c: s for c, s in holdings.items() if s > 0}
    cash = get_cash()
    qvix = fetch_qvix()
    qratio, _ = qvix_position_ratio(qvix)

    instructions = []
    # 卖出: 持仓且昨日收盘信号翻空 → 今日开盘无条件卖出
    for code, sh in holding.items():
        s = signals.get(code)
        if s and s['signal'] == 0:
            instructions.append({'type': 'sell', 'code': code, 'name': s['name'],
                                 'shares': sh, 'price': s['price'], 'reason': s['reason']})
    # 买入: 空仓且昨日收盘确认成立 → 今日开盘买入 (弱信号不发)
    for code in ORDER:
        if holdings.get(code, 0) > 0:
            continue
        s = signals.get(code)
        if s and s['signal'] == 1 and s.get('confirmed') == 1 and s['conf']['level'] != '弱':
            amt = cash * qratio * s.get('brake_w', 1.0)
            if s['conf']['level'] == '中':
                amt /= 2
            instructions.append({'type': 'buy', 'code': code, 'name': s['name'],
                                 'amount': amt, 'base': cash,
                                 'shares': int(amt / s['price'] / 100) * 100,
                                 'price': s['price'], 'brake_w': s.get('brake_w', 1.0),
                                 'conf': f'{s["conf"]["level"]}({s["conf"]["score"]}分)'})
    if not instructions:
        log('今日开盘无执行指令, 不发提醒(安静模式)')
        return
    kinds = '+'.join('卖出' if i['type'] == 'sell' else '买入' for i in instructions)
    names = '+'.join(i['name'] for i in instructions)
    subject = f'[{today} 开盘执行] {kinds}: {names}'
    html = build_morning_html(instructions, signals, holdings, qvix, qratio, missing)
    if dry_run:
        log(f'[dry-run] 将发送开盘提醒: {subject}')
        return
    ok = send_email(subject, html)
    log(f'开盘提醒{"已发送" if ok else "发送失败"}: {subject}')


def main():
    dry_run = '--dry-run' in sys.argv
    force = '--force' in sys.argv
    now = datetime.now()
    today = now.strftime('%Y-%m-%d')

    # 9:00-9:15 开盘执行提醒 (昨晚收盘邮件指令的重发, 防忘记; --morning 强制测试)
    if '--morning' in sys.argv or in_preopen_window(now):
        morning_reminder(dry_run)
        return

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
    # 卖出预告: 持仓品种盘中翻空 (立即, ≤2次/日)
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
    # 买入预告: 空仓 + 确认成立 (≤1次/日; 弱信号不预告)
    for code in ORDER:
        if holdings.get(code, 0) > 0 or code in st['buy_alerted']:
            continue
        s = signals.get(code)
        if s is None or s['signal'] != 1 or s.get('confirmed') != 1:
            continue
        if s['conf']['level'] == '弱':
            continue
        amt = cash * qratio * s.get('brake_w', 1.0)
        if s['conf']['level'] == '中':
            amt /= 2
        instructions.append({
            'type': 'buy', 'code': code, 'name': s['name'],
            'amount': amt, 'base': cash,
            'shares': int(amt / s['price'] / 100) * 100,
            'price': s['price'], 'conf': f'{s["conf"]["level"]}({s["conf"]["score"]}分)',
            'brake_w': s.get('brake_w', 1.0),
            'critical': (ASSETS[code]['adx_th'] - s['adx']) < 1})

    preview = in_preclose_window(now)
    # 明日指令预览: 14:30 无新预告但当日已有预告 → 重发(不计次数)
    if preview and not instructions and st['open_instructions']:
        instructions = [dict(i, resent=True) for i in st['open_instructions']]

    if instructions and not dry_run:
        kinds = '+'.join('卖出预告' if i['type'] == 'sell' else '买入预告' for i in instructions)
        names = '+'.join(i['name'] for i in instructions)
        tag = '明日指令预览' if preview else '盘中预警'
        subject = f'[{today} {tag}] {names} ({kinds})'
        html = build_alert_html(instructions, signals, holdings, qvix, qratio, missing, preview)
        ok = send_email(subject, html)
        log(f'邮件{"已发送" if ok else "发送失败"}: {subject}')
    elif instructions:
        log(f'[dry-run] 将发送 {len(instructions)} 条预告: ' +
            '; '.join(f"{'卖出' if i['type'] == 'sell' else '买入'} {i['name']}" for i in instructions))
    else:
        log('无操作预告, 安静模式(不发邮件)')

    # 更新状态 (预览重发不计数)
    for i in instructions:
        if i.get('resent'):
            continue
        if i['type'] == 'sell':
            st['sell_alerts'][i['code']] = st['sell_alerts'].get(i['code'], 0) + 1
        elif i['code'] not in st['buy_alerted']:
            st['buy_alerted'].append(i['code'])
    st['last_signals'] = {c: signals[c]['signal'] for c in signals}
    if instructions:
        st['open_instructions'] = [i for i in (st['open_instructions'] + instructions)
                                   if not i.get('resent')]
    save_state(st)
    log(f'检查完成: 预告{len(instructions)}条 (卖出{st["sell_alerts"]}, 买入已预告{st["buy_alerted"]})')


if __name__ == '__main__':
    main()
