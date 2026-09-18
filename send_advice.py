"""
策略邮件推送 v3 (整合版)
午间(mid): 上午收盘后分析 + 下午操作建议
收盘(close): 全天分析 + 次日操作建议

信号计算统一走 signal_core, 与 auto_refresh/signals_log 完全一致
轮动规则与 dual_rotation/intraday_signal 一致: 多品种有信号时只指向 ADX 最高者
"""
import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ===== 全局网络超时保护 =====
import socket
socket.setdefaulttimeout(15)
try:
    import requests.sessions
    import requests
    _orig_send = requests.sessions.Session.request
    def _timeout_send(self, method, url, **kwargs):
        kwargs.setdefault('timeout', 15)
        return _orig_send(self, method, url, **kwargs)
    requests.sessions.Session.request = _timeout_send
except Exception:
    pass

from config import ASSETS, ORDER, PROJECT_ROOT as ROOT, HISTORY_DB
from signal_core import (last_signal_state, compute_confidence, qvix_position_ratio,
                         fetch_qvix, CONFIRM_DAYS, vol_brake_weight, VOL_BRAKE_THRESHOLD,
                         position_shares_from_csv)

# ========== 邮件配置 ==========
# ⚠️ 授权码已移出版本库 (2026-09-10 曾泄漏到公开 GitHub 仓库, 务必已重置旧授权码)
# 读取顺序: 环境变量 SMTP_AUTH_CODE → local_secrets.py (被 .gitignore 忽略, 不上传)
def _load_auth_code():
    import os
    code = os.environ.get('SMTP_AUTH_CODE', '').strip()
    if code:
        return code
    try:
        from local_secrets import SMTP_AUTH_CODE
        return str(SMTP_AUTH_CODE).strip()
    except Exception:
        return 'YOUR_AUTH_CODE'

SMTP_CONFIG = {
    'sender': 'aassdfs@163.com',
    'auth_code': _load_auth_code(),
    'to': 'aassdfs@qq.com',
    'smtp_server': 'smtp.163.com',
    'smtp_port': 465,
}


# ========== 统一信号计算 ==========
def compute_signals(prices, append=False):
    """
    计算三品种信号 (统一逻辑)
    prices: {code: 最新价}
    append: True=把价格追加为合成K线 (午间实时价); False=只用库内数据 (收盘模式,
            价格来自库内最后一根, 再追加只会造成重复K线使指标失真)
    返回: (results, missing)  missing = 信号计算失败的品种列表
    """
    import duckdb
    import pandas as pd

    conn = duckdb.connect(HISTORY_DB)
    results = {}
    missing = []
    today_str = datetime.now().strftime('%Y-%m-%d')

    for code, info in ASSETS.items():
        try:
            if code not in prices:
                missing.append(code)
                continue
            df = conn.execute(
                f"SELECT date, open, high, low, close FROM daily_ohlc WHERE code='{code}' ORDER BY date"
            ).fetchdf()
            if df.empty:
                missing.append(code)
                continue
            df['date'] = pd.to_datetime(df['date'])
            data_date = df['date'].iloc[-1].strftime('%Y-%m-%d')
            c = df['close'].astype(float)
            h = df['high'].astype(float)
            l = df['low'].astype(float)

            if append and df['date'].iloc[-1].normalize() < pd.Timestamp.now().normalize():
                # 库内尚无当日K线才追加实时价合成K线 (防止重复K线使指标失真)
                px = float(prices[code])
                c = pd.concat([c, pd.Series([px])]).reset_index(drop=True)
                h = pd.concat([h, pd.Series([px])]).reset_index(drop=True)
                l = pd.concat([l, pd.Series([px])]).reset_index(drop=True)

            st = last_signal_state(c, h, l, info['ma_p'], info['adx_th'], info['vol_th'])
            score, level, reasons = compute_confidence(st['price'], st['ma'], st['adx'], st['vol'], info)

            if st['signal'] == 1:
                reason = f"ADX>{info['adx_th']}" if st['trend'] else ('低波动' if st['low_vol'] else '价格>MA')
            else:
                reason = '价格<MA' if not st['above_ma'] else '高波动+低趋势'

            results[code] = {
                'name': info['name'], 'price': st['price'],
                'ma': round(st['ma'], 4),
                'adx': round(st['adx'], 1),
                'vol': round(st['vol'], 1),
                'above_ma': st['above_ma'], 'low_vol': st['low_vol'], 'trend': st['trend'],
                'signal': st['signal'], 'confirmed': st['confirmed'], 'reason': reason,
                'conf': {'score': score, 'level': level},
                'brake_w': vol_brake_weight(st['vol']),
                'data_date': data_date,
            }
        except Exception as e:
            print(f'[WARN] {code} 信号计算失败: {e}')
            missing.append(code)

    conn.close()
    return results, missing


def get_prices_from_history():
    """收盘模式: 从历史库取最新收盘价 (与午间同一计算逻辑)"""
    import duckdb
    conn = duckdb.connect(HISTORY_DB)
    prices = {}
    for code in ASSETS:
        try:
            row = conn.execute(
                "SELECT close FROM daily_ohlc WHERE code=? ORDER BY date DESC LIMIT 1", [code]
            ).fetchone()
            if row:
                prices[code] = row[0]
        except Exception:
            pass
    conn.close()
    return prices


def get_prices_realtime():
    """午间模式: 新浪实时行情"""
    import requests
    headers = {"Referer": "https://finance.sina.com.cn"}
    prices = {}
    for code, info in ASSETS.items():
        try:
            r = requests.get(f"https://hq.sinajs.cn/list={info['code']}", headers=headers, timeout=8)
            r.encoding = "gbk"
            p = r.text.strip().split('"')[1].split(",")
            px = float(p[3]) if p[3] != "0.000" else float(p[2])
            prices[code] = px
        except Exception:
            pass
    return prices


def check_position(code):
    """当前持仓份额 (统一委托 signal_core.position_shares_from_csv, 与 intraday_signal 同源)"""
    return position_shares_from_csv(os.path.join(PROJECT_ROOT, 'trades', f'{code}_trades.csv'))


# ========== 邮件正文生成 ==========
def get_cash():
    """读取当前现金 (取所有交易记录中日期最新的一笔有效balance)
    若全局最新交易行的 balance 为空则告警 (OPERATION §4.3: balance 必须记录准确)"""
    import pandas as pd
    import glob
    files = glob.glob(os.path.join(PROJECT_ROOT, 'trades', '*_trades.csv'))
    rows = []  # (date_str, balance or None)
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'date' not in df.columns or len(df) == 0:
                continue
            has_bal = 'balance' in df.columns
            for _, r in df.iterrows():
                d = r.get('date')
                if pd.isna(d):
                    continue
                b = r.get('balance') if has_bal else None
                bv = None if (b is None or pd.isna(b)) else float(b)
                rows.append((str(d), bv))
        except Exception:
            pass
    if not rows:
        return 0
    rows.sort(key=lambda x: x[0])
    if rows[-1][1] is None:
        print(f'[WARN] 最新交易日 {rows[-1][0]} 的 balance 为空, 现金显示可能滞后, 请补录 trades CSV')
    for _, bv in reversed(rows):
        if bv is not None:
            return bv
    return 0


def build_email_html(mode, signals, holdings, missing=None):
    """
    生成邮件HTML (午间/收盘区分) - 明确指令式建议
    轮动规则: 有信号品种中只指向 ADX 最高者, 买入金额按QVIX比例调节
    """
    today = datetime.now().strftime('%Y-%m-%d')
    is_mid = (mode == 'mid')
    title = '午间策略快报' if is_mid else '收盘策略报告'
    sub = '上午收盘分析 + 下午操作指令' if is_mid else '全天分析 + 次日操作指令'

    missing = missing or []
    cash = get_cash()
    qvix = fetch_qvix()
    qratio, qnote = qvix_position_ratio(qvix)

    total_value = cash + sum(holdings.get(c, 0) * signals[c]['price'] for c in holdings if c in signals)

    # 轮动指向: 有信号且已确认(连续2日)的品种中选ADX最高
    # 持仓品种只要原始信号为1就保持持有(出场不等确认), 新入场才需要确认
    candidates = sorted(
        [c for c in ORDER if c in signals and signals[c]['signal'] == 1
         and (signals[c].get('confirmed') == 1 or holdings.get(c, 0) > 0)],
        key=lambda c: signals[c]['adx'], reverse=True
    )
    pick = candidates[0] if candidates else None

    # 拟卖出持仓 = 持有但非轮动指向
    sells = [c for c in holdings if holdings.get(c, 0) > 0 and c != pick]
    # 卖出回笼资金并入买入基数 (T+1开盘同日卖买, 资金可用)
    sell_value = sum(holdings[c] * signals[c]['price'] for c in sells if c in signals)
    buy_base = cash + sell_value

    L = []
    L.append(f'<h2>{title}</h2>')
    L.append(f'<p style="color:#666;">{today} | {sub} | 信号计算逻辑与主策略完全一致</p>')

    if missing:
        miss_names = '、'.join(ASSETS[c]['name'] if c in ASSETS else c for c in missing)
        L.append(f'<div style="border:2px solid #d00;background:#fff3f3;padding:8px;margin:8px 0;">'
                 f'<b style="color:#d00;">⚠ 数据异常</b>: {miss_names} 信号缺失, 以下指令仅供参考, 谨慎执行</div>')

    if not is_mid:
        stale = [c for c in ORDER if c in signals and signals[c].get('data_date', '') < today]
        if stale:
            dts = signals[stale[0]]['data_date']
            miss_names = '、'.join(ASSETS[c]['name'] for c in stale if c in ASSETS)
            L.append(f'<div style="border:2px solid #d80;background:#fffaf0;padding:8px;margin:8px 0;">'
                     f'<b style="color:#d80;">⚠ 数据滞后</b>: {miss_names} 最新数据为 {dts} 收盘, '
                     f'指令基于该日信号 (数据源可能延迟, 请确认今日行情后执行)</div>')

    # ===== 一、明确操作指令 (放最前面, 最醒目) =====
    L.append('<h3 style="background:#1a1a2e;color:white;padding:8px;">一、操作指令</h3>')

    if pick:
        s = signals[pick]
        conf = s['conf']
        hold = holdings.get(pick, 0)
        time_txt = '下午' if is_mid else '明天开盘'
        if hold > 0:
            if s.get('brake_w', 1.0) < 1.0:
                # 极端波动刹车: 持仓品种波动率超阈值, 减仓至上限
                target = int(hold * s['brake_w'] / 100) * 100
                sell_sh = hold - target
                action = f'减仓至{s["brake_w"]:.0%}(波动刹车)'
                detail = (f'20日波动率{s["vol"]:.1f}% > {VOL_BRAKE_THRESHOLD}%: '
                          f'卖出约{sell_sh}份, 保留{target}份 (回落到{VOL_BRAKE_THRESHOLD}%以下后恢复满仓)')
                color = 'orange'
            else:
                action = '继续持有'
                detail = f'不卖出, 持有 {hold} 份不动 (波动率{s["vol"]:.1f}%正常)'
                color = 'green'
        elif s.get('confirmed', 1) == 0:
            action = '观望(待确认)'
            detail = f'信号首日, 需连续{CONFIRM_DAYS}日确认, 今日不追; 明日若信号仍在则可买入'
            color = 'gray'
        elif conf['level'] == '强':
            action = '买入'
            amt = buy_base * qratio * s.get('brake_w', 1.0)
            detail = (f'买入约¥{amt:,.0f} (可用资金¥{buy_base:,.0f} × QVIX{qratio:.0%} × 波动刹车{s.get("brake_w", 1.0):.0%}), '
                      f'挂单价格 {s["price"]:.4f} 附近, 不追高超过 +1%')
            color = 'red'
        elif conf['level'] == '中':
            action = '买入(半仓)'
            amt = buy_base * qratio * s.get('brake_w', 1.0) / 2
            detail = (f'买入约¥{amt:,.0f} (可用资金¥{buy_base:,.0f} × QVIX{qratio:.0%} × 波动刹车{s.get("brake_w", 1.0):.0%} × 50%), '
                      f'挂单价格 {s["price"]:.4f} 附近, 收盘确认后再补')
            color = 'orange'
        else:
            action = '观望'
            detail = '不买入(弱信号, 等收盘确认)'
            color = 'gray'

        L.append(f'<div style="border:2px solid {color};border-radius:8px;padding:12px;margin:8px 0;background:#fff;">'
                 f'<b style="font-size:16px;color:{color};">▶ {action}</b> '
                 f'<b>{s["name"]}</b> (置信{conf["level"]} {conf["score"]}分)<br>'
                 f'<span style="color:#333;">{detail}</span><br>'
                 f'<span style="color:#999;font-size:12px;">执行时间: {time_txt} | 现价 {s["price"]:.4f} | 轮动首选(ADX {s["adx"]})</span>'
                 f'</div>')

        # 有信号但非首选的品种 → 明确不买, 避免误操作
        for code in candidates[1:]:
            s2 = signals[code]
            L.append(f'<div style="border:1px dashed #999;border-radius:8px;padding:8px;margin:8px 0;background:#fafafa;">'
                     f'<b style="color:#666;">○ {s2["name"]}</b> 有信号但非轮动首选 (ADX {s2["adx"]} < {s["adx"]}), '
                     f'<b>不操作</b></div>')
    else:
        time_txt = '下午' if is_mid else '明天'
        L.append(f'<div style="border:2px solid #999;border-radius:8px;padding:12px;margin:8px 0;background:#fff;">'
                 f'<b style="font-size:16px;color:#999;">▶ 不操作</b> '
                 f'<b>三个品种均无买入信号</b><br>'
                 f'<span style="color:#333;">继续持有现金(¥{cash:,.0f}), 不做任何买入</span><br>'
                 f'<span style="color:#999;font-size:12px;">等下一个信号出现再操作</span>'
                 f'</div>')

    # 信号首日品种提示 (不构成买入指令, 明日确认后才可买)
    first_day = [c for c in ORDER if c in signals and signals[c]['signal'] == 1
                 and signals[c].get('confirmed', 1) == 0]
    if first_day:
        names = '、'.join(ASSETS[c]['name'] for c in first_day)
        L.append(f'<div style="border:1px dashed #c80;border-radius:8px;padding:8px;margin:8px 0;background:#fffaf0;">'
                 f'<b style="color:#c80;">◔ 信号首日</b>: {names} — 今日不追, '
                 f'明日收盘若信号仍在即可买入 (连续{CONFIRM_DAYS}日确认规则)</div>')

    # 持仓但非轮动指向 → 卖出指令
    for code in sells:
        hold = holdings[code]
        s = signals.get(code, {})
        time_txt = '下午' if is_mid else '明天开盘'
        L.append(f'<div style="border:2px solid red;border-radius:8px;padding:12px;margin:8px 0;background:#fff8f8;">'
                 f'<b style="font-size:16px;color:red;">▶ 卖出</b> '
                 f'<b>{ASSETS[code]["name"]}</b> 全部 {hold} 份<br>'
                 f'<span style="color:#333;">信号已空仓或轮动转向, 必须卖出</span><br>'
                 f'<span style="color:#999;font-size:12px;">执行时间: {time_txt} | 现价 {s.get("price", 0):.4f}</span>'
                 f'</div>')

    # ===== 二、信号明细 =====
    L.append(f'<h3 style="background:#333;color:white;padding:8px;">二、信号明细</h3>')
    L.append('<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;width:100%;">')
    L.append('<tr style="background:#f0f0f0;"><th>品种</th><th>价格</th><th>MA30</th><th>ADX</th><th>波动</th><th>信号</th><th>确认</th><th>置信</th><th>持仓</th></tr>')
    for code in ORDER:
        if code not in signals:
            continue
        s = signals[code]
        sig_txt = '<b style="color:green;">持有</b>' if s['signal'] == 1 else '<b style="color:red;">空仓</b>'
        if s['signal'] == 1:
            conf_txt_flag = '✓已确认' if s.get('confirmed') == 1 else f'✗待确认(第1日)'
        else:
            conf_txt_flag = '-'
        conf = s['conf']
        conf_color = {'强': 'green', '中': 'orange', '弱': 'gray'}.get(conf['level'], 'gray')
        conf_txt = f'<b style="color:{conf_color};">{conf["level"]}({conf["score"]})</b>' if s['signal'] == 1 else '-'
        hold = holdings.get(code, 0)
        hold_txt = f'{hold}份' if hold > 0 else '-'
        L.append(f'<tr><td>{s["name"]}</td><td>{s["price"]:.4f}</td><td>{s["ma"]:.4f}</td>'
                 f'<td>{s["adx"]}</td><td>{s["vol"]}%{"" if s.get("brake_w", 1.0) >= 1 else "(刹车)"}</td>'
                 f'<td>{sig_txt}</td><td>{conf_txt_flag}</td><td>{conf_txt}</td><td>{hold_txt}</td></tr>')
    L.append('</table>')

    # ===== 三、背景参考 =====
    L.append(f'<h3 style="background:#555;color:white;padding:8px;">三、背景参考</h3>')
    if qvix:
        qvix_txt = f'QVIX={qvix} ' + ('(平静)' if qvix < 20 else '(偏紧)' if qvix < 25 else '(恐慌)' if qvix < 30 else '(极度恐慌)')
        L.append(f'<p>📊 <b>QVIX恐慌指数</b>: {qvix_txt} → 买入按 <b>{qratio:.0%}</b> 仓位执行 ({qnote})')
    else:
        L.append(f'<p>📊 <b>QVIX恐慌指数</b>: 获取失败, 按100%仓位执行')
    L.append(f'<p>💰 当前资金: 现金 ¥{cash:,.0f}, 总资产约 ¥{total_value:,.0f}</p>')
    L.append('<p style="color:#999;font-size:12px;">' + ('* 午间信号基于上午数据, 下午行情可能变化, 若下午信号转强/转弱, 以收盘邮件为准' if is_mid else '* 收盘信号为当日最终信号, 次日按此执行') + '</p>')

    L.append('<hr>')
    L.append('<p style="color:#999;font-size:12px;">本邮件由量化策略系统自动发送, 仅供参考, 不构成投资建议</p>')
    return '\n'.join(L)


def send_email(subject, html):
    msg = MIMEText(html, 'html', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = SMTP_CONFIG['sender']
    msg['To'] = SMTP_CONFIG['to']
    try:
        server = smtplib.SMTP_SSL(SMTP_CONFIG['smtp_server'], SMTP_CONFIG['smtp_port'], timeout=30)
        server.login(SMTP_CONFIG['sender'], SMTP_CONFIG['auth_code'])
        server.sendmail(SMTP_CONFIG['sender'], [SMTP_CONFIG['to']], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f'邮件发送失败: {e}')
        return False


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'close'
    if 'YOUR_AUTH_CODE' in SMTP_CONFIG['auth_code']:
        print('请先配置163邮箱授权码')
        return

    # 获取价格 (午间=实时, 收盘=历史库)
    if mode == 'mid':
        prices = get_prices_realtime()
        time_label = '午间'
    else:
        prices = get_prices_from_history()
        time_label = '收盘'

    if not prices:
        print('价格获取失败')
        return

    # 统一信号计算 (午间追加实时价合成当日盘中K线; 收盘只用库内数据)
    signals, missing = compute_signals(prices, append=(mode == 'mid'))
    if missing:
        print(f'[WARN] 信号缺失品种: {missing}')
    stale = [c for c in signals if signals[c].get('data_date', '') < datetime.now().strftime('%Y-%m-%d')]
    if stale:
        print(f"[WARN] 数据滞后品种(以库内最新收盘信号为准): {stale}")

    # 持仓
    holdings = {c: check_position(c) for c in ASSETS}

    # 生成邮件
    html = build_email_html(mode, signals, holdings, missing)
    today = datetime.now().strftime('%Y-%m-%d')

    # 轮动指向决定标题 (与正文一致: 已确认信号才提示买入)
    candidates = sorted(
        [c for c in ORDER if c in signals and signals[c]['signal'] == 1
         and (signals[c].get('confirmed') == 1 or holdings.get(c, 0) > 0)],
        key=lambda c: signals[c]['adx'], reverse=True
    )
    pick = candidates[0] if candidates else None
    if pick:
        conf_level = signals[pick]['conf']['level']
        if conf_level == '强':
            subject = f'[{today} {time_label}] 策略: 买入 {ASSETS[pick]["name"]}'
        elif conf_level == '中':
            subject = f'[{today} {time_label}] 策略: 关注 {ASSETS[pick]["name"]} (中等信号)'
        else:
            subject = f'[{today} {time_label}] 策略: 观望 ({ASSETS[pick]["name"]}弱信号待确认)'
    else:
        subject = f'[{today} {time_label}] 策略: 空仓观望'

    if missing:
        subject = f'[数据异常]{subject}'

    ok = send_email(subject, html)
    if ok:
        print(f'[OK] {time_label}邮件已发送: {subject}')
    else:
        print('[FAIL] 邮件发送失败')


if __name__ == '__main__':
    main()
