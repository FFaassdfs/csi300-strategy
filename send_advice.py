"""
策略邮件推送 v2 (整合版)
午间(mid): 上午收盘后分析 + 下午操作建议
收盘(close): 全天分析 + 次日操作建议

统一信号计算逻辑: 午间用实时价, 收盘用已入库收盘价, 计算方式完全一致
"""
import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ========== 邮件配置 ==========
SMTP_CONFIG = {
    'sender': 'aassdfs@163.com',
    'auth_code': 'CFNyDY5dEPJu6Y5U',
    'to': 'aassdfs@qq.com',
    'smtp_server': 'smtp.163.com',
    'smtp_port': 465,
}

# ========== 品种配置 (唯一权威, 与轮动一致) ==========
ASSETS = {
    '510310': {'name': '沪深300ETF', 'code': 'sh510310', 'ma_p': 30, 'adx_th': 20, 'vol_th': 18},
    '159995': {'name': '芯片ETF',    'code': 'sz159995', 'ma_p': 30, 'adx_th': 25, 'vol_th': 15},
    '512800': {'name': '银行ETF',    'code': 'sh512800', 'ma_p': 30, 'adx_th': 25, 'vol_th': 18},
}

ORDER = ['510310', '159995', '512800']


# ========== 统一信号计算 ==========
def compute_signals(prices):
    """
    计算三品种信号 (统一逻辑)
    prices: {code: 最新价}
    返回: {code: {name, price, ma, adx, vol, above_ma, low_vol, trend, signal, reason}}
    """
    import duckdb
    import pandas as pd
    import numpy as np

    conn = duckdb.connect(os.path.join(PROJECT_ROOT, 'trading_history.duckdb'))
    results = {}

    for code, info in ASSETS.items():
        try:
            df = conn.execute(
                f"SELECT date, open, high, low, close FROM daily_ohlc WHERE code='{code}' ORDER BY date"
            ).fetchdf()
            df['date'] = pd.to_datetime(df['date'])
            c = df['close'].astype(float)
            h = df['high'].astype(float)
            l = df['low'].astype(float)

            # 追加最新价 (午间=实时价, 收盘=收盘价)
            px = prices[code]
            c = pd.concat([c, pd.Series([px])]).reset_index(drop=True)
            h = pd.concat([h, pd.Series([px])]).reset_index(drop=True)
            l = pd.concat([l, pd.Series([px])]).reset_index(drop=True)

            ma_p, adx_th, vol_th = info['ma_p'], info['adx_th'], info['vol_th']
            ma = c.rolling(ma_p).mean()
            vol = c.pct_change().rolling(20).std() * np.sqrt(252) * 100
            tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1/14, adjust=False).mean()
            up = h.diff(); dn = -l.diff()
            pdm = pd.Series(0.0, index=c.index); ndm = pd.Series(0.0, index=c.index)
            pdm.loc[(up > dn) & (up > 0)] = up
            ndm.loc[(dn > up) & (dn > 0)] = dn
            pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atr
            ndi = 100 * ndm.ewm(alpha=1/14, adjust=False).mean() / atr
            adx = (100 * abs(pdi - ndi) / (pdi + ndi + 1e-10)).ewm(alpha=1/14, adjust=False).mean()

            above = bool(c.iloc[-1] > ma.iloc[-1])
            low_vol = bool(vol.iloc[-1] < vol_th)
            trend = bool(adx.iloc[-1] > adx_th)
            sig = 1 if (above and (low_vol or trend)) else 0

            # 置信度
            conf = compute_confidence(px, ma.iloc[-1], adx.iloc[-1], vol.iloc[-1], info)

            reason = ''
            if sig == 1:
                reason = 'ADX>阈值' if trend else ('低波动' if low_vol else '价格>MA')
            else:
                reason = '价格<MA' if not above else '高波动+低趋势'

            results[code] = {
                'name': info['name'], 'price': px,
                'ma': round(float(ma.iloc[-1]), 4),
                'adx': round(float(adx.iloc[-1]), 1),
                'vol': round(float(vol.iloc[-1]), 1),
                'above_ma': above, 'low_vol': low_vol, 'trend': trend,
                'signal': sig, 'reason': reason,
                'conf': conf,
            }
        except Exception:
            pass

    conn.close()
    return results


def compute_confidence(price, ma, adx, vol, info):
    """信号置信度 (0-100)"""
    score = 0
    dist = (price / ma - 1) * 100 if ma > 0 else 0
    if dist > 3: score += 30
    elif dist > 1.5: score += 22
    elif dist > 0.5: score += 12
    else: score += 4

    adx_margin = adx - info['adx_th']
    if adx_margin > 10: score += 40
    elif adx_margin > 5: score += 30
    elif adx_margin > 0: score += 18
    elif adx_margin > -5: score += 8
    else: score += 2

    vol_margin = info['vol_th'] - vol
    if vol_margin > 5: score += 30
    elif vol_margin > 2: score += 22
    elif vol_margin > 0: score += 12
    else: score += 4

    if score >= 75: level = '强'
    elif score >= 50: level = '中'
    else: level = '弱'
    return {'score': score, 'level': level}


def get_prices_from_history():
    """收盘模式: 从历史库取最新收盘价 (与午间同一计算逻辑)"""
    import duckdb
    conn = duckdb.connect(os.path.join(PROJECT_ROOT, 'trading_history.duckdb'))
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
    """检查当前是否持有该品种"""
    import pandas as pd
    f = os.path.join(PROJECT_ROOT, 'trades', f'{code}_trades.csv')
    if not os.path.exists(f):
        return 0
    try:
        df = pd.read_csv(f)
        shares = 0
        for _, row in df.iterrows():
            a = str(row['action']).strip().upper()
            sh = int(row['shares'])
            shares += sh if a == 'BUY' else -sh
        return shares
    except Exception:
        return 0


def get_qvix():
    """获取QVIX (线程超时保护, 失败时不影响邮件发送)"""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
    def _fetch():
        import akshare as ak
        df = ak.index_option_300etf_qvix()
        return round(float(df.iloc[-1]['close']), 2)
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_fetch)
            return fut.result(timeout=12)
    except FutTimeout:
        print('QVIX获取超时(12s), 跳过')
    except Exception:
        pass
    return None


# ========== 邮件正文生成 ==========
def get_cash():
    """读取当前现金 (取所有交易记录中日期最新的一笔balance)"""
    import pandas as pd
    import glob
    files = glob.glob(os.path.join(PROJECT_ROOT, 'trades', '*_trades.csv'))
    best_date = None
    best_cash = 0
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'balance' in df.columns and 'date' in df.columns and len(df) > 0:
                for _, row in df.iterrows():
                    b = row.get('balance')
                    d = row.get('date')
                    if pd.notna(b) and pd.notna(d):
                        try:
                            bv = float(b)
                            dv = str(d)
                            if best_date is None or dv > best_date:
                                best_date = dv
                                best_cash = bv
                        except Exception:
                            pass
        except Exception:
            pass
    return best_cash


def build_email_html(mode, signals, holdings):
    """生成邮件HTML (午间/收盘区分) - 明确指令式建议"""
    today = datetime.now().strftime('%Y-%m-%d')
    is_mid = (mode == 'mid')
    title = '午间策略快报' if is_mid else '收盘策略报告'
    sub = '上午收盘分析 + 下午操作指令' if is_mid else '全天分析 + 次日操作指令'

    cash = get_cash()
    total_value = cash + sum(holdings.get(c, 0) * signals[c]['price'] for c in holdings if c in signals)

    L = []
    L.append(f'<h2>{title}</h2>')
    L.append(f'<p style="color:#666;">{today} | {sub} | 信号计算逻辑与主策略完全一致</p>')

    # ===== 一、明确操作指令 (放最前面, 最醒目) =====
    picks = [c for c in ORDER if c in signals and signals[c]['signal'] == 1]
    L.append('<h3 style="background:#1a1a2e;color:white;padding:8px;">一、操作指令</h3>')

    if picks:
        # 有信号品种
        for code in picks:
            s = signals[code]
            conf = s['conf']
            hold = holdings.get(code, 0)
            if hold > 0:
                action = '继续持有'
                detail = f'不卖出, 持有 {hold} 份不动'
                color = 'green'
            elif conf['level'] == '强':
                action = '买入'
                price_zone = f'{s["price"]:.4f} 附近'
                detail = f'全仓买入(¥{total_value:,.0f}), 挂单价格 {price_zone}, 不追高超过 +1%'
                color = 'red'
            elif conf['level'] == '中':
                action = '买入(半仓)'
                price_zone = f'{s["price"]:.4f} 附近'
                detail = f'半仓买入(约¥{total_value/2:,.0f}), 挂单价格 {price_zone}, 收盘确认后再补'
                color = 'orange'
            else:
                action = '观望'
                detail = f'不买入(弱信号, 等收盘确认)'
                color = 'gray'

            time_txt = '下午' if is_mid else '明天开盘'
            L.append(f'<div style="border:2px solid {color};border-radius:8px;padding:12px;margin:8px 0;background:#fff;">'
                     f'<b style="font-size:16px;color:{color};">▶ {action}</b> '
                     f'<b>{s["name"]}</b> (置信{conf["level"]} {conf["score"]}分)<br>'
                     f'<span style="color:#333;">{detail}</span><br>'
                     f'<span style="color:#999;font-size:12px;">执行时间: {time_txt} | 现价 {s["price"]:.4f}</span>'
                     f'</div>')
    else:
        time_txt = '下午' if is_mid else '明天'
        L.append(f'<div style="border:2px solid #999;border-radius:8px;padding:12px;margin:8px 0;background:#fff;">'
                 f'<b style="font-size:16px;color:#999;">▶ 不操作</b> '
                 f'<b>三个品种均无买入信号</b><br>'
                 f'<span style="color:#333;">继续持有现金(¥{cash:,.0f}), 不做任何买入</span><br>'
                 f'<span style="color:#999;font-size:12px;">等下一个信号出现再操作</span>'
                 f'</div>')

    # 持仓但无信号的 → 卖出指令
    for code, hold in holdings.items():
        if hold > 0 and code not in picks:
            s = signals.get(code, {})
            time_txt = '下午' if is_mid else '明天开盘'
            L.append(f'<div style="border:2px solid red;border-radius:8px;padding:12px;margin:8px 0;background:#fff8f8;">'
                     f'<b style="font-size:16px;color:red;">▶ 卖出</b> '
                     f'<b>{ASSETS[code]["name"]}</b> 全部 {hold} 份<br>'
                     f'<span style="color:#333;">信号已空仓, 必须卖出</span><br>'
                     f'<span style="color:#999;font-size:12px;">执行时间: {time_txt} | 现价 {s.get("price", 0):.4f}</span>'
                     f'</div>')

    # ===== 二、信号明细 =====
    L.append(f'<h3 style="background:#333;color:white;padding:8px;">二、信号明细</h3>')
    L.append('<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;width:100%;">')
    L.append('<tr style="background:#f0f0f0;"><th>品种</th><th>价格</th><th>MA30</th><th>ADX</th><th>波动</th><th>信号</th><th>置信</th><th>持仓</th></tr>')
    for code in ORDER:
        if code not in signals:
            continue
        s = signals[code]
        sig_txt = '<b style="color:green;">持有</b>' if s['signal'] == 1 else '<b style="color:red;">空仓</b>'
        conf = s['conf']
        conf_color = {'强': 'green', '中': 'orange', '弱': 'gray'}.get(conf['level'], 'gray')
        conf_txt = f'<b style="color:{conf_color};">{conf["level"]}({conf["score"]})</b>' if s['signal'] == 1 else '-'
        hold = holdings.get(code, 0)
        hold_txt = f'{hold}份' if hold > 0 else '-'
        L.append(f'<tr><td>{s["name"]}</td><td>{s["price"]:.4f}</td><td>{s["ma"]:.4f}</td>'
                 f'<td>{s["adx"]}</td><td>{s["vol"]}%</td><td>{sig_txt}</td><td>{conf_txt}</td><td>{hold_txt}</td></tr>')
    L.append('</table>')

    # ===== 三、背景参考 =====
    L.append(f'<h3 style="background:#555;color:white;padding:8px;">三、背景参考</h3>')
    qvix = get_qvix()
    if qvix:
        qvix_txt = f'QVIX={qvix} ' + ('(平静)' if qvix < 20 else '(偏紧)' if qvix < 25 else '(恐慌)' if qvix < 30 else '(极度恐慌)')
        L.append(f'<p>📊 <b>QVIX恐慌指数</b>: {qvix_txt} ' +
                 ('→ 仓位不受限' if qvix < 20 else '→ 建议降仓' if qvix >= 25 else '→ 正常'))
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

    # 统一信号计算
    signals = compute_signals(prices)

    # 持仓
    holdings = {c: check_position(c) for c in ASSETS}

    # 生成邮件
    html = build_email_html(mode, signals, holdings)
    today = datetime.now().strftime('%Y-%m-%d')

    picks = [c for c in ORDER if c in signals and signals[c]['signal'] == 1]
    if picks:
        # 按置信度排序, 取最强信号决定标题 (与正文建议一致)
        picks.sort(key=lambda c: signals[c]['conf']['score'], reverse=True)
        top = picks[0]
        conf_level = signals[top]['conf']['level']
        if conf_level == '强':
            subject = f'[{today} {time_label}] 策略: 买入 {ASSETS[top]["name"]}'
        elif conf_level == '中':
            subject = f'[{today} {time_label}] 策略: 关注 {ASSETS[top]["name"]} (中等信号)'
        else:
            subject = f'[{today} {time_label}] 策略: 观望 ({ASSETS[top]["name"]}弱信号待确认)'
    else:
        subject = f'[{today} {time_label}] 策略: 空仓观望'

    ok = send_email(subject, html)
    if ok:
        print(f'[OK] {time_label}邮件已发送: {subject}')
    else:
        print('[FAIL] 邮件发送失败')


if __name__ == '__main__':
    main()
