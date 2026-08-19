"""
收盘策略邮件推送
每天15:10刷新数据后, 自动生成操作建议并发送邮件

配置: 修改下面 SMTP_CONFIG 中的授权码 (163邮箱设置→客户端授权密码)
"""
import os
import sys
import smtplib
import subprocess
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ========== 邮件配置 (用户修改) ==========
SMTP_CONFIG = {
    'sender': 'aassdfs@163.com',          # 发件邮箱
    'auth_code': 'CFNyDY5dEPJu6Y5U',      # 163邮箱授权码
    'to': 'aassdfs@163.com',              # 收件邮箱
    'smtp_server': 'smtp.163.com',
    'smtp_port': 465,                     # SSL
}

# ========== 品种配置 (与轮动一致) ==========
ASSETS = {
    '510310': {'name': '沪深300ETF', 'code': 'sh510310', 'ma_p': 30, 'adx_th': 20, 'vol_th': 18},
    '159995': {'name': '芯片ETF',    'code': 'sz159995', 'ma_p': 30, 'adx_th': 25, 'vol_th': 15},
    '512800': {'name': '银行ETF',    'code': 'sh512800', 'ma_p': 30, 'adx_th': 25, 'vol_th': 18},
}


def get_latest_signals():
    """从历史库读取最新信号"""
    import duckdb
    conn = duckdb.connect(os.path.join(PROJECT_ROOT, 'trading_history.duckdb'))
    today = datetime.now().strftime('%Y-%m-%d')

    signals = {}
    for code, info in ASSETS.items():
        try:
            row = conn.execute(
                "SELECT date, price, signal, reason FROM signals_log "
                "WHERE code=? ORDER BY date DESC LIMIT 1", [code]
            ).fetchone()
            if row:
                signals[code] = {'name': info['name'], 'date': str(row[0]),
                                 'price': row[1], 'signal': row[2], 'reason': row[3]}
        except Exception:
            pass
    conn.close()
    return signals


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


def build_advice(signals):
    """生成操作建议"""
    picks = [c for c, s in signals.items() if s['signal'] == 1]
    holdings = {c: check_position(c) for c in ASSETS}

    lines = []
    lines.append(f"<h3>今日信号 ({datetime.now().strftime('%Y-%m-%d')})</h3>")
    lines.append('<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">')
    lines.append('<tr style="background:#f0f0f0;"><th>品种</th><th>收盘价</th><th>信号</th><th>触发理由</th><th>持仓</th></tr>')
    for code, info in ASSETS.items():
        s = signals.get(code)
        if not s:
            continue
        sig_txt = '<b style="color:green;">持有</b>' if s['signal'] == 1 else '<b style="color:red;">空仓</b>'
        hold_txt = f"{holdings[code]}份" if holdings[code] > 0 else '-'
        lines.append(f"<tr><td>{s['name']}</td><td>{s['price']:.4f}</td><td>{sig_txt}</td><td>{s['reason']}</td><td>{hold_txt}</td></tr>")
    lines.append('</table>')

    lines.append('<h3>操作建议</h3>')
    if picks:
        for code in picks:
            s = signals[code]
            if holdings[code] > 0:
                lines.append(f"<p>✅ <b>{s['name']}</b>: 信号持有中, 继续持有 {holdings[code]} 份, 无操作</p>")
            else:
                lines.append(f"<p>🟢 <b>{s['name']}</b>: 信号翻多! 明天开盘可买入 (建议全仓 {s['price']:.4f} 附近)</p>")
    else:
        lines.append('<p>🔴 三个品种均无买入信号, 继续持币/逆回购, 等下一个信号</p>')

    # 持仓但无信号的
    for code, sh in holdings.items():
        if sh > 0 and code not in picks:
            s = signals.get(code, {})
            lines.append(f"<p>⚠️ <b>{ASSETS[code]['name']}</b>: 持有 {sh} 份但信号已空仓, 建议明天卖出转逆回购!</p>")

    lines.append('<hr>')
    lines.append('<p style="color:#999;font-size:12px;">本邮件由量化策略系统自动发送, 仅供参考, 不构成投资建议</p>')
    return '\n'.join(lines)


def send_email(subject, html):
    """发送邮件"""
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
    if 'YOUR_AUTH_CODE' in SMTP_CONFIG['auth_code']:
        print('⚠️ 请先在 send_advice.py 中配置163邮箱授权码!')
        print('   获取方式: 163邮箱 → 设置 → 客户端授权密码')
        return

    signals = get_latest_signals()
    if not signals:
        print('未获取到信号数据')
        return

    html = build_advice(signals)
    today = datetime.now().strftime('%Y-%m-%d')

    # 生成操作摘要作为主题
    picks = [c for c, s in signals.items() if s['signal'] == 1]
    if picks:
        subject = f'[{today}] 策略: 买入 {ASSETS[picks[0]]["name"]}'
    else:
        subject = f'[{today}] 策略: 空仓观望'

    ok = send_email(subject, html)
    if ok:
        print(f'[OK] 邮件已发送: {subject}')
    else:
        print('[FAIL] 邮件发送失败')


if __name__ == '__main__':
    main()
