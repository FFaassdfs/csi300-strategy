"""
CSI300 ETF 每日策略报告 (HTML版) - 用户持仓版
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
import duckdb
import csv
import json
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ========== 读取舆情数据 ==========
SENTIMENT_FILE = os.path.join(PROJECT_ROOT, 'reports', 'sentiment_data.json')
sentiment = {}
if os.path.exists(SENTIMENT_FILE):
    try:
        with open(SENTIMENT_FILE, 'r', encoding='utf-8') as f:
            sentiment = json.load(f)
    except Exception:
        pass

# ========== 多品种轮动信号 ==========
dual_signals = {}
try:
    from dual_rotation import get_rotation_signals
    dual_signals = get_rotation_signals()
except Exception:
    pass

# ========== 读取交易记录计算持仓 ==========
trades_file = os.path.join(PROJECT_ROOT, 'trades', '510310_trades.csv')
positions = {}  # code -> {shares, cost}
if os.path.exists(trades_file):
    with open(trades_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row['code']
            if code not in positions:
                positions[code] = {'shares': 0, 'cost_total': 0}
            action = row['action']
            shares = int(row['shares'])
            price = float(row['price'])
            if action == 'BUY':
                old_shares = positions[code]['shares']
                old_cost = positions[code]['cost_total']
                new_shares = old_shares + shares
                positions[code]['cost_total'] = old_cost + shares * price
                positions[code]['shares'] = new_shares
            elif action == 'SELL':
                positions[code]['shares'] -= shares

# ========== 从本地数据库获取数据 ==========
# 检查本地数据库，如果没有则向上级目录查找
local_db = os.path.join(PROJECT_ROOT, 'csi300_data.duckdb')
parent_db = os.path.join(PROJECT_ROOT, '..', 'csi300_data.duckdb')

# 选择有数据的数据库
if os.path.exists(local_db):
    conn_test = duckdb.connect(local_db)
    tables = conn_test.execute('SHOW TABLES').fetchall()
    conn_test.close()
    db_path = local_db if tables else parent_db
else:
    db_path = parent_db

conn = duckdb.connect(db_path)

# 如本地无csi300_daily但ETF表在本地，则附加父库
need_parent = False
try:
    conn.execute('SELECT 1 FROM csi300_daily LIMIT 1').fetchone()
except Exception:
    need_parent = os.path.exists(parent_db) and parent_db != db_path

if need_parent:
    conn.execute(f"ATTACH '{parent_db}' AS parent_db")
    prefix = 'parent_db.'
else:
    prefix = ''

# 尝试获取OHLC数据，若无则退化为close-only
try:
    df_300 = conn.execute(f'SELECT date, open, high, low, close FROM {prefix}csi300_daily ORDER BY date').fetchdf()
    has_ohlc = True
except Exception:
    df_300 = conn.execute(f'SELECT date, close FROM {prefix}csi300_daily ORDER BY date').fetchdf()
    has_ohlc = False

# 加载510310 ETF数据（场内基金实际交易价格）
try:
    df_etf = conn.execute('SELECT date, close FROM etf_510310_daily ORDER BY date').fetchdf()
    etf_last_close = df_etf['close'].iloc[-1]
    etf_last_date = df_etf['date'].iloc[-1]
except Exception:
    # ETF数据缺失时回退到指数价格
    df_etf = None
    etf_last_close = df_300['close'].iloc[-1] / 1000
    etf_last_date = df_300['date'].iloc[-1]

conn.close()

df_300['date'] = pd.to_datetime(df_300['date'])
close = df_300['close']

# 通用指标
vol = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
ma50 = close.rolling(50).mean()
momentum = (close / close.shift(20) - 1)

# ADX / 趋势强度计算
if has_ohlc:
    high = df_300['high']
    low = df_300['low']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(0.0, index=df_300.index)
    minus_dm = pd.Series(0.0, index=df_300.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/14, adjust=False).mean().fillna(0)
else:
    tr = abs(close - close.shift(1))
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    adx = ((close - close.shift(14)) / atr14 * 100).rolling(14).mean().fillna(0)

# 最新值
last_date = df_300['date'].max()
last_close = close.iloc[-1]
last_vol = vol.iloc[-1]
last_ma50 = ma50.iloc[-1]
last_momentum = momentum.iloc[-1]
last_adx = adx.iloc[-1]

# ── 辅助指标: 布林带 %B ──
bb_mid = close.rolling(20).mean()
bb_std = close.rolling(20).std()
bb_lower = bb_mid - 2 * bb_std
bb_upper = bb_mid + 2 * bb_std
last_bb_pct_b = (last_close - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-10)

# ── 辅助指标: MACD ──
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
macd_line = ema12 - ema26
macd_signal = macd_line.ewm(span=9, adjust=False).mean()
macd_hist = macd_line - macd_signal
last_macd_hist = macd_hist.iloc[-1]

# 策略信号
above_ma = last_close > last_ma50
low_vol = last_vol < 15
strong_trend = last_adx > 25 if pd.notna(last_adx) and last_adx > 0 else False

sig_adx = 1 if (above_ma and (low_vol or strong_trend)) else 0

tr_adx = 'ADX>25' if strong_trend else ('低波动' if low_vol else '高波动/低趋势')

# ── 辅助策略: ADX+MACD柱>0 (保守确认) ──
aux_conservative = 1 if (above_ma and (low_vol or strong_trend) and last_macd_hist > 0) else 0

# BB超卖
bb_oversold = last_bb_pct_b < 0.2 and last_close > last_ma50

# ========== 辅助参考信号 ==========
conservative_diverges = (sig_adx == 1 and aux_conservative == 0)

bb_oversold_color = '#e74c3c' if bb_oversold else '#bbb'
bb_oversold_text = '超卖区间' if bb_oversold else '正常区间'
conservative_color = '#e67e22' if conservative_diverges else '#27ae60'
conservative_text = '与主信号分歧' if conservative_diverges else '与主信号一致'

# ========== 用户持仓配置（仅场内ETF，从交易记录计算）==========
# 场外基金(003015/110020)不在此记录，仅记录场内ETF
TOTAL_CAPITAL = 10000  # 总资金（按需修改）
HOLDING_CONFIG = {}
for code, pos in positions.items():
    if pos['shares'] > 0:
        avg_cost = pos['cost_total'] / pos['shares']
        HOLDING_CONFIG[code] = {
            'name': '沪深300ETF',
            'shares': pos['shares'],
            'cost': avg_cost,
            'cost_total': pos['cost_total']
        }
# 计算持仓市值（使用510310 ETF最新收盘价）
for code in HOLDING_CONFIG:
    v = HOLDING_CONFIG[code]
    v['last_price'] = etf_last_close
    v['amount'] = v['shares'] * etf_last_close
total_holding = sum(v['amount'] for v in HOLDING_CONFIG.values())
remaining_cash = TOTAL_CAPITAL - total_holding
max_buyable_shares = int(remaining_cash / etf_last_close / 100) * 100 if etf_last_close > 0 else 0

report_date = datetime.now().strftime('%Y-%m-%d')
index_date = last_date.strftime('%Y-%m-%d')

# ========== HTML内容生成 ==========
def generate_dual_signal_panel():
    """多品种轮动信号面板"""
    if not dual_signals:
        return ''

    lines = ['<div style="margin-top:15px;background:#f0f4ff;border:2px solid #2980b9;border-radius:10px;padding:15px;">']
    lines.append('<div style="font-weight:bold;font-size:15px;color:#1a1a2e;margin-bottom:10px;">多品种轮动信号 <span style="font-size:11px;color:#999;font-weight:normal;">(510310+159995)</span></div>')
    lines.append('<table style="font-size:13px;">')
    lines.append('<tr><th>品种</th><th>价格</th><th>MA50</th><th>波动率</th><th>ADX</th><th>信号</th></tr>')

    asset_order = ['510310', '159995']
    for code in asset_order:
        s = dual_signals.get('assets', {}).get(code)
        if s is None:
            continue
        sig = '持有' if s['signal'] == 1 else '空仓'
        sig_color = '#27ae60' if s['signal'] == 1 else '#e74c3c'
        adx_color = '#27ae60' if s['adx'] > 25 else '#e67e22' if s['adx'] > 20 else '#999'
        vol_color = '#e74c3c' if s['vol'] > 15 else '#27ae60'
        lines.append(f'<tr>'
            f'<td><strong>{s["name"]}</strong></td>'
            f'<td>{s["price"]:.4f}</td>'
            f'<td>{s["ma50"]:.4f}</td>'
            f'<td style="color:{vol_color};">{s["vol"]:.1f}%</td>'
            f'<td style="color:{adx_color};">{s["adx"]:.1f}</td>'
            f'<td><span style="font-weight:bold;color:{sig_color};">{sig}</span></td>'
            f'</tr>')

    lines.append('</table>')

    # 轮动建议 (三品种)
    candidates = []
    for code in ['510310', '159995']:
        s = dual_signals.get('assets', {}).get(code, {})
        if s.get('signal') == 1:
            candidates.append((code, s.get('adx', 0), s.get('name', code)))
    
    if len(candidates) > 1:
        candidates.sort(key=lambda x: x[1], reverse=True)
        chosen = candidates[0][0]
        names = [c[2] for c in candidates]
        reason = f'{"/".join(names)}皆可，选ADX最强的'
    elif len(candidates) == 1:
        chosen = candidates[0][0]
        reason = f'仅{candidates[0][2]}符合条件'
    else:
        chosen = 'BOND'
        reason = '所有品种都不符合买入条件'

    chosen_map = {'510310': '沪深300ETF', '159995': '芯片ETF', 'BOND': '国债/逆回购'}
    chosen_name = chosen_map.get(chosen, '国债/逆回购')

    lines.append(f'<div style="margin-top:10px;padding:8px 12px;background:white;border-radius:6px;font-size:14px;">')
    lines.append(f'<strong>轮动指向:</strong> <span style="font-size:16px;color:#1a1a2e;">{chosen_name}</span>')
    lines.append(f'<span style="color:#666;margin-left:10px;">({reason})</span>')
    lines.append('</div>')

    lines.append('</div>')
    return '\n'.join(lines)


def generate_sentiment_panel():
    """生成舆情面板HTML"""
    if not sentiment:
        return '<p style="color:#999;">舆情数据暂未采集，请运行 sentiment_collector.py 或 daily_refresh.py</p>'

    lines = ['<div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px;">']

    # 北向资金
    nf = sentiment.get("north_flow", {})
    nf_color = "#27ae60" if nf.get("status") == "流入" else "#e74c3c"
    lines.append(f'''<div style="background:#f8f9fa;padding:12px;border-radius:8px;">
        <strong>北向资金</strong> <span style="font-size:11px;color:#999;">{nf.get("date","")}</span><br>
        <span style="font-size:20px;font-weight:bold;color:{nf_color};">{nf.get("net_flow",0):+.1f}亿</span>
        <span style="color:{nf_color};">{nf.get("status","")}</span>
    </div>''')

    # QVIX
    qv = sentiment.get("qvix", {})
    qv_color = "#e74c3c" if qv.get("level") == "恐慌" else "#e67e22" if qv.get("level") == "偏高" else "#27ae60"
    lines.append(f'''<div style="background:#f8f9fa;padding:12px;border-radius:8px;">
        <strong>QVIX恐慌指数</strong> <span style="font-size:11px;color:#999;">{qv.get("date","")}</span><br>
        <span style="font-size:20px;font-weight:bold;color:{qv_color};">{qv.get("qvix",0):.1f}</span>
        <span style="font-size:12px;color:#666;">({qv.get("change",0):+.1f})</span>
        <span style="color:{qv_color};">{qv.get("level","")}</span>
    </div>''')

    # 市场资金
    mf = sentiment.get("market_flow", {})
    mf_color = "#27ae60" if mf.get("direction") == "流入" else "#e74c3c"
    lines.append(f'''<div style="background:#f8f9fa;padding:12px;border-radius:8px;">
        <strong>主力资金</strong> <span style="font-size:11px;color:#999;">{mf.get("date","")}</span><br>
        <span style="font-size:20px;font-weight:bold;color:{mf_color};">{mf.get("main_net",0):+.1f}亿</span>
        <span style="color:{mf_color};">{mf.get("direction","")}</span>
    </div>''')

    # 全球概览
    gl = sentiment.get("global", {})
    gl_lines = []
    for name, v in sorted(gl.items()):
        c = v.get("change_pct", 0)
        color = "#27ae60" if c > 0 else "#e74c3c" if c < 0 else "#999"
        gl_lines.append(f'<span>{name} <span style="color:{color};">{c:+.1f}%</span></span>')
    if gl_lines:
        lines.append(f'''<div style="background:#f8f9fa;padding:12px;border-radius:8px;">
            <strong>全球指数</strong><br>
            <div style="font-size:12px;margin-top:5px;">{" | ".join(gl_lines)}</div>
        </div>''')

    lines.append('</div>')

    # 相关性矩阵
    corr = sentiment.get("correlation", {})
    if corr:
        etf_names = {"510310": "沪深300", "511260": "国债", "159995": "芯片", "159915": "创业板", "510050": "上证50"}
        lines.append('<div style="margin-top:15px;"><strong>多品种相关性 (2年日收益)</strong></div>')
        lines.append('<table style="font-size:11px;margin-top:5px;"><tr><th></th>')
        for name in sorted(corr.keys()):
            label = etf_names.get(name, name)
            lines.append(f'<th>{label}</th>')
        lines.append('</tr>')
        for name1 in sorted(corr.keys()):
            label1 = etf_names.get(name1, name1)
            lines.append(f'<tr><td><strong>{label1}</strong></td>')
            for name2 in sorted(corr.keys()):
                v = corr[name1].get(name2, 0)
                if name1 == name2:
                    lines.append('<td style="color:#ccc;">1.0</td>')
                else:
                    if v < -0.3:
                        c = f"color:#27ae60;"  # 负相关=好的对冲
                    elif v > 0.7:
                        c = f"color:#e74c3c;"  # 高相关=同涨同跌
                    elif v < 0.1:
                        c = f"color:#2980b9;"  # 低相关=好的分散
                    else:
                        c = ""
                    lines.append(f'<td style="{c}">{v:+.2f}</td>')
            lines.append('</tr>')
        lines.append('</table>')
        lines.append('''<div style="font-size:10px;color:#999;margin-top:3px;">
            <span style="color:#27ae60;">■</span>负相关(对冲) 
            <span style="color:#2980b9;">■</span>低相关(分散) 
            <span style="color:#e74c3c;">■</span>高相关(同步)
        </div>''')

    return "\n".join(lines)


def holdings_overview():
    """持仓概览卡片 - 突出显示5个核心指标"""
    if not HOLDING_CONFIG:
        return ''
    cards = []
    for code, v in HOLDING_CONFIG.items():
        cost = v.get('cost', 0)
        last_p = v.get('last_price', 0)
        shares = v.get('shares', 0)
        amount = v.get('amount', 0)
        pnl = (last_p - cost) * shares
        pnl_pct = (last_p / cost - 1) * 100 if cost > 0 else 0
        pnl_color = '#27ae60' if pnl >= 0 else '#e74c3c'
        pnl_sign = '+' if pnl >= 0 else ''
        cards.append(f"""
            <div class="overview-card">
                <div class="overview-title">{code} 沪深300ETF</div>
                <div class="overview-metrics">
                    <div class="metric">
                        <div class="metric-label">持仓成本</div>
                        <div class="metric-value">{cost:.4f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">最新收盘</div>
                        <div class="metric-value">{last_p:.4f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">持有份额</div>
                        <div class="metric-value">{shares:.0f} 份</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">持仓市值</div>
                        <div class="metric-value">¥{amount:,.2f}</div>
                    </div>
                    <div class="metric metric-pnl">
                        <div class="metric-label">浮动盈亏</div>
                        <div class="metric-value" style="color:{pnl_color};">
                            {pnl_sign}¥{pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)
                        </div>
                    </div>
                </div>
            </div>""")
    return f'<div class="overview-container">{"".join(cards)}</div>'


def holding_rows():
    rows = ''
    for code, v in HOLDING_CONFIG.items():
        pct = v['amount'] / total_holding * 100
        cost_info = ''
        if 'cost' in v:
            pnl = (v['last_price'] - v['cost']) * v['shares']
            pnl_pct = (v['last_price'] / v['cost'] - 1) * 100
            color = '#27ae60' if pnl >= 0 else '#e74c3c'
            cost_info = f'<span style="font-size:11px;color:{color};">成本{v["cost"]:.4f} | 盈亏{pnl:+.2f} ({pnl_pct:+.2f}%)</span>'
        rows += f"""
                <tr>
                    <td class="fund-name">{code} {v['name']}<div class="nav">{cost_info}</div></td>
                    <td>{v['last_price']:.4f}</td>
                    <td>¥{v['amount']:,.2f}</td>
                    <td>{v['shares']:.0f}份</td>
                    <td>{pct:.1f}%</td>
                </tr>"""
    return rows

html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>CSI300 策略信号报告 {report_date}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5; color: #333; }}
        .header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; padding: 25px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 24px; }}
        .header .subtitle {{ opacity: 0.8; font-size: 14px; }}
        .section {{ background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .section h2 {{ margin-top: 0; color: #1a1a2e; font-size: 18px; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .indicator-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 15px; }}
        .indicator {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .indicator .label {{ font-size: 12px; color: #666; margin-bottom: 5px; }}
        .indicator .value {{ font-size: 20px; font-weight: bold; color: #1a1a2e; }}
        .indicator .value.warning {{ color: #e67e22; }}
        .indicator .value.success {{ color: #27ae60; }}
        .indicator .value.danger {{ color: #e74c3c; }}
        .strategy-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #ccc; }}
        .strategy-card.adx {{ border-left-color: #27ae60; }}
        .strategy-card.momentum {{ border-left-color: #3498db; }}
        .strategy-card.absolute {{ border-left-color: #95a5a6; }}
        .strategy-card .name {{ font-weight: bold; font-size: 16px; margin-bottom: 8px; }}
        .signal {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }}
        .signal.csi300 {{ background: #27ae60; color: white; }}
        .signal.bond {{ background: #e74c3c; color: white; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .holding-table th, .holding-table td {{ text-align: right; }}
        .holding-table th {{ text-align: left; }}
        .holding-table .fund-name {{ text-align: left; font-weight: 600; }}
        .holding-table .nav {{ font-size: 12px; color: #666; }}
        .holding-table .total-row {{ font-weight: bold; background: #f8f9fa; }}
        .recommendation {{ background: linear-gradient(135deg, #f39c12, #e67e22); color: white; padding: 25px; border-radius: 10px; text-align: center; }}
        .recommendation h3 {{ margin: 0 0 15px 0; font-size: 18px; }}
        .recommendation .action {{ font-size: 28px; font-weight: bold; margin: 15px 0; }}
        .recommendation .note {{ font-size: 14px; opacity: 0.9; }}
        .info-box {{ background: #e8f4f8; border: 1px solid #b8d4e3; border-radius: 8px; padding: 15px; margin-top: 15px; }}
        .info-box .title {{ font-weight: bold; color: #2980b9; margin-bottom: 5px; }}
        .footer {{ text-align: center; font-size: 12px; color: #999; margin-top: 20px; }}
        .overview-container {{ margin-bottom: 15px; }}
        .overview-card {{ background: linear-gradient(135deg, #f8f9fa, #e8f4f8); border: 2px solid #2980b9; border-radius: 10px; padding: 20px; margin-bottom: 15px; }}
        .overview-title {{ font-size: 18px; font-weight: bold; color: #1a1a2e; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #b8d4e3; }}
        .overview-metrics {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; }}
        .metric {{ background: white; padding: 12px; border-radius: 8px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .metric-label {{ font-size: 12px; color: #666; margin-bottom: 6px; }}
        .metric-value {{ font-size: 16px; font-weight: bold; color: #1a1a2e; }}
        .metric-pnl {{ background: #fff8e1; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CSI300 趋势跟踪策略报告</h1>
        <div class="subtitle">报告生成: {report_date} | 指数数据: {index_date}</div>
    </div>

    <div class="section">
        <h2>市场状态指标（沪深300指数）</h2>
        <div class="indicator-grid">
            <div class="indicator">
                <div class="label">沪深300指数</div>
                <div class="value">{last_close:.2f}</div>
            </div>
            <div class="indicator">
                <div class="label">MA50均线</div>
                <div class="value">{"{:.2f}".format(last_ma50)}</div>
            </div>
            <div class="indicator">
                <div class="label">20日波动率</div>
                <div class="value {"warning" if last_vol > 15 else "success"}">{last_vol:.2f}%</div>
            </div>
            <div class="indicator">
                <div class="label">{'ADX趋势强度' if has_ohlc else '趋势强度指标'}</div>
                <div class="value {"success" if strong_trend else ""}">{last_adx:.2f}</div>
            </div>
        </div>
        <table>
            <tr>
                <td><strong>价格 vs MA50:</strong></td>
                <td><strong>{"✅ 在均线上方" if above_ma else "❌ 在均线下方"}</strong></td>
                <td><strong>20日动量:</strong></td>
                <td>{last_momentum*100:+.2f}%</td>
            </tr>
            <tr>
                <td><strong>波动率 vs 15%:</strong></td>
                <td>{"⚠️ 偏高" if last_vol > 15 else "✅ 偏低"}</td>
                <td><strong>趋势确认:</strong></td>
                <td>{"✅ 是" if strong_trend else "❌ 否"}</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2>主策略信号: ADX Override</h2>

        <div class="strategy-card adx" style="border-left-width: 6px; padding: 20px;">
            <div class="name" style="font-size: 18px;">ADX Override <span style="font-size:13px;color:#1a1a2e;font-weight:bold;">← 主交易信号</span></div>
            <div style="margin-top: 10px;">
                <span class="signal {"csi300" if sig_adx == 1 else "bond"}">{"持有CSI300" if sig_adx == 1 else "持有国债"}</span>
                <span style="margin-left: 15px; color: #666;">触发: {tr_adx}</span>
            </div>
            <div style="font-size: 12px; color: #666; margin-top: 8px;">
                价格 &gt; MA50 AND (波动率 &lt; 15% OR ADX &gt; 25)
            </div>
        </div>

        {generate_dual_signal_panel()}

        <details style="margin-top: 15px;">
            <summary style="cursor: pointer; color: #666; font-size: 13px;">辅助参考信号（观察，不执行）</summary>
            <div class="strategy-card" style="margin-top: 10px; border-left-color: {bb_oversold_color};">
                <div class="name">布林带极端超卖 <span style="font-size:11px;color:#999;font-weight:normal;">— %B < 0.2</span></div>
                <div style="margin-top: 5px;">
                    <span class="signal" style="background: {bb_oversold_color}; color: white;">{bb_oversold_text}</span>
                    <span style="margin-left: 10px; color: #999;">%B = {last_bb_pct_b:.2f}</span>
                </div>
                <div style="font-size: 11px; color: #999; margin-top: 5px;">%B<0.2 且价格>MA50时表示极端超卖，历史回撤仅-1.15%</div>
            </div>
            <div class="strategy-card" style="margin-top: 8px; border-left-color: {conservative_color};">
                <div class="name">保守信号确认 <span style="font-size:11px;color:#999;font-weight:normal;">— ADX + MACD柱>0</span></div>
                <div style="margin-top: 5px;">
                    <span class="signal" style="background: {conservative_color}; color: white;">{conservative_text}</span>
                    <span style="margin-left: 10px; color: #999;">MACD柱 = {last_macd_hist:+.2f}</span>
                </div>
                <div style="font-size: 11px; color: #999; margin-top: 5px;">ADX+MACD柱>0为更保守的入场确认，回测Sharpe 0.73</div>
            </div>
        </details>
    </div>

    <div class="section">
        <h2>舆情/宏观辅助参考</h2>
        {generate_sentiment_panel()}
    </div>

    <div class="section">
        <h2>当前持仓（场内ETF）</h2>
        {holdings_overview()}
        <table class="holding-table">
            <thead>
                <tr>
                    <th>基金名称</th>
                    <th>最新净值</th>
                    <th>持有金额</th>
                    <th>估算份额</th>
                    <th>占比</th>
                </tr>
            </thead>
            <tbody>
                {holding_rows()}
                <tr class="total-row">
                    <td class="fund-name"><strong>合计</strong></td>
                    <td>-</td>
                    <td><strong>¥{total_holding:,.2f}</strong></td>
                    <td>-</td>
                    <td>100%</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>交易建议</h2>
        <div class="recommendation">
            <h3>明日操作建议</h3>
"""

# 根据主策略(ADX Override)生成建议
if sig_adx == 1:
    if max_buyable_shares >= 100:
        html_content += f"""
            <div class="action">✅ 追加买入 {max_buyable_shares}股 510310</div>
            <div class="note">ADX Override 主策略看多 | 剩余资金 ¥{remaining_cash:,.2f} 可买 {max_buyable_shares}股</div>
"""
    else:
        html_content += """
            <div class="action">✅ 继续持有沪深300</div>
            <div class="note">ADX Override 主策略看多 | 剩余资金不足100股</div>
"""
else:
    html_content += """
            <div class="action">🔴 转换为国债避险</div>
            <div class="note">ADX Override 主策略看空</div>
"""

html_content += f"""
        </div>
        <div class="info-box" style="margin-top: 15px;">
            <div class="title">资金状况</div>
            <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                <div>总资金: ¥{TOTAL_CAPITAL:,.2f}</div>
                <div>已持仓: ¥{total_holding:,.2f} ({total_holding/TOTAL_CAPITAL*100:.1f}%)</div>
                <div>可用: ¥{remaining_cash:,.2f}</div>
            </div>
        </div>
"""

# 风险提示
risk_items = []
if last_vol > 15:
    risk_items.append(f'波动率 {last_vol:.2f}% 略高于 15% 阈值，{"ADX>25 覆盖了这一条件" if strong_trend else "需谨慎操作"}')
if remaining_cash > 0:
    buffer = remaining_cash * 0.085
    risk_items.append(f'建议保留约 ¥{buffer:,.0f} 现金作为缓冲')
if sig_adx == 1 and max_buyable_shares >= 200:
    reduced_shares = int(max_buyable_shares * 0.8 / 100) * 100
    risk_items.append(f'如明日开盘价高于 {etf_last_close * 1.01:.2f}，可考虑减少买入量至 {reduced_shares} 股')

if risk_items:
    html_content += """
        <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 15px; margin-top: 15px;">
            <div style="font-weight: bold; color: #856404; margin-bottom: 8px;">⚠️ 风险提示</div>
"""
    for item in risk_items:
        html_content += f'            <div style="color: #856404; font-size: 13px; margin-top: 4px;">• {item}</div>\n'
    html_content += "        </div>\n"

html_content += "    </div>\n"

html_content += """
    <div class="section">
        <h2>策略对比（真实回测 2022-08 ~ 2026-06，约3.6年）</h2>
        <table>
            <tr>
                <th>策略</th>
                <th>年化收益</th>
                <th>Sharpe</th>
                <th>最大回撤</th>
                <th>持仓比例</th>
                <th>用途</th>
            </tr>
            <tr>
                <td><strong>ADX Override</strong></td>
                <td style="color: #27ae60;"><strong>+11.49%</strong></td>
                <td style="color: #27ae60;"><strong>0.77</strong></td>
                <td>-10.99%</td>
                <td>39.3%</td>
                <td style="color: #1a1a2e;"><strong>主策略</strong></td>
            </tr>
            <tr>
                <td>ADX + MACD柱>0</td>
                <td>+9.84%</td>
                <td>0.73</td>
                <td>-12.33%</td>
                <td>22.3%</td>
                <td style="color: #666;">保守确认</td>
            </tr>
            <tr>
                <td>RSI>40 + MA50</td>
                <td>+11.71%</td>
                <td>0.73</td>
                <td>-10.99%</td>
                <td>51.3%</td>
                <td style="color: #666;">简化参考</td>
            </tr>
            <tr>
                <td>BB下轨 + MA50</td>
                <td>+4.07%</td>
                <td>0.71</td>
                <td style="color: #27ae60;">-1.15%</td>
                <td>2.0%</td>
                <td style="color: #666;">抄底辅助</td>
            </tr>
            <tr>
                <td>Buy&Hold (基准)</td>
                <td>+4.86%</td>
                <td>--</td>
                <td style="color: #e74c3c;">-24.80%</td>
                <td>100%</td>
                <td style="color: #999;">被动持有</td>
            </tr>
        </table>
        <div style="margin-top: 10px; font-size: 12px; color: #666;">
            * Wilder标准ADX算法，回测24个月周期，上证指数筛选，全市场验证<br>
            * 所有策略最大回撤 -24.8% 附近，因此 ADX Override 为执行策略，其余为观察
        </div>
    </div>
"""

# 保存HTML报告
report_dir = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(report_dir, exist_ok=True)
report_file = os.path.join(report_dir, f'daily_signal_{report_date.replace("-", "")}.html')

with open(report_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f'HTML报告已生成: {report_file}')
print()
print('='*60)
print('策略信号摘要')
print('='*60)
print(f'指数日期: {index_date}')
print(f'沪深300: {last_close:.2f}')
print(f'波动率: {last_vol:.2f}% (阈值15%)')
print(f'MA50: {last_ma50:.2f}')
print(f'20日动量: {last_momentum*100:+.2f}%')
print()
print(f'策略信号:  {"持有CSI300" if sig_adx==1 else "持有国债"} ({tr_adx})')
print(f'(辅助) BB %B={last_bb_pct_b:.2f} {"超卖!" if bb_oversold else "正常"}  |  MACD柱={last_macd_hist:+.2f} 保守信号{"分歧!" if conservative_diverges else "一致"}')
print()
print('='*60)
print('持仓状态')
print('='*60)
for code, v in HOLDING_CONFIG.items():
    print(f'{code} {v["name"]}: {v["amount"]:,.2f}')
print(f'合计: {total_holding:,.2f}')
