"""
CSI300 ETF 每日策略报告 (HTML版) - 用户持仓版
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ========== 从本地数据库获取数据 ==========
db_path = os.path.join(PROJECT_ROOT, 'csi300_data.duckdb')
conn = duckdb.connect(db_path)

# 尝试获取OHLC数据，若无则退化为close-only
try:
    df_300 = conn.execute('SELECT date, open, high, low, close FROM csi300_daily ORDER BY date').fetchdf()
    has_ohlc = True
except Exception:
    df_300 = conn.execute('SELECT date, close FROM csi300_daily ORDER BY date').fetchdf()
    has_ohlc = False
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

# ========== 用户持仓配置（按需修改）==========
HOLDING_CONFIG = {
    '003015': {'name': '中金沪深300指数增强A', 'nav': 2.1703, 'amount': 3117.96, 'nav_date': '2026-06-01'},
    '110020': {'name': '易方达沪深300ETF联接A', 'nav': 1.9702, 'amount': 10370.53, 'nav_date': '2026-05-29'},
    '510310': {'name': '沪深300ETF (场内)', 'nav': 4.727, 'shares': 1000, 'cost': 4.727, 'nav_date': '2026-06-02'},
}
total_holding = sum(v['amount'] if 'amount' in v else v['nav'] * v.get('shares', 0) for v in HOLDING_CONFIG.values())
for code in HOLDING_CONFIG:
    if 'amount' not in HOLDING_CONFIG[code]:
        HOLDING_CONFIG[code]['amount'] = HOLDING_CONFIG[code]['nav'] * HOLDING_CONFIG[code].get('shares', 0)
    HOLDING_CONFIG[code]['shares'] = HOLDING_CONFIG[code].get('shares', HOLDING_CONFIG[code]['amount'] / HOLDING_CONFIG[code]['nav'])

report_date = datetime.now().strftime('%Y-%m-%d')
index_date = last_date.strftime('%Y-%m-%d')

# ========== HTML内容生成 ==========
def holding_rows():
    rows = ''
    for code, v in HOLDING_CONFIG.items():
        pct = v['amount'] / total_holding * 100
        cost_info = ''
        if 'cost' in v:
            pnl = (v['nav'] - v['cost']) * v['shares']
            pnl_pct = (v['nav'] / v['cost'] - 1) * 100
            color = '#27ae60' if pnl >= 0 else '#e74c3c'
            cost_info = f'<span style="font-size:11px;color:{color};">成本{v["cost"]:.4f} | 盈亏{pnl:+.2f} ({pnl_pct:+.2f}%)</span>'
        rows += f"""
                <tr>
                    <td class="fund-name">{code} {v['name']}<div class="nav">净值日期: {v['nav_date']} {cost_info}</div></td>
                    <td>{v['nav']:.4f}</td>
                    <td>¥{v['amount']:,.2f}</td>
                    <td>{v['shares']:.2f}份</td>
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
        <h2>当前持仓（场外基金）</h2>
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
    html_content += """
            <div class="action">✅ 继续持有 / 买入 沪深300</div>
            <div class="note">ADX Override 主策略看多</div>
"""
else:
    html_content += """
            <div class="action">🔴 转换为国债避险</div>
            <div class="note">ADX Override 主策略看空</div>
"""

html_content += """
        </div>
    </div>

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
                <td style="color: #666;">超卖警报</td>
            </tr>
            <tr>
                <td>Buy&Hold (基准)</td>
                <td>+4.86%</td>
                <td>--</td>
                <td style="color: #e74c3c;">-24.80%</td>
                <td>100%</td>
                <td style="color: #999;">对照</td>
            </tr>
        </table>
        <div style="margin-top: 10px; font-size: 12px; color: #666;">
            * Wilder标准ADX算法，回测24个策略组合后筛选出最佳4个。<br>
            * 所有策略均将最大回撤从 -24.8% 大幅压缩。ADX Override 为主执行策略，其余为辅助观察。
        </div>
    </div>

    <div class="footer">
        <p>⚠️ 本报告仅供参考，不构成投资建议。投资有风险，决策需谨慎。</p>
        <p>策略逻辑: 趋势跟踪 + 波动率风控</p>
    </div>
</body>
</html>
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
