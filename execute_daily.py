"""
CSI300 ETF 每日策略执行脚本
主策略: ADX Override (价格>MA50 AND (波动率<15% OR ADX>25))
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from strategies.csi300_strategies import (
    compute_adx, compute_volatility, compute_ma, compute_momentum,
)

# ========== 获取数据 ==========
print('正在获取数据...')
try:
    import akshare as ak
    df_310 = ak.fund_etf_hist_sina(symbol='sh510310')
    df_310['date'] = pd.to_datetime(df_310['date'])
    df_310 = df_310.sort_values('date').reset_index(drop=True)

    df_bond = ak.fund_etf_hist_sina(symbol='sh511260')
    df_bond['date'] = pd.to_datetime(df_bond['date'])
    df_bond = df_bond.sort_values('date').reset_index(drop=True)

    df = df_310.merge(df_bond, on='date', suffixes=('_etf', '_bond'))
    df = df.sort_values('date').reset_index(drop=True)
    print('数据获取成功')
except Exception as e:
    print('数据获取失败:', e)
    sys.exit(1)

# ========== 计算指标 ==========
close = df['close_etf']
vol = compute_volatility(df)
ma50 = compute_ma(df)
momentum = compute_momentum(df)
adx = compute_adx(df)

last_close = close.iloc[-1]
last_vol = vol.iloc[-1]
last_ma50 = ma50.iloc[-1]
last_momentum = momentum.iloc[-1]
last_adx = adx.iloc[-1]
last_date = df['date'].max()

# ========== 主策略: ADX Override ==========
above_ma = last_close > last_ma50
low_vol = last_vol < 15
strong_trend = last_adx > 25

main_signal = 1 if (above_ma and (low_vol or strong_trend)) else 0
if strong_trend:
    main_trigger = 'ADX>25 趋势确认'
elif low_vol:
    main_trigger = '低波动'
else:
    main_trigger = '高波动/低趋势'

# ========== 辅助参考 ==========
# BB %B
bb_mid = close.rolling(20).mean()
bb_std = close.rolling(20).std()
bb_lower = bb_mid - 2 * bb_std
bb_upper = bb_mid + 2 * bb_std
last_bb_pct_b = (last_close - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-10)
bb_oversold = last_bb_pct_b < 0.2 and last_close > last_ma50

# MACD
ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
last_macd_hist = (ema12 - ema26 - (ema12 - ema26).ewm(span=9, adjust=False).mean()).iloc[-1]
aux_conservative = 1 if (main_signal == 1 and last_macd_hist > 0) else 0
conservative_diverges = (main_signal == 1 and aux_conservative == 0)

# ========== 输出 ==========
print()
print('=' * 60)
print('  CSI300 ETF 策略信号报告')
print('  数据日期:', last_date.strftime('%Y-%m-%d'))
print('=' * 60)
print()
print('  ── 市场指标 ──')
print(f'  沪深300:    {last_close:.2f}')
print(f'  MA50:       {last_ma50:.2f}  ({"上方" if above_ma else "下方"})')
print(f'  波动率(20d): {last_vol:.2f}%  (阈值15%)')
print(f'  ADX:        {last_adx:.2f}  (阈值25)')
print(f'  动量(20d):  {last_momentum*100:+.2f}%')
print()
print('  ── 主策略: ADX Override ──')
print(f'  信号:       {"持有CSI300" if main_signal==1 else "持有国债"}')
print(f'  触发条件:   {main_trigger}')
print()
print('  ── 辅助参考（观察，不执行）──')
print(f'  布林带 %B:   {last_bb_pct_b:.2f}  {"[!] 极端超卖区间!" if bb_oversold else "正常区间"}')
print(f'  保守信号:     {"[!] 与主信号分歧!" if conservative_diverges else "[OK] 与主信号一致"}  (MACD柱={last_macd_hist:+.2f})')
print()
print('=' * 60)

if main_signal == 1:
    print('  >>> 建议: 持有/买入 沪深300ETF (510310) <<<')
else:
    print('  >>> 建议: 持有/买入 国债ETF (511260) <<<')

# ========== 保存报告 ==========
report_dir = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(report_dir, exist_ok=True)
report_file = os.path.join(report_dir, 'daily_signal_' + last_date.strftime('%Y%m%d') + '.txt')

with open(report_file, 'w', encoding='utf-8') as f:
    f.write(f'CSI300 ETF 策略信号报告 - 主策略: ADX Override\n')
    f.write('=' * 50 + '\n')
    f.write(f'日期: {last_date.strftime("%Y-%m-%d")}\n')
    f.write(f'收盘: {last_close:.2f}  MA50: {last_ma50:.2f}  波动率: {last_vol:.2f}%  ADX: {last_adx:.2f}\n')
    f.write(f'主信号: {"持有CSI300" if main_signal==1 else "持有国债"}  ({main_trigger})\n')

print(f'\n报告已保存: {report_file}')
