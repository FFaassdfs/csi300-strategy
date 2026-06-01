"""
CSI300 ETF 每日策略执行脚本
生成明日交易信号
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.insert(0, r'D:/opencode/etf')

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
vol = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
ma50 = close.rolling(50).mean()
momentum = (close / close.shift(20) - 1)
atr = close.rolling(14).apply(lambda x: np.max(np.abs(x - x.shift(1))))
adx = (vol / atr).rolling(14).mean()

# 最新值
last_close = close.iloc[-1]
last_vol = vol.iloc[-1]
last_ma50 = ma50.iloc[-1]
last_momentum = momentum.iloc[-1]
last_adx = adx.iloc[-1]
last_date = df['date'].max()

# ========== 策略逻辑 ==========
def check_adx_override():
    """ADX Override策略"""
    above_ma = last_close > last_ma50
    low_vol = last_vol < 15
    strong_trend = last_adx > 25
    signal = 1 if (above_ma and (low_vol or strong_trend)) else 0
    trigger = 'ADX>25' if strong_trend else ('低波动' if low_vol else '高波动')
    return signal, trigger

def check_momentum_override():
    """Momentum Override策略"""
    above_ma = last_close > last_ma50
    low_vol = last_vol < 15
    strong_momentum = last_momentum > 0.10
    signal = 1 if (above_ma and (low_vol or strong_momentum)) else 0
    trigger = '动量>10%' if strong_momentum else ('低波动' if low_vol else '高波动')
    return signal, trigger

def check_absolute_15():
    """Absolute 15%策略"""
    above_ma = last_close > last_ma50
    low_vol = last_vol < 15
    signal = 1 if (above_ma and low_vol) else 0
    trigger = '低波动' if low_vol else '高波动'
    return signal, trigger

# ========== 生成报告 ==========
print()
print('='*70)
print('CSI300 ETF 策略信号报告')
print('数据日期:', last_date.strftime('%Y-%m-%d'))
print('='*70)

results = {}

# ADX Override
sig, trigger = check_adx_override()
results['ADX Override'] = {'signal': 'CSI300' if sig==1 else 'BOND', 'trigger': trigger}
print()
print('【+ADX>25 Override策略】')
print('  信号:', results['ADX Override']['signal'])
print('  触发:', trigger)
print('  收盘价:', round(last_close, 4))
print('  MA50:', round(last_ma50, 4))
print('  20日波动率:', round(last_vol, 2), '%')
print('  ADX:', round(last_adx, 2))
print('  价格>MA50:', '是' if last_close > last_ma50 else '否')

# Momentum Override
sig, trigger = check_momentum_override()
results['Momentum Override'] = {'signal': 'CSI300' if sig==1 else 'BOND', 'trigger': trigger}
print()
print('【+Momentum>10% Override策略】')
print('  信号:', results['Momentum Override']['signal'])
print('  触发:', trigger)
print('  收盘价:', round(last_close, 4))
print('  MA50:', round(last_ma50, 4))
print('  20日波动率:', round(last_vol, 2), '%')
print('  20日动量:', round(last_momentum*100, 2), '%')
print('  价格>MA50:', '是' if last_close > last_ma50 else '否')

# Absolute 15%
sig, trigger = check_absolute_15()
results['Absolute 15%'] = {'signal': 'CSI300' if sig==1 else 'BOND', 'trigger': trigger}
print()
print('【Base Absolute 15%策略】')
print('  信号:', results['Absolute 15%']['signal'])
print('  触发:', trigger)
print('  收盘价:', round(last_close, 4))
print('  MA50:', round(last_ma50, 4))
print('  20日波动率:', round(last_vol, 2), '%')
print('  价格>MA50:', '是' if last_close > last_ma50 else '否')

# ========== 汇总建议 ==========
print()
print('='*70)
print('【明日交易建议汇总】')
print('='*70)

signals = [r['signal'] for r in results.values()]
unique_signals = set(signals)

if len(unique_signals) == 1:
    final = list(unique_signals)[0]
    print('一致信号:', final)
    if final == 'CSI300':
        print()
        print('>>> 建议: 持有/买入 沪深300ETF (510310) <<<')
        print()
    else:
        print()
        print('>>> 建议: 持有/买入 国债ETF (511260) <<<')
        print()
else:
    print('策略分歧:')
    for name, r in results.items():
        print(' ', name, ':', r['signal'], '('+r['trigger']+')')

# 保存报告
report_dir = r'D:/opencode/etf/reports'
os.makedirs(report_dir, exist_ok=True)
report_file = os.path.join(report_dir, 'daily_signal_' + last_date.strftime('%Y%m%d') + '.txt')

with open(report_file, 'w', encoding='utf-8') as f:
    f.write('CSI300 ETF 策略信号报告\n')
    f.write('='*50 + '\n')
    f.write('日期: ' + last_date.strftime('%Y-%m-%d') + '\n')
    f.write('='*50 + '\n\n')
    f.write('510300 收盘: ' + str(round(last_close, 4)) + '\n')
    f.write('MA50: ' + str(round(last_ma50, 4)) + '\n')
    f.write('20日波动率: ' + str(round(last_vol, 2)) + '%\n')
    f.write('20日动量: ' + str(round(last_momentum*100, 2)) + '%\n')
    f.write('ADX: ' + str(round(last_adx, 2)) + '\n\n')
    f.write('策略信号:\n')
    for name, r in results.items():
        f.write('  ' + name + ': ' + r['signal'] + '\n')

print('报告已保存到:', report_file)
