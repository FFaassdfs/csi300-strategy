"""
多品种轮动信号系统: 510310 + 159995 + 512660
对每个品种独立运行 ADX Override，择优持仓
"""
import pandas as pd
import numpy as np
import os
import sys
import duckdb
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, 'csi300_data.duckdb')
RF = 0.025

ASSETS = {
    '510310': {'name': '沪深300ETF', 'code': 'sh510310'},
    '159995': {'name': '芯片ETF',    'code': 'sz159995'},
    '512660': {'name': '军工ETF',    'code': 'sh512660'},
}

def ensure_all_data():
    """确保所有品种数据在库中"""
    for asset_id, info in ASSETS.items():
        conn = duckdb.connect(DB_PATH)
        tbl = f'etf_{asset_id}_daily'
        try:
            conn.execute(f'SELECT 1 FROM {tbl} LIMIT 1').fetchone()
            conn.close()
            continue
        except Exception:
            conn.close()

        print(f'下载 {info["name"]} ({asset_id}) 历史数据...')
        try:
            import akshare as ak
            df = ak.fund_etf_hist_sina(symbol=info['code'])
            df['date'] = pd.to_datetime(df['date'])
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=5)
            df = df[df['date'] >= cutoff].sort_values('date')
            df = df[["date","open","high","low","close","volume","amount"]].dropna(subset=["close"])
            
            # Split detection
            c = df['close'].values
            for i in range(1, len(c)):
                if c[i] > 0 and c[i-1] > 0 and c[i-1] / c[i] > 1.8:
                    ratio = round(c[i-1] / c[i])
                    for col in ["open","high","low","close"]:
                        df.loc[df.index[:i], col] = df.loc[df.index[:i], col] / ratio
                    break
            
            conn = duckdb.connect(DB_PATH)
            conn.execute(f'DROP TABLE IF EXISTS {tbl}')
            conn.execute(f'CREATE TABLE {tbl} (date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, amount DOUBLE)')
            conn.execute(f'INSERT INTO {tbl} SELECT * FROM df')
            conn.close()
        except Exception as e:
            print(f'  下载失败: {e}')


def compute_adx_signal(df):
    """对单个ETF计算ADX Override信号"""
    close = df['close']
    high = df['high']
    low = df['low']
    ma50 = close.rolling(50).mean()
    vol = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    up = high.diff(); dn = -low.diff()
    p_dm = pd.Series(0.0, index=df.index); n_dm = pd.Series(0.0, index=df.index)
    p_dm.loc[(up > dn) & (up > 0)] = up
    n_dm.loc[(dn > up) & (dn > 0)] = dn
    pdi = 100 * p_dm.ewm(alpha=1/14, adjust=False).mean() / atr14
    ndi = 100 * n_dm.ewm(alpha=1/14, adjust=False).mean() / atr14
    adx = (100 * abs(pdi - ndi) / (pdi + ndi + 1e-10)).ewm(alpha=1/14, adjust=False).mean()
    above = close > ma50
    low_vol = vol < 15
    trend = adx > 25
    signal = (above & (low_vol | trend)).astype(int)
    return {
        'close': close, 'ma50': ma50, 'vol': vol, 'adx': adx, 'signal': signal,
        'last_close': close.iloc[-1], 'last_ma50': ma50.iloc[-1],
        'last_vol': vol.iloc[-1], 'last_adx': adx.iloc[-1],
    }


def get_rotation_signals():
    """获取所有品种信号并给出轮动指向"""
    ensure_all_data()
    
    results = {}
    conn = duckdb.connect(DB_PATH)
    for asset_id, info in ASSETS.items():
        tbl = f'etf_{asset_id}_daily'
        try:
            df = conn.execute(f'SELECT date, open, high, low, close FROM {tbl} ORDER BY date').fetchdf()
            df['date'] = pd.to_datetime(df['date'])
            r = compute_adx_signal(df)
            results[asset_id] = {
                'name': info['name'],
                'price': r['last_close'],
                'ma50': r['last_ma50'],
                'vol': r['last_vol'],
                'adx': r['last_adx'],
                'signal': int(r['signal'].iloc[-2]),  # yesterday's signal
            }
        except Exception:
            pass
    conn.close()
    
    # 轮动逻辑：选有信号中ADX最高的
    candidates = [(k, v) for k, v in results.items() if v['signal'] == 1]
    if len(candidates) >= 2:
        candidates.sort(key=lambda x: x[1]['adx'], reverse=True)
    pick = candidates[0][0] if candidates else None
    
    return {
        'assets': results,
        'pick': pick,
        'date': list(results.values())[0]['price'] if results else '',
    }


if __name__ == '__main__':
    ensure_all_data()
    sigs = get_rotation_signals()
    
    print(f"\n{'='*60}")
    print(f"  多品种轮动信号 ({sigs.get('date','')})")
    print(f"{'='*60}")
    print(f"  {'品种':<15} {'价格':>8} {'MA50':>8} {'ADX':>7} {'信号':>6}")
    
    for code in ASSETS:
        s = sigs['assets'].get(code, {})
        if not s:
            continue
        sig_text = '持有' if s['signal'] == 1 else '空仓'
        print(f"  {s['name']:<15} {s['price']:>8.4f} {s['ma50']:>8.4f} {s['adx']:>6.1f} {sig_text:>6}")
    
    pick = sigs['pick']
    if pick:
        name = ASSETS[pick]['name']
        print(f"\n  >>> 轮动指向: {name} ({pick}) <<<")
    else:
        print(f"\n  >>> 轮动指向: 国债/逆回购 (无品种符合) <<<")
