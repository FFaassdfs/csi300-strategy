"""
多品种轮动信号系统: 510310 + 159995 + 512800
对每个品种独立运行 ADX Override，择优持仓
"""
import pandas as pd
import numpy as np
import os
import sys
import duckdb
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from config import ASSETS, HISTORY_DB
from signal_core import compute_signal_core, vol_brake_weight, adjust_splits

def ensure_all_data():
    """确保三品种日线数据在历史库 daily_ohlc 中 (缺失时从新浪下载并入库)"""
    for asset_id, info in ASSETS.items():
        conn = duckdb.connect(HISTORY_DB)
        try:
            n = conn.execute('SELECT COUNT(*) FROM daily_ohlc WHERE code = ?', [asset_id]).fetchone()[0]
        except Exception:
            n = 0
        conn.close()
        if n > 0:
            continue

        print(f'下载 {info["name"]} ({asset_id}) 历史数据...')
        try:
            import akshare as ak
            df = ak.fund_etf_hist_sina(symbol=info['code'])
            df['date'] = pd.to_datetime(df['date'])
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=5)
            df = df[df['date'] >= cutoff].sort_values('date')
            df = df[["date","open","high","low","close","volume","amount"]].dropna(subset=["close"])

            # 拆分 / 份额合并 双向复权
            adjust_splits(df, log=print)

            rows = []
            for _, r in df.iterrows():
                rows.append((
                    r['date'].date(), asset_id, info['name'],
                    float(r.get('open', 0)), float(r.get('high', 0)),
                    float(r.get('low', 0)), float(r.get('close', 0)),
                    float(r.get('volume', 0)), float(r.get('amount', 0)),
                ))
            conn = duckdb.connect(HISTORY_DB)
            conn.executemany('INSERT OR REPLACE INTO daily_ohlc VALUES (?,?,?,?,?,?,?,?,?)', rows)
            conn.close()
            print(f'  入库 {len(rows)} 条')
        except Exception as e:
            print(f'  下载失败: {e}')


def compute_adx_signal(df, ma_p=30, vol_th=18, adx_th=20):
    """对单个ETF计算ADX Override信号 (参数可配置, 委托给 signal_core)"""
    c = df['close']
    h = df['high']
    l = df['low']
    r = compute_signal_core(c, h, l, ma_p=ma_p, adx_th=adx_th, vol_th=vol_th)
    return {
        'close': c, 'ma30': r['ma'], 'vol': r['vol'], 'adx': r['adx'], 'signal': r['signal'],
        'last_close': c.iloc[-1], 'last_ma30': r['ma'].iloc[-1],
        'last_vol': r['vol'].iloc[-1], 'last_adx': r['adx'].iloc[-1],
        'last_confirmed': r['confirmed'].iloc[-1],
        'ma_p': ma_p, 'adx_th': adx_th, 'vol_th': vol_th,
    }


def get_rotation_signals():
    """获取所有品种信号并给出轮动指向 (读每日更新的历史库, 与邮件同源)"""
    ensure_all_data()

    results = {}
    conn = duckdb.connect(HISTORY_DB)
    for asset_id, info in ASSETS.items():
        try:
            df = conn.execute(
                f"SELECT date, open, high, low, close FROM daily_ohlc WHERE code = '{asset_id}' ORDER BY date"
            ).fetchdf()
            if df.empty:
                continue
            df['date'] = pd.to_datetime(df['date'])
            r = compute_adx_signal(df, ma_p=info.get('ma_p', 30), adx_th=info.get('adx_th', 20), vol_th=info.get('vol_th', 18))
            results[asset_id] = {
                'name': info['name'],
                'price': r['last_close'],
                'ma30': r['last_ma30'],
                'vol': r['last_vol'],
                'adx': r['last_adx'],
                # 最新一根K线的收盘信号 (T日收盘信号, T+1开盘执行)
                'signal': int(r['signal'].iloc[-1]),
                'confirmed': int(r['last_confirmed']),
                'date': str(df['date'].iloc[-1].date()),
            }
        except Exception:
            pass
    conn.close()

    # 轮动逻辑：在"已确认信号"的品种中选ADX最高的 (新入场需连续2日确认)
    candidates = [(k, v) for k, v in results.items() if v['signal'] == 1 and v['confirmed'] == 1]
    if len(candidates) >= 2:
        candidates.sort(key=lambda x: x[1]['adx'], reverse=True)
    pick = candidates[0][0] if candidates else None

    latest_date = ''
    for v in results.values():
        if v.get('date'):
            latest_date = max(latest_date, v['date'])

    return {
        'assets': results,
        'pick': pick,
        'date': latest_date,
    }


if __name__ == '__main__':
    ensure_all_data()
    sigs = get_rotation_signals()
    
    print(f"\n{'='*60}")
    print(f"  多品种轮动信号 ({sigs.get('date','')})")
    print(f"{'='*60}")
    print(f"  {'品种':<15} {'价格':>8} {'MA30':>8} {'ADX':>7} {'信号':>6} {'确认':>6}")

    for code in ASSETS:
        s = sigs['assets'].get(code, {})
        if not s:
            continue
        sig_text = '持有' if s['signal'] == 1 else '空仓'
        conf_text = ('已确认' if s['confirmed'] == 1 else '待确认') if s['signal'] == 1 else '-'
        print(f"  {s['name']:<15} {s['price']:>8.4f} {s['ma30']:>8.4f} {s['adx']:>6.1f} {sig_text:>6} {conf_text:>6}")

    pick = sigs['pick']
    if pick:
        name = ASSETS[pick]['name']
        s = sigs['assets'][pick]
        bw = vol_brake_weight(s['vol'])
        extra = f' (波动刹车: 仓位上限{bw:.0%}, vol {s["vol"]:.1f}%)' if bw < 1 else ''
        print(f"\n  >>> 轮动指向: {name} ({pick}){extra} <<<")
    else:
        first_day = [v for v in sigs['assets'].values() if v['signal'] == 1 and v['confirmed'] == 0]
        if first_day:
            print(f"\n  >>> 信号首日待确认 (连续第2日才可买入), 暂指向: 国债/逆回购 <<<")
        else:
            print(f"\n  >>> 轮动指向: 国债/逆回购 (无品种符合) <<<")
