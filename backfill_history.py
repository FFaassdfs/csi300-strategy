"""
全量历史数据填充 + 历史信号重算 (与生产 auto_refresh/signal_core 完全一致, 2026-09-10 重写)
1. 从 akshare 拉取 config.ASSETS 各品种全部历史 (自上市起, 不截断5年) + CSI300指数
2. 拆分复权处理 (支持多次拆分)
3. 填充到 trading_history.duckdb
4. 用 signal_core 重算全部历史指标与信号
   - 信号归属日期 = 数据日期 T (T日收盘信号, 供 T+1 开盘执行), 与 auto_refresh 相同语义
   - reason 文案与 auto_refresh/signals_log 一致

用法: python backfill_history.py
"""
import pandas as pd
import numpy as np
import os
import sys
import duckdb
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from config import ASSETS, HISTORY_DB
from signal_core import compute_signal_core, VOL_BRAKE_THRESHOLD, adjust_splits

INDEX_ASSET = {'000300': {'name': '沪深300指数', 'code': 'sh000300', 'is_index': True}}


def fetch_full_history(info):
    """拉取全部历史 (支持多次拆分检测)"""
    import akshare as ak
    if info.get('is_index'):
        df = ak.stock_zh_index_daily(symbol=info['code'])
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'open', 'high', 'low', 'close']].dropna().sort_values('date')
        df['volume'] = 0.0
        df['amount'] = 0.0
    else:
        df = ak.fund_etf_hist_sina(symbol=info['code'])
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].dropna(subset=['close']).sort_values('date')

        # 拆分 / 份额合并 双向复权
        ev = adjust_splits(df)
        if ev:
            print(f'    [复权] {info["name"]}: {ev}')

    return df


def upsert_ohlc(conn, code, name, df):
    """全量写入 (主键冲突时替换)"""
    rows = []
    for _, row in df.iterrows():
        rows.append((
            row['date'].date(), code, name,
            float(row['open']), float(row['high']), float(row['low']),
            float(row['close']), float(row.get('volume', 0)), float(row.get('amount', 0))
        ))
    conn.executemany('INSERT OR REPLACE INTO daily_ohlc VALUES (?,?,?,?,?,?,?,?,?)', rows)
    return len(rows)


def _reason(above, low_v, trend, sig, conf_v, vol_v, adx_th):
    if sig == 1:
        base = f'ADX>{adx_th}' if trend else ('低波动' if low_v else '价格>MA')
        r = base + ('/2日确认' if conf_v else '/首日待确认')
        if vol_v > VOL_BRAKE_THRESHOLD:
            r += '/波动刹车'
        return r
    return '价格<MA' if not above else ('高波动+低趋势' if not low_v and not trend else '')


def recompute_indicators(conn):
    """对 config.ASSETS 各品种全历史重算指标与信号 (统一走 signal_core)
    与 auto_refresh.compute_and_log_signals 同语义: 信号记在数据日期 T, 存 T 日价格"""
    for code, info in ASSETS.items():
        df = conn.execute(
            f"SELECT date, open, high, low, close FROM daily_ohlc WHERE code = '{code}' ORDER BY date"
        ).fetchdf()
        if len(df) < 60:
            continue

        c = df['close'].astype(float); h = df['high'].astype(float); l = df['low'].astype(float)
        core = compute_signal_core(c, h, l, ma_p=info['ma_p'], adx_th=info['adx_th'], vol_th=info['vol_th'])
        ma = core['ma']; vol = core['vol']; adx = core['adx']; signal = core['signal']; confirmed = core['confirmed']
        momentum = c / c.shift(20) - 1

        bb_mid = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        bb_pct_b = (c - (bb_mid - 2*bb_std)) / ((bb_mid + 2*bb_std) - (bb_mid - 2*bb_std) + 1e-10)

        ind_rows = []
        sig_rows = []
        for i in range(len(df)):
            ma_i = ma.iloc[i]; vol_i = vol.iloc[i]; adx_i = adx.iloc[i]
            mom_i = momentum.iloc[i]; bb_i = bb_pct_b.iloc[i]
            above_v = bool(c.iloc[i] > ma_i) if not pd.isna(ma_i) else False
            low_v = bool(vol_i < info['vol_th']) if not pd.isna(vol_i) else False
            trend_v = bool(adx_i > info['adx_th']) if not pd.isna(adx_i) else False
            sig_v = int(signal.iloc[i])
            conf_v = int(confirmed.iloc[i])
            d = df['date'].iloc[i].date()
            ind_rows.append((
                d, code,
                round(float(ma_i), 6) if not pd.isna(ma_i) else None,
                round(float(adx_i), 6) if not pd.isna(adx_i) else None,
                round(float(vol_i), 6) if not pd.isna(vol_i) else None,
                round(float(mom_i), 6) if not pd.isna(mom_i) else None,
                round(float(bb_i), 6) if not pd.isna(bb_i) else None,
                above_v, low_v, trend_v, sig_v
            ))
            sig_rows.append((
                d, code, info['name'],
                round(float(c.iloc[i]), 6), sig_v,
                _reason(above_v, low_v, trend_v, sig_v, conf_v,
                        float(vol_i) if not pd.isna(vol_i) else 0.0, info['adx_th'])
            ))

        conn.executemany('INSERT OR REPLACE INTO daily_indicators VALUES (?,?,?,?,?,?,?,?,?,?,?)', ind_rows)
        conn.executemany('INSERT OR REPLACE INTO signals_log VALUES (?,?,?,?,?,?)', sig_rows)
        print(f'  {info["name"]} ({code}): 指标 {len(ind_rows)} 条, 信号 {len(sig_rows)} 条')


def main():
    print('=' * 60)
    print('  全量历史数据填充 (signal_core 统一口径)')
    print('=' * 60)

    conn = duckdb.connect(HISTORY_DB)

    all_assets = dict(INDEX_ASSET)
    all_assets.update(ASSETS)
    for code, info in all_assets.items():
        print(f'\n拉取 {info["name"]} ({code})...')
        try:
            df = fetch_full_history(info)
            n = upsert_ohlc(conn, code, info['name'], df)
            print(f'  写入 {n} 条  ({df["date"].min().date()} ~ {df["date"].max().date()})')
        except Exception as e:
            print(f'  失败: {e}')

    print('\n重算全历史指标与信号...')
    recompute_indicators(conn)

    conn.close()

    print('\n' + '=' * 60)
    print('  完成! 验证:')
    print('=' * 60)
    vconn = duckdb.connect(HISTORY_DB)
    print(vconn.execute('SELECT code, COUNT(*) cnt, MIN(date) AS "start", MAX(date) AS "end" FROM daily_ohlc GROUP BY code ORDER BY code').fetchdf().to_string())
    print()
    print(vconn.execute("SELECT code, MAX(date) latest, COUNT(*) signals FROM signals_log GROUP BY code ORDER BY code").fetchdf().to_string())
    vconn.close()


if __name__ == '__main__':
    main()
