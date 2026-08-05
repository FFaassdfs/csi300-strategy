"""
全量历史数据填充 + 历史信号重算
1. 从 akshare 拉取各品种全部历史 (不截断5年)
2. 拆分复权处理
3. 填充到 trading_history.duckdb
4. 重算全部历史的指标与信号 (形成完整历史信号轨迹)

用法: python backfill_history.py
"""
import pandas as pd
import numpy as np
import os
import duckdb
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
HISTORY_DB = os.path.join(PROJECT_ROOT, 'trading_history.duckdb')

ASSETS = {
    '000300': {'name': '沪深300指数', 'code': 'sh000300', 'is_index': True},
    '510310': {'name': '沪深300ETF', 'code': 'sh510310', 'is_index': False},
    '159995': {'name': '芯片ETF',    'code': 'sz159995', 'is_index': False},
    '512660': {'name': '军工ETF',    'code': 'sh512660', 'is_index': False},
}


def fetch_full_history(info):
    """拉取全部历史"""
    import akshare as ak
    if info['is_index']:
        df = ak.stock_zh_index_daily(symbol=info['code'])
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'open', 'high', 'low', 'close']].dropna().sort_values('date')
        df['volume'] = 0.0
        df['amount'] = 0.0
    else:
        df = ak.fund_etf_hist_sina(symbol=info['code'])
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].dropna(subset=['close']).sort_values('date')

        # 拆分复权
        c = df['close'].values
        splits = []
        for i in range(1, len(c)):
            if c[i] > 0 and c[i-1] > 0 and c[i-1] / c[i] > 1.8:
                ratio = round(c[i-1] / c[i])
                splits.append((str(df['date'].iloc[i].date()), ratio))
                for col in ['open', 'high', 'low', 'close']:
                    df.loc[df.index[:i], col] = df.loc[df.index[:i], col] / ratio
                c = df['close'].values

        if splits:
            print(f'    [SPLIT] {info["name"]}: {splits}')

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


def recompute_indicators(conn):
    """对每个品种全历史重算指标与信号, 写入 indicators + signals_log"""
    from datetime import timedelta

    for code, info in ASSETS.items():
        df = conn.execute(
            f"SELECT date, open, high, low, close FROM daily_ohlc WHERE code = '{code}' ORDER BY date"
        ).fetchdf()
        if len(df) < 60:
            continue

        c = df['close']; h = df['high']; l = df['low']
        ma50 = c.rolling(50).mean()
        vol = c.pct_change().rolling(20).std() * np.sqrt(252) * 100
        momentum = c / c.shift(20) - 1

        tr1 = h - l
        tr2 = abs(h - c.shift(1))
        tr3 = abs(l - c.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean()
        up = h.diff(); dn = -l.diff()
        pdm = pd.Series(0.0, index=df.index); ndm = pd.Series(0.0, index=df.index)
        pdm.loc[(up > dn) & (up > 0)] = up
        ndm.loc[(dn > up) & (dn > 0)] = dn
        pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atr
        ndi = 100 * ndm.ewm(alpha=1/14, adjust=False).mean() / atr
        adx = (100 * abs(pdi - ndi) / (pdi + ndi + 1e-10)).ewm(alpha=1/14, adjust=False).mean()

        bb_mid = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        bb_pct_b = (c - (bb_mid - 2*bb_std)) / ((bb_mid + 2*bb_std) - (bb_mid - 2*bb_std) + 1e-10)

        above_ma50 = (c > ma50).astype(int)
        low_vol = (vol < 15).astype(int)
        strong_trend = (adx > 25).astype(int)
        signal = (above_ma50 & (low_vol | strong_trend)).astype(int)

        # 写指标 (从第50天开始)
        ind_rows = []
        sig_rows = []
        for i in range(50, len(df)):
            d = df['date'].iloc[i].date()
            ind_rows.append((
                d, code,
                round(float(ma50.iloc[i]), 6) if not pd.isna(ma50.iloc[i]) else None,
                round(float(adx.iloc[i]), 6) if not pd.isna(adx.iloc[i]) else None,
                round(float(vol.iloc[i]), 6) if not pd.isna(vol.iloc[i]) else None,
                round(float(momentum.iloc[i]), 6) if not pd.isna(momentum.iloc[i]) else None,
                round(float(bb_pct_b.iloc[i]), 6) if not pd.isna(bb_pct_b.iloc[i]) else None,
                bool(above_ma50.iloc[i]), bool(low_vol.iloc[i]), bool(strong_trend.iloc[i]),
                int(signal.iloc[i])
            ))
            # 信号日志: 用 T 日收盘算出的信号, 记录为 T+1 操作建议
            if i < len(df) - 1:
                sig_rows.append((
                    df['date'].iloc[i+1].date(), code, info['name'],
                    round(float(c.iloc[i+1]), 6), int(signal.iloc[i]),
                    _reason(above_ma50.iloc[i], low_vol.iloc[i], strong_trend.iloc[i], int(signal.iloc[i]))
                ))

        conn.executemany('INSERT OR REPLACE INTO daily_indicators VALUES (?,?,?,?,?,?,?,?,?,?,?)', ind_rows)
        conn.executemany('INSERT OR REPLACE INTO signals_log VALUES (?,?,?,?,?,?)', sig_rows)
        print(f'  {info["name"]} ({code}): 指标 {len(ind_rows)} 条, 信号 {len(sig_rows)} 条')


def _reason(above, low_v, trend, sig):
    if sig == 1:
        return 'ADX>25' if trend else ('低波动' if low_v else '价格>MA50')
    return '价格<MA50' if not above else '高波动+低趋势'


def main():
    print('=' * 60)
    print('  全量历史数据填充')
    print('=' * 60)

    conn = duckdb.connect(HISTORY_DB)

    for code, info in ASSETS.items():
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
    print(vconn.execute("SELECT code, COUNT(*) cnt, MIN(date) start, MAX(date) end FROM daily_ohlc GROUP BY code ORDER BY code").fetchdf().to_string())
    print()
    print(vconn.execute("SELECT COUNT(*) signals FROM signals_log").fetchdf().to_string())
    vconn.close()


if __name__ == '__main__':
    main()
