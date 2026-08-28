"""
自动刷新脚本: 追加式数据收集 + 指标计算 + 信号记录
每天收盘后运行 (建议 Windows 计划任务 15:10)

用法: python auto_refresh.py
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import duckdb
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
HISTORY_DB = os.path.join(PROJECT_ROOT, 'trading_history.duckdb')
WORK_DB = os.path.join(PROJECT_ROOT, 'csi300_data.duckdb')
RF = 0.025

# ===== 全局网络超时保护 (防止akshare请求卡死) =====
import socket
socket.setdefaulttimeout(20)

# requests 全局超时
try:
    import requests.adapters
    from requests.adapters import HTTPAdapter
    import requests
    _orig_send = requests.sessions.Session.request
    def _timeout_send(self, method, url, **kwargs):
        kwargs.setdefault('timeout', 20)
        return _orig_send(self, method, url, **kwargs)
    requests.sessions.Session.request = _timeout_send
except Exception:
    pass

ASSETS = {
    '510310': {'name': '沪深300ETF', 'code': 'sh510310', 'ma_p': 30, 'adx_th': 20, 'vol_th': 18},
    '159995': {'name': '芯片ETF',    'code': 'sz159995', 'ma_p': 30, 'adx_th': 25, 'vol_th': 15},
    '512660': {'name': '军工ETF',    'code': 'sh512660', 'ma_p': 30, 'adx_th': 20, 'vol_th': 15, 'monitor_only': True},
    '512800': {'name': '银行ETF',    'code': 'sh512800', 'ma_p': 30, 'adx_th': 25, 'vol_th': 18, 'long_hold': True},
}

YEARS = 5
BOND_DAILY = (1 + RF) ** (1/252) - 1


def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}')


def fetch_all_daily():
    """获取所有品种日线 (含拆分调整, socket超时保护)"""
    import akshare as ak
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=YEARS)

    result = {}
    # CSI300 index
    try:
        df = ak.stock_zh_index_daily(symbol='sh000300')
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df = df[['date', 'open', 'high', 'low', 'close']].dropna().sort_values('date')
            df = df[df['date'] >= cutoff]
    except Exception as e:
        log(f'  [WARN] 沪深300指数获取失败: {e}')
        df = pd.DataFrame()
    result['000300'] = df

    for code, info in ASSETS.items():
        try:
            df = ak.fund_etf_hist_sina(symbol=info['code'])
            if len(df) == 0:
                log(f'  [WARN] {code} 数据为空, 跳过')
                continue
            df['date'] = pd.to_datetime(df['date'])
            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].dropna(subset=['close']).sort_values('date')

            # Split adjustment
            c = df['close'].values
            for i in range(1, len(c)):
                if c[i] > 0 and c[i-1] > 0 and c[i-1] / c[i] > 1.8:
                    ratio = round(c[i-1] / c[i])
                    log(f'  [SPLIT] {code} 1:{ratio} on {df["date"].iloc[i].date()}')
                    for col in ['open', 'high', 'low', 'close']:
                        df.loc[df.index[:i], col] = df.loc[df.index[:i], col] / ratio
                    break

            df = df[df['date'] >= cutoff]
            result[code] = df
        except Exception as e:
            log(f'  [WARN] {code} 获取失败: {e}')

    return result


def append_ohlc(conn, code, name, df):
    """追加OHLC数据 (去重: 已存在的date跳过)"""
    existing = conn.execute(
        'SELECT date FROM daily_ohlc WHERE code = ?', [code]
    ).fetchall()
    existing_dates = {d[0] for d in existing}

    new_rows = []
    for _, row in df.iterrows():
        d = row['date'].date()
        if d not in existing_dates:
            new_rows.append((
                d, code, name,
                float(row.get('open', 0)), float(row.get('high', 0)),
                float(row.get('low', 0)), float(row.get('close', 0)),
                float(row.get('volume', 0)), float(row.get('amount', 0))
            ))

    if new_rows:
        conn.executemany(
            'INSERT OR REPLACE INTO daily_ohlc VALUES (?,?,?,?,?,?,?,?,?)',
            new_rows
        )
    return len(new_rows)


def compute_and_log_signals(conn):
    """计算指标并记录信号"""
    today = datetime.now().date()

    for code, info in ASSETS.items():
        df = conn.execute(
            f"SELECT date, open, high, low, close FROM daily_ohlc WHERE code = '{code}' ORDER BY date"
        ).fetchdf()
        if len(df) < 60:
            continue

        c = df['close']; h = df['high']; l = df['low']
        ma_p = info.get('ma_p', 30)
        adx_th = info.get('adx_th', 20)
        vol_th = info.get('vol_th', 18)

        ma50 = c.rolling(ma_p).mean()
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
        low_vol = (vol < vol_th).astype(int)
        strong_trend = (adx > adx_th).astype(int)
        signal = (above_ma50 & (low_vol | strong_trend)).astype(int)

        # 最新值 (当日收盘信号)
        i = -1
        price = float(c.iloc[i])
        ma50_v = float(ma50.iloc[i]) if not pd.isna(ma50.iloc[i]) else 0
        adx_v = float(adx.iloc[i]) if not pd.isna(adx.iloc[i]) else 0
        vol_v = float(vol.iloc[i]) if not pd.isna(vol.iloc[i]) else 0
        mom_v = float(momentum.iloc[i]) if not pd.isna(momentum.iloc[i]) else 0
        bb_v = float(bb_pct_b.iloc[i]) if not pd.isna(bb_pct_b.iloc[i]) else 0
        sig_v = int(signal.iloc[-1])  # 当日收盘信号 (用于次日操作)
        above_v = bool(above_ma50.iloc[i])
        low_v = bool(low_vol.iloc[i])
        trend_v = bool(strong_trend.iloc[i])

        # 写入指标
        conn.execute(
            '''INSERT OR REPLACE INTO daily_indicators VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
            (today, code, ma50_v, adx_v, vol_v, mom_v, bb_v, above_v, low_v, trend_v, sig_v)
        )

        # 写入信号日志
        reason = ''
        if sig_v == 1:
            reason = 'ADX>25' if trend_v else ('低波动' if low_v else '价格>MA50')
        else:
            reason = '价格<MA50' if not above_v else ('高波动+低趋势' if not low_v and not trend_v else '')
        conn.execute(
            '''INSERT OR REPLACE INTO signals_log VALUES (?,?,?,?,?,?)''',
            (today, code, info['name'], price, sig_v, reason)
        )

        sig_text = '持有' if sig_v == 1 else '空仓'
        log(f'  {info["name"]} ({code}): {price:.4f} MA50={ma50_v:.4f} ADX={adx_v:.1f} vol={vol_v:.1f}% -> {sig_text} [{reason}]')

    return


def collect_sentiment():
    """采集舆情数据"""
    result = {'date': datetime.now().date(), 'qvix': None, 'north_flow': None, 'main_flow': None,
              'global_djia': None, 'global_nasdaq': None, 'global_hsi': None, 'global_n225': None}
    try:
        import akshare as ak
        try:
            df = ak.index_option_300etf_qvix()
            result['qvix'] = round(float(df.iloc[-1]['close']), 2)
        except: pass
    except: pass
    return result


def main():
    log('=== 自动数据刷新开始 ===')
    log(f'拉取 {len(ASSETS)} 个品种 + CSI300 数据...')

    all_data = fetch_all_daily()

    conn = duckdb.connect(HISTORY_DB)

    # 写入 CSI300
    n = append_ohlc(conn, '000300', '沪深300指数', all_data['000300'])
    log(f'  [OK] 000300 沪深300指数: 新增 {n} 条')

    # 写入各ETF
    for code, info in ASSETS.items():
        if code in all_data:
            n = append_ohlc(conn, code, info['name'], all_data[code])
            log(f'  [OK] {code} {info["name"]}: 新增 {n} 条')

    # 计算指标并记录信号
    log('计算指标并记录信号...')
    compute_and_log_signals(conn)

    # 舆情
    log('采集舆情...')
    senti = collect_sentiment()
    if senti['qvix']:
        conn.execute(
            '''INSERT OR REPLACE INTO daily_sentiment (date, qvix) VALUES (?,?)''',
            (senti['date'], senti['qvix'])
        )
        log(f'  QVIX: {senti["qvix"]}')

    conn.close()
    log('=== 完成 ===')


if __name__ == '__main__':
    main()
