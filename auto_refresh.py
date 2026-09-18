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
sys.path.insert(0, PROJECT_ROOT)
from config import ASSETS, HISTORY_DB, WORK_DB
from signal_core import compute_signal_core, VOL_BRAKE_THRESHOLD

YEARS = 5

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


def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}')


def fetch_today_fallback(code, info, last_date):
    """新浪日线缺当日bar时 (收盘后常滞后1-3小时), 用备用源补齐当日 OHLC:
    1) 东方财富 fund_etf_hist_em (日线, 收盘后很快含当日)
    2) 腾讯实时 qt.gtimg.cn (含当日完整 OHLC)
    仅在 15:00 后启用 (避免把盘中未定价格当收盘价); 全部失败返回 None (维持原滞后行为)"""
    import datetime as _dt
    now = datetime.now()
    if now.time() < _dt.time(15, 0):
        return None
    today = now.date()
    if last_date is not None and last_date >= today:
        return None  # 新浪已有当日数据

    # 1) 东方财富日线
    try:
        import akshare as ak
        df = ak.fund_etf_hist_em(symbol=code, period='daily',
                                 start_date=today.strftime('%Y%m%d'),
                                 end_date=today.strftime('%Y%m%d'), adjust='')
        if df is not None and len(df) > 0:
            r = df.iloc[-1]
            if pd.to_datetime(str(r['日期'])).date() == today:
                return {'date': pd.Timestamp(today), 'open': float(r['开盘']), 'high': float(r['最高']),
                        'low': float(r['最低']), 'close': float(r['收盘']),
                        'volume': float(r.get('成交量', 0)), 'amount': float(r.get('成交额', 0))}
    except Exception as e:
        log(f'  [WARN] {code} 东财补齐失败: {e}')

    # 2) 腾讯实时 (15:00 后 OHLC 即最终值)
    try:
        import requests
        r = requests.get(f"https://qt.gtimg.cn/q={info['code']}", timeout=8)
        r.encoding = 'gbk'
        p = r.text.split('"')[1].split('~')
        if len(p) > 37 and p[30][:8] == today.strftime('%Y%m%d'):
            return {'date': pd.Timestamp(today), 'open': float(p[5]), 'high': float(p[33]),
                    'low': float(p[34]), 'close': float(p[3]),
                    'volume': float(p[6]) * 100, 'amount': float(p[37]) * 10000}
    except Exception as e:
        log(f'  [WARN] {code} 腾讯补齐失败: {e}')

    return None


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

            # Split adjustment (循环检测, 支持多次拆分)
            i = 1
            while i < len(df):
                c = df['close'].values
                if c[i] > 0 and c[i-1] > 0 and c[i-1] / c[i] > 1.8:
                    ratio = round(c[i-1] / c[i])
                    log(f'  [SPLIT] {code} 1:{ratio} on {df["date"].iloc[i].date()}')
                    for col in ['open', 'high', 'low', 'close']:
                        df.loc[df.index[:i], col] = df.loc[df.index[:i], col] / ratio
                i += 1

            df = df[df['date'] >= cutoff]

            # 新浪日线滞后时 (收盘后常缺当日bar) → 东财/腾讯补齐当日 OHLC
            bar = fetch_today_fallback(code, info, df['date'].iloc[-1].date() if len(df) else None)
            if bar is not None:
                df = pd.concat([df, pd.DataFrame([bar])]).reset_index(drop=True)
                log(f'  [FALLBACK] {code} 当日bar已从东财/腾讯补齐: close={bar["close"]}')

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
    """计算指标并记录信号 (指标统一走 signal_core, 与邮件/快照完全一致)
    信号归属日期 = 数据实际日期, 而非运行当天 (新浪数据常滞后1-3小时)"""
    for code, info in ASSETS.items():
        df = conn.execute(
            f"SELECT date, open, high, low, close FROM daily_ohlc WHERE code = '{code}' ORDER BY date"
        ).fetchdf()
        if len(df) < 60:
            continue

        # 数据实际最新日期 (防止数据滞后时信号记错日期)
        data_date = df['date'].iloc[-1].date() if not pd.isna(df['date'].iloc[-1]) else datetime.now().date()

        c = df['close']; h = df['high']; l = df['low']
        ma_p = info.get('ma_p', 30)
        adx_th = info.get('adx_th', 20)
        vol_th = info.get('vol_th', 18)

        momentum = c / c.shift(20) - 1

        bb_mid = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        bb_pct_b = (c - (bb_mid - 2*bb_std)) / ((bb_mid + 2*bb_std) - (bb_mid - 2*bb_std) + 1e-10)

        core = compute_signal_core(c, h, l, ma_p=ma_p, adx_th=adx_th, vol_th=vol_th)
        ma50 = core['ma']; vol = core['vol']; adx = core['adx']; signal = core['signal']
        confirmed = core['confirmed']
        above_ma50 = (c > ma50).astype(int)
        low_vol = (vol < vol_th).astype(int)
        strong_trend = (adx > adx_th).astype(int)

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
            (data_date, code, ma50_v, adx_v, vol_v, mom_v, bb_v, above_v, low_v, trend_v, sig_v)
        )

        # 写入信号日志
        conf_v = int(confirmed.iloc[-1])
        reason = ''
        if sig_v == 1:
            base = f'ADX>{adx_th}' if trend_v else ('低波动' if low_v else '价格>MA')
            reason = base + ('/2日确认' if conf_v else '/首日待确认')
            if vol_v > VOL_BRAKE_THRESHOLD:
                reason += '/波动刹车'
        else:
            reason = '价格<MA' if not above_v else ('高波动+低趋势' if not low_v and not trend_v else '')
        conn.execute(
            '''INSERT OR REPLACE INTO signals_log VALUES (?,?,?,?,?,?)''',
            (data_date, code, info['name'], price, sig_v, reason)
        )

        sig_text = '持有' if sig_v == 1 else '空仓'
        log(f'  [{data_date}] {info["name"]} ({code}): {price:.4f} MA30={ma50_v:.4f} ADX={adx_v:.1f} vol={vol_v:.1f}% -> {sig_text} [{reason}]')

    return


def collect_sentiment():
    """采集舆情数据 (date取数据源自身日期, 避免滞后数据记到当天)"""
    result = {'date': None, 'qvix': None, 'north_flow': None, 'main_flow': None,
              'global_djia': None, 'global_nasdaq': None, 'global_hsi': None, 'global_n225': None}
    try:
        import akshare as ak
        try:
            df = ak.index_option_300etf_qvix()
            result['qvix'] = round(float(df.iloc[-1]['close']), 2)
            if 'date' in df.columns and len(df) > 0:
                d = df.iloc[-1]['date']
                if not pd.isna(d):
                    result['date'] = d
        except: pass
    except: pass
    if result['date'] is None:
        result['date'] = datetime.now().date()
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
