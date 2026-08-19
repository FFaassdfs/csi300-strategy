"""
盘中/收盘策略快照: 抓取最新数据 + 计算三品种信号 + 输出报告
用法: python intraday_signal.py [mid|close]
  mid   = 午间快照 (11:35 后)
  close = 收盘快照 (15:35 后)
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import duckdb
import requests
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WORK_DB = os.path.join(PROJECT_ROOT, 'csi300_data.duckdb')
HISTORY_DB = os.path.join(PROJECT_ROOT, 'trading_history.duckdb')

ASSETS = {
    '510310': {'name': '沪深300ETF', 'code': 'sh510310', 'ma_p': 30, 'adx_th': 20, 'vol_th': 18},
    '159995': {'name': '芯片ETF',    'code': 'sz159995', 'ma_p': 30, 'adx_th': 25, 'vol_th': 15},
    '512800': {'name': '银行ETF',    'code': 'sh512800', 'ma_p': 30, 'adx_th': 25, 'vol_th': 18},
}
RF = 0.025
BOND_DAILY = (1 + RF) ** (1/252) - 1


def get_realtime_quotes():
    """获取实时行情 (新浪)"""
    headers = {"Referer": "https://finance.sina.com.cn"}
    quotes = {}
    for code, info in ASSETS.items():
        try:
            sym = info['code']
            r = requests.get(f"https://hq.sinajs.cn/list={sym}", headers=headers, timeout=8)
            r.encoding = "gbk"
            p = r.text.strip().split('"')[1].split(",")
            px = float(p[3]) if p[3] != "0.000" else float(p[2])
            prev = float(p[2])
            quotes[code] = {'price': px, 'prev': prev, 'open': float(p[1]),
                            'high': float(p[4]), 'low': float(p[5])}
        except Exception:
            pass
    return quotes


def get_latest_ohlc(code):
    """从历史库取最近OHLC (用于MA/ADX等指标)"""
    conn = duckdb.connect(HISTORY_DB)
    df = conn.execute(
        f"SELECT date, open, high, low, close FROM daily_ohlc WHERE code='{code}' ORDER BY date DESC LIMIT 260"
    ).fetchdf()
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def compute_current_signal(df, realtime_price, info):
    """用历史数据+实时价计算当前信号"""
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    ma_p, adx_th, vol_th = info['ma_p'], info['adx_th'], info['vol_th']

    # 把实时价追加为最新close (盘中用)
    c = pd.concat([c, pd.Series([realtime_price])]).reset_index(drop=True)
    h = pd.concat([h, pd.Series([realtime_price])]).reset_index(drop=True)
    l = pd.concat([l, pd.Series([realtime_price])]).reset_index(drop=True)

    ma = c.rolling(ma_p).mean()
    vol = c.pct_change().rolling(20).std() * np.sqrt(252) * 100
    tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    up = h.diff(); dn = -l.diff()
    pdm = pd.Series(0.0, index=c.index); ndm = pd.Series(0.0, index=c.index)
    pdm.loc[(up > dn) & (up > 0)] = up
    ndm.loc[(dn > up) & (dn > 0)] = dn
    pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atr
    ndi = 100 * ndm.ewm(alpha=1/14, adjust=False).mean() / atr
    adx = (100 * abs(pdi - ndi) / (pdi + ndi + 1e-10)).ewm(alpha=1/14, adjust=False).mean()

    above = c.iloc[-1] > ma.iloc[-1]
    low_vol = vol.iloc[-1] < vol_th
    trend = adx.iloc[-1] > adx_th
    signal = 1 if (above and (low_vol or trend)) else 0

    return {
        'price': realtime_price,
        'ma': round(float(ma.iloc[-1]), 4),
        'vol': round(float(vol.iloc[-1]), 1),
        'adx': round(float(adx.iloc[-1]), 1),
        'above_ma': above,
        'low_vol': low_vol,
        'trend': trend,
        'signal': signal,
    }


def get_qvix():
    """获取QVIX恐慌指数"""
    try:
        import akshare as ak
        df = ak.index_option_300etf_qvix()
        latest = df.iloc[-1]
        return round(float(latest['close']), 2)
    except Exception:
        return None


def qvix_position_ratio(qvix):
    """
    QVIX 仓位调节规则:
    < 20: 平静, 100% 仓位
    20-25: 正常偏紧, 80% 仓位
    25-30: 恐慌, 60% 仓位
    >= 30: 极度恐慌, 40% 仓位
    """
    if qvix is None:
        return 1.0, '未知(按100%执行)'
    if qvix < 20:
        return 1.0, f'QVIX {qvix} 平静 <20'
    elif qvix < 25:
        return 0.8, f'QVIX {qvix} 正常偏紧 20-25'
    elif qvix < 30:
        return 0.6, f'QVIX {qvix} 恐慌 25-30'
    else:
        return 0.4, f'QVIX {qvix} 极度恐慌 >=30'


def compute_signal_confidence(r, info):
    """
    信号置信度评估 (0-100)
    强信号: 价格远离MA30 + ADX远超阈值 + 波动率安全
    弱信号: 贴线/刚过线
    """
    score = 0
    reasons = []

    # 1. 价格与MA30距离 (最多30分)
    dist = (r['price'] / r['ma'] - 1) * 100
    if dist > 3:
        score += 30
        reasons.append(f'价格超MA30 {dist:.1f}%')
    elif dist > 1.5:
        score += 22
        reasons.append(f'价格超MA30 {dist:.1f}%')
    elif dist > 0.5:
        score += 12
        reasons.append(f'价格贴MA30 ({dist:+.1f}%)')
    else:
        score += 4
        reasons.append(f'价格紧贴MA30 ({dist:+.1f}%)')

    # 2. ADX 余量 (最多40分)
    adx_margin = r['adx'] - info['adx_th']
    if adx_margin > 10:
        score += 40
        reasons.append(f'ADX超阈值{adx_margin:.0f}点')
    elif adx_margin > 5:
        score += 30
        reasons.append(f'ADX超阈值{adx_margin:.0f}点')
    elif adx_margin > 0:
        score += 18
        reasons.append(f'ADX刚过线({r["adx"]:.1f})')
    elif adx_margin > -5:
        score += 8
        reasons.append(f'ADX未过线但波动率触发')
    else:
        score += 2
        reasons.append(f'ADX远离阈值({r["adx"]:.1f})')

    # 3. 波动率余量 (最多30分)
    vol_margin = info['vol_th'] - r['vol']
    if vol_margin > 5:
        score += 30
        reasons.append(f'波动率低({r["vol"]:.1f}%)')
    elif vol_margin > 2:
        score += 22
        reasons.append(f'波动率安全({r["vol"]:.1f}%)')
    elif vol_margin > 0:
        score += 12
        reasons.append(f'波动率贴线({r["vol"]:.1f}%)')
    else:
        score += 4
        reasons.append(f'波动率超标({r["vol"]:.1f}%)')

    if score >= 75:
        level = '强'
    elif score >= 50:
        level = '中'
    else:
        level = '弱'

    return score, level, reasons


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'close'
    now = datetime.now()

    print(f"[{now.strftime('%H:%M:%S')}] 策略快照 ({'午间' if mode=='mid' else '收盘'})")
    quotes = get_realtime_quotes()
    if not quotes:
        print('实时行情获取失败')
        return

    results = {}
    for code, info in ASSETS.items():
        if code not in quotes:
            continue
        df = get_latest_ohlc(code)
        r = compute_current_signal(df, quotes[code]['price'], info)
        # 置信度评估
        conf_score, conf_level, conf_reasons = compute_signal_confidence(r, info)
        r['conf_score'] = conf_score
        r['conf_level'] = conf_level
        r['conf_reasons'] = conf_reasons
        results[code] = r

    # 输出
    print()
    print('=' * 62)
    print(f"  {now.strftime('%Y-%m-%d')} 三品种轮动快照 ({'盘中' if mode=='mid' else '收盘'})")
    print('=' * 62)
    print(f"  {'品种':<12} {'实时价':>8} {'MA30':>8} {'ADX':>6} {'波动':>6} {'信号':>6} {'置信':>5}")
    print('  ' + '-' * 70)
    for code in ['510310', '159995', '512800']:
        if code not in results:
            continue
        r = results[code]
        sig_txt = '持有' if r['signal'] else '空仓'
        conf_txt = r['conf_level'] if r['signal'] else '-'
        print(f"  {ASSETS[code]['name']:<10} {r['price']:>8.4f} {r['ma']:>8.4f} {r['adx']:>6.1f} {r['vol']:>5.1f}%  {sig_txt:>4}  {conf_txt:>3}")

    # 轮动指向
    candidates = [(code, results[code]['adx']) for code in results if results[code]['signal'] == 1]
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        pick = candidates[0][0]
        print(f"\n  >>> 轮动指向: {ASSETS[pick]['name']} ({pick}) <<<")
    else:
        pick = None
        print(f"\n  >>> 轮动指向: 国债/逆回购 (无品种符合) <<<")

    # QVIX 仓位调节
    qvix = get_qvix()
    ratio, qvix_note = qvix_position_ratio(qvix)
    print()
    print('  --- QVIX 仓位调节 ---')
    print(f"  QVIX: {qvix if qvix else 'N/A'}")
    print(f"  状态: {qvix_note}")
    if pick:
        base = '满仓' if ratio >= 1.0 else f'{int(ratio*100)}%仓位'
        print(f"  建议: 持有{ASSETS[pick]['name']} 按{int(ratio*100)}%仓位执行 (QVIX调节)")
    else:
        print(f"  建议: 空仓等信号 (无品种符合)")

    # ===== 下午操作建议 (mid 模式) =====
    if mode == 'mid':
        print()
        print('  ' + '=' * 58)
        print('  【下午操作建议】')
        print('  ' + '=' * 58)
        # 判断是否有持仓
        position_file = os.path.join(PROJECT_ROOT, 'trades')
        holding = {}
        for code in ASSETS:
            f = os.path.join(position_file, f'{code}_trades.csv')
            if os.path.exists(f):
                try:
                    df_pos = pd.read_csv(f)
                    shares = 0
                    for _, row in df_pos.iterrows():
                        a = str(row['action']).strip().upper()
                        sh = int(row['shares'])
                        shares += sh if a == 'BUY' else -sh
                    if shares > 0:
                        holding[code] = shares
                except Exception:
                    pass

        if pick:
            asset_name = ASSETS[pick]['name']
            conf = results[pick]['conf_level']
            conf_score = results[pick]['conf_score']
            if pick in holding:
                print(f'  [持有中] {asset_name}: 信号仍持有(置信{conf}), 下午继续持有, 无操作')
            else:
                if conf == '强':
                    print(f'  [建仓] {asset_name}: 强信号(置信{conf_score}分), 下午可果断建仓 {int(ratio*100)}%仓位')
                elif conf == '中':
                    print(f'  [分批建仓] {asset_name}: 中等信号(置信{conf_score}分), 下午建仓一半, 收盘确认后再加')
                else:
                    print(f'  [观察] {asset_name}: 弱信号(置信{conf_score}分), 贴线状态, 下午不追, 等收盘确认')
        else:
            if holding:
                for code, sh in holding.items():
                    print(f'  [减仓] {ASSETS[code]["name"]}: 午间无信号, 下午建议减仓, 转逆回购')
            else:
                print(f'  [观望] 无品种有信号, 下午继续持币/逆回购, 等收盘确认')

        # 置信度理由展示
        if pick and results[pick]['signal']:
            reasons = '; '.join(results[pick]['conf_reasons'])
            print(f'  (依据: {reasons})')

        print()
        print('  ⚠️ 午间建议基于上午收盘数据, 下午行情可能变化, 最终以收盘信号为准')

    print()
    print('  说明: 盘中快照仅供参考, 实盘操作以收盘信号为准')


if __name__ == '__main__':
    main()
