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
sys.path.insert(0, PROJECT_ROOT)

# ===== 全局网络超时保护 =====
import socket
socket.setdefaulttimeout(15)

from config import ASSETS, ORDER, HISTORY_DB
from signal_core import (last_signal_state, compute_confidence, qvix_position_ratio,
                         fetch_qvix, VOL_BRAKE_THRESHOLD, vol_brake_weight,
                         holdings_from_trades)


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
    """用历史数据+实时价计算当前信号 (委托给 signal_core, 保证与其他脚本一致)
    仅当库内尚无当日K线时才把实时价追加为合成K线;
    若 auto_refresh 已入库今日数据, 再追加会产生重复K线使指标失真
    (与 send_advice.compute_signals 的 append 门控保持同一原则)"""
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)

    today = pd.Timestamp.now().normalize()
    if not df['date'].empty and df['date'].iloc[-1] < today:
        c = pd.concat([c, pd.Series([realtime_price])]).reset_index(drop=True)
        h = pd.concat([h, pd.Series([realtime_price])]).reset_index(drop=True)
        l = pd.concat([l, pd.Series([realtime_price])]).reset_index(drop=True)

    return last_signal_state(c, h, l, info['ma_p'], info['adx_th'], info['vol_th'])


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
        r['brake_w'] = vol_brake_weight(r['vol'])
        # 置信度评估
        conf_score, conf_level, conf_reasons = compute_confidence(
            r['price'], r['ma'], r['adx'], r['vol'], info
        )
        r['conf_score'] = conf_score
        r['conf_level'] = conf_level
        r['conf_reasons'] = conf_reasons
        results[code] = r

    # 输出
    print()
    print('=' * 62)
    print(f"  {now.strftime('%Y-%m-%d')} 三品种轮动快照 ({'盘中' if mode=='mid' else '收盘'})")
    print('=' * 62)
    print(f"  {'品种':<12} {'实时价':>8} {'MA30':>8} {'ADX':>6} {'波动':>6} {'信号':>6} {'确认':>6} {'置信':>5}")
    print('  ' + '-' * 70)
    for code in ORDER:
        if code not in results:
            continue
        r = results[code]
        sig_txt = '持有' if r['signal'] else '空仓'
        conf_flag = ('已确认' if r.get('confirmed') == 1 else '待确认') if r['signal'] else '-'
        conf_txt = r['conf_level'] if r['signal'] else '-'
        vol_txt = f"{r['vol']:>5.1f}%" + ('刹车' if r.get('brake_w', 1.0) < 1 else '')
        print(f"  {ASSETS[code]['name']:<10} {r['price']:>8.4f} {r['ma']:>8.4f} {r['adx']:>6.1f} {vol_txt}  {sig_txt:>4}  {conf_flag:>4}  {conf_txt:>3}")

    # 轮动指向 (新入场需连续2日确认)
    candidates = [(code, results[code]['adx']) for code in results
                  if results[code]['signal'] == 1 and results[code].get('confirmed') == 1]
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        pick = candidates[0][0]
        print(f"\n  >>> 轮动指向: {ASSETS[pick]['name']} ({pick}) <<<")
    else:
        pick = None
        first_day = [c for c, r in results.items() if r['signal'] == 1 and r.get('confirmed') == 0]
        if first_day:
            names = '、'.join(ASSETS[c]['name'] for c in first_day)
            print(f"\n  >>> {names} 信号首日待确认(明日仍持有信号才可买入), 当前指向: 国债/逆回购 <<<")
        else:
            print(f"\n  >>> 轮动指向: 国债/逆回购 (无品种符合) <<<")

    # QVIX 仓位调节
    qvix = fetch_qvix()
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
        # 判断是否有持仓 (统一委托 signal_core, 与 send_advice 同源)
        holdings_all = holdings_from_trades(os.path.join(PROJECT_ROOT, 'trades'), list(ASSETS))
        holding = {c: s for c, s in holdings_all.items() if s > 0}
        for c, s in holdings_all.items():
            if s < 0:
                print(f'  [WARN] {c} 持仓计算为负({s}), 请检查 {c}_trades.csv')

        if pick:
            asset_name = ASSETS[pick]['name']
            conf = results[pick]['conf_level']
            conf_score = results[pick]['conf_score']
            bw = results[pick].get('brake_w', 1.0)
            if pick in holding:
                if bw < 1.0:
                    print(f'  [减仓] {asset_name}: 波动率{results[pick]["vol"]:.1f}%>{VOL_BRAKE_THRESHOLD}%, '
                          f'建议减至{bw:.0%}仓位 (波动刹车)')
                else:
                    print(f'  [持有中] {asset_name}: 信号仍持有(置信{conf}), 下午继续持有, 无操作')
            elif results[pick].get('confirmed') == 0:
                print(f'  [观察] {asset_name}: 信号首日, 需连续2日确认, 下午不追; 明日若信号仍在则可建仓')
            else:
                pct = int(ratio * bw * 100)
                if conf == '强':
                    print(f'  [建仓] {asset_name}: 强信号(置信{conf_score}分), 下午可果断建仓 {pct}%仓位'
                          + ('' if bw >= 1 else f' (含波动刹车{bw:.0%})'))
                elif conf == '中':
                    print(f'  [分批建仓] {asset_name}: 中等信号(置信{conf_score}分), 下午建仓{pct // 2}%仓位, 收盘确认后再加')
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
