"""
信号计算唯一实现 — dual_rotation / auto_refresh / send_advice / intraday_signal 共用
任何指标/阈值/评分逻辑修改只改这里, 保证各脚本信号完全一致
"""
import numpy as np
import pandas as pd

# 入场确认天数: 首日出信号只观察, 连续第2日才可买入 (出场不受影响)
# 2026-09-07 Tier-1验证采纳: 轮动回测 年化+18.8%→+20.4% / 回撤-29.3%→-21.6% / 交易261→215笔
CONFIRM_DAYS = 2

# 极端波动刹车 (2026-09-07 Tier-2验证采纳):
# 持仓品种20日年化波动率 > 阈值 时仓位上限降至上限值, 回落到阈值以下恢复满仓
# 验证(6.4年): 年化+20.4%→+16.0% / 回撤-21.6%→-16.8%, 仅触发13次
# 2026年以来: 回撤-21.6%→-13.7% / Sharpe 1.23→1.36
VOL_BRAKE_THRESHOLD = 35.0
VOL_BRAKE_CAP = 0.6


def vol_brake_weight(vol):
    """极端波动刹车权重. vol: 20日年化波动率(百分数, 如14.9=14.9%). 返回仓位权重0~1"""
    if vol is None or not np.isfinite(vol) or vol <= 0:
        return 1.0
    return VOL_BRAKE_CAP if vol > VOL_BRAKE_THRESHOLD else 1.0


def adjust_splits(df, min_ratio=1.8, log=None):
    """拆分 / 份额合并 双向复权 (所有取数脚本统一调用, 2026-09-17)
    df: 含 date/open/high/low/close, 按日期升序 (原地修改)
    正向拆分 (价格下跌, c[i-1]/c[i] > min_ratio): 此前价格 / ratio
    反向合并 (价格跳升, c[i]/c[i-1] > min_ratio): 此前价格 * ratio
      — 新浪对部分品种(如510310 于2024-09-23份额合并)不做复权, 会留下 +100% 假跳变
    返回事件列表 [(日期, 比例, 类型)]
    """
    i = 1
    events = []
    while i < len(df):
        c = df['close'].values
        if c[i] > 0 and c[i - 1] > 0:
            r_down, r_up = c[i - 1] / c[i], c[i] / c[i - 1]
            if r_down > min_ratio:
                ratio = round(r_down)
                for col in ['open', 'high', 'low', 'close']:
                    df.loc[df.index[:i], col] = df.loc[df.index[:i], col] / ratio
                events.append((str(pd.Timestamp(df['date'].iloc[i]).date()), f'1:{ratio}', '拆分'))
            elif r_up > min_ratio:
                ratio = round(r_up)
                for col in ['open', 'high', 'low', 'close']:
                    df.loc[df.index[:i], col] = df.loc[df.index[:i], col] * ratio
                events.append((str(pd.Timestamp(df['date'].iloc[i]).date()), f'{ratio}:1', '份额合并'))
        i += 1
    if log is not None:
        for d, r, t in events:
            log(f'  [{t}] {d} {r}')
    return events


def compute_signal_core(c, h, l, ma_p=30, adx_th=20, vol_th=18):
    """
    ADX Override 核心计算
    c/h/l: close/high/low 序列 (可含盘中追加的合成K线)
    返回: {'ma','vol','adx','signal','confirmed'} 均为与输入等长的 Series
          signal = 原始信号 (决定持仓/出场)
          confirmed = 连续CONFIRM_DAYS日为1 (决定新入场)
    """
    ma = c.rolling(ma_p).mean()
    vol = c.pct_change().rolling(20).std() * np.sqrt(252) * 100

    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    up = h.diff(); dn = -l.diff()
    pdm = pd.Series(0.0, index=c.index); ndm = pd.Series(0.0, index=c.index)
    pdm.loc[(up > dn) & (up > 0)] = up
    ndm.loc[(dn > up) & (dn > 0)] = dn
    pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atr
    ndi = 100 * ndm.ewm(alpha=1/14, adjust=False).mean() / atr
    adx = (100 * abs(pdi - ndi) / (pdi + ndi + 1e-10)).ewm(alpha=1/14, adjust=False).mean()

    signal = ((c > ma) & ((vol < vol_th) | (adx > adx_th))).astype(int)
    confirmed = (signal.rolling(CONFIRM_DAYS).min() == 1).astype(int)
    return {'ma': ma, 'vol': vol, 'adx': adx, 'signal': signal, 'confirmed': confirmed}


def last_signal_state(c, h, l, ma_p=30, adx_th=20, vol_th=18):
    """最后一根K线的信号状态 (T日收盘信号, T+1执行)"""
    r = compute_signal_core(c, h, l, ma_p, adx_th, vol_th)

    def f(s):
        v = s.iloc[-1]
        return 0.0 if pd.isna(v) else float(v)

    return {
        'price': float(c.iloc[-1]),
        'ma': f(r['ma']),
        'vol': f(r['vol']),
        'adx': f(r['adx']),
        'above_ma': bool(c.iloc[-1] > r['ma'].iloc[-1]),
        'low_vol': bool(r['vol'].iloc[-1] < vol_th),
        'trend': bool(r['adx'].iloc[-1] > adx_th),
        'signal': int(r['signal'].iloc[-1]),
        'confirmed': int(r['confirmed'].iloc[-1]),
    }


def compute_confidence(price, ma, adx, vol, info):
    """
    信号置信度 (0-100): 价格距MA(30分) + ADX余量(40分) + 波动率余量(30分)
    返回 (score, level, reasons); level: 强>=75 / 中>=50 / 弱<50
    """
    score = 0
    reasons = []

    dist = (price / ma - 1) * 100 if ma > 0 else 0
    if dist > 3:
        score += 30; reasons.append(f'价格超MA{info["ma_p"]} {dist:.1f}%')
    elif dist > 1.5:
        score += 22; reasons.append(f'价格超MA{info["ma_p"]} {dist:.1f}%')
    elif dist > 0.5:
        score += 12; reasons.append(f'价格贴MA{info["ma_p"]} ({dist:+.1f}%)')
    else:
        score += 4; reasons.append(f'价格紧贴MA{info["ma_p"]} ({dist:+.1f}%)')

    adx_margin = adx - info['adx_th']
    if adx_margin > 10:
        score += 40; reasons.append(f'ADX超阈值{adx_margin:.0f}点')
    elif adx_margin > 5:
        score += 30; reasons.append(f'ADX超阈值{adx_margin:.0f}点')
    elif adx_margin > 0:
        score += 18; reasons.append(f'ADX刚过线({adx:.1f})')
    elif adx_margin > -5:
        score += 8; reasons.append('ADX未过线但波动率触发')
    else:
        score += 2; reasons.append(f'ADX远离阈值({adx:.1f})')

    vol_margin = info['vol_th'] - vol
    if vol_margin > 5:
        score += 30; reasons.append(f'波动率低({vol:.1f}%)')
    elif vol_margin > 2:
        score += 22; reasons.append(f'波动率安全({vol:.1f}%)')
    elif vol_margin > 0:
        score += 12; reasons.append(f'波动率贴线({vol:.1f}%)')
    else:
        score += 4; reasons.append(f'波动率超标({vol:.1f}%)')

    if score >= 75:
        level = '强'
    elif score >= 50:
        level = '中'
    else:
        level = '弱'
    return score, level, reasons


def qvix_position_ratio(qvix):
    """
    QVIX 仓位调节: <20→100% / 20-25→80% / 25-30→60% / >=30→40%
    返回 (ratio, 说明)
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


def fetch_qvix():
    """获取QVIX恐慌指数 (失败返回None, 不阻塞)"""
    try:
        import akshare as ak
        df = ak.index_option_300etf_qvix()
        if df is not None and len(df) > 0:
            return round(float(df.iloc[-1]['close']), 2)
    except Exception:
        pass
    return None


def position_shares_from_csv(path):
    """当前持仓份额 (从单个 trades CSV 累计; send_advice/intraday_signal 统一使用此实现)
    action 语义: BUY=+shares, SELL=-shares, SPLIT/ADJUST=份额增量(+shares)
    未知 action 跳过并告警, 不误按卖出处理"""
    import os
    if not os.path.exists(path):
        return 0
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f'[WARN] 持仓计算失败 {path}: {e}')
        return 0
    shares = 0
    for _, row in df.iterrows():
        a = str(row.get('action', '')).strip().upper()
        sh = row.get('shares')
        if pd.isna(sh):
            continue
        sh = int(float(sh))
        if a == 'BUY':
            shares += sh
        elif a == 'SELL':
            shares -= sh
        elif a in ('SPLIT', 'ADJUST'):
            shares += sh
        else:
            print(f'[WARN] {os.path.basename(path)} 未知action: {a} (跳过)')
    return shares


def holdings_from_trades(trades_dir, codes):
    """多品种当前持仓 {code: shares} (shares 可为 0/负数, 由调用方过滤/告警)"""
    import os
    return {code: position_shares_from_csv(os.path.join(trades_dir, f'{code}_trades.csv'))
            for code in codes}
