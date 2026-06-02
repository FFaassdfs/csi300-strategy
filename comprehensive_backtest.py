"""
综合策略回测系统
测试 20+ 单策略 + 交叉组合，寻找最优辅助策略
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, 'csi300_data.duckdb')
RF = 0.025

def load_data():
    conn = duckdb.connect(DB_PATH)
    df = conn.execute('SELECT date, open, high, low, close FROM csi300_daily ORDER BY date').fetchdf()
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df


def compute_all_indicators(df):
    """一次性计算所有技术指标"""
    close = df['close']
    high = df['high']
    low = df['low']
    idx = {}

    # --- 均线 ---
    for p in [5, 10, 20, 50, 100, 200]:
        idx[f'ma{p}'] = close.rolling(p).mean()

    # --- 波动率 ---
    idx['vol_20'] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

    # --- ADX ---
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
    idx['adx'] = (100 * abs(pdi - ndi) / (pdi + ndi + 1e-10)).ewm(alpha=1/14, adjust=False).mean()

    # --- ATR ---
    idx['atr14'] = atr14
    idx['atr20'] = tr.ewm(alpha=1/20, adjust=False).mean()

    # --- RSI ---
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    idx['rsi'] = 100 - 100 / (1 + gain / (loss + 1e-10))

    # --- MACD ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    idx['macd'] = ema12 - ema26
    idx['macd_signal'] = idx['macd'].ewm(span=9, adjust=False).mean()
    idx['macd_hist'] = idx['macd'] - idx['macd_signal']

    # --- 布林带 ---
    idx['bb_mid'] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    idx['bb_upper'] = idx['bb_mid'] + 2 * bb_std
    idx['bb_lower'] = idx['bb_mid'] - 2 * bb_std
    idx['bb_pct_b'] = (close - idx['bb_lower']) / (idx['bb_upper'] - idx['bb_lower'] + 1e-10)

    # --- 唐奇安通道 ---
    idx['dc_high20'] = high.rolling(20).max()
    idx['dc_low20'] = low.rolling(20).min()
    idx['dc_mid20'] = (idx['dc_high20'] + idx['dc_low20']) / 2

    # --- 动量 ---
    for p in [5, 10, 20, 60]:
        idx[f'mom{p}'] = close / close.shift(p) - 1

    # --- 成交量 (如有) ---
    if 'volume' in df.columns:
        idx['vol_ma20'] = df['volume'].rolling(20).mean()
        idx['vol_ratio'] = df['volume'] / idx['vol_ma20']

    # --- Keltner Channel ---
    idx['kc_mid'] = close.ewm(span=20, adjust=False).mean()
    idx['kc_upper'] = idx['kc_mid'] + 2 * atr14
    idx['kc_lower'] = idx['kc_mid'] - 2 * atr14

    return close, idx


# ============================================================
# 策略工厂: 每个 strategy 是一个函数，返回 (signal_series, name, description)
# ============================================================

def make_strategies(close, idx):
    """生成所有待测策略"""
    strategies = []

    # ── 基准 ──
    def baseline():
        s = pd.Series(1, index=close.index)
        return s, 'Buy & Hold', '一直持有沪深300'
    strategies.append(baseline)

    # ── 原ADX Override ──
    def adx_override():
        above = close > idx['ma50']
        low_vol = idx['vol_20'] < 15
        trend = idx['adx'] > 25
        s = (above & (low_vol | trend)).astype(int)
        return s, 'ADX Override (原版基准)', '价格>MA50 & (vol<15% | ADX>25)'
    strategies.append(adx_override)

    # ── 双均线交叉 ──
    macross_params = [(5, 20), (10, 50), (20, 50), (20, 100), (50, 200)]
    for fast, slow in macross_params:
        def make_macross(f=fast, s=slow):
            sig = (idx[f'ma{f}'] > idx[f'ma{s}']).astype(int)
            return sig, f'MA{f}>{s}交叉', f'快线MA{f}上穿慢线MA{s}买入'
        strategies.append(make_macross)

    # ── MACD ──
    def make_macd():
        s = (idx['macd'] > idx['macd_signal']).astype(int)
        return s, 'MACD 金叉死叉', 'MACD>Signal买入'
    strategies.append(make_macd)

    def make_macd_zero():
        s = (close > idx['ma50']) & (idx['macd_hist'] > 0)
        s = s.astype(int)
        return s, 'MACD柱+MA50过滤', '价格>MA50 & MACD柱>0'
    strategies.append(make_macd_zero)

    # ── RSI超卖买入 ──
    def make_rsi_oversold():
        in_pos = pd.Series(0, index=close.index)
        holding = False
        for i in range(1, len(close)):
            if holding:
                if idx['rsi'].iloc[i] > 65:
                    holding = False
                else:
                    in_pos.iloc[i] = 1
            else:
                if idx['rsi'].iloc[i] < 40 and close.iloc[i] > idx['ma50'].iloc[i]:
                    holding = True
                    in_pos.iloc[i] = 1
        return in_pos, 'RSI回调买入+MA50', 'RSI<40且价格>MA50买入, RSI>65卖出'
    strategies.append(make_rsi_oversold)

    def make_rsi_filter():
        s = (close > idx['ma50']) & (idx['rsi'] > 40)
        return s.astype(int), 'RSI>40+MA50过滤', '价格>MA50 & RSI>40(非弱势)'
    strategies.append(make_rsi_filter)

    # ── 布林带 ──
    def make_bb_mean_rev():
        s = (idx['bb_pct_b'] < 0.2) & (close > idx['ma50'])
        return s.astype(int), '布林下轨+MA50', '%B<0.2超卖且趋势向上时买入'
    strategies.append(make_bb_mean_rev)

    def make_bb_trend():
        s = (close > idx['bb_mid']) & (idx['bb_pct_b'] < 0.8)
        return s.astype(int), '布林中轨趋势', '价格>BB中轨且不过热'
    strategies.append(make_bb_trend)

    # ── 唐奇安通道 ──
    def make_dc_breakout():
        s = (close > idx['dc_high20'].shift(1)) & (close > idx['ma50'])
        return s.astype(int), '唐奇安突破+MA50', '突破20日高点且趋势向上'
    strategies.append(make_dc_breakout)

    # ── 动量 ──
    def make_mom_ma50():
        s = (close > idx['ma50']) & (idx['mom20'] > 0)
        return s.astype(int), '动量>0+MA50', '价格>MA50且20日动量为正'
    strategies.append(make_mom_ma50)

    def make_mom_accel():
        s = (close > idx['ma50']) & (idx['mom5'] > 0) & (idx['mom20'] > -0.05)
        return s.astype(int), '短期动量加速', '价格>MA50 & 5日动量为正 & 20日未大跌'
    strategies.append(make_mom_accel)

    # ── 海龟/ATR通道 ──
    def make_atr_channel():
        prev_high = idx['dc_high20'].shift(1)
        s = (close > prev_high) & (close > idx['ma50'])
        return s.astype(int), '海龟突破', '突破20日高点(昨)且>MA50'
    strategies.append(make_atr_channel)

    # ── Keltner 通道 ──
    def make_kc():
        s = (close > idx['kc_upper'].shift(1))
        return s.astype(int), 'Keltner突破', '价格突破KC上轨'
    strategies.append(make_kc)

    # ── 组合策略: ADX框架 + 附加过滤器 ──
    # ADX基础信号
    base_adx = ((close > idx['ma50']) & ((idx['vol_20'] < 15) | (idx['adx'] > 25))).astype(int)

    def make_adx_rsi():
        s = base_adx & (idx['rsi'] > 40)
        return s.astype(int), 'ADX+RSI>40', 'ADX信号 & RSI不弱势'
    strategies.append(make_adx_rsi)

    def make_adx_bb():
        s = base_adx & (close > idx['bb_mid'])
        return s.astype(int), 'ADX+BB中轨上', 'ADX信号 & 价格>BB中轨'
    strategies.append(make_adx_bb)

    def make_adx_macd():
        s = base_adx & (idx['macd_hist'] > 0)
        return s.astype(int), 'ADX+MACD柱>0', 'ADX信号 & MACD柱为正'
    strategies.append(make_adx_macd)

    def make_adx_mom():
        s = base_adx & (idx['mom20'] > 0)
        return s.astype(int), 'ADX+动量>0', 'ADX信号 & 20日动量>0'
    strategies.append(make_adx_mom)

    def make_adx_strict():
        s = base_adx & (idx['mom20'] > 0) & (idx['rsi'] > 45)
        return s.astype(int), 'ADX+动量>0+RSI>45', 'ADX信号+动量+RSI三重确认'
    strategies.append(make_adx_strict)

    # ── 不同离场策略 ──
    # 用ATR追踪止损替换MA50离场
    def make_adx_atr_exit():
        s = pd.Series(0, index=close.index)
        in_pos = False
        stop = 0
        for i in range(300, len(close)):
            if in_pos:
                stop = max(stop, close.iloc[i] - 3 * idx['atr20'].iloc[i])
                if close.iloc[i] < stop:
                    in_pos = False
                elif idx['vol_20'].iloc[i] >= 15 and idx['adx'].iloc[i] <= 25:
                    in_pos = False
                else:
                    s.iloc[i] = 1
            else:
                if close.iloc[i] > idx['ma50'].iloc[i] and (idx['vol_20'].iloc[i] < 15 or idx['adx'].iloc[i] > 25):
                    in_pos = True
                    stop = close.iloc[i] - 3 * idx['atr20'].iloc[i]
                    s.iloc[i] = 1
        return s, 'ADX+ATR追踪止损', 'ADX Override入场, 3xATR追踪止损离场'
    strategies.append(make_adx_atr_exit)

    return strategies


# ============================================================
# 回测引擎
# ============================================================
def backtest(daily_ret, signal_raw, warmup=300):
    bond_daily = (1 + RF) ** (1/252) - 1
    signal = signal_raw.shift(1).fillna(0).astype(float).iloc[warmup:]
    daily_ret = daily_ret.iloc[warmup:]

    ret = signal * daily_ret + (1 - signal) * bond_daily
    cum = (1 + ret).cumprod()

    total = cum.iloc[-1] - 1
    ny = len(ret) / 252
    ann = (1 + total) ** (1 / ny) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ann - RF) / vol if vol > 0 else -99

    peak = cum.expanding().max()
    dd = (cum / peak - 1).min()

    hold_pct = signal.sum() / len(signal) * 100
    trades = signal.diff().abs().sum()

    return ann, sharpe, dd, vol, hold_pct, trades, total


# ============================================================
# 主流程
# ============================================================
if __name__ == '__main__':
    df = load_data()
    close, idx = compute_all_indicators(df)
    print(f'数据: {len(df)}条, {df["date"].min().date()} ~ {df["date"].max().date()}')
    warmup = 300

    strategies = make_strategies(close, idx)
    results = []

    print(f'\n测试 {len(strategies)} 个策略...')
    for i, factory in enumerate(strategies):
        try:
            signal, name, desc = factory()
            ann, sharpe, dd, vol, hold, trades, total = backtest(close.pct_change(), signal, warmup)
            results.append({
                'name': name, 'desc': desc,
                'ann': ann, 'sharpe': sharpe, 'dd': dd,
                'vol': vol, 'hold': hold, 'trades': trades, 'total': total,
            })
        except Exception as e:
            pass

    results.sort(key=lambda x: x['sharpe'], reverse=True)

    print()
    print('=' * 110)
    print(f'{"#":>3}  {"策略":<38} {"年化":>7} {"Sharpe":>7} {"回撤":>7} {"波动":>7} {"持仓%":>6} {"交易":>5}  描述')
    print('=' * 110)

    bh_ret = close.pct_change().iloc[warmup:]
    bh_total = (1 + bh_ret).cumprod().iloc[-1] - 1
    bh_ann = (1 + bh_total) ** (1 / (len(bh_ret)/252)) - 1
    bh_vol = bh_ret.std() * np.sqrt(252)
    bh_peak = (1 + bh_ret).cumprod().expanding().max()
    bh_dd = ((1 + bh_ret).cumprod() / bh_peak - 1).min()
    print(f'{"--":>3}  {"Buy & Hold":<38} {bh_ann:>+6.2%}  {"--":>6}  {bh_dd:>6.2%}  {bh_vol:>6.2%}')
    print('-' * 110)

    for rank, r in enumerate(results, 1):
        marker = ' ★' if rank <= 5 else ''
        print(f'{rank:>3}{marker}  {r["name"]:<38} {r["ann"]:>+6.2%}  {r["sharpe"]:>6.2f}  {r["dd"]:>6.2%}  {r["vol"]:>6.2%}  {r["hold"]:>5.1f}%  {r["trades"]:>5.0f}  {r["desc"]}')

    print('=' * 110)
    print(f'\n回测区间: {df["date"].iloc[warmup].date()} ~ {df["date"].iloc[-1].date()}  (约{(len(close)-warmup)/252:.1f}年)')
    print(f'无风险利率: {RF:.1%}')
    print()
    print('── 前5名策略详情 ──')
    for r in results[:5]:
        print(f'  {r["name"]}:  年化{r["ann"]:+.2%}  Sharpe{r["sharpe"]:.2f}  回撤{r["dd"]:.2%}  交易{r["trades"]:.0f}次')
        print(f'         {r["desc"]}')
