"""
沪深300 策略回测系统
1. 下载5年CSI300指数数据 → duckdb
2. 对三个策略进行真实数据回测
3. 输出回测报告
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import duckdb

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from strategies.csi300_strategies import (
    compute_volatility, compute_ma, compute_momentum, compute_adx
)

DB_PATH = os.path.join(PROJECT_ROOT, 'csi300_data.duckdb')
RISK_FREE_RATE = 0.025
YEARS = 5

# ============ 1. 下载数据并入库 ============
def download_and_store():
    print('=' * 60)
    print('步骤1: 下载 CSI300 指数数据')
    print('=' * 60)

    try:
        import akshare as ak
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=YEARS * 365)).strftime('%Y%m%d')

        print(f'  数据范围: {start_date} ~ {end_date}')
        df = ak.stock_zh_index_daily(symbol='sh000300')
        print(f'  原始数据: {len(df)} 条')
    except Exception as e:
        print(f'  akshare 下载失败: {e}')
        print('  尝试备用方案: baostock...')
        try:
            import baostock as bs
            bs.login()
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=YEARS * 365)).strftime('%Y-%m-%d')
            rs = bs.query_history_k_data_plus(
                'sh.000300',
                'date,open,high,low,close,volume',
                start_date=start_date, end_date=end_date,
                frequency='d', adjustflag='3'
            )
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            df = pd.DataFrame(data_list, columns=rs.fields)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            bs.logout()
            print(f'  baostock 数据: {len(df)} 条')
        except Exception as e2:
            print(f'  baostock 也失败: {e2}')
            return None

    df = df.rename(columns={'date': 'date'})
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # 过滤最近5年
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=YEARS)
    if 'date' in df.columns:
        df = df[df['date'] >= cutoff].copy()

    # 确保有 OHLC 列，列名统一
    need_cols = ['date', 'open', 'high', 'low', 'close']
    for c in need_cols:
        if c not in df.columns:
            print(f'  缺失列: {c}')
            return None

    df = df[need_cols].dropna().sort_values('date').reset_index(drop=True)
    print(f'  有效数据: {len(df)} 条')
    print(f'  日期范围: {df["date"].min().date()} ~ {df["date"].max().date()}')
    print(f'  最新收盘: {df["close"].iloc[-1]:.2f}')

    # 存入 duckdb
    conn = duckdb.connect(DB_PATH)
    conn.execute('DROP TABLE IF EXISTS csi300_daily')
    conn.execute('CREATE TABLE csi300_daily (date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE)')
    conn.execute('INSERT INTO csi300_daily SELECT * FROM df')
    conn.close()
    print(f'  已写入 duckdb: {DB_PATH}')
    return df


# ============ 2. 回测引擎 ============
def compute_indicators(df):
    """计算所有指标"""
    close = df['close']
    ma50 = close.rolling(50).mean()
    vol = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    momentum = close / close.shift(20) - 1

    # ADX (Wilder 标准)
    high = df['high']
    low = df['low']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_dm.loc[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm.loc[(down_move > up_move) & (down_move > 0)] = down_move
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()

    return close, ma50, vol, momentum, adx


def run_backtest(df, strategy_name):
    """运行单个策略回测"""
    close, ma50, vol, momentum, adx = compute_indicators(df)

    # 计算信号 (偏移一天，避免前视偏差)
    above_ma = close.shift(1) > ma50.shift(1)
    low_vol = vol.shift(1) < 15

    if strategy_name == 'adx_override':
        strong_trend = adx.shift(1) > 25
        signal = (above_ma & (low_vol | strong_trend)).astype(int)
    elif strategy_name == 'momentum_override':
        strong_momentum = momentum.shift(1) > 0.10
        signal = (above_ma & (low_vol | strong_momentum)).astype(int)
    elif strategy_name == 'absolute_15':
        signal = (above_ma & low_vol).astype(int)
    else:
        raise ValueError(f'Unknown strategy: {strategy_name}')

    # 日收益率
    daily_ret = close.pct_change()
    bond_daily = (1 + RISK_FREE_RATE) ** (1/252) - 1

    # 策略日收益 = 信号*指数收益 + (1-信号)*债券收益
    strat_ret = signal * daily_ret + (1 - signal) * bond_daily

    # 从有足够数据开始
    start_idx = 60  # 跳过前60天(MA50+ADX需要)
    strat_ret = strat_ret.iloc[start_idx:]
    daily_ret = daily_ret.iloc[start_idx:]
    signal = signal.iloc[start_idx:]
    dates = df['date'].iloc[start_idx:]

    # 累计收益
    cum_ret = (1 + strat_ret).cumprod()
    cum_bh = (1 + daily_ret).cumprod()

    # 统计指标
    total_ret = cum_ret.iloc[-1] - 1
    bh_total_ret = cum_bh.iloc[-1] - 1
    n_days = len(strat_ret)
    n_years = n_days / 252

    ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    bh_ann_ret = (1 + bh_total_ret) ** (1 / n_years) - 1

    excess_ret = strat_ret - bond_daily
    ann_vol = strat_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    # 最大回撤
    peak = cum_ret.expanding().max()
    drawdown = (cum_ret / peak - 1)
    max_dd = drawdown.min()

    # 胜率
    win_rate = (strat_ret > bond_daily).sum() / n_days

    # 持仓天数
    hold_days = signal.sum()
    hold_pct = hold_days / len(signal) * 100

    # 交易次数
    signal_change = signal.diff().abs()
    trades = signal_change.sum()

    return {
        'strategy': strategy_name,
        'total_return': total_ret,
        'annual_return': ann_ret,
        'annual_vol': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'hold_days': hold_days,
        'hold_pct': hold_pct,
        'trades': trades,
        'start_date': dates.iloc[0],
        'end_date': dates.iloc[-1],
        'n_days': n_days,
        'n_years': n_years,
        'bh_total_return': bh_total_ret,
        'bh_annual_return': bh_ann_ret,
        'cum_ret': cum_ret,
        'cum_bh': cum_bh,
        'signal': signal,
        'daily_ret': strat_ret,
        'dates': dates,
    }


def print_result(r, title=None):
    if title:
        print(f'\n--- {title} ---')
    print(f'  回测区间: {r["start_date"].date()} ~ {r["end_date"].date()} ({r["n_years"]:.1f}年, {r["n_days"]}天)')
    print(f'  总收益:    {r["total_return"]:+.2%}')
    print(f'  年化收益:  {r["annual_return"]:+.2%}')
    print(f'  年化波动:  {r["annual_vol"]:.2%}')
    print(f'  Sharpe:    {r["sharpe"]:.2f}')
    print(f'  最大回撤:  {r["max_drawdown"]:.2%}')
    print(f'  胜率:      {r["win_rate"]:.1%}')
    print(f'  持仓比例:  {r["hold_pct"]:.1f}%')
    print(f'  交易次数:  {r["trades"]:.0f}')
    print(f'  买入持有年化: {r["bh_annual_return"]:+.2%}')
    print(f'  超额收益:  {r["annual_return"] - r["bh_annual_return"]:+.2%}')


# ============ 3. 主流程 ============
def main():
    # 1. 下载并入库
    df = download_and_store()
    if df is None:
        print('数据获取失败，退出')
        return

    # 2. 回测
    print()
    print('=' * 60)
    print('步骤2: 策略回测')
    print('=' * 60)

    strategies = {
        'adx_override': '+ADX>25 Override',
        'momentum_override': '+Momentum>10% Override',
        'absolute_15': 'Base Absolute 15%',
    }

    results = {}
    for key, name in strategies.items():
        print(f'\n正在回测: {name}...')
        results[key] = run_backtest(df, key)
        print_result(results[key], name)

    # 3. 对比汇总
    print()
    print('=' * 60)
    print('步骤3: 策略对比汇总')
    print('=' * 60)
    print(f'  无风险利率: {RISK_FREE_RATE:.1%} (国债近似)')
    print()

    header = f'{"策略":<28} {"年化收益":>8} {"Sharpe":>7} {"最大回撤":>8} {"年化波动":>8} {"交易次数":>6} {"持仓%":>6}'
    print(header)
    print('-' * len(header))

    for key, name in strategies.items():
        r = results[key]
        print(f'{name:<24} {r["annual_return"]:>+7.2%}  {r["sharpe"]:>6.2f}  {r["max_drawdown"]:>7.2%}  {r["annual_vol"]:>7.2%}  {r["trades"]:>6.0f}  {r["hold_pct"]:>5.1f}%')

    r0 = list(results.values())[0]
    print(f'{"Buy & Hold (基准)":<24} {r0["bh_annual_return"]:>+7.2%}  {"--":>6}  {((r0["cum_bh"] / r0["cum_bh"].expanding().max() - 1).min()):>7.2%}  {r0["daily_ret"].std() * np.sqrt(252):>7.2%}')

    # 4. 保存回测报告
    report_path = os.path.join(PROJECT_ROOT, 'reports', 'backtest_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('CSI300 策略回测报告\n')
        f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
        f.write(f'无风险利率: {RISK_FREE_RATE:.1%}\n')
        f.write('=' * 60 + '\n\n')
        for key, name in strategies.items():
            r = results[key]
            f.write(f'【{name}】\n')
            f.write(f'  总收益: {r["total_return"]:+.2%}\n')
            f.write(f'  年化:   {r["annual_return"]:+.2%}\n')
            f.write(f'  Sharpe: {r["sharpe"]:.2f}\n')
            f.write(f'  回撤:   {r["max_drawdown"]:.2%}\n')
            f.write(f'  胜率:   {r["win_rate"]:.1%}\n')
            f.write(f'  交易:   {r["trades"]:.0f}次\n\n')

    print(f'\n回测报告已保存: {report_path}')

    return results


if __name__ == '__main__':
    main()
