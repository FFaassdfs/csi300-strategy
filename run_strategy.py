"""
沪深300 ETF 策略执行器
每天收盘后运行，生成交易信号
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from strategies.csi300_strategies import (
    STRATEGIES, get_strategy, generate_trade_signal
)


def get_latest_data(days=300):
    """获取最新数据"""
    try:
        import akshare as ak
        df_310 = ak.fund_etf_hist_sina(symbol='sh510310')
        df_310['date'] = pd.to_datetime(df_310['date'])
        df_310 = df_310.sort_values('date').reset_index(drop=True)

        df_bond = ak.fund_etf_hist_sina(symbol='sh511260')
        df_bond['date'] = pd.to_datetime(df_bond['date'])
        df_bond = df_bond.sort_values('date').reset_index(drop=True)

        df = df_310.merge(df_bond, on='date', suffixes=('_etf', '_bond'))
        df = df.sort_values('date').reset_index(drop=True)

        return df
    except Exception as e:
        print(f'Failed to fetch data: {e}')
        return None


def run_strategy_report(df, output_path=None):
    """
    运行策略并生成报告

    Args:
        df: 包含历史数据的DataFrame
        output_path: 报告输出路径

    Returns:
        dict: 各策略的交易信号
    """
    results = {}

    print('=' * 70)
    print(f'CSI300 ETF 策略信号报告')
    print(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'数据日期: {df["date"].max().strftime("%Y-%m-%d")}')
    print('=' * 70)
    print()

    for strategy_key in ['adx_override', 'momentum_override', 'absolute_15']:
        strategy = get_strategy(strategy_key)

        result = generate_trade_signal(df, strategy_key, position='bond')
        results[strategy_key] = result

        print(f'【{strategy["name"]}】')
        print(f'  策略描述: {strategy["description"]}')
        print(f'  信号: {result["signal"]}')
        print(f'  建议操作: {result["action"]}')
        print()

        ind = result['indicators']
        print(f'  指标详情:')
        print(f'    收盘价:     {ind["close"]:.4f}')
        print(f'    MA50:       {ind["ma50"]:.4f}')
        print(f'    20日波动率: {ind["vol_20"]:.2f}%')
        print(f'    波动率阈值: {ind["vol_threshold"]:.2f}%')
        print(f'    价格>MA50:  {"是" if ind["above_ma50"] else "否"}')
        print(f'    低波动:     {"是" if ind["low_vol"] else "否"}')
        print('-' * 70)
        print()

    # 汇总建议
    print()
    print('=' * 70)
    print('交易建议汇总')
    print('=' * 70)

    signals_list = [r['signal'] for r in results.values()]
    actions_list = [r['action'] for r in results.values()]

    if len(set(signals_list)) == 1:
        print(f'【一致】所有策略信号: {signals_list[0]}')
        print(f'建议操作: {actions_list[0]}')
    else:
        print('【分歧】各策略信号不一致:')
        for key, r in results.items():
            print(f'  {STRATEGIES[key]["name"]}: {r["signal"]} -> {r["action"]}')

    # 保存报告
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f'CSI300 ETF 策略信号报告\n')
            f.write(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
            f.write(f'数据日期: {df["date"].max().strftime("%Y-%m-%d")}\n')
            f.write('=' * 70 + '\n\n')
            for key, r in results.items():
                f.write(f'【{STRATEGIES[key]["name"]}】\n')
                f.write(f'  信号: {r["signal"]}\n')
                f.write(f'  操作: {r["action"]}\n')
                ind = r['indicators']
                f.write(f'  收盘: {ind["close"]:.4f}, MA50: {ind["ma50"]:.4f}\n')
                f.write(f'  波动率: {ind["vol_20"]:.2f}%, 阈值: {ind["vol_threshold"]:.2f}%\n')
                f.write(f'  价格>MA50: {ind["above_ma50"]}, 低波动: {ind["low_vol"]}\n')
                f.write('-' * 40 + '\n\n')

    return results


def main():
    """主函数"""
    print('正在获取数据...')
    df = get_latest_data()

    if df is None:
        print('数据获取失败')
        return

    print(f'获取到 {len(df)} 条数据, 日期范围: {df["date"].min()} 到 {df["date"].max()}')

    # 生成报告
    report_dir = os.path.join(PROJECT_ROOT, 'reports')
    os.makedirs(report_dir, exist_ok=True)
    output_path = os.path.join(report_dir, 'daily_signal.txt')

    results = run_strategy_report(df, output_path)

    print()
    print(f'报告已保存到: {output_path}')

    return results


if __name__ == '__main__':
    main()
