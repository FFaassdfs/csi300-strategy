"""
下载 510310 沪深300ETF 近5年价格数据并入库
- 数据源: akshare (ak.fund_etf_hist_sina, 新浪财经)
- 表名: etf_510310_daily
- 存储位置: D:/opencode/etf/csi300_strategy/csi300_data.duckdb (本地项目库)
"""
import akshare as ak
import pandas as pd
import os
import duckdb
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, 'csi300_data.duckdb')
YEARS = 5
SYMBOL = 'sh510310'


def download_etf_data():
    """下载510310 ETF历史数据"""
    print(f'下载 {SYMBOL} 沪深300ETF 数据...')
    df = ak.fund_etf_hist_sina(symbol=SYMBOL)
    print(f'  原始数据: {len(df)} 条')
    print(f'  列名: {df.columns.tolist()}')

    df['date'] = pd.to_datetime(df['date'])

    # 过滤近5年
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=YEARS)
    df = df[df['date'] >= cutoff].copy()
    df = df.sort_values('date').reset_index(drop=True)

    print(f'  5年数据: {len(df)} 条')
    print(f'  日期范围: {df["date"].min().date()} ~ {df["date"].max().date()}')
    print(f'  最新收盘: {df["close"].iloc[-1]:.4f}')
    print(f'  5年最高: {df["high"].max():.4f} ({df.loc[df["high"].idxmax(), "date"].date()})')
    print(f'  5年最低: {df["low"].min():.4f} ({df.loc[df["low"].idxmin(), "date"].date()})')

    return df


def store_to_db(df):
    """存储到duckdb"""
    conn = duckdb.connect(DB_PATH)

    # 创建新表 (列名加etf后缀以与csi300_daily区分)
    conn.execute('DROP TABLE IF EXISTS etf_510310_daily')
    conn.execute('''
        CREATE TABLE etf_510310_daily (
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            amount DOUBLE
        )
    ''')
    conn.execute('INSERT INTO etf_510310_daily SELECT * FROM df')

    # 验证
    count = conn.execute('SELECT COUNT(*) FROM etf_510310_daily').fetchone()[0]
    last = conn.execute('SELECT date, close FROM etf_510310_daily ORDER BY date DESC LIMIT 1').fetchone()
    print(f'  已写入: {count} 条')
    print(f'  最新一条: {last}')

    conn.close()


def main():
    print('=' * 60)
    print('510310 沪深300ETF 数据更新')
    print('=' * 60)
    df = download_etf_data()
    print()
    print('=' * 60)
    print('写入数据库')
    print('=' * 60)
    store_to_db(df)
    print()
    print('完成！')


if __name__ == '__main__':
    main()
