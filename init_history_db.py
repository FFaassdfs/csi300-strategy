"""
trading_history.duckdb 数据库初始化
追加式存储: 只插入不覆盖，保留完整历史
"""
import duckdb
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, 'trading_history.duckdb')

SCHEMA = {
    'daily_ohlc': '''
        CREATE TABLE IF NOT EXISTS daily_ohlc (
            date DATE NOT NULL,
            code VARCHAR NOT NULL,
            name VARCHAR,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            amount DOUBLE,
            PRIMARY KEY (date, code)
        )
    ''',
    'daily_indicators': '''
        CREATE TABLE IF NOT EXISTS daily_indicators (
            date DATE NOT NULL,
            code VARCHAR NOT NULL,
            ma50 DOUBLE,
            adx DOUBLE,
            vol20 DOUBLE,
            momentum20 DOUBLE,
            bb_pct_b DOUBLE,
            above_ma50 BOOLEAN,
            low_vol BOOLEAN,
            strong_trend BOOLEAN,
            signal INT,
            PRIMARY KEY (date, code)
        )
    ''',
    'daily_sentiment': '''
        CREATE TABLE IF NOT EXISTS daily_sentiment (
            date DATE NOT NULL PRIMARY KEY,
            qvix DOUBLE,
            north_flow DOUBLE,
            main_flow DOUBLE,
            sh_close DOUBLE,
            sz_close DOUBLE,
            global_djia DOUBLE,
            global_nasdaq DOUBLE,
            global_hsi DOUBLE,
            global_n225 DOUBLE
        )
    ''',
    'signals_log': '''
        CREATE TABLE IF NOT EXISTS signals_log (
            date DATE NOT NULL,
            code VARCHAR NOT NULL,
            name VARCHAR,
            price DOUBLE,
            signal INT,
            reason VARCHAR,
            PRIMARY KEY (date, code)
        )
    ''',
    'portfolio_snapshot': '''
        CREATE TABLE IF NOT EXISTS portfolio_snapshot (
            date DATE NOT NULL PRIMARY KEY,
            cash DOUBLE,
            market_value DOUBLE,
            total_assets DOUBLE,
            day_pnl DOUBLE,
            note VARCHAR
        )
    ''',
}

def init_db():
    conn = duckdb.connect(DB_PATH)
    for name, ddl in SCHEMA.items():
        conn.execute(ddl)
        print(f'  [OK] {name}')
    conn.close()
    print(f'\n数据库初始化完成: {DB_PATH}')

if __name__ == '__main__':
    init_db()
