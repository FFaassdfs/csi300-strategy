"""
每日一键刷新: 登记交易 → 更新数据 → 生成报告 → 提交历史
用法: python daily_refresh.py
"""
import os
import sys
import csv
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TRADES_FILE = os.path.join(PROJECT_ROOT, 'trades', '510310_trades.csv')
DB_FILE = os.path.join(PROJECT_ROOT, 'csi300_data.duckdb')


def run(cmd_args, **kw):
    return subprocess.run(cmd_args, cwd=PROJECT_ROOT, **kw)


# ============ 1. 登记交易 ============
print()
print('=' * 50)
print('  每日刷新流程')
print('=' * 50)
print()

# 显示现有持仓
existing_trades = []
if os.path.exists(TRADES_FILE):
    with open(TRADES_FILE, 'r', encoding='utf-8') as f:
        existing_trades = list(csv.DictReader(f))

if existing_trades:
    print('已有交易记录:')
    total_shares = 0
    total_cost = 0.0
    for r in existing_trades:
        shares = int(r['shares'])
        amount = float(r['amount'])
        sign = 1 if r['action'].strip().upper() == 'BUY' else -1
        total_shares += shares * sign
        total_cost += amount * sign
        print(f'  {r["date"]} {r["action"]:>4s} {shares:>5}份 @{r["price"]}  ¥{amount}')
    if total_shares > 0:
        print(f'  当前持仓: {total_shares}份  成本¥{total_cost:,.2f}  均价¥{total_cost/total_shares:.4f}')
else:
    print('暂无交易记录')

print()
print('-' * 50)
action = input('今日有新的交易要登记吗? (y/n, 回车跳过): ').strip().lower()

trade_recorded = False
if action == 'y':
    today = datetime.now().strftime('%Y-%m-%d')
    trade_type = input('  操作 (BUY/SELL): ').strip().upper()
    price = input('  成交价: ').strip()
    shares = input('  份额: ').strip()
    note = input('  备注 (回车默认): ').strip() or '按ADX策略'

    amount = float(price) * int(shares)

    # 确保文件有表头
    file_exists = os.path.exists(TRADES_FILE) and os.path.getsize(TRADES_FILE) > 0
    with open(TRADES_FILE, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['date', 'action', 'code', 'price', 'shares', 'amount', 'balance', 'note'])
        writer.writerow([today, trade_type, '510310', price, shares, f'{amount:.2f}', '', note])

    print(f'  ✅ 已登记: {today} {trade_type} {shares}份 @{price}  ¥{amount:,.2f}')
    trade_recorded = True

# ============ 2. 刷新数据 ============
print()
print('=' * 50)
print('  刷新数据...')
print('=' * 50)

today_str = datetime.now().strftime('%Y%m%d')

print('  更新沪深300指数...')
run([sys.executable, '-c',
    "import akshare as ak, pandas as pd, duckdb;"
    "from datetime import datetime, timedelta;"
    "df = ak.stock_zh_index_daily(symbol='sh000300');"
    "df['date'] = pd.to_datetime(df['date']);"
    "df = df[['date','open','high','low','close']].dropna().sort_values('date');"
    "cutoff = pd.Timestamp.now() - pd.DateOffset(years=5);"
    "df = df[df['date'] >= cutoff];"
    "conn = duckdb.connect('csi300_data.duckdb');"
    "conn.execute('DROP TABLE IF EXISTS csi300_daily');"
    "conn.execute('CREATE TABLE csi300_daily (date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE)');"
    "conn.execute('INSERT INTO csi300_daily SELECT * FROM df');"
    "conn.close();"
    "print(f'  {len(df)}条, {df[\"date\"].max().date()} 收{df[\"close\"].iloc[-1]:.2f}')"
])

print('  更新510310 ETF...')
run([sys.executable, 'update_etf_510310.py'])

print('  采集舆情/宏观数据...')
run([sys.executable, 'sentiment_collector.py'], capture_output=True)

# ============ 3. 生成报告 ============
print()
print('=' * 50)
print('  生成报告...')
print('=' * 50)

run([sys.executable, 'generate_html_report.py'])

# ============ 4. 提交历史 ============
print()
print('=' * 50)
print('  保存历史记录...')
print('=' * 50)

report_file = f'reports/daily_signal_{today_str}.html'

files_to_commit = [report_file]
if trade_recorded:
    files_to_commit.append(TRADES_FILE)

# 添加并提交
for f in files_to_commit:
    if os.path.exists(os.path.join(PROJECT_ROOT, f)):
        run(['git', 'add', f], capture_output=True)

msg = f'auto: {today_str} refresh'
if trade_recorded:
    msg += ' + trade'
result = run(['git', 'commit', '-m', msg], capture_output=True, text=True)

if 'nothing to commit' not in result.stdout + result.stderr:
    run(['git', 'push', 'origin', 'main'], capture_output=True)
    print('  已推送到 GitHub')
else:
    print('  无变更需提交')

print()
print(f'  报告: {report_file}')
print('=' * 50)
