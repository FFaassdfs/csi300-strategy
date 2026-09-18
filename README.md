# CSI300 多品种 ADX Override 轮动策略

沪深300 指数量化择时系统：在 3 只 ETF 之间轮动，有信号持有、无信号持币/逆回购。不做空、不加杠杆、不买个股。

> 完整操作手册：**[OPERATION.md](OPERATION.md)**（权威）。本 README 只是速览。
> PROJECT_OVERVIEW.md 为 v1 单品种方案的历史快照，已被 OPERATION.md 取代。

## 策略规则（现行 v2）

```
持有条件 = 价格 > MA30 AND (20日波动率 < 品种阈值 OR ADX > 品种阈值)
入场需连续 2 日确认；出场信号消失立即卖出（不等确认）
T日收盘算信号 → T+1 开盘执行
多品种有信号 → 持有 ADX 最高者；全无 → 国债/逆回购
```

| 品种 | MA | ADX阈值 | 波动率阈值 | 定位 |
|------|----|---------|-----------|------|
| 510310 沪深300ETF | 30 | 20 | 18% | 主进攻 |
| 159995 芯片ETF | 30 | 25 | 15% | 高弹性 |
| 512800 银行ETF | 30 | 25 | 18% | 防守仓 |

仓位调节：QVIX 恐慌指数分档（100/80/60/40%）× 极端波动刹车（vol>35%→60%）。

## 每日流程（Windows 计划任务自动，Python 3.12）

| 时间 | 脚本 | 作用 |
|------|------|------|
| 11:35 | `intraday_signal.py mid` | 盘中快照 |
| 12:00 | `send_advice.py mid` | 午间邮件（下午操作建议） |
| 15:10 | `daily_close.py` → `auto_refresh.py` | 行情入库 + 指标/信号记录（不发邮件） |
| 16:00 | `send_advice.py close` | 收盘邮件（**次日操作指令权威来源**） |

手动命令：

```bash
python auto_refresh.py      # 刷新数据入库
python dual_rotation.py     # 只看三品种轮动信号(不发邮件)
python send_advice.py close # 手动发收盘邮件
```

## 目录结构

- **权威核心**：`config.py`（品种池/路径）、`signal_core.py`（信号/置信度/QVIX/刹车/持仓唯一实现）
- **生产脚本**：`auto_refresh.py`、`send_advice.py`、`intraday_signal.py`、`daily_close.py`、`dual_rotation.py`
- **数据**：`trading_history.duckdb`（历史库，追加式）、`trades/*_trades.csv`（持仓与现金权威来源）
- **研究/回测**：`validate_tier1.py`（walk-forward/成本/过滤器）、`validate_expansion.py`（扩池，已否决）、`validate_voltarget.py`（仓位模式，采纳极端刹车）、`validate_signals.py`、`compare_bank_inclusion.py`
- **废弃 v1**（勿用于实盘）：`generate_html_report.py`、`execute_daily.py`、`run_strategy.py`、`strategies/`、`daily_refresh.py`、`csi300_data.duckdb` 等，详见 OPERATION.md §7.1 目录树

## 关键回测数字（3 品种轮动，2020-04 ~ 2026-09，含 10bp 单边成本，修正数据后，见 OPERATION.md §11）

| 方案 | 年化 | Sharpe | 最大回撤 |
|------|------|--------|---------|
| 现行（2日确认） | +20.3% | 0.78 | -21.6% |
| + 极端波动刹车(35%/60%) | +17.2% | 0.80 | -16.8%（2026年以来回撤 -13.7%，Sharpe 1.24） |

**长历史压力测试**（指数代理 2015-08~2026-09，覆盖 2015 股灾/2018 熊市/2022-24 长熊）：策略 **+7.7~8.7%/年、Sharpe 0.37、回撤 -24.9%**，沪深300 Buy&Hold 同期 +1.4%/年、-45.6%。危机段回撤控制显著（2015 第二轮 -7.8% vs 指数 -25.4%；2018 -17.2% vs -30.8%；2022-24 -4.2% vs -31.8%），但 2024-09 急涨跑输（+8.3% vs +34.2%）。

> **预期收益请按"含成本 +10~15%/年、回撤 -20~-30%"规划**（6.4年回测的 +20% 属结构性行情偏乐观；全周期跨牛熊约 +8%）。
> 详见 `reports/tier1_validation_*.md`、`reports/voltarget_validation_*.md`、`reports/longhistory_validation_*.md`。

预期请按"含成本 +15~20%/年、回撤 -20~-30%"规划。

## 依赖

```
baostock  akshare  pandas  duckdb   # requirements.txt
```

## 注意事项

- 所有策略输出仅供研究参考，不构成投资建议
- 信号唯一实现 `signal_core.py`，任何口径修改只改这里
- 交易执行后必须登记 `trades/<代码>_trades.csv` 且 balance 必填（邮件现金显示依赖它）
