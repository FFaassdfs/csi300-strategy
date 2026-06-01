# CSI300 ETF 趋势跟踪策略

沪深300指数的量化择时策略，结合波动率风控和趋势确认。

## 策略概述

基于三个策略的集成：
- **+ADX>25 Override**: 趋势确认时持有
- **+Momentum>10% Override**: 动量确认时持有
- **Base Absolute 15%**: 波动率阈值控制

## 策略逻辑

```
买入信号: 价格 > MA50 AND (波动率 < 15% OR 趋势确认信号)
卖出信号: 价格 < MA50 OR 波动率 >= 15%
```

## 核心指标

| 指标 | 计算方式 |
|------|---------|
| MA50 | 50日简单移动平均 |
| 波动率 | 20日收益率标准差，年化 |
| 动量 | 20日价格变化率 |
| ADX | 趋势强度指标 |

## 目录结构

```
csi300_strategy/
├── README.md                    # 本文件
├── requirements.txt             # 依赖列表
├── generate_html_report.py       # 生成HTML策略报告（主程序）
├── execute_daily.py             # 每日执行脚本
├── run_strategy.py             # 策略运行器
├── strategies/
│   └── csi300_strategies.py    # 三个策略定义
└── reports/
    └── daily_signal_*.html     # 每日报告输出
```

## 使用方法

```bash
cd D:/opencode/etf/csi300_strategy
py -3.11 generate_html_report.py
```

## 回测结果（2021-2025）

| 策略 | 年化收益 | Sharpe | 最大回撤 |
|------|---------|--------|---------|
| ADX Override | +8.44% | 0.68 | -13.60% |
| Momentum Override | +7.63% | 0.73 | -11.74% |
| Absolute 15% | +3.29% | 0.48 | -11.74% |
| Buy&Hold | +16.24% | 0.32 | -36.05% |

## 策略对比说明

| 策略 | 适用场景 | 优点 |
|------|---------|------|
| ADX Override | 强趋势市场 | 牛市参与度高 |
| Momentum Override | 趋势/动量市场 | Sharpe最高 |
| Absolute 15% | 保守操作 | 熊市保护强 |

## 注意事项

- 本策略仅供参考，不构成投资建议
- 历史回测不代表未来收益
- 投资有风险，决策需谨慎
