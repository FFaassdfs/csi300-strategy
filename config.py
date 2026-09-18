"""
品种与路径唯一权威配置 — 所有脚本统一从这里导入
与 OPERATION.md §1.2 权威配置保持一致, 勿在其他文件复制 ASSETS
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WORK_DB = os.path.join(PROJECT_ROOT, 'csi300_data.duckdb')
HISTORY_DB = os.path.join(PROJECT_ROOT, 'trading_history.duckdb')

RF = 0.025  # 无风险利率 (逆回购/国债参考)

# 品种池 (已淘汰: 512660军工/512010医药/512880证券, 勿重新纳入)
ASSETS = {
    '510310': {'name': '沪深300ETF', 'code': 'sh510310', 'ma_p': 30, 'adx_th': 20, 'vol_th': 18},
    '159995': {'name': '芯片ETF',    'code': 'sz159995', 'ma_p': 30, 'adx_th': 25, 'vol_th': 15},
    '512800': {'name': '银行ETF',    'code': 'sh512800', 'ma_p': 30, 'adx_th': 25, 'vol_th': 18},
}

ORDER = ['510310', '159995', '512800']
