"""
舆情/宏观辅助参考模块
采集: 北向资金, QVIX(中国恐慌指数), 市场资金流向, 全球指数, 相关性矩阵
"""
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SENTIMENT_FILE = os.path.join(PROJECT_ROOT, 'reports', 'sentiment_data.json')


def get_north_flow():
    """北向资金净流向"""
    try:
        import akshare as ak
        df = ak.stock_hsgt_fund_flow_summary_em()
        if len(df) == 0:
            return {"date": "N/A", "net_flow": 0, "status": "未知"}
        latest_date = str(df["日期"].iloc[-1])
        # 北上资金 = 沪股通+深股通当日净流入
        day_data = df[df["日期"] == latest_date]
        total = 0
        for _, r in day_data.iterrows():
            try:
                total += float(r["当日资金净流入"]) if str(r["当日资金净流入"]) != "暂无" else 0
            except Exception:
                pass
        return {
            "date": latest_date,
            "net_flow": round(total, 2),
            "status": "流入" if total > 0 else "流出" if total < 0 else "暂停",
        }
    except Exception:
        pass
    return {"date": "N/A", "net_flow": 0, "status": "未知"}


def get_qvix():
    """中国版恐慌指数 (300ETF期权隐含波动率)"""
    try:
        import akshare as ak
        df = ak.index_option_300etf_qvix()
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        return {
            "date": str(latest["date"]),
            "qvix": round(float(latest["close"]), 2),
            "change": round(float(latest["close"]) - float(prev["close"]), 2),
            "level": "恐慌" if float(latest["close"]) > 25 else "偏高" if float(latest["close"]) > 20 else "正常",
        }
    except Exception:
        pass
    return {"date": "N/A", "qvix": 0, "change": 0, "level": "未知"}


def get_market_flow():
    """市场主力资金流向"""
    try:
        import akshare as ak
        df = ak.stock_market_fund_flow()
        latest = df.iloc[-1]
        # column: 主力净流入-净额 (in yuan)
        main_col = [c for c in df.columns if "主力" in c and "净额" in c]
        if main_col:
            main_net = float(latest[main_col[0]]) / 1e8
        else:
            main_net = 0
        return {
            "date": str(latest.iloc[0]),
            "main_net": round(main_net, 2),
            "direction": "流入" if main_net > 0 else "流出",
        }
    except Exception:
        pass
    return {"date": "N/A", "main_net": 0, "direction": "未知"}


def get_global_indexes():
    """全球主要指数"""
    try:
        import akshare as ak
        df = ak.index_global_spot_em()
        targets = {
            "道琼斯": "DJIA",
            "纳斯达克": "NASDAQ",
            "标普500": "SPX",
            "恒生指数": "HSI",
            "日经225": "N225",
        }
        result = {}
        for cn, en in targets.items():
            row = df[df["名称"].str.contains(cn, na=False)]
            if len(row) > 0:
                r = row.iloc[0]
                result[en] = {
                    "price": float(r["最新价"]) if r["最新价"] != "-" else 0,
                    "change_pct": float(r["涨跌幅"]) if r["涨跌幅"] != "-" else 0,
                }
        return result
    except Exception:
        pass
    return {}


def get_correlation_matrix():
    """多品种相关性矩阵"""
    try:
        import akshare as ak
        etfs = {
            "510300": "sh510300",  # 沪深300ETF
            "510310": "sh510310",  # 沪深300ETF易方达
            "159995": "sz159995",  # 芯片ETF
            "511260": "sh511260",  # 国债ETF
            "159915": "sz159915",  # 创业板ETF
            "510050": "sh510050",  # 上证50ETF
        }
        cutoff = pd.Timestamp.now() - pd.DateOffset(years=2)
        closes = {}
        for name, code in etfs.items():
            try:
                df = ak.fund_etf_hist_sina(symbol=code)
                df["date"] = pd.to_datetime(df["date"])
                df = df[df["date"] >= cutoff][["date", "close"]]
                df = df.rename(columns={"close": name})
                closes[name] = df
            except Exception:
                pass

        if len(closes) < 3:
            return {}

        merged = None
        for name, df in closes.items():
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on="date", how="inner")

        returns = {}
        for col in closes:
            returns[col] = merged[col].pct_change()

        ret_df = pd.DataFrame(returns).dropna()
        corr = ret_df.corr().round(3)

        return corr.to_dict()
    except Exception:
        return {}


def collect_all():
    """采集所有舆情数据"""
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "north_flow": get_north_flow(),
        "qvix": get_qvix(),
        "market_flow": get_market_flow(),
        "global": get_global_indexes(),
        "correlation": get_correlation_matrix(),
    }
    os.makedirs(os.path.dirname(SENTIMENT_FILE), exist_ok=True)
    with open(SENTIMENT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return data


if __name__ == "__main__":
    d = collect_all()
    print(json.dumps(d, ensure_ascii=False, indent=2, default=str))
