# -*- coding: utf-8 -*-
"""
P1-数据采集：通过 tushare 代理接口拉取豆粕(M)、玉米(C)、菜粕(RM)期货日线。
候选合约限定为 1/5/9 月合约（农产品期货主力集中于这三个月），主力认定按持仓量最大。
限流策略：每次调用间隔 >=1.2s，遇到 upstream_pool_exhausted 等待 35s 重试，最多 5 次。
断点续传：每个合约单独存 CSV，已存在的跳过。
用法: python collect_futures.py M   (或 C / RM / ALL)
"""
import os, sys, time
import pandas as pd
import tushare as ts

KEY = "tsr_mVGs0M9gLUG2-apuk1l5Qhsi4afinuaEowj1Z0hGwag"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_futures")
os.makedirs(RAW_DIR, exist_ok=True)

# 合约乘数/交易所
SYMBOLS = {
    "M":  {"exchange": "DCE", "name": "豆粕"},
    "C":  {"exchange": "DCE", "name": "玉米"},
    "RM": {"exchange": "CZCE", "name": "菜粕"},
}
MONTHS = [1, 5, 9]
YEARS = [23, 24, 25, 26, 27]   # 合约年份（两位）

def contract_window(year2, month):
    """合约活跃窗口：上市约=到期前13个月，到期=合约月份15日左右。"""
    full_y = 2000 + year2
    end = pd.Timestamp(full_y, month, 20)
    start = end - pd.DateOffset(months=13)
    return max(start, pd.Timestamp("2022-07-01")), min(end, pd.Timestamp("2026-10-10"))

def chunks(start, end, months=6):
    """把 [start, end] 切成 <=6 个月的区间。"""
    out, s = [], start
    while s <= end:
        e = min(s + pd.DateOffset(months=months) - pd.DateOffset(days=1), end)
        out.append((s.strftime("%Y%m%d"), e.strftime("%Y%m%d")))
        s = e + pd.DateOffset(days=1)
    return out

def fetch(pro, ts_code, s, e):
    for attempt in range(5):
        try:
            df = pro.fut_daily(ts_code=ts_code, start_date=s, end_date=e)
            time.sleep(1.2)
            return df
        except Exception as ex:
            msg = str(ex)
            if "pool_exhausted" in msg or "Read timed out" in msg or "timeout" in msg.lower():
                wait = 35 * (attempt + 1)
                print(f"  retry {attempt+1} after {wait}s ({msg[:60]})")
                time.sleep(wait)
            elif "date_range_too_large" in msg:
                raise
            else:
                print(f"  error: {msg[:120]}")
                time.sleep(5)
    return pd.DataFrame()

def main(symbols):
    pro = ts.pro_api(KEY)
    pro._DataApi__http_url = "https://tl.kaixin8.top/tushare/sdk"
    for sym in symbols:
        ex = SYMBOLS[sym]["exchange"]
        for y in YEARS:
            for m in MONTHS:
                if sym == "RM":
                    code = f"RM{y:02d}{m:02d}.CZCE"
                else:
                    code = f"{sym}{y:02d}{m:02d}.DCE"
                fp = os.path.join(RAW_DIR, f"{code}.csv")
                if os.path.exists(fp):
                    continue
                s, e = contract_window(y, m)
                parts = []
                for cs, ce in chunks(s, e):
                    df = fetch(pro, code, cs, ce)
                    if df is not None and not df.empty:
                        parts.append(df)
                if parts:
                    out = pd.concat(parts).drop_duplicates(
                        subset=["ts_code", "trade_date"]).sort_values("trade_date")
                    out.to_csv(fp, index=False, encoding="utf-8-sig")
                    print(f"{code}: {len(out)} rows saved")
                else:
                    print(f"{code}: no data")

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    syms = list(SYMBOLS) if arg == "ALL" else [arg]
    main(syms)
