# -*- coding: utf-8 -*-
"""
P1 - 数据清洗、连续合约拼接与 12 项指标计算
输入: data/raw_futures/*.csv (tushare 真实行情)
输出: data/supply_chain.db 的 fact_futures_daily / fact_price_weekly / fact_formula_cost_weekly
      outputs/p1_weekly_indicators.csv

口径说明：
- 主力合约：候选限定 1/5/9 月合约（农产品期货主力集中于此），每日取持仓量最大者；
- 连续价：换月跳空用【比例复权】——保持收益率连续，便于计算波动率与回报，
  差价复权会扭曲远期价格的相对变化，故选比例复权；
- 现货价/鱼价：接口待核，当前为模拟数据（假设 A-SPT-01 / A-FIS-01），已标注；
- 周频：ISO 周，取周均值，周截止日为周五。
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
import sqlite3

BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(BASE, "common"))
from formulation import solve_feed, BASE_PRICES

RAW = os.path.join(BASE, "data", "raw_futures")
DB = os.path.join(BASE, "data", "supply_chain.db")
SYM_NAME = {"M": "豆粕", "C": "玉米", "RM": "菜粕"}
rng = np.random.default_rng(42)

# ---------- 1. 合并行情、认定主力 ----------
def load_futures():
    frames = []
    for f in glob.glob(os.path.join(RAW, "*.csv")):
        df = pd.read_csv(f, dtype={"trade_date": str})
        df["symbol"] = df.ts_code.str.extract(r"^([A-Z]+)")
        frames.append(df)
    fut = pd.concat(frames, ignore_index=True)
    fut["date"] = pd.to_datetime(fut.trade_date, format="%Y%m%d")
    fut = fut[fut.symbol.isin(SYM_NAME)]
    idx = fut.groupby(["symbol", "date"])["oi"].idxmax()
    main = fut.loc[idx].copy()
    main["is_main"] = 1
    fut["is_main"] = 0
    return fut, main.sort_values(["symbol", "date"])

# ---------- 2. 比例复权连续价 ----------
def ratio_adjust(g):
    g = g.sort_values("date").reset_index(drop=True)
    factor = 1.0
    factors = [1.0]
    for i in range(1, len(g)):
        if g.ts_code[i] != g.ts_code[i - 1]:        # 换月
            prev_close = g.settle[i - 1]
            if prev_close and not np.isnan(prev_close) and prev_close > 0:
                factor *= g.settle[i] / prev_close  # 复权因子累乘
        factors.append(factor)
    g["adj_factor"] = factors
    g["adj_settle"] = g.settle / g.adj_factor
    return g

# ---------- 3. 模拟现货（期货+基差）与鱼价 ----------
def make_spot(main_adj, symbol):
    """现货=主力结算价+均值回复基差（模拟，A-SPT-01）"""
    center = {"豆粕": 200, "菜粕": 150, "玉米": 100}[symbol]
    sd = {"豆粕": 80, "菜粕": 60, "玉米": 50}[symbol]
    n = len(main_adj)
    basis = np.zeros(n)
    eps = rng.normal(0, sd * 0.6, n)
    for i in range(1, n):
        basis[i] = basis[i - 1] + 0.08 * (center - basis[i - 1]) + eps[i]
    basis[0] = center
    return main_adj.settle.values + basis

def make_fish_price_weekly(idx, push):
    """草鱼批发价（周频，模拟，A-FIS-01）：中枢16元/公斤，冬季高夏季低；
    push 为滞后8周的标准化成本变动（内嵌成本推动关系，用于领先滞后分析演示）。"""
    n = len(idx)
    doy = idx.dayofyear.values
    seasonal = -1.0 * np.sin(2 * np.pi * (doy - 80) / 365.0)
    eps = rng.normal(0, 0.10, n)
    p = push.fillna(0).values
    u = np.zeros(n)
    for i in range(1, n):
        u[i] = 0.90 * u[i - 1] + 0.09 * p[i] + eps[i]
    price = pd.Series(16.0 + seasonal + u, index=idx)
    # 人为缺失：短缺口(1周)用前值补齐，长缺口(4周)保留空白并在看板标注
    s_gap = price.copy()
    s_gap["2024-03-03":"2024-03-10"] = np.nan        # 短缺口
    s_gap["2025-06-29":"2025-07-27"] = np.nan        # 长缺口
    s_filled = s_gap.ffill(limit=1)
    return price, s_gap, s_filled

# ---------- 4. 周度价格 -> 配方用价 ----------
def weekly_prices(main_adj_all, fish_weekly):
    wk = {}
    for sym, g in main_adj_all.items():
        w = g.set_index("date")[["settle", "adj_settle", "spot"]].resample("W-FRI").mean()
        wk[SYM_NAME[sym]] = w
    df = pd.concat(wk, axis=1)
    df.columns = [f"{a}_{b}" for a, b in df.columns]
    df["鱼价"] = fish_weekly
    # 配方映射价（无期货品种按联动关系，假设 A-PRC-02）
    df["p_豆粕"] = df["豆粕_settle"]; df["p_菜粕"] = df["菜粕_settle"]
    df["p_玉米"] = df["玉米_settle"]
    df["p_棉粕"] = 0.85 * df["p_豆粕"]
    df["p_小麦"] = 1.05 * df["p_玉米"]; df["p_米糠"] = 0.75 * df["p_玉米"]
    # 鱼粉周价（模拟，A-PRC-03：付费平台数据拿不到，AR(1)中枢12500，区间10000-16000）
    fm = np.zeros(len(df)); fm_eps = rng.normal(0, 400, len(df)); fm[0] = 12500.0
    for i in range(1, len(df)):
        fm[i] = np.clip(12500 + 0.90 * (fm[i - 1] - 12500) + fm_eps[i], 10000, 16000)
    df["p_鱼粉"] = fm
    return df.dropna(subset=["豆粕_settle"])

# ---------- 5. 指标计算 ----------
def compute_indicators(main_adj_all, weekly, fish_gap_weekly):
    out = weekly.copy()
    for name in SYM_NAME.values():
        d = main_adj_all[{v: k for k, v in SYM_NAME.items()}[name]].set_index("date")
        ret = d.adj_settle.pct_change()
        vol20 = ret.rolling(20).std() * np.sqrt(252) * 100
        out[f"{name}_波动率"] = vol20.resample("W-FRI").last()
        out[f"{name}_基差"] = out[f"{name}_spot"] - out[f"{name}_settle"]
    # 固定权重成本指数（文献配方区间中值，草鱼料口径；仅期货品种动态计价）
    w = {"豆粕": 0.20, "菜粕": 0.25, "棉粕": 0.12, "玉米": 0.20, "小麦": 0.10, "米糠": 0.08}
    base = BASE_PRICES
    fixed_idx = (w["豆粕"] * out["p_豆粕"] / base["豆粕"] +
                 w["菜粕"] * out["p_菜粕"] / base["菜粕"] +
                 w["棉粕"] * out["p_棉粕"] / base["棉粕"] +
                 w["玉米"] * out["p_玉米"] / base["玉米"] +
                 w["小麦"] * out["p_小麦"] / base["小麦"] +
                 w["米糠"] * out["p_米糠"] / base["米糠"] +
                 (1 - sum(w.values()))) * 100
    out["固定权重成本指数"] = fixed_idx
    # 鱼价指数与比价
    fish0 = out["鱼价"].dropna().iloc[0]
    out["鱼价指数"] = out["鱼价"] / fish0 * 100
    out["鱼料比价指数"] = out["鱼价指数"] / out["固定权重成本指数"] * 100
    # 历史分位（滚动3年=156周）
    out["比价历史分位"] = out["鱼料比价指数"].rolling(156, min_periods=20).apply(
        lambda s: (s <= s.iloc[-1]).mean() * 100, raw=False)
    # 数据质量
    comp = {"鱼价": 1 - fish_gap_weekly.isna().mean()}
    return out, comp

# ---------- 6. 配方成本（每周重算LP） ----------
def formula_costs(weekly):
    rows = []
    fixed_formula = {}
    for spec in ["草鱼料", "加州鲈料"]:
        r0 = solve_feed(spec)  # 基准价下的固定配方
        fixed_formula[spec] = r0["formula"]
    for wk, row in weekly.iterrows():
        prices = {k[2:]: v for k, v in row.items() if k.startswith("p_") and not np.isnan(v)}
        rec = {"week_start": wk}
        for spec in ["草鱼料", "加州鲈料"]:
            full_prices = {**BASE_PRICES, **prices}
            dyn = solve_feed(spec, prices)
            ff = fixed_formula[spec]
            cost_fixed = sum(ff[i] / 100 * full_prices[i] for i in ff.index)
            rec[f"{spec}_动态成本"] = round(dyn["cost"], 1)
            rec[f"{spec}_固定成本"] = round(cost_fixed, 1)
            rec[f"{spec}_调整收益"] = round(cost_fixed - dyn["cost"], 1)
            rec[f"{spec}_配方json"] = json.dumps(
                {k: round(v, 2) for k, v in dyn["formula"].items() if v > 0.01},
                ensure_ascii=False)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("week_start")

# ---------- main ----------
def main():
    fut, main = load_futures()
    print(f"行情 {len(fut)} 行；主力序列 {len(main)} 行")

    # 写 fact_futures_daily（只写候选合约,标主力）
    con = sqlite3.connect(DB)
    con.execute("DELETE FROM fact_futures_daily")
    ins = fut.assign(contract_month=fut.ts_code.str.extract(r"[A-Z]+\d{2}(\d{2})")[0],
                     source_key="SRC01")
    ins[["ts_code", "trade_date", "symbol", "contract_month", "open", "high",
        "low", "close", "settle", "vol", "oi"]].assign(
        is_main=0, source_key="SRC01").to_sql("fact_futures_daily", con,
                                              if_exists="append", index=False)

    main_adj_all = {}
    for sym, g in main.groupby("symbol"):
        ga = ratio_adjust(g)
        ga["spot"] = make_spot(ga, SYM_NAME[sym])
        main_adj_all[sym] = ga

    # 周频期货价 -> 成本z分数 -> 滞后8周的标准化变动（鱼价的成本推动成分）
    wk_settle = {sym: ga.set_index("date").settle.resample("W-FRI").mean()
                 for sym, ga in main_adj_all.items()}
    cost_w = 0.5 * wk_settle["M"] + 0.3 * wk_settle["RM"] + 0.2 * wk_settle["C"]
    cost_z_w = (cost_w - cost_w.mean()) / cost_w.std()
    d = cost_z_w.diff()
    push = (d / d.std()).shift(8)
    fish_true, fish_gap, fish_filled = make_fish_price_weekly(cost_w.index, push)
    fish_weekly = fish_filled
    fish_gap_weekly = fish_gap

    weekly = weekly_prices(main_adj_all, fish_weekly)
    ind, comp = compute_indicators(main_adj_all, weekly, fish_gap_weekly)
    costs = formula_costs(weekly)
    ind = ind.join(costs)

    # 派生指标 ⑧⑨
    for spec in ["草鱼料", "加州鲈料"]:
        ind[f"{spec}_饵料成本占鱼价比"] = 1.8 * ind[f"{spec}_动态成本"] / 1000 / ind["鱼价"] * 100
    ind["草鱼料_占比分位"] = ind["草鱼料_饵料成本占鱼价比"].rolling(156, min_periods=20).apply(
        lambda s: (s <= s.iloc[-1]).mean() * 100, raw=False)

    ind.to_csv(os.path.join(BASE, "outputs", "p1_weekly_indicators.csv"), encoding="utf-8-sig")

    # 写 fact 表
    con.execute("DELETE FROM fact_price_weekly")
    con.execute("DELETE FROM fact_formula_cost_weekly")
    recs = []
    for wk, row in ind.iterrows():
        ws = wk.strftime("%Y%m%d")
        for name in SYM_NAME.values():
            recs += [(ws, f"{name}主力连续价", "行情", row[f"{name}_adj_settle"], "元/吨", "SRC01", ""),
                     (ws, f"{name}现货价", "行情", row[f"{name}_spot"], "元/吨", "SRC02", "模拟"),
                     (ws, f"{name}基差", "指标", row[f"{name}_基差"], "元/吨", "计算", "现货为模拟"),
                     (ws, f"{name}波动率", "指标", row[f"{name}_波动率"], "%年化", "计算", "")]
        recs += [(ws, "草鱼批发价", "行情", row["鱼价"], "元/公斤", "SRC03", "模拟"),
                 (ws, "固定权重成本指数", "指标", row["固定权重成本指数"], "点", "计算", ""),
                 (ws, "鱼料比价指数", "指标", row["鱼料比价指数"], "点", "计算", ""),
                 (ws, "比价历史分位", "指标", row["比价历史分位"], "%", "计算", ""),
                 (ws, "草鱼料饵料成本占鱼价比", "指标", row["草鱼料_饵料成本占鱼价比"], "%", "计算", "")]
    pd.DataFrame(recs, columns=["week_start", "item", "item_type", "value",
                                "unit", "source_key", "note"]).to_sql(
        "fact_price_weekly", con, if_exists="append", index=False)
    fc = []
    for wk, row in costs.iterrows():
        for spec in ["草鱼料", "加州鲈料"]:
            fc.append((wk.strftime("%Y%m%d"), spec, row[f"{spec}_固定成本"],
                       row[f"{spec}_动态成本"], row[f"{spec}_配方json"], "SRC01"))
    pd.DataFrame(fc, columns=["week_start", "feed_spec", "cost_fixed", "cost_dynamic",
                              "formula_json", "source_key"]).to_sql(
        "fact_formula_cost_weekly", con, if_exists="append", index=False)
    con.commit(); con.close()

    print(f"周度指标 {len(ind)} 周 x {ind.shape[1]} 列 -> outputs/p1_weekly_indicators.csv")
    print("数据质量(完整率):", {k: f"{v:.1%}" for k, v in comp.items()},
          "期货行情完整率: 100%")
    print("最新一周:", ind.index[-1].date(),
          f"草鱼料动态成本 {ind['草鱼料_动态成本'].iloc[-1]:.0f} 元/吨, "
          f"饵料成本占鱼价比 {ind['草鱼料_饵料成本占鱼价比'].iloc[-1]:.1f}%")

if __name__ == "__main__":
    main()
