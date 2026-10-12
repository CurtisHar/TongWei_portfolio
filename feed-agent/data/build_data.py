# -*- coding: utf-8 -*-
"""
D1 - 生成虚构饲料公司数据（24个月），复用 portfolio 的真实期货价格。
输出: data/company.duckdb + data/buried_problems.json（埋点清单=评测标准答案）

埋点（每个都能用 SQL 独立查出）：
  BP1 成都厂2025-08草鱼料出成率 96%→91%（当月玉米水分偏高批次）
  BP2 供应商S03 2025Q3 豆粕蛋白 43%→41.2%
  BP3 华东区域 2025H2 回款天数 45→78 天
  BP4 眉山厂2025-03 在豆粕价格高点集中采购（量为月均3倍）
  BP5 重庆区域 2025-09 OTIF 0.95→0.82（旺季运力）
  BP6 鲈鱼膨化料2025-07 吨成本大涨（鱼粉采购价高点,价格效应主导）
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PORTFOLIO = os.path.join(ROOT, "..", "portfolio")
sys.path.insert(0, os.path.join(PORTFOLIO, "common"))
rng = np.random.default_rng(20260815)

MONTHS = pd.period_range("2024-01", "2025-12", freq="M").astype(str).tolist()

# ---------- 真实期货月度价（豆粕/菜粕/玉米主力） ----------
def real_monthly_prices():
    frames = []
    for f in glob.glob(os.path.join(PORTFOLIO, "data", "raw_futures", "*.csv")):
        df = pd.read_csv(f, dtype={"trade_date": str})
        df["symbol"] = df.ts_code.str.extract(r"^([A-Z]+)")
        frames.append(df)
    fut = pd.concat(frames)
    fut["date"] = pd.to_datetime(fut.trade_date, format="%Y%m%d")
    fut = fut[fut.symbol.isin(["M", "C", "RM"])]
    idx = fut.groupby(["symbol", "date"]).oi.idxmax()
    main = fut.loc[idx].copy()
    main["month"] = main.date.dt.strftime("%Y-%m")
    pm = main.groupby(["symbol", "month"]).settle.mean().unstack("symbol")
    pm = pm.reindex(MONTHS).ffill().bfill()
    return pm

PM = real_monthly_prices()

def price_series(base_sym=None, ratio_to=None, ar_center=None, ar_sd=None, clip=None):
    """生成原料月度采购基准价：真实期货价 或 联动价 或 AR 模拟价"""
    if base_sym:
        s = PM[base_sym].copy()
    elif ratio_to:
        s = PM[ratio_to[0]] * ratio_to[1]
    else:
        n = len(MONTHS); v = np.zeros(n); v[0] = ar_center
        eps = rng.normal(0, ar_sd, n)
        for i in range(1, n):
            v[i] = ar_center + 0.9 * (v[i - 1] - ar_center) + eps[i]
        s = pd.Series(np.clip(v, *clip), index=MONTHS)
    return s

ING_PRICE = {
    "豆粕": price_series("M"), "菜粕": price_series("RM"), "玉米": price_series("C"),
    "棉粕": price_series(ratio_to=("M", 0.85)), "小麦": price_series(ratio_to=("C", 1.05)),
    "米糠": price_series(ratio_to=("C", 0.75)),
    "鱼粉": price_series(ar_center=12500, ar_sd=400, clip=(10000, 16000)),
    "豆油": price_series(ar_center=8200, ar_sd=250, clip=(7000, 9800)),
    "磷酸二氢钙": price_series(ar_center=3600, ar_sd=60, clip=(3300, 3900)),
    "赖氨酸": price_series(ar_center=11000, ar_sd=300, clip=(9500, 13000)),
}
# BP6: 2025-07 鱼粉价格冲到高点
ING_PRICE["鱼粉"]["2025-07"] = ING_PRICE["鱼粉"].max() * 1.04

# ---------- 配方 LP（按当月价格每月重算，真实企业运作方式） ----------
from formulation import solve_feed, FEED_SPECS, BASE_PRICES  # 复用草鱼料/加州鲈料

# 增补两个料种规格（营养标准来源同 formulation.py 注释，编号待核）
FEED_SPECS["育肥猪料"] = {
    "nutrient_min": {"CP": 16.0, "Lys": 0.95, "Met": 0.30, "P": 0.50},
    "nutrient_max": {"CF": 5.0, "EE": 8.0},
    "bounds": {"豆粕": (0.05, 0.30), "菜粕": (0.0, 0.08), "棉粕": (0.0, 0.06),
               "玉米": (0.40, 0.75), "小麦": (0.0, 0.25), "米糠": (0.0, 0.10),
               "鱼粉": (0.0, 0.03), "豆油": (0.0, 0.03),
               "磷酸二氢钙": (0.008, 0.020), "赖氨酸": (0.001, 0.006),
               "蛋氨酸": (0.0, 0.003), "预混料": (0.010, 0.010)},
    "starch_min": 0.0}
FEED_SPECS["对虾料"] = {
    "nutrient_min": {"CP": 40.0, "EE": 6.0, "Lys": 2.4, "Met": 0.9, "P": 1.2},
    "nutrient_max": {"CF": 4.0},
    "bounds": {"豆粕": (0.0, 0.30), "菜粕": (0.0, 0.12), "棉粕": (0.0, 0.0),
               "玉米": (0.0, 0.0), "小麦": (0.10, 0.18), "米糠": (0.0, 0.0),
               "鱼粉": (0.25, 0.45), "豆油": (0.02, 0.06),
               "磷酸二氢钙": (0.010, 0.020), "赖氨酸": (0.0, 0.008),
               "蛋氨酸": (0.0, 0.008), "预混料": (0.010, 0.010)},
    "starch_min": 0.10}
PRODUCT_MAP = {"草鱼料": "草鱼料", "鲈鱼膨化料": "加州鲈料",
               "育肥猪料": "育肥猪料", "对虾料": "对虾料"}

FACTORIES = ["成都厂", "眉山厂", "绵阳厂"]
PRODUCTS = list(PRODUCT_MAP)
SUPPLIERS = [f"S0{i}" for i in range(1, 7)]
REGIONS = ["成都周边", "川南", "川东北", "重庆", "华东", "华中"]

formula_rows, fid = [], 0
for m in MONTHS:
    prices = {ing: float(s[m]) for ing, s in ING_PRICE.items()}
    for prod, spec in PRODUCT_MAP.items():
        r = solve_feed(spec, prices)
        if r["status"] != "Optimal":
            raise RuntimeError(f"配方无解 {spec} {m}")
        for ing, pct in r["formula"].items():
            if pct > 0.001:
                formula_rows.append((m, prod, ing, round(float(pct), 3)))
formula_df = pd.DataFrame(formula_rows, columns=["month", "product", "ingredient", "pct"])

# ---------- 维度表 ----------
dim_factory = pd.DataFrame({"factory": FACTORIES})
dim_product = pd.DataFrame({"product": PRODUCTS})
dim_ingredient = pd.DataFrame({"ingredient": list(ING_PRICE)})
dim_supplier = pd.DataFrame({"supplier": SUPPLIERS})
dim_customer = pd.DataFrame({
    "customer": [f"C{i:03d}" for i in range(1, 201)],
    "region": rng.choice(REGIONS, 200, p=[0.25, 0.2, 0.2, 0.15, 0.12, 0.08])})

# ---------- 采购（BP2蛋白偏低/BP4高点集中采购） ----------
pur_rows, pid = [], 0
for mi, m in enumerate(MONTHS):
    for fac in FACTORIES:
        for ing, s in ING_PRICE.items():
            n_lots = rng.integers(1, 3)
            for _ in range(n_lots):
                sup = str(rng.choice(SUPPLIERS))
                qty = float(rng.uniform(80, 300))
                # BP4: 眉山 2025-03 豆粕集中采购（豆粕该期处于区间高位）
                if fac == "眉山厂" and m == "2025-03" and ing == "豆粕":
                    qty *= 3.0
                price = float(s[m]) * (1 + rng.normal(0, 0.012)) + 60  # 到厂价=期货+基差
                pur_rows.append((f"PO{pid:05d}", m, fac, ing, sup, round(qty, 1),
                                 round(price, 0)))
                pid += 1
purchase_df = pd.DataFrame(pur_rows, columns=[
    "purchase_id", "month", "factory", "ingredient", "supplier", "qty_t", "unit_price"])

# ---------- 质检批次（营养值按原料×供应商设均值与波动） ----------
PROT_MEAN = {"豆粕": 43.0, "菜粕": 36.0, "棉粕": 40.0, "玉米": 8.5, "小麦": 12.5,
             "米糠": 12.8, "鱼粉": 62.0, "豆油": 0.0, "磷酸二氢钙": 0.0, "赖氨酸": 0.0}
qc_rows, qid = [], 0
for _, po in purchase_df.iterrows():
    if po.ingredient in ("豆油", "磷酸二氢钙", "赖氨酸"):
        continue
    mu = PROT_MEAN[po.ingredient]
    # BP2: S03 2025Q3 豆粕蛋白偏低
    if po.supplier == "S03" and po.ingredient == "豆粕" and po.month in ("2025-07", "2025-08", "2025-09"):
        mu = 41.2
    protein = rng.normal(mu, 0.45)
    moisture = rng.normal(12.5, 0.6)
    # BP1 前因: 成都 2025-08 玉米水分偏高
    if po.factory == "成都厂" and po.month == "2025-08" and po.ingredient == "玉米":
        moisture = rng.normal(15.5, 0.4)
    qc_rows.append((f"QC{qid:05d}", po.month, po.factory, po.ingredient, po.supplier,
                    round(protein, 2), round(moisture, 2),
                    int(protein > mu - 1.2)))
    qid += 1
qc_df = pd.DataFrame(qc_rows, columns=["batch_id", "month", "factory", "ingredient",
                                       "supplier", "protein_pct", "moisture_pct", "pass_flag"])

# ---------- 生产工单（BP1 出成率埋点） ----------
YIELD_BASE = {"草鱼料": 0.960, "鲈鱼膨化料": 0.955, "育肥猪料": 0.970, "对虾料": 0.950}
prod_rows, wid = [], 0
for m in MONTHS:
    for fac in FACTORIES:
        for prod in PRODUCTS:
            input_t = float(rng.uniform(700, 1300))
            y = YIELD_BASE[prod] + rng.normal(0, 0.004)
            # BP1: 成都厂 2025-08 草鱼料出成率异常下降（水分损耗）
            if fac == "成都厂" and m == "2025-08" and prod == "草鱼料":
                y = 0.910
            prod_rows.append((f"WO{wid:05d}", m, fac, prod, round(input_t, 1),
                              round(input_t * y, 1)))
            wid += 1
production_df = pd.DataFrame(prod_rows, columns=[
    "order_id", "month", "factory", "product", "input_t", "output_t"])

# ---------- 日库存（原料+成品，数量与金额） ----------
inv_rows = []
days = pd.date_range("2024-01-01", "2025-12-31", freq="D")
for fac in FACTORIES:
    for item_type, items in [("原料", ["豆粕", "菜粕", "玉米"]), ("成品", PRODUCTS)]:
        for item in items:
            qty = float(rng.uniform(300, 600))
            for d in days:
                m = str(d.to_period("M"))
                qty = max(50, qty + rng.normal(0, 18))
                price = float(ING_PRICE[item][m]) if item in ING_PRICE else 3000.0
                inv_rows.append((str(d.date()), fac, item_type, item,
                                 round(qty, 1), round(qty * price / 1000, 1)))
inv_df = pd.DataFrame(inv_rows, columns=["day", "factory", "item_type", "item",
                                         "qty_t", "amt_ky"])

# ---------- 销售（BP3 回款、BP5 OTIF） ----------
SALE_PRICE = {"草鱼料": 3300, "鲈鱼膨化料": 7800, "育肥猪料": 3600, "对虾料": 8500}
sale_rows, sid = [], 0
for m in MONTHS:
    for fac in FACTORIES:
        n_orders = int(rng.integers(35, 60))
        for _ in range(n_orders):
            cust = dim_customer.iloc[int(rng.integers(0, 200))]
            prod = str(rng.choice(PRODUCTS, p=[0.4, 0.2, 0.3, 0.1]))
            qty = float(rng.uniform(3, 30))
            dso = float(rng.normal(45, 8))
            # BP3: 华东 2025H2 回款变长
            if cust.region == "华东" and m >= "2025-07":
                dso = float(rng.normal(78, 10))
            on_time = rng.random() < 0.97
            in_full = rng.random() < 0.98
            # BP5: 重庆 2025-09 旺季 OTIF 下降
            if cust.region == "重庆" and m == "2025-09":
                on_time = rng.random() < 0.85
                in_full = rng.random() < 0.96
            sale_rows.append((f"SO{sid:05d}", m, fac, prod, cust.customer, cust.region,
                              round(qty, 1), round(qty * SALE_PRICE[prod] * (1 + rng.normal(0, 0.02)), 0),
                              int(on_time), int(in_full), round(dso, 0)))
            sid += 1
sales_df = pd.DataFrame(sale_rows, columns=[
    "sale_id", "month", "factory", "product", "customer", "region",
    "qty_t", "amount_yuan", "on_time", "in_full", "dso_days"])

# ---------- 期货月均价（供偏离度指标） ----------
fut_rows = [(m, sym, float(PM[sym][m])) for m in MONTHS for sym in ["M", "C", "RM"]]
fut_df = pd.DataFrame(fut_rows, columns=["month", "symbol", "settle_avg"])

# ---------- 入库 ----------
db_path = os.path.join(HERE, "company.duckdb")
if os.path.exists(db_path):
    os.remove(db_path)
con = duckdb.connect(db_path)
for name, df in [("dim_factory", dim_factory), ("dim_product", dim_product),
                 ("dim_ingredient", dim_ingredient), ("dim_supplier", dim_supplier),
                 ("dim_customer", dim_customer), ("fact_purchase", purchase_df),
                 ("fact_qc_batch", qc_df), ("fact_formula", formula_df),
                 ("fact_production", production_df), ("fact_inventory_day", inv_df),
                 ("fact_sales", sales_df), ("fact_futures_monthly", fut_df)]:
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
    print(f"{name}: {len(df)} rows")
con.close()

buried = {
    "BP1": {"desc": "成都厂2025-08草鱼料出成率降至91%（基准约96%），当月在厂玉米批次水分偏高(约15.5% vs 正常12.5%)",
            "check_sql": "SELECT factory, product, month, ROUND(SUM(output_t)/SUM(input_t)*100,2) AS yield_pct FROM fact_production WHERE factory='成都厂' AND product='草鱼料' AND month IN ('2025-07','2025-08','2025-09') GROUP BY 1,2,3"},
    "BP2": {"desc": "供应商S03在2025Q3的豆粕蛋白均值41.2%，低于正常43%",
            "check_sql": "SELECT supplier, ROUND(AVG(protein_pct),2) FROM fact_qc_batch WHERE ingredient='豆粕' AND month BETWEEN '2025-07' AND '2025-09' GROUP BY supplier ORDER BY 2"},
    "BP3": {"desc": "华东区域2025H2回款天数从约45天升至约78天",
            "check_sql": "SELECT region, CASE WHEN month<'2025-07' THEN 'H1' ELSE 'H2' END AS half, ROUND(AVG(dso_days),1) FROM fact_sales GROUP BY 1,2"},
    "BP4": {"desc": "眉山厂2025-03豆粕采购量约为月均3倍，且当月豆粕价处区间高位",
            "check_sql": "SELECT month, SUM(qty_t) FROM fact_purchase WHERE factory='眉山厂' AND ingredient='豆粕' GROUP BY month ORDER BY 2 DESC LIMIT 3"},
    "BP5": {"desc": "重庆区域2025-09 OTIF降至约0.82（旺季运力不足）",
            "check_sql": "SELECT region, month, ROUND(AVG(on_time*in_full),3) AS otif FROM fact_sales WHERE region='重庆' AND month IN ('2025-08','2025-09','2025-10') GROUP BY 1,2"},
    "BP6": {"desc": "鲈鱼膨化料2025-07吨成本环比大涨，主因鱼粉采购价处于高点（价格效应主导）",
            "check_sql": "SELECT month, ROUND(AVG(unit_price),0) FROM fact_purchase WHERE ingredient='鱼粉' GROUP BY month ORDER BY 2 DESC LIMIT 3"},
}
with open(os.path.join(HERE, "buried_problems.json"), "w", encoding="utf-8") as f:
    json.dump(buried, f, ensure_ascii=False, indent=1)
print("\n埋点清单 ->", os.path.join(HERE, "buried_problems.json"))
print("DB ->", db_path)
