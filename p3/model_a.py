# -*- coding: utf-8 -*-
"""
P3 模型A - 多月份生产、库存、运输统筹（线性规划, PuLP）
四种策略：
  ① 按订单生产，就近供货（基准：不允许提前生产/调货）
  ② 允许淡季提前生产
  ③ 允许工厂之间调货
  ④ 提前生产 + 调货都允许
情景（在④上叠加）：柴油+10% / 成都厂8月停产两周 / 需求+15%
约束：产线月产能、仓库容量、保质期(库存≤未来2个月需求)、需求尽量满足(缺货惩罚)
输出：outputs/p3_model_a_results.json, p3_strategy_compare.csv
      figures/p3_model_a.png, fact_shipment_plan / fact_inventory_plan
"""
import os, sys, json
import numpy as np
import pandas as pd
import pulp
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
BASE = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(BASE, "outputs", "figures")
DB = os.path.join(BASE, "data", "supply_chain.db")

con = sqlite3.connect(DB)
regions = pd.read_sql("SELECT * FROM dim_region", con)
factories = pd.read_sql("SELECT * FROM dim_factory", con)
dem = pd.read_sql("SELECT * FROM fact_demand_monthly WHERE scenario='基准'", con)
dem15 = pd.read_sql("SELECT * FROM fact_demand_monthly WHERE scenario='需求+15%'", con)
cost_df = pd.read_csv(os.path.join(BASE, "outputs", "p3_distance_cost.csv"),
                      index_col=0)

MONTHS = list(range(1, 13))
FKEYS = list(factories.factory_key)
RCODES = list(regions.region_code)
PRODUCTS = ["颗粒料", "膨化料"]
CAP = {f: {"颗粒料": factories.set_index("factory_key").loc[f, "cap_pellet_t_year"] / 12,
           "膨化料": factories.set_index("factory_key").loc[f, "cap_extruded_t_year"] / 12}
       for f in FKEYS}
WH = factories.set_index("factory_key").warehouse_cap_t.to_dict()
PROD_COST = 60.0        # 加工成本 元/吨(统一,差异不是重点)
HOLD_COST = 42.5        # 库存持有 元/吨/月(A-HLD-01: 资金年化6%+仓储1元/吨/天)
SHORT_PEN = 500.0       # 缺货惩罚 元/吨(丢失毛利的机会成本代理)
SHELF_K = 2             # 保质期: 库存不超过未来2个月需求(A-SHL-01)
COST_T_KM = 0.287       # 元/吨·公里

D = {(r, p, m): 0.0 for r in RCODES for p in PRODUCTS for m in MONTHS}
for _, row in dem.iterrows():
    D[(row.region_code, row["product"], int(row.month))] = row.demand_t
D15 = {(r, p, m): 0.0 for r in RCODES for p in PRODUCTS for m in MONTHS}
for _, row in dem15.iterrows():
    D15[(row.region_code, row["product"], int(row.month))] = row.demand_t

# 工厂间调货距离成本
def haversine(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))
TR_COST = {}
for f in FKEYS:
    for g in FKEYS:
        if f != g:
            ff = factories.set_index("factory_key")
            TR_COST[(f, g)] = haversine(*ff.loc[f, ["lon_wgs84", "lat_wgs84"]],
                                        *ff.loc[g, ["lon_wgs84", "lat_wgs84"]]) * 1.25 * COST_T_KM

def solve_strategy(name, allow_preprod, allow_transfer, demand=D,
                   cost_mult=1.0, cap_override=None):
    prob = pulp.LpProblem(name, pulp.LpMinimize)
    P = pulp.LpVariable.dicts("prod", (FKEYS, PRODUCTS, MONTHS), lowBound=0)
    I = pulp.LpVariable.dicts("inv", (FKEYS, PRODUCTS, MONTHS), lowBound=0)
    S = pulp.LpVariable.dicts("ship", (FKEYS, RCODES, PRODUCTS, MONTHS), lowBound=0)
    T = pulp.LpVariable.dicts("trans", (FKEYS, FKEYS, PRODUCTS, MONTHS), lowBound=0)
    U = pulp.LpVariable.dicts("short", (RCODES, PRODUCTS, MONTHS), lowBound=0)

    prob += (pulp.lpSum(PROD_COST * P[f][p][m] for f in FKEYS for p in PRODUCTS for m in MONTHS)
             + pulp.lpSum(HOLD_COST * I[f][p][m] for f in FKEYS for p in PRODUCTS for m in MONTHS)
             + pulp.lpSum(cost_df.loc[f, r] * cost_mult * S[f][r][p][m]
                          for f in FKEYS for r in RCODES for p in PRODUCTS for m in MONTHS)
             + pulp.lpSum(TR_COST.get((f, g), 0) * T[f][g][p][m]
                          for f in FKEYS for g in FKEYS if f != g
                          for p in PRODUCTS for m in MONTHS)
             + pulp.lpSum(SHORT_PEN * U[r][p][m] for r in RCODES for p in PRODUCTS for m in MONTHS))

    cap_cons = {}
    for f in FKEYS:
        for p in PRODUCTS:
            for m in MONTHS:
                cap = CAP[f][p]
                if cap_override and (f, p, m) in cap_override:
                    cap = cap_override[(f, p, m)]
                c = P[f][p][m] <= cap
                prob += c, f"cap_{f}_{p}_{m}"
                cap_cons[(f, p, m)] = c
                prob += I[f][p][m] <= WH[f]
                # 保质期: 库存不超过未来k个月该产品总需求
                fut = sum(demand[(r, p, mm)] for r in RCODES
                          for mm in range(m + 1, min(m + SHELF_K, 12) + 1))
                prob += I[f][p][m] <= fut
                # 流量平衡
                prev = I[f][p][m - 1] if m > 1 else 0
                prob += (prev + P[f][p][m]
                         + pulp.lpSum(T[g][f][p][m] for g in FKEYS if g != f)
                         - pulp.lpSum(T[f][g][p][m] for g in FKEYS if g != f)
                         - pulp.lpSum(S[f][r][p][m] for r in RCODES)
                         == I[f][p][m])
                if not allow_preprod:
                    prob += I[f][p][m] == 0
                if not allow_transfer:
                    for g in FKEYS:
                        if g != f:
                            prob += T[f][g][p][m] == 0
    for r in RCODES:
        for p in PRODUCTS:
            for m in MONTHS:
                prob += (pulp.lpSum(S[f][r][p][m] for f in FKEYS) + U[r][p][m]
                         == demand[(r, p, m)])

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return {"strategy": name, "status": status}
    total = pulp.value(prob.objective)
    short = sum(U[r][p][m].value() for r in RCODES for p in PRODUCTS for m in MONTHS)
    prod_m = {m: sum(P[f][p][m].value() for f in FKEYS for p in PRODUCTS) for m in MONTHS}
    inv_m = {m: sum(I[f][p][m].value() for f in FKEYS for p in PRODUCTS) for m in MONTHS}
    ship_m = {m: sum(S[f][r][p][m].value() for f in FKEYS for r in RCODES
                     for p in PRODUCTS) for m in MONTHS}
    trans_total = sum(T[f][g][p][m].value() for f in FKEYS for g in FKEYS if f != g
                      for p in PRODUCTS for m in MONTHS)
    # 库存周转天数 = 平均库存 ÷ 日均出库(出库≈发运)
    avg_inv = np.mean(list(inv_m.values()))
    daily_out = sum(ship_m.values()) / 365
    turn_days = avg_inv / daily_out if daily_out > 0 else 0
    # 产能利用率(按月)
    cap_m_total = sum(CAP[f][p] for f in FKEYS for p in PRODUCTS)
    util_m = {m: prod_m[m] / cap_m_total for m in MONTHS}
    # 旺季产能影子价格(仅报告8月各厂各产线)
    shadow_cap = {}
    if allow_preprod and allow_transfer:
        for (f, p, m), c in cap_cons.items():
            if m == 8 and c.pi is not None and abs(c.pi) > 1e-6:
                shadow_cap[f"{f}_{p}"] = round(abs(c.pi), 1)
    detail = {"prod_m": prod_m, "inv_m": inv_m, "ship_m": ship_m}
    plan = None
    if name.startswith("④"):
        ship_rows = [(name, m, f, r, p, round(S[f][r][p][m].value(), 1),
                      round(S[f][r][p][m].value() * cost_df.loc[f, r] * cost_mult, 1))
                     for f in FKEYS for r in RCODES for p in PRODUCTS for m in MONTHS
                     if S[f][r][p][m].value() > 0.01]
        inv_rows = [(name, m, f, p, round(I[f][p][m].value(), 1),
                     round(P[f][p][m].value() / CAP[f][p] * 100, 1))
                    for f in FKEYS for p in PRODUCTS for m in MONTHS]
        plan = (ship_rows, inv_rows)
    return {"strategy": name, "status": status, "total_cost_wy": round(total / 1e4, 1),
            "shortage_t": round(short, 0), "transfer_t": round(trans_total, 0),
            "turnover_days": round(turn_days, 1),
            "peak_util": round(max(util_m.values()) * 100, 1),
            "util_m": {m: round(u * 100, 1) for m, u in util_m.items()},
            "shadow_cap_aug": shadow_cap, "detail": detail, "plan": plan}

# ---------- 四策略 ----------
strategies = [
    ("①按订单生产+就近供货", False, False),
    ("②允许提前生产", True, False),
    ("③允许工厂间调货", False, True),
    ("④提前生产+调货", True, True),
]
results = []
for name, pre, tr in strategies:
    r = solve_strategy(name, pre, tr)
    results.append(r)
    print(f"{name}: 总成本 {r['total_cost_wy']} 万元, 缺货 {r['shortage_t']:.0f} 吨, "
          f"调货 {r['transfer_t']:.0f} 吨, 周转 {r['turnover_days']} 天, 峰值利用率 {r['peak_util']}%")

# ---------- 情景（叠加在④上） ----------
scen = []
cap_half = {(f, p, 8): CAP[f][p] * 0.5 for p in PRODUCTS for f in ["F01"]}  # 成都8月停产两周
for sname, kw in [("基准(④)", {}),
                  ("柴油+10%", {"cost_mult": 1.039}),   # 油费占车公里成本39%
                  ("成都厂8月停产两周", {"cap_override": cap_half}),
                  ("需求+15%", {"demand": D15})]:
    r = solve_strategy(sname, True, True, **kw)
    scen.append(r)
    print(f"情景 {sname}: 总成本 {r['total_cost_wy']} 万元, 缺货 {r['shortage_t']:.0f} 吨")

# ---------- 对比表与图 ----------
cmp_rows = []
for r in results:
    cmp_rows.append({k: r[k] for k in ["strategy", "total_cost_wy", "shortage_t",
                                       "transfer_t", "turnover_days", "peak_util"]})
cmp_df = pd.DataFrame(cmp_rows)
cmp_df["成本节约_vs基准"] = (cmp_df.total_cost_wy[0] - cmp_df.total_cost_wy).round(1)
cmp_df["缺货下降_vs基准"] = (cmp_df.shortage_t[0] - cmp_df.shortage_t).round(0)
cmp_df.to_csv(os.path.join(BASE, "outputs", "p3_strategy_compare.csv"),
              index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
dem_m = [sum(D[(r, p, m)] for r in RCODES for p in PRODUCTS) for m in MONTHS]
for ax, r in zip(axes.ravel(), results):
    d = r["detail"]
    ax.bar(MONTHS, dem_m, color="#dddddd", label="需求")
    ax.plot(MONTHS, [d["prod_m"][m] for m in MONTHS], "o-", color="#2E6FBA", label="产量")
    ax.plot(MONTHS, [d["inv_m"][m] for m in MONTHS], "s-", color="#E8813A", label="月末库存")
    ax.plot(MONTHS, [d["ship_m"][m] for m in MONTHS], "^-", color="#5BA55B", label="发运量")
    ax.set_title(f"{r['strategy']}  成本{r['total_cost_wy']}万 缺货{r['shortage_t']:.0f}吨")
    ax.set_xlabel("月"); ax.set_ylabel("吨"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p3_model_a.png"), dpi=150); plt.close()

# ---------- 写 fact 表 ----------
con.execute("DELETE FROM fact_shipment_plan")
con.execute("DELETE FROM fact_inventory_plan")
plan4 = results[3]["plan"]
if plan4:
    pd.DataFrame(plan4[0], columns=["strategy", "month", "factory_key", "region_code",
                                    "product", "qty_t", "cost"]).to_sql(
        "fact_shipment_plan", con, if_exists="append", index=False)
    pd.DataFrame(plan4[1], columns=["strategy", "month", "factory_key", "product",
                                    "end_inventory_t", "capacity_util"]).to_sql(
        "fact_inventory_plan", con, if_exists="append", index=False)
con.commit(); con.close()

out = {"strategies": [{k: v for k, v in r.items() if k not in ("detail", "plan")} for r in results],
       "scenarios": [{k: v for k, v in r.items() if k not in ("detail", "plan")} for r in scen],
       "params": {"PROD_COST": PROD_COST, "HOLD_COST": HOLD_COST,
                  "SHORT_PEN": SHORT_PEN, "SHELF_K": SHELF_K}}
with open(os.path.join(BASE, "outputs", "p3_model_a_results.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("影子价格(8月产能):", results[3]["shadow_cap_aug"])
