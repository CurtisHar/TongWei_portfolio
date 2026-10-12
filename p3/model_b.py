# -*- coding: utf-8 -*-
"""
P3 模型B - 旺季单厂每日派车排线（模拟订单，标注 A-ORD-01）
对象: 成都双流工厂(F01) 8月旺季某一天
三种派车方式对比: ①一单一车直送 ②按片区固定线路 ③Clarke-Wright拼车优化
约束: 车辆载重30吨(GB 1589口径)、日驾时上限11小时、送货时间窗(8:00-18:00)
输出: outputs/p3_model_b_results.json, figures/p3_routes.png
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
BASE = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(BASE, "outputs", "figures")
rng = np.random.default_rng(2026)

regions = pd.read_csv(os.path.join(BASE, "outputs", "p3_regions.csv"))
dem = pd.read_csv(os.path.join(BASE, "outputs", "p3_demand_monthly.csv"))
dem = dem[(dem.scenario == "基准") & (dem.month == 8)]
F = {"lon": 103.92, "lat": 30.50}   # F01 成都双流(示例)

def haversine(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a)) * 1.25   # 绕行系数

CAP_T = 30.0          # 载重上限(吨)
SPEED = 60.0          # 平均时速(公里/小时)
UNLOAD_H = 0.6        # 每单卸货时长(小时)
MAX_DRIVE_H = 11.0    # 日驾时上限
COST_PER_KM = 5.75    # 元/公里
FIXED_PER_TRIP = 300  # 每车次固定成本(装卸排队等, 元)

# ---------- 模拟订单：F01服务圈的8月某日 ----------
svc = regions[regions.nearest_factory == "F01"].copy()
day_dem = dem[dem.region_code.isin(svc.region_code)].groupby("region_code").demand_t.sum() / 30
orders = []
oid = 0
for _, r in svc.iterrows():
    left = day_dem.get(r.region_code, 0.0)
    while left > 0.5:
        q = min(left, float(rng.uniform(2, 15)))
        left -= q
        orders.append({"order_id": f"O{oid:03d}", "city": r.city,
                       "lon": r.lon_wgs84 + rng.normal(0, 0.08),   # 客户在市内散布
                       "lat": r.lat_wgs84 + rng.normal(0, 0.08),
                       "qty": round(q, 1),
                       "tw_end": rng.choice([14, 16, 18])})        # 送货截止时点
        oid += 1
orders = pd.DataFrame(orders)
print(f"模拟订单 {len(orders)} 单, 共 {orders.qty.sum():.0f} 吨")
orders["dist_f"] = [haversine(F["lon"], F["lat"], o.lon, o.lat) for o in orders.itertuples()]

def route_time(route):
    """路线(订单索引列表)的行驶时长: 工厂->各点->工厂 + 卸货"""
    pts = [orders.iloc[i] for i in route]
    t, lon, lat = 0.0, F["lon"], F["lat"]
    for p in pts:
        d = haversine(lon, lat, p.lon, p.lat)
        t += d / SPEED + UNLOAD_H
        lon, lat = p.lon, p.lat
    t += haversine(lon, lat, F["lon"], F["lat"]) / SPEED
    return t

def route_km(route):
    pts = [orders.iloc[i] for i in route]
    km, lon, lat = 0.0, F["lon"], F["lat"]
    for p in pts:
        km += haversine(lon, lat, p.lon, p.lat)
        lon, lat = p.lon, p.lat
    return km + haversine(lon, lat, F["lon"], F["lat"])

def ontime_rate(route):
    """按路线顺序检查时间窗(假定8:00发车)"""
    t, lon, lat = 8.0, F["lon"], F["lat"]
    ok, n = 0, 0
    for i in route:
        p = orders.iloc[i]
        t += haversine(lon, lat, p.lon, p.lat) / SPEED + UNLOAD_H
        ok += int(t <= p.tw_end); n += 1
        lon, lat = p.lon, p.lat
    return ok / n if n else 1.0

# ---------- 方式① 一单一车直送 ----------
routes_direct = [[i] for i in range(len(orders))]

# ---------- 方式② 片区固定线路：按城市分组，组内按吨数贪心装满 ----------
routes_zone = []
for city, grp in orders.groupby("city"):
    cur, load = [], 0.0
    for i in grp.sort_values("qty", ascending=False).index:
        if load + orders.loc[i, "qty"] > CAP_T and cur:
            routes_zone.append(cur); cur, load = [], 0.0
        cur.append(i); load += orders.loc[i, "qty"]
    if cur:
        routes_zone.append(cur)

# ---------- 方式③ Clarke-Wright 拼车 ----------
def clarke_wright():
    routes = {i: [i] for i in range(len(orders))}
    load_of = {i: orders.qty[i] for i in range(len(orders))}
    owner = {i: i for i in range(len(orders))}   # 订单->路线key
    sav = []
    for i in range(len(orders)):
        for j in range(i + 1, len(orders)):
            oi, oj = orders.iloc[i], orders.iloc[j]
            s = (haversine(F["lon"], F["lat"], oi.lon, oi.lat)
                 + haversine(F["lon"], F["lat"], oj.lon, oj.lat)
                 - haversine(oi.lon, oi.lat, oj.lon, oj.lat))
            sav.append((s, i, j))
    sav.sort(reverse=True)
    for s, i, j in sav:
        ri, rj = owner[i], owner[j]
        if ri == rj:
            continue
        A, B = routes[ri], routes[rj]
        # 只在两端拼接
        for cand in ([A[-1], B[0]] if False else []):
            pass
        merged = None
        if A[-1] == i and B[0] == j: merged = A + B
        elif A[0] == i and B[-1] == j: merged = B + A
        elif A[0] == i and B[0] == j: merged = A[::-1] + B
        elif A[-1] == i and B[-1] == j: merged = A + B[::-1]
        if merged is None:
            continue
        newload = load_of[ri] + load_of[rj]
        if newload > CAP_T or route_time(merged) > MAX_DRIVE_H:
            continue
        routes[ri] = merged; load_of[ri] = newload
        for k in merged:
            owner[k] = ri
        routes.pop(rj); load_of.pop(rj)
    return list(routes.values())
routes_cw = clarke_wright()

# ---------- 评估 ----------
def evaluate(routes, name):
    trips = len(routes)
    km = sum(route_km(r) for r in routes)
    tons = sum(orders.qty[r].sum() for r in routes)
    load_rate = tons / (trips * CAP_T)
    cost = km * COST_PER_KM + trips * FIXED_PER_TRIP
    ot = np.mean([ontime_rate(r) for r in routes])
    return {"方式": name, "车次": trips, "总里程km": round(km),
            "装载率%": round(load_rate * 100, 1), "吨公里成本": round(cost / tons / km, 3),
            "总成本元": round(cost), "每吨成本元": round(cost / tons, 1),
            "按时送达率%": round(ot * 100, 1)}

res = [evaluate(routes_direct, "①一单一车直送"),
       evaluate(routes_zone, "②片区固定线路"),
       evaluate(routes_cw, "③拼车优化(CW)")]
res_df = pd.DataFrame(res)
res_df.to_csv(os.path.join(BASE, "outputs", "p3_dispatch_compare.csv"),
              index=False, encoding="utf-8-sig")
print(res_df.to_string(index=False))

# 小单临界分析: 直送时每吨成本 vs 订单吨位
small = []
for q in [2, 4, 6, 8, 10, 15, 20, 25, 30]:
    d = 120.0  # 中位配送距离
    cost = 2 * d * COST_PER_KM + FIXED_PER_TRIP
    small.append({"订单吨位": q, "直送每吨成本": round(cost / q, 1)})
small_df = pd.DataFrame(small)
small_df.to_csv(os.path.join(BASE, "outputs", "p3_small_order.csv"),
                index=False, encoding="utf-8-sig")

# ---------- 路线示意图 ----------
fig, ax = plt.subplots(figsize=(9, 8))
for r in routes_cw:
    xs = [F["lon"]] + [orders.iloc[i].lon for i in r] + [F["lon"]]
    ys = [F["lat"]] + [orders.iloc[i].lat for i in r] + [F["lat"]]
    ax.plot(xs, ys, lw=0.8, alpha=0.7)
ax.scatter(orders.lon, orders.lat, s=orders.qty * 8, c="#E8813A",
           edgecolor="k", zorder=5, label="客户(气泡=吨位)")
ax.scatter(F["lon"], F["lat"], marker="*", s=500, c="#2E6FBA",
           edgecolor="k", zorder=6, label="成都双流工厂(示例)")
ax.set_title(f"拼车优化后的配送路线(共{len(routes_cw)}车次, 模拟订单)")
ax.legend(); ax.set_xlabel("经度"); ax.set_ylabel("纬度")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p3_routes.png"), dpi=150); plt.close()

base = res[0]; cw = res[2]
out = {"compare": res, "small_order": small,
       "结论": f"拼车比直送车次少{base['车次']-cw['车次']}趟"
               f"({(1-cw['车次']/base['车次'])*100:.0f}%), 每吨成本{base['每吨成本元']}→{cw['每吨成本元']}元",
       "小单临界": "直送每吨成本随吨位快速下降, 小于约6吨的订单宜拼车或客户自提"}
with open(os.path.join(BASE, "outputs", "p3_model_b_results.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(out["结论"])
print("小单临界表:", small_df.to_string(index=False))
