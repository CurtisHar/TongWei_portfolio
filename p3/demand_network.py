# -*- coding: utf-8 -*-
"""
P3 - 需求测算 + 运输网络成本 + 现状诊断
数据逻辑：
- 年需求 = 水产养殖产量 × 饲料系数 × 配合饲料使用率 × 通威服务份额（鲢鳙滤食性鱼不喂料，已在使用率口径中剔除）
  四川养殖产量按 170 万吨（A-DEM-02，待年鉴校准），FCR 1.8，使用率 55%，份额 30%
- 月度分配：用成都2025年真实日气温（Open-Meteo）驱动投喂强度曲线
- 距离：球面直线距离 × 绕行系数1.25（A-TRC-02；坐标WGS84与GCJ-02偏移问题在本口径下不混用）
- 运输成本：自下而上 车公里成本 = 油费+过路费+司机+折旧保险，除以实际装载，考虑回程空车
输出: fact_demand_monthly, outputs/p3_demand_monthly.csv, p3_distance_cost.csv
      figures/p3_demand_curve.png, p3_service_cost_map.png
"""
import os, sys, json
import numpy as np
import pandas as pd
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

# ---------- 1. 年需求 ----------
TOTAL_AQUA_T = 1_700_000     # 四川水产养殖产量(吨/年, 待年鉴校准)
FCR = 1.8                    # 综合饲料系数(草鱼口径)
USAGE = 0.55                 # 配合饲料使用率(已剔除鲢鳙等滤食性鱼)
SHARE = 0.30                 # 通威服务份额假设
total_feed_t = TOTAL_AQUA_T * FCR * USAGE * SHARE      # ≈ 50.5 万吨
regions["annual_demand_t"] = regions.aqua_weight / regions.aqua_weight.sum() * total_feed_t
PRODUCTS = {"颗粒料": 0.70, "膨化料": 0.30}

# ---------- 2. 月度分布：气温驱动投喂 ----------
temp = pd.read_csv(os.path.join(BASE, "data", "chengdu_temp_2025.csv"), parse_dates=["date"])
def feed_intensity(t):
    """投喂强度曲线(文献口径: <12℃基本停喂, 12-28℃线性上升, 28-32℃平台, >33℃略降)"""
    t = np.asarray(t, float)
    i = np.clip((t - 12) / 16, 0, 1)
    i = np.where(t > 33, i * 0.9, i)
    return i
temp["intensity"] = feed_intensity(temp.t_mean)
temp["month"] = temp.date.dt.month
monthly_w = temp.groupby("month").intensity.mean()
monthly_w = monthly_w / monthly_w.sum()
print("月度需求分布(%):", (monthly_w * 100).round(1).to_dict())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].plot(temp.date, temp.t_mean, lw=0.8, color="#2E6FBA", label="日均气温")
axes[0].plot(temp.date, temp.intensity * temp.t_mean.max(), lw=1.2, color="#E8813A",
             label="投喂强度(缩放)")
axes[0].legend(); axes[0].set_title("成都2025年日气温与投喂强度(Open-Meteo)"); axes[0].grid(alpha=0.3)
axes[1].bar(monthly_w.index, monthly_w * 100, color="#5BA55B")
axes[1].set_title("水产料月度需求分布(气温驱动)"); axes[1].set_xlabel("月"); axes[1].set_ylabel("%")
axes[1].grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p3_demand_curve.png"), dpi=150); plt.close()

# 需求表：region × month × product × scenario
scenarios = {"基准": 1.0, "需求+15%": 1.15}
rows = []
for _, r in regions.iterrows():
    for m in range(1, 13):
        for p, share in PRODUCTS.items():
            for sc, mult in scenarios.items():
                rows.append((r.region_code, 2026, m,
                             round(r.annual_demand_t * monthly_w[m] * share * mult, 1),
                             p, sc, "SRC05"))
dem = pd.DataFrame(rows, columns=["region_code", "year", "month", "demand_t",
                                  "product", "scenario", "source_key"])
con.execute("DELETE FROM fact_demand_monthly")
dem.to_sql("fact_demand_monthly", con, if_exists="append", index=False)
dem.to_csv(os.path.join(BASE, "outputs", "p3_demand_monthly.csv"),
           index=False, encoding="utf-8-sig")

# ---------- 3. 距离矩阵 + 运输成本 ----------
def haversine(lon1, lat1, lon2, lat2):
    Rr = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * Rr * np.arcsin(np.sqrt(a))

DETOUR = 1.25   # 绕行系数(A-TRC-02)
# 运输成本参数(A-TRC-01, 本人物流经验+公开规则)
DIESEL = 7.5          # 元/升
L_PER_100KM = 30      # 13米重卡百公里油耗
TOLL = 2.0            # 过路费 元/公里
DRIVER = 0.8          # 司机 元/公里
DEPR = 0.7            # 折旧保险 元/公里
LOAD_T = 30           # 额定载重(吨, GB 1589六轴49吨总重→净载约30吨)
EMPTY_BACK = 0.5      # 回程空车率
COST_PER_KM = DIESEL * L_PER_100KM / 100 + TOLL + DRIVER + DEPR     # ≈5.75 元/公里
COST_T_KM = COST_PER_KM * (1 + EMPTY_BACK) / LOAD_T                  # ≈0.29 元/吨·公里
print(f"车公里成本 {COST_PER_KM:.2f} 元/公里, 吨公里成本 {COST_T_KM:.3f} 元/吨·公里")

dist = np.zeros((len(factories), len(regions)))
for i, f in factories.iterrows():
    for j, r in regions.iterrows():
        dist[i, j] = haversine(f.lon_wgs84, f.lat_wgs84, r.lon_wgs84, r.lat_wgs84) * DETOUR
cost_t = dist * COST_T_KM
dist_df = pd.DataFrame(dist, index=factories.factory_key, columns=regions.region_code)
cost_df = pd.DataFrame(cost_t, index=factories.factory_key, columns=regions.region_code)
cost_df.to_csv(os.path.join(BASE, "outputs", "p3_distance_cost.csv"), encoding="utf-8-sig")

# ---------- 4. 现状诊断：就近供货 + 每吨服务成本 ----------
assign = dist.argmin(axis=0)          # 每个需求点的最近工厂
regions["nearest_factory"] = factories.factory_key.values[assign]
regions["service_cost_t"] = cost_t[assign, np.arange(len(regions))].round(1)
wavg_dist = float((dist[assign, np.arange(len(regions))] *
                   regions.annual_demand_t).sum() / regions.annual_demand_t.sum())
regions.to_csv(os.path.join(BASE, "outputs", "p3_regions.csv"),
               index=False, encoding="utf-8-sig")
avg_cost = float((regions.service_cost_t * regions.annual_demand_t).sum()
                 / regions.annual_demand_t.sum())
print(f"就近供货下: 加权平均配送距离 {wavg_dist:.0f} 公里, 平均服务成本 {avg_cost:.1f} 元/吨")
top = regions.nlargest(3, "service_cost_t")[["city", "nearest_factory", "service_cost_t"]]
print("服务成本最高:", top.to_string(index=False),
      f"(全区平均的 {top.service_cost_t.iloc[0]/avg_cost:.1f} 倍)")

fig, ax = plt.subplots(figsize=(9, 8))
sc = ax.scatter(regions.lon_wgs84, regions.lat_wgs84, s=regions.annual_demand_t / 300,
                c=regions.service_cost_t, cmap="RdYlGn_r", edgecolor="k", zorder=5)
for _, r in regions.iterrows():
    ax.annotate(f"{r.city}\n{r.service_cost_t:.0f}", (r.lon_wgs84, r.lat_wgs84),
                fontsize=7, ha="center", va="bottom")
for _, f in factories.iterrows():
    ax.scatter(f.lon_wgs84, f.lat_wgs84, marker="*", s=400, c="#2E6FBA",
               edgecolor="k", zorder=6)
    ax.annotate(f.factory_name, (f.lon_wgs84, f.lat_wgs84), fontsize=8,
                ha="center", va="top", fontweight="bold")
plt.colorbar(sc, label="每吨服务成本(元/吨)")
ax.set_title("每吨服务成本地图（气泡=年需求规模, 星=工厂[示例位置]）")
ax.set_xlabel("经度"); ax.set_ylabel("纬度")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p3_service_cost_map.png"), dpi=150); plt.close()

params = {"total_feed_t": round(total_feed_t), "cost_per_km": round(COST_PER_KM, 2),
          "cost_t_km": round(COST_T_KM, 3), "wavg_dist_km": round(wavg_dist, 1),
          "avg_service_cost": round(avg_cost, 1),
          "monthly_dist_pct": (monthly_w * 100).round(2).to_dict()}
with open(os.path.join(BASE, "outputs", "p3_network_params.json"), "w", encoding="utf-8") as f:
    json.dump(params, f, ensure_ascii=False, indent=1)
con.commit(); con.close()
print("OK ->", json.dumps(params, ensure_ascii=False)[:200])
