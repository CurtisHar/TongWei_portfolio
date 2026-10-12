# -*- coding: utf-8 -*-
"""
P1 - 分析与看板图表
1) 成本敏感度：原料±100元/吨，配方不动 vs 允许调配方
2) 比价低位后的走势统计：跌破20分位后4/8/12周，鱼价先回升还是原料先回落
3) 领先滞后：差分序列互相关（只报告相关与滞后，不推断因果）
4) 套保回测：滚动最小方差套保比例，比较0%/50%/最优/100%四种比例的成本波动
输出: outputs/p1_analysis_results.json + outputs/figures/p1_page*.png
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(BASE, "common"))
from formulation import solve_feed, sensitivity, replacement_price, BASE_PRICES

FIG = os.path.join(BASE, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)
ind = pd.read_csv(os.path.join(BASE, "outputs", "p1_weekly_indicators.csv"),
                  index_col=0, parse_dates=True)
R = {}   # 结果汇总

# ============ 1. 成本敏感度 ============
sens_rows = []
for spec in ["草鱼料", "加州鲈料"]:
    for ing in (["豆粕", "菜粕", "棉粕", "玉米"] if spec == "草鱼料"
                else ["豆粕", "菜粕", "鱼粉"]):
        s = sensitivity(spec, ingredient=ing, delta=100)
        s["feed_spec"] = spec
        if s["fixed_delta"] > 0:
            s["吸收比例"] = round((float(s["fixed_delta"]) - s["dynamic_delta"])
                                  / float(s["fixed_delta"]) * 100, 1)
        else:
            s["吸收比例"] = 0.0
        sens_rows.append(s)
sens = pd.DataFrame(sens_rows)
sens.to_csv(os.path.join(BASE, "outputs", "p1_sensitivity.csv"),
            index=False, encoding="utf-8-sig")
R["sensitivity"] = sens_rows

rep_rows = []
for spec in ["草鱼料", "加州鲈料"]:
    for ing in ["菜粕", "棉粕", "豆粕"]:
        rp = replacement_price(spec, ingredient=ing)
        rp["feed_spec"] = spec
        rep_rows.append(rp)
R["replacement"] = [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                     for k, v in r.items()} for r in rep_rows]

# ============ 2. 比价低位后的走势 ============
ratio = ind["鱼料比价指数"].dropna()
thr = ratio.quantile(0.20)
below = ratio < thr
# 切成独立时段（连续低于阈值为一段）
episodes, start = [], None
for d, b in below.items():
    if b and start is None:
        start = d
    if not b and start is not None:
        episodes.append(start); start = None
if start is not None:
    episodes.append(start)
fish = ind["鱼价"]; cost = ind["固定权重成本指数"]
stats = {4: {"鱼价先回升": 0, "原料先回落": 0, "都未发生": 0, "同时": 0},
         8: {"鱼价先回升": 0, "原料先回落": 0, "都未发生": 0, "同时": 0},
         12: {"鱼价先回升": 0, "原料先回落": 0, "都未发生": 0, "同时": 0}}
for ep in episodes:
    f0, c0 = fish.get(ep, np.nan), cost.get(ep, np.nan)
    if np.isnan(f0):
        continue
    for h in [4, 8, 12]:
        fw = fish.loc[ep:].iloc[1:h + 1]; cw = cost.loc[ep:].iloc[1:h + 1]
        if len(fw) == 0:
            stats[h]["都未发生"] += 1; continue
        f_hit = fw[fw >= f0 * 1.03].index.min()   # 鱼价回升3%
        c_hit = cw[cw <= c0 * 0.98].index.min()   # 成本回落2%
        if pd.isna(f_hit) and pd.isna(c_hit):
            stats[h]["都未发生"] += 1
        elif pd.isna(c_hit) or (not pd.isna(f_hit) and f_hit < c_hit):
            stats[h]["鱼价先回升"] += 1
        elif pd.isna(f_hit) or c_hit < f_hit:
            stats[h]["原料先回落"] += 1
        else:
            stats[h]["同时"] += 1
R["low_ratio"] = {"threshold_pct20": round(float(thr), 1),
                  "episodes": len(episodes), "stats": stats}

# ============ 3. 领先滞后（差分互相关） ============
def ccf(a, b, max_lag=12):
    a = (a - a.mean()) / a.std(); b = (b - b.mean()) / b.std()
    av, bv = a.values, b.values   # 用位置对齐，避免pandas按索引对齐抵消平移
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = av[:lag], bv[-lag:]
        elif lag > 0:
            x, y = av[lag:], bv[:-lag]
        else:
            x, y = av, bv
        out[lag] = float(np.corrcoef(x, y)[0, 1])
    return pd.Series(out)

ccf_res = {}
pairs = [("豆粕", "豆粕_settle"), ("菜粕", "菜粕_settle"), ("玉米", "玉米_settle")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, col) in zip(axes, pairs):
    d_cost = ind[col].diff().dropna()
    d_fish = ind["鱼价"].diff().dropna()
    both = pd.concat([d_cost, d_fish], axis=1).dropna()
    c = ccf(both.iloc[:, 0], both.iloc[:, 1], 12)
    peak_lag = c.abs().idxmax()
    ccf_res[name] = {"peak_lag_weeks": int(peak_lag),
                     "peak_corr": round(float(c[peak_lag]), 3),
                     "interpretation": "负滞后=原料领先鱼价" if peak_lag < 0 else
                                       ("正滞后=鱼价领先原料" if peak_lag > 0 else "同步")}
    ax.bar(c.index, c.values, color="#2E6FBA")
    ax.axhline(2 / np.sqrt(len(both)), ls="--", c="r", lw=1)
    ax.axhline(-2 / np.sqrt(len(both)), ls="--", c="r", lw=1)
    ax.set_title(f"{name}价格变动与鱼价变动 互相关\n峰值: 滞后{peak_lag}周, r={c[peak_lag]:.2f}")
    ax.set_xlabel("滞后周数(负=原料领先鱼价)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p1_ccf.png"), dpi=150); plt.close()
R["ccf"] = ccf_res

# ============ 4. 套保回测 ============
hedge_res = {}
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for k, (name, col) in enumerate([("豆粕", "豆粕"), ("菜粕", "菜粕")]):
    d = ind[[f"{col}_spot", f"{col}_settle"]].dropna()
    dS = d[f"{col}_spot"].diff(); dF = d[f"{col}_settle"].diff()
    # 滚动26周最小方差套保比例
    h = dS.rolling(26).cov(dF) / dF.rolling(26).var()
    h = h.clip(0, 1.2).shift(1)   # 用上周估计的比例套本周,避免前视
    strat = {"不套保": 0.0, "套50%": 0.5, "最优比例": None, "套100%": 1.0}
    vr = {}
    var0 = dS.var()
    for sname, ratio_h in strat.items():
        hh = h if ratio_h is None else pd.Series(ratio_h, index=d.index)
        hedged = (dS - hh * dF).dropna()
        vr[sname] = 1 - hedged.var() / var0
    hedge_res[name] = {"variance_reduction_pct": {k2: round(v2 * 100, 1) for k2, v2 in vr.items()},
                       "avg_optimal_ratio": round(float(h.mean()), 2)}
    ax = axes[k][0]
    ax.plot(h.index, h.values, lw=1, color="#2E6FBA")
    ax.set_title(f"{name}：滚动26周最小方差套保比例(均值{h.mean():.2f})")
    ax.set_ylabel("套保比例"); ax.grid(alpha=0.3)
    ax = axes[k][1]
    ax.bar(vr.keys(), [v * 100 for v in vr.values()], color="#E8813A")
    ax.set_title(f"{name}：套保后周度采购成本方差缩减率(%)")
    ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p1_hedge.png"), dpi=150); plt.close()
R["hedge"] = hedge_res
R["hedge_note"] = ("现货价为模拟数据(基差均值回复),回测仅作方法演示;"
                   "未计入保证金占用与手续费,不构成套保比例建议;基差风险另述。")

# ============ 看板图表 ============
# 第1页 行情监测
fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
for name in ["豆粕", "菜粕", "玉米"]:
    axes[0].plot(ind.index, ind[f"{name}_adj_settle"], label=f"{name}主力连续价", lw=1.2)
axes[0].legend(); axes[0].set_title("第1页 | 行情监测：主力连续价（比例复权）"); axes[0].grid(alpha=0.3)
axes[0].set_ylabel("元/吨")
for name in ["豆粕", "菜粕", "玉米"]:
    axes[1].plot(ind.index, ind[f"{name}_基差"], label=f"{name}基差", lw=1)
axes[1].axhline(0, c="k", lw=0.5); axes[1].legend()
axes[1].set_title("基差 = 现货(模拟) − 期货主力"); axes[1].set_ylabel("元/吨"); axes[1].grid(alpha=0.3)
for name in ["豆粕", "菜粕", "玉米"]:
    axes[2].plot(ind.index, ind[f"{name}_波动率"], label=f"{name}波动率", lw=1)
axes[2].legend(); axes[2].set_title("20日滚动年化波动率"); axes[2].set_ylabel("%"); axes[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p1_page1_行情监测.png"), dpi=150); plt.close()

# 第2页 成本与剪刀差
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
ax = axes[0][0]
ax.plot(ind.index, ind["草鱼料_固定成本"], label="固定配方成本", lw=1.3)
ax.plot(ind.index, ind["草鱼料_动态成本"], label="动态配方成本", lw=1.3)
ax.fill_between(ind.index, ind["草鱼料_动态成本"], ind["草鱼料_固定成本"],
                alpha=0.25, color="green", label="配方调整收益")
ax.legend(); ax.set_title("草鱼料：固定 vs 动态配方吨成本"); ax.set_ylabel("元/吨"); ax.grid(alpha=0.3)
ax = axes[0][1]
ax.plot(ind.index, ind["草鱼料_饵料成本占鱼价比"], color="#2E6FBA", lw=1.3,
        label="饵料成本占鱼价比(草鱼料)")
ax2 = ax.twinx()
ax2.plot(ind.index, ind["草鱼料_占比分位"], color="#E8813A", lw=1, ls="--", label="历史分位(右)")
ax.set_ylabel("%"); ax2.set_ylabel("分位%")
ax.set_title("养1公斤鱼,原料成本占鱼价的比例"); ax.grid(alpha=0.3)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
ax = axes[1][0]
ax.plot(ind.index, ind["加州鲈料_固定成本"], label="固定配方", lw=1.2)
ax.plot(ind.index, ind["加州鲈料_动态成本"], label="动态配方", lw=1.2)
ax.legend(); ax.set_title("加州鲈料：固定 vs 动态配方吨成本"); ax.set_ylabel("元/吨"); ax.grid(alpha=0.3)
ax = axes[1][1]
sens_gc = sens[sens.feed_spec == "草鱼料"]
x = np.arange(len(sens_gc))
ax.bar(x - 0.2, sens_gc.fixed_delta.astype(float), 0.35, label="配方不动")
ax.bar(x + 0.2, sens_gc.dynamic_delta, 0.35, label="允许调配方")
ax.set_xticks(x); ax.set_xticklabels(sens_gc.ingredient)
ax.set_title("草鱼料敏感度：原料+100元/吨对吨成本的影响"); ax.set_ylabel("元/吨")
ax.legend(); ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p1_page2_成本与剪刀差.png"), dpi=150); plt.close()

# 第3页 预警与解读
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
ax = axes[0]
ax.plot(ind.index, ind["鱼料比价指数"], lw=1.2, color="#2E6FBA", label="鱼料比价指数")
for q, c in [(0.2, "red"), (0.5, "gray"), (0.8, "green")]:
    v = ind["鱼料比价指数"].quantile(q)
    ax.axhline(v, ls="--", c=c, lw=1, label=f"{int(q*100)}分位 {v:.0f}")
low_pts = ind[ind["鱼料比价指数"] < ind["鱼料比价指数"].quantile(0.2)]
ax.scatter(low_pts.index, low_pts["鱼料比价指数"], c="red", s=12, zorder=5, label="跌破20分位(预警)")
ax.legend(fontsize=9); ax.set_title("第3页 | 预警：鱼料比价指数与历史分位"); ax.grid(alpha=0.3)
ax = axes[1]
ax.plot(ind.index, ind["鱼价"], lw=1.2, color="#E8813A", label="草鱼批发价(模拟)")
ax.plot(ind.index, ind["固定权重成本指数"] / 100 * ind["鱼价"].dropna().iloc[0],
        lw=1.2, color="#2E6FBA", label="原料成本指数(换算至鱼价起点)")
ax.legend(); ax.set_title("鱼价 vs 原料成本（剪刀差）"); ax.set_ylabel("元/公斤"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p1_page3_预警.png"), dpi=150); plt.close()

# ============ 结果落盘 ============
latest = ind.iloc[-1]
R["latest"] = {
    "week": str(ind.index[-1].date()),
    "草鱼料_动态成本": round(float(latest["草鱼料_动态成本"]), 1),
    "草鱼料_固定成本": round(float(latest["草鱼料_固定成本"]), 1),
    "草鱼料_调整收益_年均": round(float(ind["草鱼料_调整收益"].mean()), 1),
    "加州鲈料_调整收益_年均": round(float(ind["加州鲈料_调整收益"].mean()), 1),
    "饵料成本占鱼价比": round(float(latest["草鱼料_饵料成本占鱼价比"]), 1),
    "比价指数": round(float(latest["鱼料比价指数"]), 1),
    "比价分位": round(float(latest["比价历史分位"]), 1),
}
with open(os.path.join(BASE, "outputs", "p1_analysis_results.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2, default=str)

print(json.dumps(R["latest"], ensure_ascii=False, indent=1))
print("低位时段数:", R["low_ratio"]["episodes"], "| 12周统计:", stats[12])
print("套保:", json.dumps(hedge_res, ensure_ascii=False))
print("领先滞后:", json.dumps(ccf_res, ensure_ascii=False))
print("figures ->", FIG)
