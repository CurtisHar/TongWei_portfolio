# -*- coding: utf-8 -*-
"""
P2 - 异常样本拦截 + 仪器间转移 + 省钱测算
输出: outputs/p2_qc_transfer_results.json
      outputs/figures/p2_qc_control.png, p2_transfer.png, p2_savings.png
      并写入 fact_nir_result 表
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import chi2
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
import sqlite3

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(BASE, "common"))
from formulation import solve_feed

FIG = os.path.join(BASE, "outputs", "figures")
D = np.load(os.path.join(BASE, "data", "corn_nir.npz"))
WL = np.arange(1100, 2499, 2).astype(float)
PROPS = D["props"]
rng = np.random.default_rng(11)
R = {}

def sg1(X, window=15, poly=2):
    return savgol_filter(X, window, poly, deriv=1, axis=1)

def kennard_stone(X, n_train):
    Xc = (X - X.mean(0)) / X.std(0)
    dist = np.linalg.norm(Xc[:, None, :] - Xc[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    sel = [i, j]
    while len(sel) < n_train:
        dmin = np.min(dist[:, sel], axis=1); dmin[sel] = -1
        sel.append(int(np.argmax(dmin)))
    mask = np.zeros(len(X), bool); mask[sel] = True
    return mask, ~mask

Xm5, Xmp5, Xmp6 = sg1(D["m5"]), sg1(D["mp5"]), sg1(D["mp6"])
tr_mask, te_mask = kennard_stone(sg1(D["m5"]), 60)

# ============ 1. Hotelling T² / Q 控制图 ============
N_PC = 5
pca = PCA(N_PC).fit(Xm5[tr_mask])
T_train = pca.transform(Xm5[tr_mask])
lams = pca.explained_variance_
T2_train = np.sum(T_train ** 2 / lams, axis=1)
E_train = Xm5[tr_mask] - pca.inverse_transform(T_train)
Q_train = np.sum(E_train ** 2, axis=1)
# 控制限：T² 用卡方近似，Q 用 Jackson-Mudholkar
t2_lim95 = chi2.ppf(0.95, N_PC); t2_lim99 = chi2.ppf(0.99, N_PC)
def q_limit(Q, alpha):
    lam = np.array([np.mean(Q ** i) for i in (1, 2, 3)])
    h0 = 1 - 2 * lam[0] * lam[2] / (3 * lam[1] ** 2)
    z = chi2.ppf(alpha, 1)   # 近似
    from scipy.stats import norm
    z = norm.ppf(alpha)
    return lam[0] * (z * np.sqrt(2 * lam[1] * h0 ** 2) / lam[0] + 1
                     + lam[1] * h0 * (h0 - 1) / lam[0] ** 2) ** (1 / h0)
q_lim95 = q_limit(Q_train, 0.95); q_lim99 = q_limit(Q_train, 0.99)

def t2q(Xnew):
    Tn = pca.transform(Xnew)
    T2 = np.sum(Tn ** 2 / lams, axis=1)
    E = Xnew - pca.inverse_transform(Tn)
    return T2, np.sum(E ** 2, axis=1)

# 正常测试样本 + 三类人为异常
Xte = Xm5[te_mask]
n_anom = 10
base = Xte[:n_anom].copy()
anom_noise = base + rng.normal(0, 0.15, base.shape)                       # 噪声污染
anom_drift = base + np.linspace(0, 0.3, base.shape[1]) * rng.normal(1, 0.2, (n_anom, 1))  # 基线漂移
anom_inst = Xmp6[te_mask][:n_anom]                                        # 另一台仪器的光谱
X_qc = np.vstack([Xte, anom_noise, anom_drift, anom_inst])
labels = (["正常"] * len(Xte) + ["噪声异常"] * n_anom +
          ["基线漂移"] * n_anom + ["异机光谱"] * n_anom)
T2_all, Q_all = t2q(X_qc)
flag = (T2_all > t2_lim95) | (Q_all > q_lim95)
qc = pd.DataFrame({"label": labels, "T2": T2_all, "Q": Q_all, "flagged": flag})
det = qc[qc.label != "正常"].groupby("label").flagged.mean()
far = qc[qc.label == "正常"].flagged.mean()
R["qc"] = {"拦截率_按类型": det.round(2).to_dict(), "正常样本误报率": round(float(far), 2),
           "总拦截率": round(float(det.mean()), 2)}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
colors = {"正常": "#999999", "噪声异常": "#E8813A", "基线漂移": "#D44E52", "异机光谱": "#8172B3"}
for ax, vals, lim95, lim99, name in [(axes[0], T2_all, t2_lim95, t2_lim99, "Hotelling T²"),
                                     (axes[1], Q_all, q_lim95, q_lim99, "Q残差")]:
    for lab in colors:
        idx = [i for i, l in enumerate(labels) if l == lab]
        ax.scatter(idx, vals[idx], s=18, c=colors[lab], label=lab)
    ax.axhline(lim95, ls="--", c="orange", lw=1, label="95%控制限")
    ax.axhline(lim99, ls="--", c="red", lw=1, label="99%控制限")
    ax.set_title(f"{name} 控制图"); ax.set_xlabel("样本序号"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p2_qc_control.png"), dpi=150); plt.close()

# ============ 2. 仪器间转移（蛋白模型, m5 -> mp5/mp6） ============
y = PROPS[:, 2]   # 蛋白
LV = 7
model = PLSRegression(LV).fit(Xm5[tr_mask], y[tr_mask])

transfer = {}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (iname, Xi) in zip(axes, [("mp5", Xmp5), ("mp6", Xmp6)]):
    pred_te = model.predict(Xi[te_mask]).ravel()
    rmsep_before = float(np.sqrt(np.mean((pred_te - y[te_mask]) ** 2)))
    # 斜率/截距校正：用训练集中10个转移样本
    tr_idx = np.where(tr_mask)[0][:10]
    pred_tr = model.predict(Xi[tr_idx]).ravel()
    a, b = np.polyfit(pred_tr, y[tr_idx], 1)      # y_ref = a*pred + b
    pred_corr = a * pred_te + b
    rmsep_after = float(np.sqrt(np.mean((pred_corr - y[te_mask]) ** 2)))
    transfer[iname] = {"RMSEP_转移前": round(rmsep_before, 3),
                       "RMSEP_校正后": round(rmsep_after, 3),
                       "转移样本数": 10, "斜率": round(float(a), 3), "截距": round(float(b), 3)}
    ax.scatter(y[te_mask], pred_te, c="#D44E52", label=f"直接转移 RMSEP={rmsep_before:.3f}")
    ax.scatter(y[te_mask], pred_corr, c="#5BA55B", label=f"SBC校正后 RMSEP={rmsep_after:.3f}")
    lo, hi = y.min(), y.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title(f"m5模型 → {iname}"); ax.set_xlabel("化验值 %"); ax.set_ylabel("预测值 %")
    ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p2_transfer.png"), dpi=150); plt.close()
R["transfer"] = transfer
R["transfer"]["m5_本机RMSEP"] = 0.071   # 来自 nir_model.py 最优结果

# ============ 3. 省钱测算 ============
sigma_batch = float(np.std(PROPS[:, 2], ddof=1))     # 批次间蛋白标准差
rmsep = 0.071                                         # 快检后的误差
base_formula = solve_feed("草鱼料")
corn_incl = base_formula["formula"]["玉米"] / 100     # 玉米在配方中的占比
z = 1.28                                              # 单侧90%置信（A-NIR-01）
unc_old = corn_incl * sigma_batch                     # 成品蛋白不确定性：无快检
unc_new = corn_incl * rmsep                           # 有快检
margin_old, margin_new = z * unc_old, z * unc_new
cost_old = solve_feed("草鱼料", cp_margin=margin_old)["cost"]
cost_new = solve_feed("草鱼料", cp_margin=margin_new)["cost"]
save_per_t = cost_old - cost_new
# 影子价格快速估算交叉验证
sp = base_formula["shadow"].get("min_CP", 0.0)
save_shadow = abs(sp) * (margin_old - margin_new)
scenarios = {"年产5万吨": 5e4, "年产20万吨": 2e5, "年产50万吨": 5e5}
R["savings"] = {
    "玉米批次间蛋白std": round(sigma_batch, 3), "快检RMSEP": rmsep,
    "玉米配方占比": round(float(corn_incl), 4), "z": z,
    "余量_无快检_pct": round(margin_old, 4), "余量_有快检_pct": round(margin_new, 4),
    "吨成本节约_元": round(save_per_t, 2), "影子价格估算_元": round(save_shadow, 2),
    "年节约_万元": {k: round(save_per_t * v / 1e4, 1) for k, v in scenarios.items()},
    "注": "玉米主要提供能量、蛋白占比小,测算偏保守;换豆粕/菜粕/鱼粉需用真实数据重新建模",
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(["无快检\n(批次间波动)", "有快检\n(模型误差)"],
            [unc_old, unc_new], color=["#D44E52", "#5BA55B"])
axes[0].set_title("成品蛋白的不确定性来源收窄"); axes[0].set_ylabel("成品蛋白标准差(百分点)")
for i, v in enumerate([unc_old, unc_new]):
    axes[0].text(i, v, f"{v:.4f}", ha="center", va="bottom")
sc_names = list(scenarios)
axes[1].bar(sc_names, [save_per_t * scenarios[k] / 1e4 for k in sc_names], color="#2E6FBA")
axes[1].set_title(f"年节约测算(每吨省 {save_per_t:.2f} 元)"); axes[1].set_ylabel("万元/年")
for i, k in enumerate(sc_names):
    axes[1].text(i, save_per_t * scenarios[k] / 1e4, f"{save_per_t*scenarios[k]/1e4:.1f}",
                 ha="center", va="bottom")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p2_savings.png"), dpi=150); plt.close()

# ============ 4. 写 fact_nir_result ============
con = sqlite3.connect(os.path.join(BASE, "data", "supply_chain.db"))
con.execute("DELETE FROM fact_nir_result")
recs = []
for i, idx in enumerate(np.where(te_mask)[0]):
    for aname, kk in [("蛋白", 2), ("水分", 0)]:
        mdl = PLSRegression(7 if aname == "蛋白" else 13).fit(
            Xm5[tr_mask], PROPS[tr_mask, kk])
        recs.append((f"S{idx:03d}", "m5", aname, float(PROPS[idx, kk]),
                     float(mdl.predict(Xm5[idx:idx + 1]).ravel()[0]), "test",
                     "PLS", "SG一阶导数", float(T2_all[i]), float(Q_all[i]),
                     int(flag[i]), "SRC04"))
pd.DataFrame(recs, columns=["sample_id", "instrument", "analyte", "reference",
                            "predicted", "split", "model", "preprocess",
                            "t2", "q_res", "flagged", "source_key"]).to_sql(
    "fact_nir_result", con, if_exists="append", index=False)
con.commit(); con.close()

with open(os.path.join(BASE, "outputs", "p2_qc_transfer_results.json"), "w",
          encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
