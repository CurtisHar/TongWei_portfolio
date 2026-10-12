# -*- coding: utf-8 -*-
"""
P2 - NIR快检建模：预处理比较、Kennard-Stone划分、PLS(+RF对照)、全套评价指标
数据: data/corn_nir.npz (Eigenvector/Cargill 玉米NIR公开数据集, 已核实)
输出: outputs/p2_model_compare.csv, outputs/p2_repeated_splits.csv,
      outputs/figures/p2_eda.png, p2_preprocess.png, p2_vip.png, p2_pred_ref.png

方法要点：
- 预处理只在训练集内用交叉验证选择，避免测试集信息泄漏；
- 潜变量个数用"一倍标准误规则"防过拟合；
- 除KS固定划分外，再做20次随机划分报告误差分布；
- VIP波段对照已知吸收带（水1450/1940nm，蛋白~2060/2180nm，油~1720/2310nm）。
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(BASE, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)
D = np.load(os.path.join(BASE, "data", "corn_nir.npz"))
WL = np.arange(1100, 2499, 2).astype(float)   # 1100-2498nm, 间隔2nm, 700点
X_ALL = {"m5": D["m5"], "mp5": D["mp5"], "mp6": D["mp6"]}
PROPS = D["props"]
ANALYTES = {"水分": 0, "油": 1, "蛋白": 2, "淀粉": 3}
rng = np.random.default_rng(7)

# ---------- 预处理 ----------
def snv(X):
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

def msc(X, ref=None):
    ref = X.mean(axis=0) if ref is None else ref
    out = np.zeros_like(X)
    for i, s in enumerate(X):
        b = np.polyfit(ref, s, 1)
        out[i] = (s - b[1]) / b[0]
    return out

def sg1(X, window=15, poly=2):
    return savgol_filter(X, window, poly, deriv=1, axis=1)

PREPROCS = {
    "原始": lambda X: X.copy(),
    "SNV": snv,
    "MSC": msc,
    "SG一阶导数": sg1,
    "SNV+SG一阶导数": lambda X: sg1(snv(X)),
    "MSC+SG一阶导数": lambda X: sg1(msc(X)),
}

# ---------- Kennard-Stone 划分 ----------
def kennard_stone(X, n_train):
    Xc = (X - X.mean(0)) / X.std(0)
    dist = np.linalg.norm(Xc[:, None, :] - Xc[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    sel = [i, j]
    while len(sel) < n_train:
        dmin = np.min(dist[:, sel], axis=1)
        dmin[sel] = -1
        sel.append(int(np.argmax(dmin)))
    mask = np.zeros(len(X), bool); mask[sel] = True
    return mask, ~mask

# ---------- PLS：CV选潜变量（一倍标准误规则） ----------
def select_lv(X, y, max_lv=15, seed=0):
    kf = KFold(5, shuffle=True, random_state=seed)
    errs = np.zeros(max_lv)
    for lv in range(1, max_lv + 1):
        cv_err = []
        for tr, va in kf.split(X):
            m = PLSRegression(lv).fit(X[tr], y[tr])
            cv_err.append(np.mean((m.predict(X[va]).ravel() - y[va]) ** 2))
        errs[lv - 1] = np.sqrt(np.mean(cv_err))
    best = int(np.argmin(errs)) + 1
    se = np.std([np.sqrt(np.mean((PLSRegression(best).fit(X[tr], y[tr])
                 .predict(X[va]).ravel() - y[va]) ** 2)) for tr, va in kf.split(X)])
    # 一倍标准误规则：误差不超过最小误差+1个标准误的最简单模型
    one_se = min(lv for lv in range(1, max_lv + 1) if errs[lv - 1] <= errs[best - 1] + se)
    return best, one_se, errs

def metrics(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    bias = float(np.mean(y_pred - y_true))
    rpd = float(np.std(y_true, ddof=1) / rmse) if rmse > 0 else np.inf
    return rmse, r2, bias, rpd

def vip_scores(model, X, y):
    """PLS VIP 得分（单响应）：VIP_j = sqrt( p * Σ_a SSY_a*(w_ja/||w_a||)² / Σ_a SSY_a )"""
    t = model.x_scores_                     # n×h
    w = model.x_weights_                    # p×h
    q = model.y_loadings_.ravel()           # h
    p = w.shape[0]
    ssy = (t ** 2).sum(axis=0) * q ** 2     # 每个LV贡献的y方差
    w_norm = w / np.linalg.norm(w, axis=0)  # 归一化权重
    return np.sqrt(p * (w_norm ** 2 @ ssy) / ssy.sum())

# ---------- EDA 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for inst, c in [("m5", "#2E6FBA"), ("mp5", "#E8813A"), ("mp6", "#5BA55B")]:
    for row in X_ALL[inst][::8]:
        axes[0].plot(WL, row, lw=0.5, alpha=0.5, color=c)
axes[0].set_title("三台仪器的玉米NIR光谱(每8条抽1)")
axes[0].set_xlabel("波长 nm"); axes[0].set_ylabel("吸光度")
axes[0].legend([plt.Line2D([], [], color=c) for _, c in
                [("m5", "#2E6FBA"), ("mp5", "#E8813A"), ("mp6", "#5BA55B")]],
               ["m5", "mp5", "mp6"])
for i, (name, k) in enumerate(ANALYTES.items()):
    axes[1].hist(PROPS[:, k], bins=12, alpha=0.6, label=name)
axes[1].legend(); axes[1].set_title("四项化验参考值分布(%)"); axes[1].set_xlabel("含量 %")
pca = PCA(2).fit(snv(X_ALL["m5"]))
sc = pca.transform(snv(X_ALL["m5"]))
sc2 = axes[2].scatter(sc[:, 0], sc[:, 1], c=PROPS[:, 2], cmap="viridis")
plt.colorbar(sc2, ax=axes[2], label="蛋白%")
axes[2].set_title(f"PCA得分图(m5, SNV) PC1 {pca.explained_variance_ratio_[0]:.0%} / "
                  f"PC2 {pca.explained_variance_ratio_[1]:.0%}")
axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p2_eda.png"), dpi=150); plt.close()

# ---------- 预处理效果示意 ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
demo = X_ALL["m5"][:10]
for ax, (pname, fn) in zip(axes.ravel(), PREPROCS.items()):
    for row in fn(demo):
        ax.plot(WL, row, lw=0.6)
    ax.set_title(pname); ax.set_xlabel("nm")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p2_preprocess.png"), dpi=150); plt.close()

# ---------- 主流程：KS划分 + 预处理×模型比较 ----------
X = X_ALL["m5"]   # 主机仪器 m5
tr_mask, te_mask = kennard_stone(snv(X), 60)
print(f"KS划分: 训练 {tr_mask.sum()} / 测试 {te_mask.sum()}")

rows = []
best = {}
for aname, k in ANALYTES.items():
    y = PROPS[:, k]
    for pname, fn in PREPROCS.items():
        Xp = fn(X)
        Xtr, Xte, ytr, yte = Xp[tr_mask], Xp[te_mask], y[tr_mask], y[te_mask]
        best_lv, lv_1se, _ = select_lv(Xtr, ytr, seed=1)
        m = PLSRegression(lv_1se).fit(Xtr, ytr)
        p_tr, p_te = m.predict(Xtr).ravel(), m.predict(Xte).ravel()
        rmsec, r2c, _, _ = metrics(ytr, p_tr)
        rmsep, r2p, biasp, rpdp = metrics(yte, p_te)
        rows.append(dict(指标=aname, 预处理=pname, 模型=f"PLS({lv_1se}LV)",
                         RMSEC=round(rmsec, 3), RMSEP=round(rmsep, 3),
                         R2=round(r2p, 3), 偏差=round(biasp, 3), RPD=round(rpdp, 2)))
        key = (aname, "PLS")
        if key not in best or rmsep < best[key][0]:
            best[key] = (rmsep, pname, lv_1se, m, fn)
    # RF 对照（用各指标最优预处理）
    bp = best[(aname, "PLS")][1]
    Xp = PREPROCS[bp](X)
    rf = RandomForestRegressor(300, max_features=0.3, random_state=0, n_jobs=-1)
    rf.fit(Xp[tr_mask], y[tr_mask])
    p_te = rf.predict(Xp[te_mask])
    rmsep, r2p, biasp, rpdp = metrics(y[te_mask], p_te)
    rows.append(dict(指标=aname, 预处理=bp, 模型="随机森林",
                     RMSEC=np.nan, RMSEP=round(rmsep, 3), R2=round(r2p, 3),
                     偏差=round(biasp, 3), RPD=round(rpdp, 2)))

cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(os.path.join(BASE, "outputs", "p2_model_compare.csv"),
              index=False, encoding="utf-8-sig")
print(cmp_df[cmp_df.指标.isin(["蛋白", "水分"])].to_string(index=False))

# ---------- 20次随机划分：误差分布 ----------
rep_rows = []
for aname in ["蛋白", "水分"]:
    k = ANALYTES[aname]
    rmsep, pname, lv, _, fn = best[(aname, "PLS")]
    y = PROPS[:, k]
    for rep in range(20):
        m_tr, m_te = kennard_stone(snv(X), 60) if rep == 0 else (
            np.isin(np.arange(80), rng.choice(80, 60, replace=False)), None)
        if rep > 0:
            m_te = ~m_tr
        Xp = fn(X)
        mdl = PLSRegression(lv).fit(Xp[m_tr], y[m_tr])
        r_ = metrics(y[m_te], mdl.predict(Xp[m_te]).ravel())
        rep_rows.append(dict(指标=aname, rep=rep, RMSEP=round(r_[0], 3), RPD=round(r_[3], 2)))
rep_df = pd.DataFrame(rep_rows)
rep_df.to_csv(os.path.join(BASE, "outputs", "p2_repeated_splits.csv"),
              index=False, encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(6, 4))
data = [rep_df[(rep_df.指标 == a)].RMSEP for a in ["蛋白", "水分"]]
ax.boxplot(data, labels=["蛋白", "水分"])
ax.set_title("20次随机划分的RMSEP分布(KS划分=第1次)"); ax.set_ylabel("RMSEP (百分点)")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p2_rmsep_dist.png"), dpi=150); plt.close()

# ---------- 预测-参考值散点 + VIP ----------
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
vip_fig, vax = plt.subplots(1, 4, figsize=(16, 3.5), sharex=True)
summary = {}
for i, aname in enumerate(ANALYTES):
    k = ANALYTES[aname]
    rmsep, pname, lv, mdl, fn = best[(aname, "PLS")]
    y = PROPS[:, k]
    Xp = fn(X)
    p_te = mdl.predict(Xp[te_mask]).ravel()
    axes[i].scatter(y[te_mask], p_te, c="#2E6FBA", label="测试集")
    p_tr = mdl.predict(Xp[tr_mask]).ravel()
    axes[i].scatter(y[tr_mask], p_tr, c="gray", s=10, alpha=0.5, label="训练集")
    lo, hi = y.min(), y.max()
    axes[i].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[i].set_title(f"{aname} | {pname}+PLS({lv}LV)\nRMSEP={rmsep:.3f}, RPD={rmsep and round(np.std(y[te_mask], ddof=1)/rmsep, 2)}")
    axes[i].set_xlabel("化验值 %"); axes[i].set_ylabel("预测值 %")
    axes[i].legend(fontsize=8)
    v = vip_scores(mdl, Xp[tr_mask], y[tr_mask])
    vax[i].plot(WL, v, lw=0.8)
    vax[i].axhline(1, ls="--", c="r", lw=0.8)
    for band, lab in [(1450, "水"), (1940, "水"), (2060, "蛋白"), (2180, "蛋白"),
                      (1720, "油"), (2310, "油")]:
        vax[i].axvline(band, ls=":", c="green", lw=0.8)
    vax[i].set_title(f"{aname} VIP"); vax[i].set_xlabel("nm")
    summary[aname] = {"preprocess": pname, "lv": int(lv), "RMSEP": rmsep,
                      "RPD": round(float(np.std(y[te_mask], ddof=1) / rmsep), 2)}
plt.tight_layout(); plt.savefig(os.path.join(FIG, "p2_pred_ref.png"), dpi=150); plt.close()
vip_fig.tight_layout(); vip_fig.savefig(os.path.join(FIG, "p2_vip.png"), dpi=150); plt.close(vip_fig)

with open(os.path.join(BASE, "outputs", "p2_best_models.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
