# -*- coding: utf-8 -*-
"""
公共底座 - 最低成本配方模型（PuLP 线性规划）
三个作品共用：
  P1 用它算动态配方吨成本、敏感度、替换临界价；
  P2 用它算"测得准"带来的安全余量收窄收益。

数据口径（全部写入假设登记表）：
  - 原料营养值：《中国饲料成分及营养价值表》典型值（版次待核），单位 %；
  - 营养约束：水产行业草鱼配合饲料标准（编号待核）与 NRC《鱼虾营养需要》(2011) 的常用区间；
  - 用量上限：由抗营养因子限制（棉粕-游离棉酚、菜粕-硫苷）及淀粉粘合需求给出；
  - 价格：默认给 2026 年典型价，实际由 P1 的周度价格序列驱动。
"""
import pulp
import pandas as pd

# ---------------- 原料营养表（%，典型值，来源见假设登记表 A-NUT-01） ----------------
# 列：粗蛋白CP 粗脂肪EE 粗纤维CF 赖氨酸Lys 蛋氨酸Met 总磷P
NUTRITION = {
    #            CP     EE    CF    Lys    Met    P
    "豆粕":   (43.0,  1.5,  5.0,  2.70,  0.60, 0.65),
    "菜粕":   (36.0,  2.0, 11.0,  1.90,  0.80, 1.00),
    "棉粕":   (40.0,  1.0, 11.0,  1.60,  0.50, 0.90),
    "玉米":   ( 8.5,  3.8,  2.2,  0.24,  0.18, 0.27),
    "小麦":   (12.5,  1.8,  2.5,  0.35,  0.20, 0.35),
    "米糠":   (12.8, 15.0,  9.0,  0.55,  0.25, 1.50),
    "鱼粉":   (62.0,  8.0,  0.5,  4.80,  1.80, 3.00),
    "豆油":   ( 0.0, 99.0,  0.0,  0.00,  0.00, 0.00),
    "磷酸二氢钙": (0.0, 0.0,  0.0,  0.00,  0.00, 22.0),
    "赖氨酸": ( 0.0,  0.0,  0.0, 78.00,  0.00, 0.00),
    "蛋氨酸": ( 0.0,  0.0,  0.0,  0.00, 99.00, 0.00),
    "预混料": ( 0.0,  0.0,  0.0,  0.00,  0.00, 0.00),
}

# 基准价格（元/吨，2026 年典型水平；鱼粉为假设值，见假设登记表 A-PRC-01）
BASE_PRICES = {
    "豆粕": 3100, "菜粕": 2400, "棉粕": 2600, "玉米": 2400, "小麦": 2500,
    "米糠": 1800, "鱼粉": 12500, "豆油": 8200, "磷酸二氢钙": 3600,
    "赖氨酸": 11000, "蛋氨酸": 22000, "预混料": 6000,
}

# ---------------- 料种规格 ----------------
# 用量上下限为小数（占配方比例）；营养约束为百分含量
FEED_SPECS = {
    "草鱼料": {
        "nutrient_min": {"CP": 28.0, "EE": 3.0, "Lys": 1.5, "Met": 0.5, "P": 0.9},
        "nutrient_max": {"CF": 10.0, "EE": 7.0},
        "bounds": {  # (下限, 上限)
            "豆粕": (0.00, 0.40), "菜粕": (0.00, 0.35), "棉粕": (0.00, 0.20),
            "玉米": (0.00, 0.25), "小麦": (0.05, 0.20), "米糠": (0.00, 0.15),
            "鱼粉": (0.00, 0.08), "豆油": (0.00, 0.04),
            "磷酸二氢钙": (0.010, 0.025), "赖氨酸": (0.00, 0.005), "蛋氨酸": (0.00, 0.005),
            "预混料": (0.010, 0.010),
        },
        "starch_min": 0.15,   # 玉米+小麦+米糠之和下限：制粒粘合所需淀粉（简化口径）
    },
    "加州鲈料": {
        "nutrient_min": {"CP": 42.0, "EE": 8.0, "Lys": 2.6, "Met": 1.0, "P": 1.2},
        "nutrient_max": {"CF": 5.0},
        "bounds": {
            "豆粕": (0.00, 0.35), "菜粕": (0.00, 0.15), "棉粕": (0.00, 0.00),
            "玉米": (0.00, 0.00), "小麦": (0.08, 0.15), "米糠": (0.00, 0.00),
            "鱼粉": (0.20, 0.45), "豆油": (0.04, 0.08),
            "磷酸二氢钙": (0.010, 0.020), "赖氨酸": (0.00, 0.010), "蛋氨酸": (0.00, 0.010),
            "预混料": (0.010, 0.010),
        },
        "starch_min": 0.08,
    },
}
STARCH_SOURCES = ["玉米", "小麦", "米糠"]
NUTRIENTS = ["CP", "EE", "CF", "Lys", "Met", "P"]


def solve_feed(spec_name, prices=None, cp_margin=0.0):
    """
    求解最低成本配方。
    prices: dict 元/吨，缺省用 BASE_PRICES
    cp_margin: 蛋白安全余量（百分点），P2 省钱测算用 —— 相当于把 CP 下限抬高
    返回 dict: status/cost/formula(pd.Series 百分比)/shadow(营养约束影子价格, 元每吨每百分点)
    """
    spec = FEED_SPECS[spec_name]
    prices = {**BASE_PRICES, **(prices or {})}
    ings = list(NUTRITION.keys())

    prob = pulp.LpProblem(f"feed_{spec_name}", pulp.LpMinimize)
    x = {i: pulp.LpVariable(f"x_{i}", lowBound=spec["bounds"][i][0],
                            upBound=spec["bounds"][i][1]) for i in ings}

    prob += pulp.lpSum(prices[i] * x[i] for i in ings), "cost_yuan_per_t"
    prob += pulp.lpSum(x[i] for i in ings) == 1.0, "mass_balance"

    for n, lb in spec["nutrient_min"].items():
        idx = NUTRIENTS.index(n)
        val = lb + (cp_margin if n == "CP" else 0.0)
        prob += pulp.lpSum(NUTRITION[i][idx] * x[i] for i in ings) >= val, f"min_{n}"
    for n, ub in spec["nutrient_max"].items():
        idx = NUTRIENTS.index(n)
        prob += pulp.lpSum(NUTRITION[i][idx] * x[i] for i in ings) <= ub, f"max_{n}"
    if spec["starch_min"] > 0:
        prob += pulp.lpSum(x[i] for i in STARCH_SOURCES) >= spec["starch_min"], "starch_min"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        return {"status": status}

    formula = pd.Series({i: x[i].value() * 100 for i in ings})  # 百分比
    shadow = {}
    for name, c in prob.constraints.items():
        if name.startswith(("min_", "max_")):
            shadow[name] = c.pi  # 元/吨 每百分点
    return {"status": status, "cost": pulp.value(prob.objective),
            "formula": formula.round(3), "shadow": shadow}


def sensitivity(spec_name, prices=None, ingredient="菜粕", delta=100.0):
    """
    原料价格 ±delta 元/吨 时，"配方不动"与"允许调配方"的吨成本变化对比。
    返回 dict: base_cost, fixed_delta, dynamic_delta
    """
    prices = {**BASE_PRICES, **(prices or {})}
    base = solve_feed(spec_name, prices)
    p_up = {**prices, ingredient: prices[ingredient] + delta}
    dyn = solve_feed(spec_name, p_up)
    # 配方不动：沿用 base 配方，直接按新价格结算
    fixed_cost = sum((base["formula"][i] / 100) * p_up[i] for i in NUTRITION)
    return {
        "ingredient": ingredient, "delta": delta,
        "base_cost": round(base["cost"], 1),
        "fixed_delta": round(fixed_cost - base["cost"], 1),
        "dynamic_delta": round(dyn["cost"] - base["cost"], 1),
    }


def replacement_price(spec_name, prices=None, ingredient="菜粕", grid=None):
    """
    替换临界价：该原料涨到多少元/吨时被挤出配方（用量降为 0）。
    """
    prices = {**BASE_PRICES, **(prices or {})}
    if grid is None:
        grid = range(int(prices[ingredient]), int(prices[ingredient] * 2.5) + 50, 50)
    base_incl = None
    for p in grid:
        r = solve_feed(spec_name, {**prices, ingredient: p})
        if r["status"] != "Optimal":
            return {"ingredient": ingredient, "critical_price": None, "note": "无解"}
        incl = r["formula"][ingredient]
        if base_incl is None:
            base_incl = incl
            if incl <= 1e-6:
                return {"ingredient": ingredient, "critical_price": prices[ingredient],
                        "note": "基准价下即未使用"}
        if incl <= 1e-6:
            return {"ingredient": ingredient, "critical_price": p,
                    "base_inclusion": round(base_incl, 2)}
    return {"ingredient": ingredient, "critical_price": None,
            "note": "扫描区间内仍在使用", "base_inclusion": round(base_incl, 2)}


if __name__ == "__main__":
    pd.set_option("display.unicode.east_asian_width", True)
    for spec in FEED_SPECS:
        r = solve_feed(spec)
        print(f"\n===== {spec} =====  状态: {r['status']}  吨成本: {r['cost']:.0f} 元/吨")
        print(r["formula"][r["formula"] > 0.01].to_string())
        print("影子价格(元/吨/百分点):",
              {k: round(v, 1) for k, v in r["shadow"].items()})
    print("\n--- 敏感度：菜粕 +100 元/吨 ---")
    for spec in FEED_SPECS:
        print(spec, sensitivity(spec, ingredient="菜粕", delta=100))
    print("\n--- 敏感度：豆粕 +100 元/吨 ---")
    for spec in FEED_SPECS:
        print(spec, sensitivity(spec, ingredient="豆粕", delta=100))
    print("\n--- 替换临界价 ---")
    for spec in FEED_SPECS:
        for ing in ["菜粕", "棉粕", "豆粕"]:
            print(spec, replacement_price(spec, ingredient=ing))
