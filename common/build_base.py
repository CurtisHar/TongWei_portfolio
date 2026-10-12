# -*- coding: utf-8 -*-
"""
公共底座
输出: portfolio/data/supply_chain.db (SQLite)

维度表: dim_date / dim_ingredient / dim_region / dim_factory / dim_source
事实表(由 P1/P2/P3 脚本填充): fact_futures_daily / fact_price_weekly /
  fact_formula_cost_weekly / fact_nir_result / fact_demand_monthly /
  fact_shipment_plan / fact_inventory_plan
"""
import os
import sqlite3
import pandas as pd
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..")
DB = os.path.join(BASE, "data", "supply_chain.db")

# ---------------- dim_date ----------------
def build_dim_date():
    dates = pd.date_range("2022-01-01", "2026-12-31", freq="D")
    df = pd.DataFrame({"date": dates})
    df["date_key"] = df.date.dt.strftime("%Y%m%d")
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df["iso_week"] = df.date.dt.isocalendar().week.astype(int)
    df["week_start"] = (df.date - pd.to_timedelta(df.date.dt.weekday, unit="D")).dt.strftime("%Y%m%d")
    # 旬
    df["xun"] = df.date.dt.day.apply(lambda d: "上旬" if d <= 10 else ("中旬" if d <= 20 else "下旬"))
    # 交易日标记：工作日且不在法定节假日近似清单（精确交易日以期货行情实际日期为准）
    df["is_weekday"] = (df.date.dt.weekday < 5).astype(int)
    # 养殖季节标签（四川口径，假设登记表 A-SEA-01）
    def season(m):
        if m in (3, 4, 5): return "投苗期"
        if m in (6, 7, 8, 9, 10): return "旺季"
        return "越冬期"
    df["culture_season"] = df.month.map(season)
    df["region_scope"] = "四川"
    return df[["date_key", "date", "year", "month", "iso_week", "week_start",
               "xun", "is_weekday", "culture_season", "region_scope"]]

# ---------------- dim_ingredient ----------------
def build_dim_ingredient():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from formulation import NUTRITION, BASE_PRICES
    meta = {  # 类别, 交易所, 期货代码
        "豆粕": ("植物蛋白", "DCE", "M"), "菜粕": ("植物蛋白", "CZCE", "RM"),
        "棉粕": ("植物蛋白", None, None), "玉米": ("能量原料", "DCE", "C"),
        "小麦": ("能量原料", None, None), "米糠": ("能量原料", None, None),
        "鱼粉": ("动物蛋白", None, None), "豆油": ("油脂", "DCE", "Y"),
        "磷酸二氢钙": ("矿物质", None, None), "赖氨酸": ("氨基酸", None, None),
        "蛋氨酸": ("氨基酸", None, None), "预混料": ("添加剂", None, None),
    }
    rows = []
    for i, (name, nut) in enumerate(NUTRITION.items(), 1):
        cat, ex, sym = meta[name]
        rows.append({
            "ingredient_key": f"IG{i:03d}", "name": name, "category": cat,
            "exchange": ex, "futures_symbol": sym,
            "cp": nut[0], "ee": nut[1], "cf": nut[2],
            "lys": nut[3], "met": nut[4], "p": nut[5],
            "base_price_yuan_t": BASE_PRICES[name],
            "nutrition_source": "中国饲料成分及营养价值表典型值(版次待核)",
        })
    return pd.DataFrame(rows)

# ---------------- dim_region（四川+重庆市级需求点，P3 用；坐标为市中心近似值） ----------------
def build_dim_region():
    # name, code(市级区划), lon, lat, 水产权重(相对值, 假设 A-DEM-01)
    data = [
        ("四川省", "成都市", "510100", 104.07, 30.57, 1.00),
        ("四川省", "眉山市", "511400", 103.83, 30.05, 0.85),
        ("四川省", "乐山市", "511100", 103.77, 29.55, 0.70),
        ("四川省", "德阳市", "510600", 104.40, 31.13, 0.55),
        ("四川省", "绵阳市", "510700", 104.68, 31.47, 0.60),
        ("四川省", "遂宁市", "510900", 105.57, 30.53, 0.55),
        ("四川省", "南充市", "511300", 106.08, 30.78, 0.75),
        ("四川省", "资阳市", "512000", 104.65, 30.12, 0.50),
        ("四川省", "内江市", "511000", 105.06, 29.58, 0.55),
        ("四川省", "自贡市", "510300", 104.78, 29.35, 0.40),
        ("四川省", "宜宾市", "511500", 104.62, 28.75, 0.60),
        ("四川省", "泸州市", "510500", 105.44, 28.87, 0.50),
        ("四川省", "广安市", "511600", 106.63, 30.46, 0.50),
        ("四川省", "达州市", "511700", 107.50, 31.21, 0.55),
        ("四川省", "雅安市", "511800", 103.00, 29.98, 0.35),
        ("四川省", "广元市", "510800", 105.84, 32.43, 0.30),
        ("四川省", "巴中市", "511900", 106.75, 31.87, 0.30),
        ("重庆市", "重庆市", "500100", 106.55, 29.56, 0.90),
    ]
    return pd.DataFrame(data, columns=["province", "city", "region_code",
                                       "lon_wgs84", "lat_wgs84", "aqua_weight"])

# ---------------- dim_factory（通威四川工厂：名单/产能为示例假设，待核，A-FAC-01） ----------------
def build_dim_factory():
    data = [
        # key, 名称, 城市, lon, lat, 颗粒料产能吨/年, 膨化料产能吨/年, 仓库容量吨
        ("F01", "通威成都双流工厂(示例)", "成都市", 103.92, 30.50, 180000, 90000, 25000),
        ("F02", "通威眉山工厂(示例)",     "眉山市", 103.83, 30.08, 120000, 60000, 15000),
        ("F03", "通威绵阳工厂(示例)",     "绵阳市", 104.70, 31.45,  80000, 30000, 10000),
        ("F04", "通威南充工厂(示例)",     "南充市", 106.10, 30.80,  80000, 30000, 10000),
    ]
    return pd.DataFrame(data, columns=["factory_key", "factory_name", "city",
                                       "lon_wgs84", "lat_wgs84",
                                       "cap_pellet_t_year", "cap_extruded_t_year",
                                       "warehouse_cap_t"])

# ---------------- dim_source ----------------
def build_dim_source():
    data = [
        ("SRC01", "tushare期货日线(代理接口)", "日频", "A", "期货收盘价/结算价/持仓量，真实数据"),
        ("SRC02", "生意社现货价(模拟,接口待核)", "日频", "C", "现货价=期货主力+模拟基差,标注模拟"),
        ("SRC03", "农业农村部批发鱼价(模拟,接口待核)", "日频", "C", "草鱼批发价,按历史公开水平模拟"),
        ("SRC04", "Eigenvector玉米NIR数据集", "静态", "A", "80样本×3仪器×700波长,已核实"),
        ("SRC05", "Open-Meteo历史天气API", "日频", "A", "日平均气温,免费接口"),
        ("SRC06", "文献/行业参数(饲料系数等)", "静态", "B", "取值区间见假设登记表"),
        ("SRC07", "模拟订单/产能/运费参数", "静态", "C", "本人物流经验+公开规则,标注模拟"),
        ("SRC08", "中国饲料成分及营养价值表", "静态", "B", "原料营养典型值,版次待核"),
    ]
    return pd.DataFrame(data, columns=["source_key", "source_name", "update_freq",
                                       "confidence", "note"])

FACT_DDL = {
    "fact_futures_daily": """CREATE TABLE fact_futures_daily(
        ts_code TEXT, trade_date TEXT, symbol TEXT, contract_month TEXT,
        open REAL, high REAL, low REAL, close REAL, settle REAL,
        vol REAL, oi REAL, is_main INT, source_key TEXT)""",
    "fact_price_weekly": """CREATE TABLE fact_price_weekly(
        week_start TEXT, item TEXT, item_type TEXT, value REAL,
        unit TEXT, source_key TEXT, note TEXT)""",
    "fact_formula_cost_weekly": """CREATE TABLE fact_formula_cost_weekly(
        week_start TEXT, feed_spec TEXT, cost_fixed REAL, cost_dynamic REAL,
        formula_json TEXT, source_key TEXT)""",
    "fact_nir_result": """CREATE TABLE fact_nir_result(
        sample_id TEXT, instrument TEXT, analyte TEXT, reference REAL,
        predicted REAL, split TEXT, model TEXT, preprocess TEXT,
        t2 REAL, q_res REAL, flagged INT, source_key TEXT)""",
    "fact_demand_monthly": """CREATE TABLE fact_demand_monthly(
        region_code TEXT, year INT, month INT, demand_t REAL,
        product TEXT, scenario TEXT, source_key TEXT)""",
    "fact_shipment_plan": """CREATE TABLE fact_shipment_plan(
        strategy TEXT, month INT, factory_key TEXT, region_code TEXT,
        product TEXT, qty_t REAL, cost REAL)""",
    "fact_inventory_plan": """CREATE TABLE fact_inventory_plan(
        strategy TEXT, month INT, factory_key TEXT, product TEXT,
        end_inventory_t REAL, capacity_util REAL)""",
}

def main():
    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    dims = {"dim_date": build_dim_date(), "dim_ingredient": build_dim_ingredient(),
            "dim_region": build_dim_region(), "dim_factory": build_dim_factory(),
            "dim_source": build_dim_source()}
    for name, df in dims.items():
        df.to_sql(name, con, index=False)
        df.to_csv(os.path.join(BASE, "data", f"{name}.csv"), index=False, encoding="utf-8-sig")
        print(f"{name}: {len(df)} rows")
    for t, ddl in FACT_DDL.items():
        con.execute(ddl)
        print(f"{t}: created")
    con.commit(); con.close()
    print("DB ->", os.path.abspath(DB))

if __name__ == "__main__":
    main()
