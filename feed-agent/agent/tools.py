# -*- coding: utf-8 -*-
"""
工具层（确定性计算）：list_metrics / query_metric / decompose_cost_change /
optimize_formula / run_readonly_sql / search_docs
护栏：DuckDB 只读连接；SQL 白名单校验 + 行数上限 + 全程日志。
"""
import os, re, sys, json, datetime
import duckdb
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WS = os.path.dirname(ROOT)                       # 工作区根（含 portfolio/）
sys.path.insert(0, WS)                           # 保证 portfolio 可导入，与运行目录无关
from portfolio.common import formulation

PORTFOLIO = os.path.join(WS, "portfolio")
sys.path.insert(0, os.path.join(PORTFOLIO, "common"))
DB = os.path.join(ROOT, "data", "company.duckdb")
LOG = os.path.join(ROOT, "logs", "sql_log.jsonl")
os.makedirs(os.path.dirname(LOG), exist_ok=True)

import yaml
METRICS = yaml.safe_load(open(os.path.join(ROOT, "semantic", "metrics.yaml"),
                              encoding="utf-8"))["metrics"]
CARDS = {m["name"]: m for m in METRICS}

def con_ro():
    return duckdb.connect(DB, read_only=True)

def _log(entry):
    entry["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ---------------- 1. 指标目录 ----------------
def list_metrics():
    out = []
    for m in METRICS:
        item = {"name": m["name"], "label": m["label"], "status": m["status"]}
        if m["status"] == "needs_clarification":
            item["clarify"] = m["clarify"]
            item["variants"] = list(m["variants"].keys())
        out.append(item)
    return {"metrics": out}

# ---------------- 2. 指标查询（SQL 由语义层拼接） ----------------
def _quote(v):
    return "'" + str(v).replace("'", "''") + "'"

def build_sql(card, dims, filters, month_from, month_to, variant=None):
    if card["status"] == "needs_clarification":
        if not variant:
            return None, {"needs_clarification": True, "clarify": card["clarify"],
                          "variants": {k: v["label"] for k, v in card["variants"].items()}}
        card = card["variants"][variant]
    if card.get("custom"):
        return "CUSTOM", card
    dims = [d for d in (dims or []) if d in card["grain"]]
    filters = {k: v for k, v in (filters or {}).items() if k in card.get("filters", [])}
    ctes = card.get("ctes", {})
    parts = []
    if ctes:
        parts.append("WITH " + ", ".join(f"{k} AS ({v})" for k, v in ctes.items()))
    frm = card.get("from") or card.get("join")
    select_dims = ", ".join(dims) + ", " if dims else ""
    where = []
    for k, v in filters.items():
        if isinstance(v, list):
            where.append(f"{k} IN ({', '.join(_quote(x) for x in v)})")
        else:
            where.append(f"{k} = {_quote(v)}")
    if month_from and "month" in card.get("grain", []):
        where.append(f"month >= {_quote(month_from)}")
    if month_to and "month" in card.get("grain", []):
        where.append(f"month <= {_quote(month_to)}")
    # CTE 列上的筛选需要落到子查询层——简化：主查询引用 CTE 别名时的列名与维度同名
    sql = f"{parts[0] + ' ' if parts else ''}SELECT {select_dims}{card['expr']} AS value FROM {frm}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if dims:
        sql += " GROUP BY " + ", ".join(dims)
        if "month" in dims:
            sql += " ORDER BY month"
    return sql, card

def query_metric(metric, dims=None, filters=None, month_from=None, month_to=None,
                 compare=None, variant=None):
    if metric not in CARDS:
        return {"error": f"指标 '{metric}' 不在目录中",
                "available": [m["name"] for m in METRICS]}
    card = CARDS[metric]
    sql, used_card = build_sql(card, dims, filters, month_from, month_to, variant)
    if sql is None:
        return used_card          # 需要澄清
    if sql == "CUSTOM":
        return _unit_protein_cost(dims, filters, month_from, month_to, compare)
    con = con_ro()
    try:
        df = con.execute(sql).fetchdf()
    except Exception as e:
        return {"error": str(e), "sql": sql}
    finally:
        con.close()
    _log({"tool": "query_metric", "metric": metric, "sql": sql, "rows": len(df)})
    result = {"metric": metric, "label": CARDS[metric]["label"], "sql": sql,
              "unit": used_card.get("unit", CARDS[metric].get("unit", "元/吨")),
              "caveats": used_card.get("caveats", CARDS[metric].get("caveats", "")),
              "rows": df.where(pd.notna(df), None).values.tolist(),
              "columns": list(df.columns)}
    # 环比/同比：要求 dims 含 month，按月序列计算
    if compare and "month" in (dims or []) and "month" in df.columns:
        df = df.sort_values("month")
        periods = sorted(df.month.unique())
        freq = 1 if compare == "环比" else 12
        gcols = [c for c in df.columns if c not in ("month", "value")]
        cmp_rows = []
        for key, g in df.groupby(gcols) if gcols else [(None, df)]:
            g = g.set_index("month").sort_index()
            chg = g.value.pct_change(freq) * 100
            for mth, c in chg.dropna().items():
                cmp_rows.append((list(key) if isinstance(key, tuple) else [key], mth, round(c, 2)))
        result["compare"] = compare
        result["compare_rows"] = cmp_rows
    return result

def _unit_protein_cost(dims, filters, month_from, month_to, compare):
    """单位蛋白成本 = Σ(r×price) / Σ(r×质检蛋白均值)"""
    con = con_ro()
    q = """
    WITH f AS (SELECT month, product, ingredient, pct/100.0 AS r FROM fact_formula),
    p AS (SELECT month, factory, ingredient, SUM(qty_t*unit_price)/SUM(qty_t) AS price
          FROM fact_purchase GROUP BY 1,2,3),
    q AS (SELECT ingredient, supplier, AVG(protein_pct) AS prot FROM fact_qc_batch GROUP BY 1,2),
    pp AS (SELECT p.month, p.factory, p.ingredient, p.price,
                  AVG(q.prot) AS prot
           FROM p JOIN fact_purchase po ON po.month=p.month AND po.factory=p.factory
                AND po.ingredient=p.ingredient
           JOIN q ON q.ingredient=po.ingredient AND q.supplier=po.supplier
           GROUP BY 1,2,3,4)
    SELECT f.month, f.product, pp.factory,
           SUM(f.r*pp.price)/NULLIF(SUM(f.r*pp.prot),0) AS value
    FROM f JOIN pp ON f.month=pp.month AND f.ingredient=pp.ingredient
    GROUP BY 1,2,3
    """
    df = con.execute(q).fetchdf()
    con.close()
    if filters:
        for k, v in filters.items():
            if k in df.columns:
                df = df[df[k] == v]
    if month_from:
        df = df[df.month >= month_from]
    if month_to:
        df = df[df.month <= month_to]
    dims = [d for d in (dims or ["month", "factory", "product"]) if d in df.columns]
    df = df.groupby(dims).value.mean().reset_index() if dims else df
    _log({"tool": "query_metric", "metric": "unit_protein_cost", "sql": q, "rows": len(df)})
    return {"metric": "unit_protein_cost", "label": "单位蛋白成本", "sql": q,
            "unit": "元/吨/蛋白百分点", "rows": df.values.tolist(), "columns": list(df.columns)}

# ---------------- 3. 吨成本因素分解（价格/配方/出成率三效应，和恒等） ----------------
def decompose_cost_change(product, factory, base_month, report_month):
    con = con_ro()
    def get(month):
        f = con.execute("SELECT ingredient, pct/100.0 AS r FROM fact_formula "
                        "WHERE month=? AND product=?", [month, product]).fetchdf()
        p = con.execute("SELECT ingredient, SUM(qty_t*unit_price)/SUM(qty_t) AS price "
                        "FROM fact_purchase WHERE month=? AND factory=? "
                        "GROUP BY ingredient", [month, factory]).fetchdf()
        y = con.execute("SELECT SUM(output_t)/SUM(input_t) FROM fact_production "
                        "WHERE month=? AND factory=? AND product=?",
                        [month, factory, product]).fetchone()[0]
        df = f.merge(p, on="ingredient", how="outer").fillna(0)
        return df.set_index("ingredient"), y
    b, y0 = get(base_month)
    r, y1 = get(report_month)
    idx = sorted(set(b.index) | set(r.index))
    b = b.reindex(idx).fillna(0); r = r.reindex(idx).fillna(0)
    C0 = float((b.r * b.price).sum() / y0)
    C1 = float((r.r * r.price).sum() / y1)
    # 三效应（顺序：先价格、后配方、再出成率；顺序敏感，见README关于Shapley的说明）
    price_eff = float((b.r * (r.price - b.price)).sum() / y0)
    formula_eff = float(((r.r - b.r) * r.price).sum() / y0)
    yield_eff = float((r.r * r.price).sum() * (1 / y1 - 1 / y0))
    detail = pd.DataFrame({
        "ingredient": idx,
        "价格效应": (b.r * (r.price - b.price) / y0).round(2),
        "配方效应": ((r.r - b.r) * r.price / y0).round(2),
    }).sort_values("价格效应", key=abs, ascending=False)
    con.close()
    total = price_eff + formula_eff + yield_eff
    return {
        "product": product, "factory": factory,
        "base": base_month, "report": report_month,
        "C0": round(C0, 1), "C1": round(C1, 1), "delta": round(C1 - C0, 1),
        "价格效应": round(price_eff, 2), "配方效应": round(formula_eff, 2),
        "出成率效应": round(yield_eff, 2),
        "效应合计": round(total, 2), "恒等校验": round(total - (C1 - C0), 6),
        "yield_base": round(y0 * 100, 2), "yield_report": round(y1 * 100, 2),
        "分原料明细": detail.head(6).to_dict("records"),
        "note": "分解采用连环替代法（价格→配方→出成率），结果受替代顺序影响；"
                "顺序无关的分解可用 Shapley 法。"}

# ---------------- 4. 配方优化（LP，标注“建议”） ----------------
PRODUCT_SPEC = {"草鱼料": "草鱼料", "鲈鱼膨化料": "加州鲈料",
                "育肥猪料": "育肥猪料", "对虾料": "对虾料"}

def optimize_formula(product, price_overrides=None):
    import importlib
    importlib.reload(formulation)
    from portfolio.common.formulation import solve_feed, BASE_PRICES, FEED_SPECS
    if product not in PRODUCT_SPEC:
        return {"error": f"产品 '{product}' 无配方规格", "available": list(PRODUCT_SPEC)}
    # 增补料种规格（与 data/build_data.py 保持一致）
    from data_extra import ensure_specs
    ensure_specs(FEED_SPECS)
    # 基线价格：最新月份实际采购均价
    con = con_ro()
    latest = con.execute("SELECT MAX(month) FROM fact_purchase").fetchone()[0]
    base = con.execute("SELECT ingredient, SUM(qty_t*unit_price)/SUM(qty_t) AS price "
                       "FROM fact_purchase WHERE month=? GROUP BY ingredient",
                       [latest]).fetchdf()
    con.close()
    prices = {**BASE_PRICES, **dict(zip(base.ingredient, base.price))}
    for k, v in (price_overrides or {}).items():
        if k in prices:
            prices[k] = prices[k] * (1 + v) if abs(v) <= 1 else float(v)
    cur = solve_feed(PRODUCT_SPEC[product], {k: v for k, v in prices.items()})
    res = {"product": product, "price_month_basis": latest,
           "price_overrides": price_overrides or {},
           "cost_yuan_per_t": round(cur["cost"], 1),
           "formula_pct": {k: round(float(v), 2) for k, v in cur["formula"].items() if v > 0.01},
           "紧约束影子价格": {k: round(float(v), 2) for k, v in cur["shadow"].items()
                            if abs(v) > 1e-6},
           "标注": "建议——需配方师确认后方可执行"}
    # 若给了价格假设，同时返回相对基准的变化
    if price_overrides:
        base_prices = {**BASE_PRICES, **dict(zip(base.ingredient, base.price))}
        ref = solve_feed(PRODUCT_SPEC[product], base_prices)
        res["基准吨成本"] = round(ref["cost"], 1)
        res["成本变化"] = round(cur["cost"] - ref["cost"], 1)
        res["基准配方_pct"] = {k: round(float(v), 2) for k, v in ref["formula"].items() if v > 0.01}
    _log({"tool": "optimize_formula", "product": product, "overrides": price_overrides})
    return res

# ---------------- 5. 只读 SQL（兜底，记日志） ----------------
FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|ATTACH|COPY|EXPORT|IMPORT|PRAGMA|SET)\b",
                       re.IGNORECASE)
MAX_ROWS = 500

def run_readonly_sql(sql):
    sql = sql.strip().rstrip(";")
    if not re.match(r"^(SELECT|WITH|EXPLAIN)\b", sql, re.IGNORECASE):
        return {"error": "只允许 SELECT/WITH 只读查询"}
    if FORBIDDEN.search(sql):
        return {"error": "检测到写操作关键字，已拒绝"}
    if " LIMIT " not in sql.upper():
        sql += f" LIMIT {MAX_ROWS}"
    con = con_ro()
    try:
        df = con.execute(sql).fetchdf()
        _log({"tool": "run_readonly_sql", "sql": sql, "rows": len(df)})
        return {"columns": list(df.columns),
                "rows": df.where(pd.notna(df), None).values.tolist(),
                "truncated_to": MAX_ROWS, "sql": sql}
    except Exception as e:
        _log({"tool": "run_readonly_sql", "sql": sql, "error": str(e)})
        return {"error": str(e), "sql": sql}
    finally:
        con.close()

# ---------------- 6. 简易 RAG：检索既有文档 ----------------
_DOCS_CACHE = None
def _load_docs():
    global _DOCS_CACHE
    if _DOCS_CACHE is not None:
        return _DOCS_CACHE
    chunks = []
    def add(source, text):
        text = text.strip()
        if len(text) > 30:
            chunks.append({"source": source, "text": text})
    # portfolio 报告与说明
    for rel in ["reports/作品集报告.md", "docs/业务说明文档.md", "README.md"]:
        p = os.path.join(PORTFOLIO, rel)
        if os.path.exists(p):
            txt = open(p, encoding="utf-8").read()
            for sec in re.split(r"\n#{1,3} ", txt):
                add(rel, sec)
    # 指标字典与假设登记表（逐行）；openpyxl 缺失时跳过，不影响主流程
    for rel in ["outputs/指标字典.xlsx", "outputs/假设登记表.xlsx"]:
        p = os.path.join(PORTFOLIO, rel)
        if os.path.exists(p):
            try:
                df = pd.read_excel(p)
            except ImportError:
                continue
            for _, row in df.iterrows():
                add(rel, " | ".join(str(v) for v in row.values))
    _DOCS_CACHE = chunks
    return chunks

def search_docs(query, top_k=3):
    """TF-IDF 字符 n-gram 检索（中文免分词），返回最相关的文档片段。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    chunks = _load_docs()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    X = vec.fit_transform([c["text"] for c in chunks])
    q = vec.transform([query])
    scores = (X @ q.T).toarray().ravel()
    top = scores.argsort()[::-1][:top_k]
    return {"query": query, "results": [
        {"source": chunks[i]["source"], "text": chunks[i]["text"][:600],
         "score": round(float(scores[i]), 3)} for i in top if scores[i] > 0]}

TOOLS = {"list_metrics": list_metrics, "query_metric": query_metric,
         "decompose_cost_change": decompose_cost_change,
         "optimize_formula": optimize_formula,
         "run_readonly_sql": run_readonly_sql, "search_docs": search_docs}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_metrics", "description": "列出指标目录（名称/口径/是否需澄清）", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "query_metric",
        "description": "按指标卡查询指标。dims 维度、filters 筛选、month_from/to 期间、compare 环比|同比、variant 口径变体（库存周转天数必须指定）",
        "parameters": {"type": "object", "properties": {
            "metric": {"type": "string"}, "dims": {"type": "array", "items": {"type": "string"}},
            "filters": {"type": "object"}, "month_from": {"type": "string"},
            "month_to": {"type": "string"}, "compare": {"type": "string"},
            "variant": {"type": "string"}}, "required": ["metric"]}}},
    {"type": "function", "function": {
        "name": "decompose_cost_change",
        "description": "吨料原料成本的两期因素分解：价格效应/配方效应/出成率效应，含分原料明细",
        "parameters": {"type": "object", "properties": {
            "product": {"type": "string"}, "factory": {"type": "string"},
            "base_month": {"type": "string"}, "report_month": {"type": "string"}},
            "required": ["product", "factory", "base_month", "report_month"]}}},
    {"type": "function", "function": {
        "name": "optimize_formula",
        "description": "最低成本配方优化（线性规划）。price_overrides 为价格假设，如 {\"豆粕\": 0.1} 表示涨价10%。输出标注为建议",
        "parameters": {"type": "object", "properties": {
            "product": {"type": "string"}, "price_overrides": {"type": "object"}},
            "required": ["product"]}}},
    {"type": "function", "function": {
        "name": "run_readonly_sql",
        "description": "只读 SQL 兜底查询（自动加行数上限，记日志）",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}},
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "检索既有分析文档/指标字典/假设登记表（RAG）",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]}}},
]
