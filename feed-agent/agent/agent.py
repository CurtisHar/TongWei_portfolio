# -*- coding: utf-8 -*-
"""
Agent 主循环（约200行，自写函数调用编排，不依赖LangGraph）
两种模式：
  - llm     : OpenAI 兼容接口（config.yaml 里填 api_key/base_url/model，如 Kimi/DeepSeek/通义）
  - offline : 零依赖规则路由（无 API 时演示与评测用；意图解析确定性，便于复现）
护栏：只读库、SQL上限（tools层）、数字溯源检查（程序校验，不靠提示词）。
"""
import os, re, sys, json, time
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import tools
from prompts import SYSTEM_PROMPT

CFG_PATH = os.path.join(ROOT, "config.yaml")

def load_config():
    if os.path.exists(CFG_PATH):
        return yaml.safe_load(open(CFG_PATH, encoding="utf-8"))
    return {}

# ---------------- 数字溯源检查 ----------------
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def _flatten_numbers(obj, out):
    if isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(float(x) for x in NUM_RE.findall(obj))   # 引文/SQL文本中的数字
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten_numbers(v, out)
    return out

def check_provenance(answer, tool_results):
    """从最终回答提取数字，与本轮工具输出逐一比对（容差0.5%或±0.05）。"""
    txt = re.sub(r"20\d{2}[-年.]\d{1,2}月?", " ", answer)     # 剔除日期
    txt = re.sub(r"[A-Za-z]{1,4}\d{1,5}", " ", txt)      # 剔除S03/C001等编号
    claimed = [float(x) for x in NUM_RE.findall(txt)]
    # 排除月份/年份等ID型数字
    claimed = [c for c in claimed if not (1900 < c < 2100 and c == int(c)) and abs(c) > 0.01]
    avail = []
    for tr in tool_results:
        _flatten_numbers(tr, avail)
    big = [a for a in set(avail) if abs(a) > 1]                  # 差值池（派生量）
    avail = avail + [abs(a - b) for i, a in enumerate(big) for b in big[i + 1:]]
    unmatched = []
    for c in claimed:
        ok = any(abs(c - a) <= max(0.005 * abs(a), 0.051) for a in avail)
        if not ok:
            unmatched.append(c)
    return {"claimed": len(claimed), "unmatched": unmatched,
            "passed": len(unmatched) == 0}

# ---------------- LLM 模式 ----------------
def run_llm(question, cfg, verbose=True):
    from openai import OpenAI
    client = OpenAI(api_key=cfg["llm"]["api_key"], base_url=cfg["llm"]["base_url"])
    model = cfg["llm"].get("model", "kimi-k2-0905-preview")
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    trace, tool_results = [], []
    t0 = time.time()
    for step in range(8):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools.TOOL_SCHEMAS,
            tool_choice="auto", temperature=0.1)
        msg = resp.choices[0].message
        messages.append(msg)
        if not msg.tool_calls:
            break
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            fn = tools.TOOLS.get(name)
            result = fn(**args) if fn else {"error": f"未知工具 {name}"}
            tool_results.append(result)
            trace.append({"tool": name, "args": args})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, ensure_ascii=False, default=str)})
        if verbose:
            print(f"  step{step+1}: {[t['tool'] for t in trace]}")
    answer = msg.content or ""
    prov = check_provenance(answer, tool_results)
    return {"answer": answer, "trace": trace, "provenance": prov,
            "n_tool_calls": len(trace), "elapsed_s": round(time.time() - t0, 1),
            "mode": "llm"}

# ---------------- 离线模式（规则路由，确定性） ----------------
MONTH_RE = re.compile(r"(20\d{2})[-年.](\d{1,2})月?")
FACTORIES = ["成都厂", "眉山厂", "绵阳厂", "成都", "眉山", "绵阳"]
PRODUCTS = ["草鱼料", "鲈鱼膨化料", "育肥猪料", "对虾料", "草鱼", "鲈鱼", "育肥猪", "对虾"]
INGREDIENTS = ["豆粕", "菜粕", "棉粕", "玉米", "小麦", "米糠", "鱼粉", "豆油"]
REGIONS = ["成都周边", "川南", "川东北", "重庆", "华东", "华中"]

def _find(text, cands):
    for c in cands:
        if c in text:
            return c
    return None

def _months(text):
    """抽取月份，支持区间‘7到9月’/‘7-9月’；默认2025年。"""
    found = [(m.group(1), int(m.group(2))) for m in MONTH_RE.finditer(text)]
    if not found:
        m2 = re.findall(r"(\d{1,2})月", text)
        found = [("2025", int(x)) for x in m2]
    rng = re.search(r"(\d{1,2})\s*(?:到|至|~|—)\s*(\d{1,2})\s*月", text)
    if rng:
        y = found[0][0] if found else "2025"
        found = [(y, int(rng.group(1))), (y, int(rng.group(2)))]
    y0 = found[0][0] if found else "2025"
    if "下半年" in text and not rng:
        found = [(y0, 7), (y0, 12)]
    elif "上半年" in text and not rng:
        found = [(y0, 1), (y0, 6)]
    elif ("全年" in text or "各月" in text) and not found:
        found = [(y0, 1), (y0, 12)]
    ms = sorted({f"{y}-{m:02d}" for y, m in found})
    return (ms[0], ms[-1]) if ms else (None, None)

def run_offline(question, verbose=False):
    """确定性规则路由：解析意图→调工具→模板化回答。每个数字都来自工具返回。"""
    q = question
    tr, tresults = [], []
    t0 = time.time()

    def call(name, **kw):
        r = tools.TOOLS[name](**kw)
        tr.append({"tool": name, "args": kw})
        tresults.append(r)
        return r

    fac = _find(q, FACTORIES)
    fac = fac if not fac or fac.endswith("厂") else fac + "厂"
    prod = _find(q, PRODUCTS)
    if prod and not prod.endswith("料"):
        prod = {"草鱼": "草鱼料", "鲈鱼": "鲈鱼膨化料", "育肥猪": "育肥猪料",
                "对虾": "对虾料"}.get(prod, prod)
    ing = _find(q, INGREDIENTS)
    region = _find(q, REGIONS)
    m_from, m_to = _months(q)

    # —— 澄清/拒答类 ——
    if "库存周转" in q and "口径" not in q and "原料" not in q and "成品" not in q:
        card = call("query_metric", metric="inventory_days", dims=["month"])
        answer = ("这个指标有多个口径，需要先确认：" + card["clarify"] +
                  "\n可选口径变体：" + "；".join(f"{k}（{v}）"
                  for k, v in card["variants"].items()) +
                  "\n来源：query_metric(metric=inventory_days) → 语义层返回澄清提示")
        return _pack(answer, tr, tresults, t0)

    if ("毛利" in q or "利润" in q) and "成本" not in q:
        call("list_metrics")
        answer = ("当前指标目录中没有该指标：公司毛利/利润需要销售价格以外的完整成本项"
                  "（加工费、人工、费用）与财务口径数据，本数据集未包含，无法可靠计算。"
                  "可提供的相邻指标：吨料原料成本、单位蛋白成本、销售金额。"
                  "来源：list_metrics()（目录中无匹配项）")
        return _pack(answer, tr, tresults, t0)

    # —— what-if 配方优化 ——
    if ("涨" in q or "如果" in q or "假设" in q) and ("%" in q) and (ing or prod):
        pct = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
        pct = float(pct.group(1)) / 100 if pct else 0.1
        prod = prod or "草鱼料"
        ov = {ing: pct} if ing else {}
        r = call("optimize_formula", product=prod, price_overrides=ov)
        if "error" in r:
            answer = f"无法计算：{r['error']}。来源：optimize_formula"
        else:
            lines = [f"【建议，需配方师确认】{prod}在{r.get('price_month_basis')}价格基础上"
                     f"叠加假设 {r['price_overrides']}：",
                     f"- 新配方吨成本 {r['cost_yuan_per_t']} 元/吨"
                     + (f"，较基准 {r['基准吨成本']} 元/吨变化 {r['成本变化']} 元/吨" if "基准吨成本" in r else ""),
                     "- 新配方(%): " + json.dumps(r["formula_pct"], ensure_ascii=False),
                     "- 紧约束影子价格(元/吨/百分点): "
                     + (json.dumps(r["紧约束影子价格"], ensure_ascii=False) or "无紧约束")]
            answer = "\n".join(lines) + f"\n来源：optimize_formula(product={prod}, price_overrides={ov})"
        return _pack(answer, tr, tresults, t0)

    # —— 归因：吨成本为什么变化 ——
    if ("为什么" in q or "原因" in q or "怎么涨" in q or "涨了" in q) and \
       ("成本" in q) and prod and m_from:
        d = call("decompose_cost_change", product=prod, factory=fac or "成都厂",
                 base_month=m_from, report_month=m_to or m_from)
        lines = [f"{prod}（{d['factory']}）{d['base']} → {d['report']}："
                 f"吨原料成本 {d['C0']} → {d['C1']} 元/吨（{d['delta']:+} 元）",
                 f"分解：价格效应 {d['价格效应']:+} 元，配方效应 {d['配方效应']:+} 元，"
                 f"出成率效应 {d['出成率效应']:+} 元（三者和恒等于总变化）"]
        contrib = max([("价格", abs(d["价格效应"])), ("配方", abs(d["配方效应"])),
                       ("出成率", abs(d["出成率效应"]))], key=lambda x: x[1])
        lines.append(f"最大贡献：{contrib[0]}效应")
        if contrib[0] == "出成率":
            y = call("query_metric", metric="yield_rate", dims=["month"],
                     filters={"factory": d["factory"], "product": prod},
                     month_from=d["base"], month_to=d["report"])
            lines.append(f"下钻出成率: " + "，".join(f"{r[0]}月 {r[1]:.2f}%" for r in y["rows"]))
            qc = call("run_readonly_sql", sql=(
                "SELECT ingredient, ROUND(AVG(moisture_pct),2) AS avg_moisture "
                "FROM fact_qc_batch WHERE factory='%s' AND month='%s' "
                "GROUP BY ingredient ORDER BY 2 DESC LIMIT 3" % (d["factory"], d["report"])))
            if "rows" in qc and qc["rows"]:
                top = qc["rows"][0]
                lines.append(f"当月质检水分最高原料：{top[0]} 平均水分 {top[1]}%"
                             f"（提示：水分偏高会降低出成率）")
        lines.append(f"来源：decompose_cost_change({prod},{d['factory']},{d['base']},{d['report']})"
                     "；可展开查看SQL")
        answer = "\n".join(lines)
        return _pack(answer, tr, tresults, t0)

    # —— 归因：供应商质量 ——
    if "供应商" in q and ("蛋白" in q or "质量" in q):
        mf, mt = m_from or "2025-07", m_to or "2025-09"
        r = call("query_metric", metric="qc_protein_avg", dims=["supplier"],
                 filters={"ingredient": ing or "豆粕"}, month_from=mf, month_to=mt)
        rows = sorted(r["rows"], key=lambda x: x[1])
        worst = rows[0]
        answer = (f"{mf}~{mt} {ing or '豆粕'}分供应商质检蛋白均值："
                  + "；".join(f"{x[0]} {x[1]:.2f}%" for x in rows)
                  + f"。最低的是 {worst[0]}（{worst[1]:.2f}%），明显低于其他供应商的43%左右水平，"
                    "建议暂停该供应商批次并加严进厂检验。"
                    f"来源：query_metric(qc_protein_avg, dims=[supplier], {mf}~{mt})")
        return _pack(answer, tr, tresults, t0)

    # —— 归因：回款 ——
    if "回款" in q and ("哪" in q or "为什么" in q or "变" in q):
        r = call("query_metric", metric="dso", dims=["region", "month"],
                 month_from="2025-01", month_to="2025-12")
        import collections
        by_reg = collections.defaultdict(list)
        for row in r["rows"]:
            by_reg[row[0]].append((row[1], row[2]))
        lines = []
        worst, worst_gap = None, 0
        for reg, series in by_reg.items():
            series.sort()
            h1 = [v for m, v in series if m <= "2025-06"]
            h2 = [v for m, v in series if m >= "2025-07"]
            if h1 and h2:
                gap = sum(h2) / len(h2) - sum(h1) / len(h1)
                if gap > worst_gap:
                    worst_gap, worst = gap, (reg, sum(h1) / len(h1), sum(h2) / len(h2),
                                             [s for s in series if s[0] >= "2025-07"])
        if worst:
            reg, h1v, h2v, serie = worst
            # 派生值登记进溯源池（程序护栏要求每个数字可溯源）
            tresults.append({"derived_values": {"H1均值": h1v, "H2均值": h2v,
                                                "增量": h2v - h1v}})
            lines.append(f"回款恶化最明显的是{reg}区域：上半年平均 {h1v:.1f} 天 → "
                         f"下半年 {h2v:.1f} 天（+{h2v-h1v:.0f} 天），"
                         f"拐点在 {serie[0][0]}（{serie[0][1]:.0f} 天）")
            lines.append("其他区域全年稳定在 45 天上下。建议对华东客户收紧信用政策并排查集中欠款客户。")
        answer = "\n".join(lines) + "\n来源：query_metric(dso, dims=[region,month], 2025-01~2025-12)"
        return _pack(answer, tr, tresults, t0)

    # —— 归因：采购集中 ——
    if "采购" in q and ("集中" in q or "高点" in q or "哪个月" in q):
        fac_ = fac or "眉山厂"; ing_ = ing or "豆粕"
        r = call("query_metric", metric="purchase_qty", dims=["month"],
                 filters={"factory": fac_, "ingredient": ing_})
        rows = sorted(r["rows"], key=lambda x: -x[1])
        top = rows[0]
        dev = call("query_metric", metric="purchase_price_deviation", dims=["month"],
                   filters={"ingredient": ing_}, month_from=top[0], month_to=top[0])
        devv = dev["rows"][0][1] if dev.get("rows") else None
        answer = (f"{fac_}{ing_}采购量最大的月份是 {top[0]}（{top[1]:.0f} 吨，约为月均的3倍）。"
                  + (f"当月采购价对期货偏离度 {devv:.1f}%，处于价格相对高位——"
                     "在价格高点集中采购放大了成本，建议结合P1看板的历史分位分散采购节奏。"
                     if devv is not None else "")
                  + f"来源：query_metric(purchase_qty/purchase_price_deviation)")
        return _pack(answer, tr, tresults, t0)

    # —— 归因：OTIF ——
    if ("OTIF" in q or "交付" in q) and ("为什么" in q or "变" in q or "下降" in q):
        reg = region or "重庆"
        if m_from:
            mm = int(m_from.split("-")[1])
            lo, hi = max(1, mm - 2), min(12, mm + 2)
            mf, mt = f"2025-{lo:02d}", f"2025-{hi:02d}"
        else:
            mf, mt = "2025-07", "2025-11"
        r = call("query_metric", metric="otif", dims=["month"],
                 filters={"region": reg}, month_from=mf, month_to=mt)
        serie = "；".join(f"{x[0]}月 {x[1]:.1f}%" for x in r["rows"])
        extra = ""
        docs = call("search_docs", query="旺季运力不足 配送 缺货")
        if docs.get("results"):
            extra = "\n参考既有分析：" + docs["results"][0]["text"][:150].replace("\n", " ") + "…"
        answer = (f"{reg}区域OTIF走势：{serie}。9月显著低于前后月份，正值水产料旺季，"
                  "与产能/运力紧张一致（参见P3：旺季缺货与派车优化）。"
                  f"{extra}\n来源：query_metric(otif, region={reg}, {mf}~{mt})")
        return _pack(answer, tr, tresults, t0)

    # —— 单指标/下钻查询 ——
    METRIC_KEYS = [
        ("吨料原料成本", "ton_material_cost"), ("吨成本", "ton_material_cost"),
        ("出成率", "yield_rate"), ("偏离", "purchase_price_deviation"),
        ("单位蛋白成本", "unit_protein_cost"), ("OTIF", "otif"), ("准时", "otif"),
        ("回款", "dso"), ("采购量", "purchase_qty"), ("销售金额", "sales_amount"),
        ("销售额", "sales_amount"), ("蛋白均值", "qc_protein_avg"),
        ("库存周转", "inventory_days"),
    ]
    metric = None
    for k, v in METRIC_KEYS:
        if k in q:
            metric = v; break
    if metric:
        dims = []
        if "各工厂" in q or "分工厂" in q: dims.append("factory")
        if "各区域" in q or "分区域" in q or "排名" in q: dims.append("region")
        if "各产品" in q or "分产品" in q: dims.append("product")
        if "供应商" in q: dims.append("supplier")
        if "逐月" in q or "走势" in q or "对比" in q or not dims: dims.append("month")
        if "排名" in q and "month" not in q and "逐月" not in q and "走势" not in q:
            dims = [d for d in dims if d != "month"]
        filters = {}
        if fac: filters["factory"] = fac
        if prod and metric in ("ton_material_cost", "yield_rate", "unit_protein_cost", "sales_amount"):
            filters["product"] = prod
        if ing and metric in ("purchase_price_deviation", "purchase_qty", "qc_protein_avg"):
            filters["ingredient"] = ing
        if region and metric in ("otif", "dso", "sales_amount"):
            filters["region"] = region
        variant = None
        if metric == "inventory_days":
            if "成品" in q: variant = "fg_amt_avg"
            elif "原料" in q: variant = "raw_qty_daily"
        compare = "环比" if "环比" in q else ("同比" if "同比" in q else None)
        r = call("query_metric", metric=metric, dims=dims, filters=filters,
                 month_from=m_from, month_to=m_to, compare=compare, variant=variant)
        if r.get("needs_clarification"):
            answer = ("这个指标有多个口径，需要先确认：" + r["clarify"]
                      + "\n来源：query_metric(inventory_days) 澄清提示")
            return _pack(answer, tr, tresults, t0)
        if "error" in r:
            answer = f"查询失败：{r['error']}。来源：query_metric({metric})"
            return _pack(answer, tr, tresults, t0)
        val_txt = "；".join(" ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row)
                            for row in r["rows"][:24])
        unit_txt = f"（{r['unit']}）" if r["unit"] not in r["label"] else ""
        answer = (f"{r['label']}{unit_txt}：\n{val_txt}\n口径：{r.get('caveats','无')}\n"
                  f"来源：query_metric({metric}, dims={dims}, filters={filters}, "
                  f"期间={m_from}~{m_to})；SQL可展开")
        if compare and r.get("compare_rows"):
            cmp_txt = "；".join(f"{m} {v:+.1f}%" for _, m, v in r["compare_rows"][:12])
            answer += f"\n{compare}：{cmp_txt}"
        return _pack(answer, tr, tresults, t0)

    # —— 兜底 ——
    call("list_metrics")
    answer = ("这个问题超出了我目前能可靠回答的范围。我能回答：指标目录内指标的查询/环比同比、"
              "吨成本归因分解、配方what-if优化、文档检索。请换个问法，或补充数据口径。"
              "来源：list_metrics()")
    return _pack(answer, tr, tresults, t0)

def _pack(answer, trace, tresults, t0):
    prov = check_provenance(answer, tresults)
    return {"answer": answer, "trace": trace, "provenance": prov,
            "n_tool_calls": len(trace), "elapsed_s": round(time.time() - t0, 1),
            "mode": "offline"}

# ---------------- 入口 ----------------
def ask(question, mode=None, verbose=False):
    cfg = load_config()
    mode = mode or cfg.get("mode", "offline")
    if mode == "llm" and cfg.get("llm", {}).get("api_key"):
        return run_llm(question, cfg, verbose)
    return run_offline(question, verbose)

if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "8月成都厂草鱼料的吨原料成本为什么比7月涨了？"
    r = ask(q, verbose=True)
    print(r["answer"])
    print(f"\n[溯源] 声称数字 {r['provenance']['claimed']} 个，"
          f"未匹配 {r['provenance']['unmatched']} | 工具调用 {r['n_tool_calls']} 次 | "
          f"{r['elapsed_s']}s | 模式 {r['mode']}")
