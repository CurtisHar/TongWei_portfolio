# -*- coding: utf-8 -*-
"""
评测脚本：加载 questions.yaml，逐题跑 agent.ask(mode="offline")，按 check 规则判分。
用法：python eval/run_eval.py v0.1
输出：eval/results_<version>.json + 控制台汇总表
"""
import os, re, sys, json, time
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
import agent

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def extract_numbers(text):
    """与 agent.check_provenance 同款预处理：剔除日期/编号，再提数字。"""
    txt = re.sub(r"20\d{2}[-年.]\d{1,2}月?", " ", text)
    txt = re.sub(r"\b[A-Za-z]{1,4}\d{1,5}\b", " ", txt)
    nums = [float(x) for x in NUM_RE.findall(txt)]
    return [c for c in nums if not (1900 < c < 2100 and c == int(c)) and abs(c) > 0.01]

def judge(check, answer):
    """返回 (passed, detail)"""
    if "numeric" in check:
        target, tol = check["numeric"], check.get("tol_pct", 1.0)
        for n in extract_numbers(answer):
            if abs(n - target) <= max(abs(target) * tol / 100, 0.051):
                extra = True
                if "contains" in check and not all(s in answer for s in check["contains"]):
                    return False, f"数值{target}命中但缺少关键词{check['contains']}"
                return True, f"数值命中 {n}≈{target}"
        return False, f"未找到数值 {target}±{tol}%"
    if "numeric_range" in check:
        lo, hi = check["numeric_range"]
        for n in extract_numbers(answer):
            if lo <= n <= hi:
                if "contains" in check and not all(s in answer for s in check["contains"]):
                    return False, f"数值{n}在区间但缺少关键词{check['contains']}"
                return True, f"数值命中 {n}∈[{lo},{hi}]"
        return False, f"未找到数值落在 [{lo},{hi}]"
    if "contains" in check:
        missing = [s for s in check["contains"] if s not in answer]
        return (not missing), ("关键词齐" if not missing else f"缺少关键词 {missing}")
    if check.get("must_clarify"):
        ok = ("口径" in answer) and ("确认" in answer or "请" in answer)
        return ok, ("已澄清口径" if ok else "未给出口径澄清")
    if check.get("must_refuse"):
        ok = bool(re.search(r"无法|不在目录|未包含|缺", answer)) and not re.search(
            r"毛利率.{0,20}\d+\.?\d*\s*%|净利润.{0,20}\d+\.?\d*\s*万", answer)
        return ok, ("已拒答并说明" if ok else "应拒答但未拒（或编造了数字）")
    return False, f"未知check类型 {check}"

def main(version):
    qs = yaml.safe_load(open(os.path.join(HERE, "questions.yaml"), encoding="utf-8"))["questions"]
    results = []
    t_start = time.time()
    for item in qs:
        r = agent.ask(item["q"], mode="offline")
        passed, detail = judge(item["check"], r["answer"])
        root_hit = None
        if "root_cause" in item:
            key = item["root_cause"].split("：")[0]
            # 根因命中：归因主词出现在回答里
            probe = {"出成率效应": "出成率", "S03蛋白偏低": "S03",
                     "华东H2回款恶化": "华东", "2025-03高点集中采购": "2025-03",
                     "旺季运力不足": "运力"}.get(item["root_cause"], key)
            root_hit = probe in r["answer"]
        results.append({
            "id": item["id"], "type": item["type"], "q": item["q"],
            "passed": passed, "judge": detail, "root_cause_hit": root_hit,
            "provenance_passed": r["provenance"]["passed"],
            "unmatched": r["provenance"]["unmatched"],
            "n_tool_calls": r["n_tool_calls"], "elapsed_s": r["elapsed_s"],
            "answer": r["answer"],
        })
        mark = "✓" if passed else "✗"
        print(f"{mark} {item['id']} [{item['type']}] {detail} "
              f"(溯源{'过' if r['provenance']['passed'] else '挂:'+str(r['provenance']['unmatched'][:3])}, "
              f"{r['n_tool_calls']}调用, {r['elapsed_s']}s)")

    n = len(results)
    npass = sum(1 for x in results if x["passed"])
    prov = sum(1 for x in results if x["provenance_passed"])
    by_type = {}
    for x in results:
        d = by_type.setdefault(x["type"], [0, 0])
        d[0] += x["passed"]; d[1] += 1
    attrib = [x for x in results if x["root_cause_hit"] is not None]
    summary = {
        "version": version, "mode": "offline", "n_questions": n,
        "accuracy": round(npass / n, 4),
        "provenance_pass_rate": round(prov / n, 4),
        "avg_tool_calls": round(sum(x["n_tool_calls"] for x in results) / n, 2),
        "avg_elapsed_s": round(sum(x["elapsed_s"] for x in results) / n, 2),
        "total_elapsed_s": round(time.time() - t_start, 1),
        "by_type": {k: {"acc": round(a / b, 3), "n": b} for k, (a, b) in by_type.items()},
        "root_cause_hit_rate": (round(sum(x["root_cause_hit"] for x in attrib) / len(attrib), 3)
                                if attrib else None),
        "failed_ids": [x["id"] for x in results if not x["passed"]],
        "provenance_failed_ids": [x["id"] for x in results if not x["provenance_passed"]],
    }
    out = {"summary": summary, "results": results}
    fp = os.path.join(HERE, f"results_{version}.json")
    json.dump(out, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n===== 汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"已写入 {fp}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v0.1")
