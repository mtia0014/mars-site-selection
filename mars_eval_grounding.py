"""
MARS 生成层 + 工具层指标（离线后处理，读 eval_product_result.json 里的 trajectory / final_answer）。

本脚本补上两块之前空白的评测，全部确定性地从「query 信息需求」推导，不引入 LLM-judge：

1. 数字溯源率（不是"幻觉率"）
   = 答案里的事实性数字，有多大比例能在工具返回(observation)里找到同值来源。
   量的是「数字有出处」，不是「解读正确」——引用≠正确，这个边界在卡里要讲清。
   - 数值归一化：3.5万→35000；千分位 33,822→33822（避免误判为编造）
   - 排除 15 位以上长数字 = mall_id 标识符（由 agent 从返回里复制，天然有源，不计入事实断言）

2. 工具召回率 / 工具精确率（只统计 7 个非守卫工具）
   - 守卫工具 score_customer_match / analyze_brand_competition 由护栏强制，天然 100%，无信息量，排除。
   - 标注规则（从 query 信息需求推导，不是从 Router/实现规则推导）：
       "有X"/"X聚集"/"X丰富"/"奢侈"/"珠宝"/"运动" → 召回类 = {vector_search_malls, search_mall}
       "客流最大"                                     → 客流类 = {get_traffic_ranking, search_mall}
       "评分/档次/区域/最好"                           → 结构化 = {search_mall}
       "客群/人群/年轻/银发/亲子"                      → 客群类 = {match_customer_profile}
     "必需工具类" = 满足该信息需求的一组可互换工具（用类而非单工具，避免"关键词检索 vs 语义检索"这种
     同效选择被误判为选错）。
   - 工具召回率 = 必需工具类里至少一个被调用的 query 占比（"该用的工具用没用"）
   - 工具精确率 = 实际调用的非守卫工具里，属于「相关工具集」的比例（"用没用废工具"）

用法: python mars_eval_grounding.py
"""

import json
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULT_FILE = "eval_product_result.json"

GUARD = {"score_customer_match", "analyze_brand_competition"}
NON_GUARD = {
    "search_mall", "get_traffic_ranking", "match_customer_profile",
    "get_mall_detail", "compare_malls", "analyze_customer_profile", "vector_search_malls",
}

MAG = {"万": 1e4, "亿": 1e8, "千": 1e3}
_NUM = re.compile(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(万|亿|千)?")


def extract_numbers(text):
    """提取数值：千分位 + 量级归一化（33,822→33822，3.5万→35000），排除 mall_id（≥15 位整数）。"""
    out = []
    for m in _NUM.finditer(text or ""):
        raw, mag = m.group(1), m.group(2)
        if mag is None and "," not in raw and "." not in raw and len(raw) >= 15:
            continue
        val = float(raw.replace(",", ""))
        if mag:
            val *= MAG[mag]
        out.append(val)
    return out


def grounding(case):
    fa_nums = extract_numbers(case.get("final_answer", ""))
    obs_text = "\n".join(s.get("observation", "") for s in case.get("trajectory", []))
    obs_nums = set(extract_numbers(obs_text))
    if not fa_nums:
        return 1.0, 0, []
    unmatched = [v for v in fa_nums if v not in obs_nums]
    return (len(fa_nums) - len(unmatched)) / len(fa_nums), len(fa_nums), unmatched


def annotate(query):
    """返回 (必需工具类, 相关工具集)。必需类 = 可互换、任一满足即算选对。"""
    q = query
    # 1) 客群/人群（复合评分/高端时 search_mall 也相关）
    if any(k in q for k in ["白领", "高收入", "亲子", "银发", "老年", "年轻"]):
        rel = {"match_customer_profile", "analyze_customer_profile"}
        if "评分" in q or "高端" in q:
            rel.add("search_mall")
        return {"match_customer_profile"}, rel
    # 2) 语义召回（品牌/聚集/奢侈/珠宝/运动/丰富）
    if any(k in q for k in ["有", "聚集", "丰富", "奢侈", "珠宝", "运动"]):
        rel = {"vector_search_malls", "search_mall"}
        if "客流大" in q:
            rel.add("get_traffic_ranking")
        return {"vector_search_malls", "search_mall"}, rel
    # 3) 客流
    if "客流" in q:
        if "高端" in q or "评分" in q:  # 高客流的高端 = 复合硬约束 → 结构化
            return {"search_mall"}, {"search_mall", "get_traffic_ranking"}
        return {"get_traffic_ranking", "search_mall"}, {"get_traffic_ranking", "search_mall"}
    # 4) 结构化（评分/档次/区域/最好）
    return {"search_mall"}, {"search_mall"}


def non_guard_calls(case):
    calls = set()
    for s in case.get("trajectory", []):
        a = (s.get("action") or "").strip()
        if a and a in NON_GUARD:
            calls.add(a)
    return calls


def _type_of(required):
    if "match_customer_profile" in required:
        return "客群"
    if "get_traffic_ranking" in required:
        return "客流"
    if "vector_search_malls" in required:
        return "语义/品牌"
    return "结构化"


def main():
    data = json.load(open(RESULT_FILE, encoding="utf-8"))
    cases = data["cases"]

    # ===== 1. 数字溯源率 =====
    rates, total_nums, all_unmatched = [], 0, []
    for c in cases:
        r, n, um = grounding(c)
        rates.append(r)
        total_nums += n
        if um:
            all_unmatched.append((c["query"], um))
    rate = sum(rates) / len(rates)
    print("=" * 70)
    print("1. 数字溯源率（答案数字有工具返回来源的比例）")
    print("=" * 70)
    print(f"  答案事实数字总数      : {total_nums}")
    print(f"  数字溯源率（按数字）  : {rate:.4f}")
    print(f"  数字溯源率（按 query）: {sum(r == 1.0 for r in rates)}/{len(rates)} 条答案所有数字都有源")
    print(f"  有未溯源数字的 query  : {len(all_unmatched)}/{len(cases)}")
    if all_unmatched:
        print("  —— 未溯源数字明细（复核：自指计数/取整/真编造）:")
        for q, um in all_unmatched:
            print(f"    [{q}] {um}")

    # ===== 2. 工具召回率 / 精确率 =====
    print("\n" + "=" * 70)
    print("2. 工具召回率 / 工具精确率（7 个非守卫工具）")
    print("=" * 70)
    recalled = 0
    prec_sum = 0.0
    prec_n = 0
    missed = []
    impure = []
    type_stat = {}
    for c in cases:
        required, relevant = annotate(c["query"])
        calls = non_guard_calls(c)
        ok = bool(calls & required)
        recalled += 1 if ok else 0
        if not ok:
            missed.append((c["query"], sorted(required), sorted(calls)))
        if calls:
            hit = len(calls & relevant)
            prec_sum += hit / len(calls)
            prec_n += 1
            junk = sorted(calls - relevant)
            if junk:
                impure.append((c["query"], junk, sorted(calls)))
        t = _type_of(required)
        tr, td, tp, tn = type_stat.get(t, (0, 0, 0.0, 0))
        type_stat[t] = (tr + (1 if ok else 0), td + 1, tp + (hit / len(calls) if calls else 0), tn + (1 if calls else 0))

    recall = recalled / len(cases)
    prec = prec_sum / prec_n if prec_n else 0.0
    print(f"  工具召回率（必需工具类被调用）: {recall:.4f}  ({recalled}/{len(cases)})")
    print(f"  工具精确率（非守卫调用纯净度）: {prec:.4f}")
    print("\n  按 query 型（召回率 / 精确率）:")
    for t, (tr, td, tp, tn) in sorted(type_stat.items()):
        print(f"    {t:<10} (n={td}): 召回 {tr/td:.3f} | 精确 {tp/tn:.3f}")

    if missed:
        print(f"\n  必需工具类没被调用的 query ({len(missed)}):")
        for q, req, calls in missed:
            print(f"    [{q}] 需要 {req}，实际调用 {calls}")
    if impure:
        print(f"\n  调用了无关工具的 query ({len(impure)}):")
        for q, junk, calls in impure:
            print(f"    [{q}] 无关 {junk}（全部 {calls}）")


if __name__ == "__main__":
    main()
