"""
MARS 产品层（第二层）指标评估 —— 跑真实 Agent（ReAct + 护栏），统计「任务完成率 / 一次解决率 /
平均步数 / 无效步数 / 答案召回率」，区别于 mars_eval_50.py 只评检索层（Recall/MRR/nDCG）。

检索层指标回答「找得准不准」，本脚本回答「端到端能不能完整完成任务」：
- 任务完成率   = 最终答案含 ≥1 个可溯源 mall_id 的查询占比（产出可用、带证据的推荐）
- 一次解决率   = 全程没有触发「必需工具守卫拒绝」的查询占比（协议第一次就走对）
- 平均步数     = ReAct 平均步数（收敛速度）
- 无效步数率   = 无 Action / 被守卫拒绝的「废步」占总步数比例
- 终局收敛率   = 靠「最后一步强制交卷」才结束的查询占比（没自然收敛）
- 推荐精度     = 最终推荐的商场里，真正命中 ground-truth 的比例（端到端「推荐得对不对」）
- 答案召回率   = 最终推荐里命中 ground-truth 商场的比例（桥接检索层；注意被推荐数 3~5 天然封顶）

用法: python mars_eval_product.py [--limit N]   # --limit 只跑前 N 条（验证用）
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# 先加载 .env（mars_agent 从 os.environ 读 DEEPSEEK_API_KEY）
load_dotenv()

from rag_utils import get_embeddings, load_vectorstore  # noqa: E402
from mars_agent import MARSAgent  # noqa: E402
from mars_eval_50 import build_queries  # noqa: E402
from mars_eval_retrieval import DATA_FILE, recall_at_k  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REQUIRED_TOOLS = ["score_customer_match", "analyze_brand_competition"]  # 与 recommend() 生产一致


def precision_of(retrieved, relevant):
    """推荐精度：推荐的商场里命中 ground-truth 的比例（推荐任务看这个，不看 recall）。"""
    if not retrieved:
        return 0.0
    hit = len(set(retrieved) & set(relevant))
    return hit / len(retrieved)


def run_one(agent: MARSAgent, query: str):
    """跑一条查询，返回原始指标字典。"""
    t0 = time.time()
    final_answer, trajectory = agent.react_agent.run(query, required_tools=REQUIRED_TOOLS)
    latency = time.time() - t0

    steps = len(trajectory)
    final_mall_ids = agent._extract_mall_ids(final_answer)

    rejected = sum(1 for s in trajectory if "系统拒绝" in (s.get("observation") or ""))
    no_action = sum(1 for s in trajectory if "未检测到 Action" in (s.get("observation") or ""))
    forced = bool(trajectory) and ("已到最后一步" in (trajectory[-1].get("observation") or ""))

    # 落盘轨迹（瘦身版）与最终答案原文，供后续计算「数字溯源率」「工具召回/精确率」
    traj_slim = [
        {
            "step": s.get("step"),
            "action": str(s.get("action") or ""),
            "action_input": str(s.get("action_input") or ""),
            "observation": str(s.get("observation") or ""),
        }
        for s in trajectory
    ]

    return {
        "steps": steps,
        "latency_s": round(latency, 2),
        "final_mall_ids": final_mall_ids,
        "completed": len(final_mall_ids) >= 1,
        "first_pass": rejected == 0,
        "rejected_steps": rejected,
        "no_action_steps": no_action,
        "forced_terminal": forced,
        "final_answer": final_answer or "",
        "trajectory": traj_slim,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    args = ap.parse_args()

    df = pd.read_csv(DATA_FILE, dtype={"mall_id": str})
    queries = build_queries(df)
    if args.limit:
        queries = queries[: args.limit]

    print(f"评估集: {len(queries)} 条（产品层指标，真实 Agent）")
    print("加载向量库 + 嵌入模型 ...")
    embeddings = get_embeddings()
    vs = load_vectorstore("./chroma_db", embeddings)

    agent = MARSAgent(vs, df)
    status = agent.init_llm()
    print(status)
    if "成功" not in status:
        print("❌ LLM 未初始化成功，指标将基于模拟模式，无意义。请检查 .env 的 DEEPSEEK_API_KEY。")
        return

    rows = []
    for i, (query, diff, label_ids) in enumerate(queries, 1):
        r = run_one(agent, query)
        r["query"] = query
        r["difficulty"] = diff
        r["label_count"] = len(label_ids)
        r["label_ids"] = label_ids
        r["precision"] = precision_of(r["final_mall_ids"], label_ids)
        r["recall@1"] = recall_at_k(r["final_mall_ids"], label_ids, 1)
        r["recall@5"] = recall_at_k(r["final_mall_ids"], label_ids, 5)
        rows.append(r)
        print(f"[{i}/{len(queries)}] 步数={r['steps']:<2} 完成={'✓' if r['completed'] else '✗'} "
              f"一次={'✓' if r['first_pass'] else '✗'} 精度={r['precision']:.2f} "
              f"R@5={r['recall@5']:.2f}  {query}")

    n = len(rows)
    total_steps = sum(r["steps"] for r in rows)
    invalid_steps = sum(r["rejected_steps"] + r["no_action_steps"] for r in rows)

    summary = {
        "n": n,
        "任务完成率": round(float(np.mean([r["completed"] for r in rows])), 4),
        "平均步数": round(float(np.mean([r["steps"] for r in rows])), 2),
        "一次解决率": round(float(np.mean([r["first_pass"] for r in rows])), 4),
        "无效步数率": round(invalid_steps / total_steps, 4) if total_steps else 0.0,
        "终局收敛率": round(float(np.mean([r["forced_terminal"] for r in rows])), 4),
        "推荐精度": round(float(np.mean([r["precision"] for r in rows])), 4),
        "答案Recall@1": round(float(np.mean([r["recall@1"] for r in rows])), 4),
        "答案Recall@5": round(float(np.mean([r["recall@5"] for r in rows])), 4),
        "平均耗时_s": round(float(np.mean([r["latency_s"] for r in rows])), 2),
    }

    print("\n" + "=" * 70)
    print(f"产品层指标汇总（n={n}，真实 Agent）")
    print("=" * 70)
    for k, v in summary.items():
        if k == "n":
            continue
        print(f"  {k:<12}: {v}")

    print("\n按难度（任务完成率 / 推荐精度 / 答案Recall@5）:")
    for diff in ["easy", "medium", "hard"]:
        sub = [r for r in rows if r["difficulty"] == diff]
        if sub:
            c = np.mean([r["completed"] for r in sub])
            p = np.mean([r["precision"] for r in sub])
            r5 = np.mean([r["recall@5"] for r in sub])
            print(f"  {diff:<8}(n={len(sub)}): 完成率 {c:.3f} | 精度 {p:.3f} | 答案R@5 {r5:.3f}")

    with open("eval_product_result.json", "w", encoding="utf-8") as f:
        json.dump({"cases": [{k: v for k, v in r.items()} for r in rows], "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print("\n结果已存 eval_product_result.json")


if __name__ == "__main__":
    main()
