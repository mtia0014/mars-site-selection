"""
MARS 检索质量评估 —— 用人工标注的评估集，计算 Recall@k / MRR / nDCG@10。

评估集（EVAL_SET）为 10 条真实选址需求，每条标注了「相关商场」（ground truth）。
标注依据客观数据：客流排名 / 行政区 / 评分档次 / 品牌（奢侈/运动）等，均可在宽表中核对。

用法:
    python mars_eval_retrieval.py
"""

import json
import time

import numpy as np
import pandas as pd

from rag_utils import get_embeddings, load_vectorstore, hybrid_vector_search

DATA_FILE = "mall_wide_table_shanghai_final_v3.csv"

# 每条：query=需求描述, relevant=相关商场名(精确匹配宽表 mall_name), difficulty=easy/medium/hard
EVAL_SET = [
    {"id": "Q01", "query": "上海客流量最大的商场有哪些", "difficulty": "easy",
     "relevant": ["上海港汇恒隆广场", "上海徐汇绿地缤纷城", "上海中山公园龙之梦"]},
    {"id": "Q02", "query": "徐汇区的高客流商场", "difficulty": "easy",
     "relevant": ["上海港汇恒隆广场", "上海徐汇绿地缤纷城", "上海环贸iapm"]},
    {"id": "Q03", "query": "评分最高的顶级高端商场", "difficulty": "easy",
     "relevant": ["上海恒隆广场", "上海BFC外滩金融中心", "上海洛克外滩源"]},
    {"id": "Q04", "query": "奢侈品牌聚集的高端商场", "difficulty": "medium",
     "relevant": ["上海前滩太古里", "上海恒隆广场", "上海久光百货", "上海环贸iapm"]},
    {"id": "Q05", "query": "运动户外品牌商场，耐克始祖鸟", "difficulty": "medium",
     "relevant": ["上海力宝广场", "上海天安千树", "百联中环购物广场", "上海新世界城"]},
    {"id": "Q06", "query": "高收入人群高消费能力的商场", "difficulty": "medium",
     "relevant": ["上海恒隆广场", "上海IFC", "上海久光百货"]},
    {"id": "Q07", "query": "亲子家庭客群多的商场", "difficulty": "hard",
     "relevant": ["上海启源儿童文化城", "上海浦江欢乐颂", "上海大华锦绣嘉年华"]},
    {"id": "Q08", "query": "年轻白领咖啡茶饮聚集的商场", "difficulty": "hard",
     "relevant": ["上海中山公园龙之梦", "上海月星环球港", "上海中庚漫游城"]},
]


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def mrr(retrieved, relevant):
    rel = set(relevant)
    for i, x in enumerate(retrieved, 1):
        if x in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved, relevant, k):
    rel = set(relevant)
    dcg = sum(1.0 / np.log2(i + 2) for i, x in enumerate(retrieved[:k]) if x in rel)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal > 0 else 0.0


def main():
    df = pd.read_csv(DATA_FILE, dtype={"mall_id": str})
    name_to_id = dict(zip(df["mall_name"].astype(str), df["mall_id"].astype(str)))

    # 解析标注 → mall_id，找不到的告警
    print("=" * 66)
    print("MARS 检索质量评估")
    print("=" * 66)

    embeddings = get_embeddings()
    vs = load_vectorstore("./chroma_db", embeddings)
    if vs is None:
        print("❌ 向量库加载失败")
        return

    rows = []
    unresolved = []
    for case in EVAL_SET:
        rel_ids = []
        for name in case["relevant"]:
            mid = name_to_id.get(name)
            if mid is None:
                # 子串兜底
                hit = df[df["mall_name"].astype(str).str.contains(name.replace("上海", ""), na=False)]
                if len(hit):
                    mid = str(hit.iloc[0]["mall_id"])
                else:
                    unresolved.append((case["id"], name))
                    continue
            rel_ids.append(mid)
        case["rel_ids"] = rel_ids

        t0 = time.time()
        results = hybrid_vector_search(vs, case["query"], top_k=10)
        dt = time.time() - t0
        ret_ids = [str(r["mall_id"]) for r in results if "mall_id" in r]

        m = {
            "id": case["id"],
            "query": case["query"],
            "difficulty": case["difficulty"],
            "recall@1": recall_at_k(ret_ids, rel_ids, 1),
            "recall@3": recall_at_k(ret_ids, rel_ids, 3),
            "recall@5": recall_at_k(ret_ids, rel_ids, 5),
            "recall@10": recall_at_k(ret_ids, rel_ids, 10),
            "mrr": mrr(ret_ids, rel_ids),
            "ndcg@10": ndcg_at_k(ret_ids, rel_ids, 10),
            "top3": [r.get("mall_name", "") for r in results[:3]],
            "latency_s": round(dt, 2),
        }
        rows.append(m)

    # 逐条打印
    print(f"\n{'ID':<4}{'难度':<8}{'R@1':<7}{'R@3':<7}{'R@5':<7}{'R@10':<8}{'MRR':<7}{'nDCG@10':<9}  查询")
    for m in rows:
        print(f"{m['id']:<4}{m['difficulty']:<8}{m['recall@1']:<7.2f}{m['recall@3']:<7.2f}"
              f"{m['recall@5']:<7.2f}{m['recall@10']:<8.2f}{m['mrr']:<7.2f}{m['ndcg@10']:<9.2f}  {m['query']}")
        print(f"        Top3: {' | '.join(m['top3'])}")

    # 汇总
    keys = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10"]
    print("\n" + "=" * 66)
    print("汇总（平均）")
    print("=" * 66)
    for k in keys:
        print(f"  {k:<10}: {np.mean([m[k] for m in rows]):.3f}")

    # 按难度分组
    print("\n按难度分组（平均 Recall@5 / MRR）:")
    for diff in ["easy", "medium", "hard"]:
        sub = [m for m in rows if m["difficulty"] == diff]
        if sub:
            print(f"  {diff:<8}: n={len(sub)}, Recall@5={np.mean([m['recall@5'] for m in sub]):.3f}, "
                  f"MRR={np.mean([m['mrr'] for m in sub]):.3f}")

    print(f"\n平均单查询耗时: {np.mean([m['latency_s'] for m in rows]):.2f}s")
    if unresolved:
        print("\n⚠️ 未解析到的标注名:")
        for cid, nm in unresolved:
            print(f"  {cid}: {nm}")

    # 存结果
    with open("eval_retrieval_result.json", "w", encoding="utf-8") as f:
        json.dump({"cases": rows,
                   "summary": {k: round(float(np.mean([m[k] for m in rows])), 4) for k in keys}},
                  f, ensure_ascii=False, indent=2)
    print("\n结果已存 eval_retrieval_result.json")


if __name__ == "__main__":
    main()
