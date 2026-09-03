"""
GraphRAG A/B 对比（同一套 50 条评测集）

三路：基线(纯向量多查询) vs 混合(BM25+稠密) vs GraphRAG(单向量+2-hop 图谱)
标注复用 mars_eval_50.build_queries，确保与既有结果口径完全一致。

用法: python eval_graphrag_ab.py
"""

import json
import time

import numpy as np
import pandas as pd

from rag_utils import get_embeddings, load_vectorstore, hybrid_vector_search
from mars_hybrid_retrieval import HybridRetriever
from mars_eval_retrieval import DATA_FILE, recall_at_k, mrr, ndcg_at_k
from mars_eval_50 import build_queries
from mars_graphrag import MallKnowledgeGraph, GraphRAGRetriever

KEYS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10"]


def _metrics(ranked, label_ids):
    return {
        "recall@1": recall_at_k(ranked, label_ids, 1),
        "recall@3": recall_at_k(ranked, label_ids, 3),
        "recall@5": recall_at_k(ranked, label_ids, 5),
        "recall@10": recall_at_k(ranked, label_ids, 10),
        "mrr": mrr(ranked, label_ids),
        "ndcg@10": ndcg_at_k(ranked, label_ids, 10),
    }


def main():
    df = pd.read_csv(DATA_FILE, dtype={"mall_id": str})
    Q = build_queries(df)

    embeddings = get_embeddings()
    vs = load_vectorstore("./chroma_db", embeddings)
    retriever = HybridRetriever(vs)

    # 构建知识图谱 + GraphRAG 检索器（一次性，含同区 O(n²) 相似度）
    t0 = time.time()
    kg = MallKnowledgeGraph()
    kg.build_from_dataframe(df)
    gretriever = GraphRAGRetriever(vs, kg, df)
    print(f"\n[计时] 图谱构建 {time.time() - t0:.1f}s")

    rows = {"base": [], "hybrid": [], "graph": []}

    print("\n" + "=" * 100)
    print(f"{'ID':<5}{'难度':<7}{'标数':<5}{'基线R@5':<9}{'混合R@5':<9}{'图R@5':<9}{'基线MRR':<9}{'混合MRR':<9}{'图MRR':<9}")
    print("=" * 100)
    t0 = time.time()
    for i, (query, diff, label_ids) in enumerate(Q, 1):
        base_ids = [str(r["mall_id"]) for r in hybrid_vector_search(vs, query, top_k=10)
                    if r.get("mall_id")]
        hyb_ids = [str(r["mall_id"]) for r in retriever.search(query, top_k=10)]
        graph_ids = [str(r["mall_id"]) for r in gretriever.hybrid_search(query, top_k=10)]

        bm = _metrics(base_ids, label_ids)
        hm = _metrics(hyb_ids, label_ids)
        gm = _metrics(graph_ids, label_ids)
        rows["base"].append(bm)
        rows["hybrid"].append(hm)
        rows["graph"].append(gm)
        print(f"{i:<5}{diff:<7}{len(label_ids):<5}{bm['recall@5']:<9.2f}{hm['recall@5']:<9.2f}"
              f"{gm['recall@5']:<9.2f}{bm['mrr']:<9.2f}{hm['mrr']:<9.2f}{gm['mrr']:<9.2f}  {query}")

    print(f"\n[计时] 50 条 × 3 路检索 {time.time() - t0:.1f}s")

    print("\n" + "=" * 100)
    print("汇总（平均，n=%d）" % len(Q))
    print("=" * 100)
    print(f"{'指标':<12}{'基线':<10}{'混合':<10}{'GraphRAG':<10}{'图-混':<10}")
    summary = {}
    for k in KEYS:
        b = np.mean([r[k] for r in rows["base"]])
        h = np.mean([r[k] for r in rows["hybrid"]])
        g = np.mean([r[k] for r in rows["graph"]])
        summary[k] = {"base": round(float(b), 4), "hybrid": round(float(h), 4),
                      "graph": round(float(g), 4)}
        print(f"{k:<12}{b:<10.3f}{h:<10.3f}{g:<10.3f}{g-h:+10.3f}")

    print("\n按难度（Recall@5 / MRR）：")
    for diff in ["easy", "medium", "hard"]:
        idx = [i for i, (_, d, _) in enumerate(Q) if d == diff]
        if not idx:
            continue
        b5 = np.mean([rows["base"][i]["recall@5"] for i in idx])
        h5 = np.mean([rows["hybrid"][i]["recall@5"] for i in idx])
        g5 = np.mean([rows["graph"][i]["recall@5"] for i in idx])
        bm_ = np.mean([rows["base"][i]["mrr"] for i in idx])
        hm_ = np.mean([rows["hybrid"][i]["mrr"] for i in idx])
        gm_ = np.mean([rows["graph"][i]["mrr"] for i in idx])
        print(f"  {diff:<8}(n={len(idx)}): R@5 {b5:.3f}/{h5:.3f}/{g5:.3f} | "
              f"MRR {bm_:.3f}/{hm_:.3f}/{gm_:.3f}")

    with open("eval_graphrag_ab_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "cases": [{"id": i, "query": q, "difficulty": d, "label_count": len(l),
                       "base": rows["base"][i], "hybrid": rows["hybrid"][i],
                       "graph": rows["graph"][i]}
                      for i, (q, d, l) in enumerate(Q)],
            "summary": summary,
        }, f, ensure_ascii=False, indent=2)
    print("\n结果已存 eval_graphrag_ab_result.json")


if __name__ == "__main__":
    main()
