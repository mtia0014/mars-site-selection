"""
GraphRAG 融合 A/B（4 路，同一套 50 条评测集）

基线(纯向量多查询) vs 混合(BM25+稠密) vs GraphRAG(单向量+图谱)
vs 融合(BM25+稠密 RRF 之上，把图谱 2-hop 扩展作为第三路召回并进 RRF)

标注复用 mars_eval_50.build_queries，与既有结果口径一致。
用法: python eval_graphrag_fusion.py
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


def hybrid_graph_search(retriever, kg, gretriever, query, top_k=10,
                        candidate_k=50, graph_weight=0.35):
    """hybrid(BM25+稠密) 的 RRF 之上，把图谱 2-hop 扩展作为第三路召回融合"""
    dense_res = hybrid_vector_search(retriever.vs, query, top_k=candidate_k)
    dense_rank = [str(r["mall_id"]) for r in dense_res if r.get("mall_id")]
    dense_by_mid = {str(r["mall_id"]): r for r in dense_res if r.get("mall_id")}

    bm25_rank = retriever._bm25_rank(query, candidate_k)
    sig = retriever._signal(query)
    w_bm25 = float(np.clip((sig - 1.2) / 3.0, 0.0, 1.0))

    entities = gretriever._extract_entities(query)
    graph_expanded = kg.graph_enhanced_search(entities, hop=2)
    vec_score = {str(r["mall_id"]): (r.get("similarity", 0) or 0)
                 for r in dense_res if r.get("mall_id")}
    graph_rank = sorted(graph_expanded, key=lambda m: -vec_score.get(m, 0.0))

    fused = retriever._rrf([dense_rank, bm25_rank, graph_rank],
                           weights=[1.0, w_bm25, graph_weight])
    out = []
    for mid in fused[:top_k]:
        if mid in dense_by_mid:
            out.append(dense_by_mid[mid])
        else:
            meta = retriever._meta_by_id.get(mid, {})
            out.append({
                "mall_id": mid,
                "mall_name": meta.get("mall_name", ""),
                "district": meta.get("district", ""),
                "score_market": meta.get("score_market", 0) or 0,
                "traffic_daily": meta.get("traffic_daily", 0) or 0,
                "brand_count": meta.get("brand_count", 0) or 0,
                "similarity": 0.0,
            })
    return out


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

    t0 = time.time()
    kg = MallKnowledgeGraph()
    kg.build_from_dataframe(df)
    gretriever = GraphRAGRetriever(vs, kg, df)
    print(f"\n[计时] 图谱构建 {time.time() - t0:.1f}s")

    methods = ["base", "hybrid", "graph", "fusion"]
    rows = {m: [] for m in methods}

    print("\n" + "=" * 116)
    print(f"{'ID':<5}{'难度':<7}{'标数':<5}{'基线R@5':<8}{'混合R@5':<8}{'图R@5':<8}{'融合R@5':<9}"
          f"{'混合MRR':<9}{'图MRR':<9}{'融合MRR':<9}")
    print("=" * 116)
    t0 = time.time()
    for i, (query, diff, label_ids) in enumerate(Q, 1):
        base_ids = [str(r["mall_id"]) for r in hybrid_vector_search(vs, query, top_k=10)
                    if r.get("mall_id")]
        hyb_ids = [str(r["mall_id"]) for r in retriever.search(query, top_k=10)]
        graph_ids = [str(r["mall_id"]) for r in gretriever.hybrid_search(query, top_k=10)]
        fusion_ids = [str(r["mall_id"]) for r in hybrid_graph_search(retriever, kg, gretriever, query)]

        bm = _metrics(base_ids, label_ids)
        hm = _metrics(hyb_ids, label_ids)
        gm = _metrics(graph_ids, label_ids)
        fm = _metrics(fusion_ids, label_ids)
        for m, r in zip(methods, [bm, hm, gm, fm]):
            rows[m].append(r)
        print(f"{i:<5}{diff:<7}{len(label_ids):<5}{bm['recall@5']:<8.2f}{hm['recall@5']:<8.2f}"
              f"{gm['recall@5']:<8.2f}{fm['recall@5']:<9.2f}{hm['mrr']:<9.2f}{gm['mrr']:<9.2f}"
              f"{fm['mrr']:<9.2f}  {query}")

    print(f"\n[计时] 50 条 × 4 路检索 {time.time() - t0:.1f}s")

    print("\n" + "=" * 116)
    print("汇总（平均，n=%d）" % len(Q))
    print("=" * 116)
    print(f"{'指标':<12}{'基线':<9}{'混合':<9}{'GraphRAG':<9}{'融合':<9}{'融合-混':<9}")
    summary = {}
    for k in KEYS:
        vals = {m: np.mean([r[k] for r in rows[m]]) for m in methods}
        summary[k] = {m: round(float(v), 4) for m, v in vals.items()}
        print(f"{k:<12}{vals['base']:<9.3f}{vals['hybrid']:<9.3f}{vals['graph']:<9.3f}"
              f"{vals['fusion']:<9.3f}{vals['fusion'] - vals['hybrid']:+9.3f}")

    print("\n按难度（Recall@5 / MRR）：")
    for diff in ["easy", "medium", "hard"]:
        idx = [i for i, (_, d, _) in enumerate(Q) if d == diff]
        if not idx:
            continue
        line = []
        for m in methods:
            r5 = np.mean([rows[m][i]["recall@5"] for i in idx])
            mm = np.mean([rows[m][i]["mrr"] for i in idx])
            line.append(f"{m[0].upper()}:{r5:.3f}/{mm:.3f}")
        print(f"  {diff:<8}(n={len(idx)}): " + "  ".join(line))

    with open("eval_graphrag_fusion_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "cases": [{"id": i, "query": q, "difficulty": d, "label_count": len(l),
                       **{m: rows[m][i] for m in methods}}
                      for i, (q, d, l) in enumerate(Q)],
            "summary": summary,
        }, f, ensure_ascii=False, indent=2)
    print("\n结果已存 eval_graphrag_fusion_result.json")


if __name__ == "__main__":
    main()
