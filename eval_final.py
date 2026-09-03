"""
三路对比：纯向量(基线) vs 混合检索(BM25+稠密) vs 混合+Reranker。
用法: python eval_final.py
"""

import os

import numpy as np
import pandas as pd

from rag_utils import get_embeddings, load_vectorstore, hybrid_vector_search
from mars_hybrid_retrieval import HybridRetriever, load_reranker
from mars_eval_retrieval import EVAL_SET, DATA_FILE, recall_at_k, mrr, ndcg_at_k

KEYS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10"]
RERANK_DIR = "./models/bge-reranker-base"


def resolve(df, names):
    name_to_id = dict(zip(df["mall_name"].astype(str), df["mall_id"].astype(str)))
    ids = []
    for name in names:
        mid = name_to_id.get(name)
        if mid is None:
            hit = df[df["mall_name"].astype(str).str.contains(name.replace("上海", ""), na=False)]
            if len(hit):
                mid = str(hit.iloc[0]["mall_id"])
        if mid:
            ids.append(mid)
    return ids


def metrics(ret_ids, rel_ids):
    return {
        "recall@1": recall_at_k(ret_ids, rel_ids, 1),
        "recall@3": recall_at_k(ret_ids, rel_ids, 3),
        "recall@5": recall_at_k(ret_ids, rel_ids, 5),
        "recall@10": recall_at_k(ret_ids, rel_ids, 10),
        "mrr": mrr(ret_ids, rel_ids),
        "ndcg@10": ndcg_at_k(ret_ids, rel_ids, 10),
    }


def main():
    df = pd.read_csv(DATA_FILE, dtype={"mall_id": str})
    embeddings = get_embeddings()
    vs = load_vectorstore("./chroma_db", embeddings)
    retriever = HybridRetriever(vs)
    reranker = load_reranker(RERANK_DIR) if os.path.isdir(RERANK_DIR) else None
    print(f"reranker: {'已加载 ' + RERANK_DIR if reranker else '未找到，跳过'}")

    rows = {"base": [], "hyb": [], "rr": []}
    print("\n" + "=" * 80)
    print(f"{'ID':<4}{'难度':<7}{'基线R@5':<10}{'混合R@5':<10}{'+RR R@5':<10}  查询")
    print("=" * 80)
    for case in EVAL_SET:
        rel_ids = resolve(df, case["relevant"])

        base_ids = [str(r["mall_id"]) for r in hybrid_vector_search(vs, case["query"], top_k=10) if r.get("mall_id")]
        hyb_ids = [str(r["mall_id"]) for r in retriever.search(case["query"], top_k=10)]
        rr_ids = [str(r["mall_id"]) for r in retriever.search(case["query"], top_k=10, reranker=reranker)] if reranker else hyb_ids

        rows["base"].append(metrics(base_ids, rel_ids))
        rows["hyb"].append(metrics(hyb_ids, rel_ids))
        rows["rr"].append(metrics(rr_ids, rel_ids))

        print(f"{case['id']:<4}{case['difficulty']:<7}{rows['base'][-1]['recall@5']:<10.2f}"
              f"{rows['hyb'][-1]['recall@5']:<10.2f}{rows['rr'][-1]['recall@5']:<10.2f}  {case['query']}")

    print("\n" + "=" * 80)
    print("汇总（平均）")
    print("=" * 80)
    print(f"{'指标':<12}{'基线':<10}{'混合':<10}{'混合+RR':<10}")
    for k in KEYS:
        print(f"{k:<12}{np.mean([r[k] for r in rows['base']]):<10.3f}"
              f"{np.mean([r[k] for r in rows['hyb']]):<10.3f}"
              f"{np.mean([r[k] for r in rows['rr']]):<10.3f}")


if __name__ == "__main__":
    main()
