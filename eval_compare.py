"""
对比 纯向量检索(基线) vs BM25+稠密混合检索(RRF)，同一评估集，打印逐条与汇总增量。
用法: python eval_compare.py
"""

import numpy as np
import pandas as pd

from rag_utils import get_embeddings, load_vectorstore, hybrid_vector_search
from mars_hybrid_retrieval import HybridRetriever
from mars_eval_retrieval import EVAL_SET, DATA_FILE, recall_at_k, mrr, ndcg_at_k

KEYS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10"]


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
    return {k: v for k, v in {
        "recall@1": recall_at_k(ret_ids, rel_ids, 1),
        "recall@3": recall_at_k(ret_ids, rel_ids, 3),
        "recall@5": recall_at_k(ret_ids, rel_ids, 5),
        "recall@10": recall_at_k(ret_ids, rel_ids, 10),
        "mrr": mrr(ret_ids, rel_ids),
        "ndcg@10": ndcg_at_k(ret_ids, rel_ids, 10),
    }.items()}


def main():
    df = pd.read_csv(DATA_FILE, dtype={"mall_id": str})
    embeddings = get_embeddings()
    vs = load_vectorstore("./chroma_db", embeddings)
    retriever = HybridRetriever(vs)

    base_rows, hyb_rows = [], []
    print("=" * 78)
    print(f"{'ID':<4}{'难度':<7}{'基线R@5':<10}{'混合R@5':<10}{'基线MRR':<10}{'混合MRR':<10}  查询")
    print("=" * 78)
    for case in EVAL_SET:
        rel_ids = resolve(df, case["relevant"])

        base_res = hybrid_vector_search(vs, case["query"], top_k=10)
        base_ids = [str(r["mall_id"]) for r in base_res if r.get("mall_id")]
        bm = metrics(base_ids, rel_ids)

        hyb_res = retriever.search(case["query"], top_k=10)
        hyb_ids = [str(r["mall_id"]) for r in hyb_res]
        hm = metrics(hyb_ids, rel_ids)

        base_rows.append(bm)
        hyb_rows.append(hm)

        flag = "▲" if hm["recall@5"] > bm["recall@5"] else ("▼" if hm["recall@5"] < bm["recall@5"] else "=")
        print(f"{case['id']:<4}{case['difficulty']:<7}{bm['recall@5']:<10.2f}{hm['recall@5']:<10.2f}"
              f"{bm['mrr']:<10.2f}{hm['mrr']:<10.2f}  {flag} {case['query']}")

    print("\n" + "=" * 78)
    print("汇总（平均）")
    print("=" * 78)
    print(f"{'指标':<12}{'基线':<10}{'混合':<10}{'增量':<10}")
    for k in KEYS:
        b = np.mean([r[k] for r in base_rows])
        h = np.mean([r[k] for r in hyb_rows])
        print(f"{k:<12}{b:<10.3f}{h:<10.3f}{h-b:+10.3f}")

    # 按难度
    print("\n按难度（Recall@5 / MRR，基线→混合）:")
    for diff in ["easy", "medium", "hard"]:
        idx = [i for i, c in enumerate(EVAL_SET) if c["difficulty"] == diff]
        if idx:
            b5 = np.mean([base_rows[i]["recall@5"] for i in idx])
            h5 = np.mean([hyb_rows[i]["recall@5"] for i in idx])
            bm_ = np.mean([base_rows[i]["mrr"] for i in idx])
            hm_ = np.mean([hyb_rows[i]["mrr"] for i in idx])
            print(f"  {diff:<8}: R@5 {b5:.3f}→{h5:.3f}  |  MRR {bm_:.3f}→{hm_:.3f}")


if __name__ == "__main__":
    main()
