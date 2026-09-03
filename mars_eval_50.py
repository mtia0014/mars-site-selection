"""
MARS 检索质量评估（50 条，最终版）—— 标注由宽表客观字段自动生成，
每条查询的相关商场是可复核的客观集合（非手拍）。跑 基线(纯向量) vs 混合(BM25+稠密)。

查询类型配比（贴近选址真实需求）：
- 品牌实体（18）：某品牌已入驻的商场（精确词面命中）
- 奢侈/珠宝（4）、运动（3）：品牌品类聚集
- 客流/评分/档次（7）：客观数值阈值
- 客群/定位（8）：position_text 关键词 + 数值约束
- 区县（6）：各区评分最高的商场
- 组合（4）：品牌/客群 × 档次/客流 的复合约束

用法: python mars_eval_50.py
"""

import json

import numpy as np
import pandas as pd

from rag_utils import get_embeddings, load_vectorstore, hybrid_vector_search
from mars_hybrid_retrieval import HybridRetriever
from mars_eval_retrieval import DATA_FILE, recall_at_k, mrr, ndcg_at_k

KEYS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10"]

# 品牌词典（中英合并，小写子串匹配）
LUXURY = ["gucci", "古驰", "dior", "迪奥", "chanel", "香奈儿", "hermes", "爱马仕", "hermès",
          "louis vuitton", "路易威登", "prada", "普拉达", "burberry", "versace", "givenchy",
          "celine", "fendi", "balenciaga", "giada", "kenzo", "valentino", "tiffany", "蒂芙尼",
          "cartier", "卡地亚", "宝格丽", "bvlgari"]
SPORT = ["nike", "耐克", "adidas", "阿迪达斯", "始祖鸟", "arc", "fila", "李宁", "安踏",
         "under armour", "迪桑特", "descente", "columbia", "哥伦比亚", "the north face",
         "北面", "new balance", "asics", "puma", "彪马", "hoka", "salomon", "萨洛蒙",
         "lululemon", "露露乐檬"]
JEWELRY = ["周大福", "老凤祥", "卡地亚", "cartier", "蒂芙尼", "tiffany", "周生生", "六福",
           "梵克雅宝", "宝格丽", "bvlgari", "老庙", "i do"]

# (query, 品牌词典, 难度) —— 单一品牌实体查询
BRAND_QUERIES = [
    ("有瑞幸咖啡的商场", ["瑞幸"], "easy"),
    ("有喜茶的商场", ["喜茶"], "easy"),
    ("有奈雪的茶的商场", ["奈雪", "奈雪的茶"], "easy"),
    ("有霸王茶姬的商场", ["霸王茶姬"], "easy"),
    ("有Manner咖啡的商场", ["manner"], "easy"),
    ("有蜜雪冰城的商场", ["蜜雪冰城"], "easy"),
    ("有优衣库UNIQLO的商场", ["优衣库", "uniqlo"], "easy"),
    ("有无印良品MUJI的商场", ["无印良品", "muji"], "easy"),
    ("有波司登的商场", ["波司登"], "easy"),
    ("有耐克NIKE的商场", ["nike", "耐克"], "easy"),
    ("有阿迪达斯的商场", ["adidas", "阿迪达斯"], "easy"),
    ("有李宁的商场", ["李宁"], "easy"),
    ("有始祖鸟的商场", ["始祖鸟", "arc"], "easy"),
    ("有周大福的商场", ["周大福"], "easy"),
    ("有老凤祥的商场", ["老凤祥"], "easy"),
    ("有华为的商场", ["华为"], "easy"),
    ("有小米之家的商场", ["小米"], "easy"),
    ("有名创优品的商场", ["名创优品"], "easy"),
]


def build_queries(df):
    """由宽表客观字段生成 50 条 (query, difficulty, label_ids)"""
    for c in ["traffic_daily", "score_market", "brand_count",
              "餐饮_brand_count", "服装_brand_count", "运动_brand_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    def _brands(row):
        s = row.get("brand_data_json")
        if not s or (isinstance(s, float) and pd.isna(s)):
            return []
        try:
            return [str(b).lower() for b in (json.loads(s).get("top_brands", []) or [])]
        except Exception:
            return []

    df["_brands"] = df.apply(_brands, axis=1)

    def has_kw(row, kws):
        bl = row["_brands"]
        return any(k.lower() in b for b in bl for k in kws)

    def ids(mask):
        return df.loc[mask, "mall_id"].astype(str).tolist()

    def top(col, n):
        return df.nlargest(n, col)["mall_id"].astype(str).tolist()

    pos = df["position_text"].astype(str)
    Q = []  # (query, difficulty, label_ids)

    # ---- 品牌实体（18）----
    for q, kws, diff in BRAND_QUERIES:
        Q.append((q, diff, ids(df.apply(lambda r: has_kw(r, kws), axis=1))))

    # ---- 奢侈 / 珠宝（4）----
    Q.append(("奢侈品牌聚集的高端商场", "medium", ids(df.apply(lambda r: has_kw(r, LUXURY), axis=1))))
    Q.append(("珠宝手表大牌聚集的商场", "medium", ids(df.apply(lambda r: has_kw(r, JEWELRY), axis=1))))
    Q.append(("有古驰GUCCI的商场", "easy", ids(df.apply(lambda r: has_kw(r, ["gucci", "古驰"]), axis=1))))
    Q.append(("有迪奥DIOR的商场", "easy", ids(df.apply(lambda r: has_kw(r, ["dior", "迪奥"]), axis=1))))

    # ---- 运动（3）----
    Q.append(("运动户外品牌聚集的商场", "medium", ids(df.apply(lambda r: has_kw(r, SPORT), axis=1))))
    Q.append(("有FILA的商场", "easy", ids(df.apply(lambda r: has_kw(r, ["fila"]), axis=1))))
    Q.append(("运动品牌聚集的高端商场", "hard",
              ids(df.apply(lambda r: has_kw(r, SPORT), axis=1) & (df["score_market"] >= 85))))

    # ---- 客流 / 评分 / 档次（7）----
    Q.append(("上海客流量最大的商场", "easy", ids(df["traffic_daily"] >= 30000)))
    Q.append(("评分最高的顶级高端商场", "easy", ids(df["score_market"] >= 93)))
    Q.append(("综合评分最高的商场", "easy", top("score_market", 10)))
    Q.append(("大众平价商场", "medium", ids(df["score_market"] < 65)))
    Q.append(("静安区的高端商场", "medium", ids((df["district"] == "静安区") & (df["score_market"] >= 88))))
    Q.append(("徐汇区评分高的商场", "medium", ids((df["district"] == "徐汇区") & (df["score_market"] >= 85))))
    Q.append(("高客流的高端商场", "medium", ids((df["traffic_daily"] >= 15000) & (df["score_market"] >= 88))))

    # ---- 客群 / 定位（8）----
    Q.append(("亲子家庭客群多的商场", "medium", ids(pos.str.contains("亲子|儿童", na=False))))
    Q.append(("白领商务人群的商场", "hard",
              ids(pos.str.contains("白领|商务", na=False) & (df["traffic_daily"] >= 15000))))
    Q.append(("高收入人群的商场", "medium", ids(pos.str.contains("高收入", na=False))))
    Q.append(("高评分且年轻客群的商场", "hard",
              ids((df["score_market"] >= 88) & pos.str.contains("年轻", na=False))))
    Q.append(("银发老年客群的商场", "medium", ids(pos.str.contains("银发|老年", na=False))))
    Q.append(("亲子客群的高端商场", "hard",
              ids(pos.str.contains("亲子|儿童", na=False) & (df["score_market"] >= 85))))
    Q.append(("高收入高评分人群的商场", "hard",
              ids(pos.str.contains("高收入", na=False) & (df["score_market"] >= 88))))
    Q.append(("有乐高的商场", "easy", ids(df.apply(lambda r: has_kw(r, ["乐高", "lego"]), axis=1))))

    # ---- 区县（6，各区评分最高的商场）----
    for dist in ["徐汇区", "浦东新区", "静安区", "黄浦区", "长宁区", "虹口区"]:
        sub = df[df["district"] == dist]
        Q.append((f"{dist}最好的商场", "easy", sub.nlargest(8, "score_market")["mall_id"].astype(str).tolist()))

    # ---- 组合（4）----
    Q.append(("奢侈品牌且客流大的商场", "hard",
              ids(df.apply(lambda r: has_kw(r, LUXURY), axis=1) & (df["traffic_daily"] >= 10000))))
    Q.append(("餐饮品牌丰富的商场", "medium", top("餐饮_brand_count", 10)))
    Q.append(("服装品牌丰富的商场", "medium", top("服装_brand_count", 10)))
    Q.append(("运动品类丰富的商场", "medium", top("运动_brand_count", 10)))

    # 清洗：去掉空标签
    return [(q, d, l) for q, d, l in Q if l]


def main():
    df = pd.read_csv(DATA_FILE, dtype={"mall_id": str})
    Q = build_queries(df)
    label_sizes = [len(l) for _, _, l in Q]
    print(f"评估集: {len(Q)} 条")
    print(f"每查询相关商场数: min={min(label_sizes)} max={max(label_sizes)} "
          f"中位={int(np.median(label_sizes))} 平均={np.mean(label_sizes):.1f}")

    # ============ 检索 ============
    embeddings = get_embeddings()
    vs = load_vectorstore("./chroma_db", embeddings)
    retriever = HybridRetriever(vs)

    base_rows, hyb_rows = [], []
    print("\n" + "=" * 84)
    print(f"{'ID':<5}{'难度':<7}{'标数':<5}{'基线R@5':<9}{'混合R@5':<9}{'基线MRR':<9}{'混合MRR':<9}")
    print("=" * 84)
    for i, (query, diff, label_ids) in enumerate(Q, 1):
        base_ids = [str(r["mall_id"]) for r in hybrid_vector_search(vs, query, top_k=10)
                    if r.get("mall_id")]
        hyb_ids = [str(r["mall_id"]) for r in retriever.search(query, top_k=10)]

        def _metrics(ranked):
            return {
                "recall@1": recall_at_k(ranked, label_ids, 1),
                "recall@3": recall_at_k(ranked, label_ids, 3),
                "recall@5": recall_at_k(ranked, label_ids, 5),
                "recall@10": recall_at_k(ranked, label_ids, 10),
                "mrr": mrr(ranked, label_ids),
                "ndcg@10": ndcg_at_k(ranked, label_ids, 10),
            }

        bm, hm = _metrics(base_ids), _metrics(hyb_ids)
        base_rows.append(bm)
        hyb_rows.append(hm)
        print(f"{i:<5}{diff:<7}{len(label_ids):<5}{bm['recall@5']:<9.2f}{hm['recall@5']:<9.2f}"
              f"{bm['mrr']:<9.2f}{hm['mrr']:<9.2f}  {query}")

    print("\n" + "=" * 84)
    print("汇总（平均，n=%d）" % len(Q))
    print("=" * 84)
    print(f"{'指标':<12}{'基线':<10}{'混合':<10}{'增量':<10}")
    summary = {}
    for k in KEYS:
        b = np.mean([r[k] for r in base_rows])
        h = np.mean([r[k] for r in hyb_rows])
        summary[k] = {"base": round(float(b), 4), "hybrid": round(float(h), 4)}
        print(f"{k:<12}{b:<10.3f}{h:<10.3f}{h-b:+10.3f}")

    print("\n按难度（Recall@5 / MRR，基线→混合）:")
    for diff in ["easy", "medium", "hard"]:
        idx = [i for i, (_, d, _) in enumerate(Q) if d == diff]
        if idx:
            b5 = np.mean([base_rows[i]["recall@5"] for i in idx])
            h5 = np.mean([hyb_rows[i]["recall@5"] for i in idx])
            bm_ = np.mean([base_rows[i]["mrr"] for i in idx])
            hm_ = np.mean([hyb_rows[i]["mrr"] for i in idx])
            print(f"  {diff:<8}(n={len(idx)}): R@5 {b5:.3f}→{h5:.3f} | MRR {bm_:.3f}→{hm_:.3f}")

    with open("eval_50_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "cases": [{"id": i, "query": q, "difficulty": d, "label_count": len(l),
                       "base": base_rows[i], "hybrid": hyb_rows[i]}
                      for i, (q, d, l) in enumerate(Q)],
            "summary": summary,
        }, f, ensure_ascii=False, indent=2)
    print("\n结果已存 eval_50_result.json")


if __name__ == "__main__":
    main()
