"""
MARS 混合检索：BM25（词面）+ 稠密向量，RRF 融合，可选 Cross-Encoder Rerank。

设计：
- 稠密召回：复用 rag_utils.hybrid_vector_search（多查询拆分 + 相似度合并）
- 词面召回：Okapi BM25，语料直接取向量库 page_content（与嵌入文本完全一致）
- 融合：Reciprocal Rank Fusion（RRF，k=60）
- 中文分词：字符二元组（零依赖，无需 jieba），英文/数字整体成词

可选 rerank：若本地有 bge-reranker 模型，用 CrossEncoder 对融合候选精排。
"""

import math
import os
import re
from collections import Counter
from typing import List, Dict

import numpy as np

from rag_utils import get_embeddings, load_vectorstore, hybrid_vector_search, split_query

_JIEBA = None


def _get_jieba():
    """惰性加载 jieba 分词；不可用时返回 None（回退字符二元组）"""
    global _JIEBA
    if _JIEBA is None:
        try:
            import jieba
            jieba.setLogLevel(60)
            _JIEBA = jieba.lcut
        except Exception:
            _JIEBA = None
    return _JIEBA


def _tokenize(text: str) -> List[str]:
    """jieba 中文分词（≥2 字）+ 拉丁/数字整体成词；无 jieba 时回退字符二元组"""
    text = str(text).lower()
    tokens: List[str] = []
    for seg in re.findall(r"[a-z0-9]+", text):
        tokens.append(seg)
    lcut = _get_jieba()
    for seg in re.findall(r"[一-鿿]+", text):
        if lcut is not None:
            for w in lcut(seg):
                w = w.strip()
                if len(w) >= 2:
                    tokens.append(w)
        elif len(seg) == 1:
            tokens.append(seg)
        else:
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return tokens


class BM25:
    """标准 Okapi BM25（k1=1.5, b=0.75）"""

    def __init__(self, corpus: List[List[str]]):
        self.n = len(corpus)
        self.doc_len = np.array([len(d) for d in corpus], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.doc_freqs = [Counter(d) for d in corpus]

        df = Counter()
        for freq in self.doc_freqs:
            for w in freq:
                df[w] += 1
        self.idf = {w: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def score(self, query_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        qf = Counter(query_tokens)
        scores = np.zeros(self.n, dtype=np.float32)
        for w, qtf in qf.items():
            idf = self.idf.get(w, 0.0)
            if idf == 0.0:
                continue
            k1_plus = k1 + 1.0
            for i, freq in enumerate(self.doc_freqs):
                f = freq.get(w, 0)
                if f == 0:
                    continue
                denom = f + k1 * (1 - b + b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (f * k1_plus / denom) * qtf
        return scores


class HybridRetriever:
    """BM25 + 稠密向量的混合检索器，一次加载，多次查询"""

    def __init__(self, vectorstore=None, chroma_dir: str = "./chroma_db"):
        if vectorstore is None:
            vectorstore = load_vectorstore(chroma_dir, get_embeddings())
        self.vs = vectorstore

        data = self.vs.get(include=["documents", "metadatas"])
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []

        self.docs: List[Dict] = []
        self._meta_by_id: Dict[str, dict] = {}
        for text, meta in zip(documents, metadatas):
            mid = str((meta or {}).get("mall_id", ""))
            self._meta_by_id[mid] = meta or {}
            self.docs.append({"mid": mid, "text": text or ""})

        corpus = [_tokenize(d["text"]) for d in self.docs]
        self.bm25 = BM25(corpus)
        self._bm25_mids = [d["mid"] for d in self.docs]

    def _bm25_rank(self, query: str, top_k: int) -> List[str]:
        """对整句 + 各子查询分别算 BM25，取每文档最高分"""
        sub = split_query(query) or [query]
        best = np.zeros(len(self.docs), dtype=np.float32)
        for sq in sub:
            tok = _tokenize(sq)
            if not tok:
                continue
            s = self.bm25.score(tok)
            best = np.maximum(best, s)
        order = np.argsort(-best)[:top_k]
        return [self._bm25_mids[i] for i in order if best[i] > 0]

    def _signal(self, query: str) -> float:
        """查询里最「稀有」词项的 IDF —— 判断 BM25 是否有真实词面信号。

        概念型查询（客流/白领等转述词）在语料里没有字面词，稀有词 IDF≈0，
        BM25 只能靠常见二元组凑数 → 应降权；品牌/区名等字面词 IDF 高 → 可用。
        """
        toks = set()
        for sq in (split_query(query) or [query]):
            toks.update(_tokenize(sq))
        if not toks:
            return 0.0
        return max((self.bm25.idf.get(t, 0.0) for t in toks), default=0.0)

    def search(self, query: str, top_k: int = 10, candidate_k: int = 50,
               reranker=None) -> List[Dict]:
        """加权 RRF 融合 BM25 与稠密排序，返回 top_k 个结果。

        BM25 权重由稀有词信号门控：无稀有词时 w→0（纯稠密，不引入噪声）。
        """
        # 稠密排序（多查询混合检索）
        dense_res = hybrid_vector_search(self.vs, query, top_k=candidate_k)
        dense_rank = [str(r["mall_id"]) for r in dense_res if r.get("mall_id")]
        dense_by_mid = {str(r["mall_id"]): r for r in dense_res if r.get("mall_id")}

        # BM25 排序 + 稀有词信号门控权重
        bm25_rank = self._bm25_rank(query, candidate_k)
        sig = self._signal(query)
        w_bm25 = float(np.clip((sig - 1.2) / 3.0, 0.0, 1.0))

        # 加权 RRF 融合
        fused = self._rrf([dense_rank, bm25_rank], weights=[1.0, w_bm25])
        candidates = fused[:max(top_k, 20)]

        # 可选 rerank
        if reranker is not None and len(candidates) > 1:
            pairs = [(query, self._rerank_passage(mid)) for mid in candidates]
            try:
                scores = reranker.predict(pairs)
                order = np.argsort(-np.asarray(scores))
                candidates = [candidates[i] for i in order]
            except Exception as e:
                print(f"⚠️ rerank 失败，回退融合序: {e}")

        out = []
        for mid in candidates[:top_k]:
            if mid in dense_by_mid:
                out.append(dense_by_mid[mid])
            else:
                meta = self._meta_by_id.get(mid, {})
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

    def _doc_text(self, mid: str) -> str:
        for d in self.docs:
            if d["mid"] == mid:
                return d["text"]
        return ""

    _TIER_TRAFFIC = frozenset({
        "顶级高端商场", "高端商场", "中高端商场", "中端商场", "大众商场",
        "超高人气", "高人气", "中等人气",
    })

    def _rerank_passage(self, mid: str) -> str:
        """rerank 用精简 passage：名称+区县+档次+客流+品牌+品类+客群，去掉长 position_text"""
        text = self._doc_text(mid)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return self._meta_by_id.get(mid, {}).get("mall_name", "")
        keep = [lines[0]]  # 名称
        for ln in lines[1:]:
            if ln in self._TIER_TRAFFIC or ln.startswith(("品牌：", "品类：", "客群：")):
                keep.append(ln[:250])
            elif len(ln) <= 10 and ln.startswith("上海"):  # 区县短行
                keep.append(ln)
        return "\n".join(keep)

    @staticmethod
    def _rrf(ranked_lists: List[List[str]], weights: List[float] = None, k: int = 60) -> List[str]:
        if weights is None:
            weights = [1.0] * len(ranked_lists)
        scores: Dict[str, float] = {}
        for w, rlist in zip(weights, ranked_lists):
            if w <= 0:
                continue
            for rank, mid in enumerate(rlist):
                if mid:
                    scores[mid] = scores.get(mid, 0.0) + w / (k + rank + 1)
        return [m for m, _ in sorted(scores.items(), key=lambda x: -x[1])]


def load_reranker(model_dir: str):
    """从本地目录加载 Cross-Encoder reranker（若可用）"""
    if not (model_dir and os.path.isdir(model_dir)):
        return None
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_dir, max_length=512)
    except Exception as e:
        print(f"⚠️ reranker 加载失败: {e}")
        return None


if __name__ == "__main__":
    # 快速自测
    import sys
    PY = os.environ.get("PYTHONIOENCODING", "utf-8")
    retriever = HybridRetriever()
    q = sys.argv[1] if len(sys.argv) > 1 else "奢侈品牌聚集的高端商场"
    print(f"查询: {q}")
    for r in retriever.search(q, top_k=5):
        print(f"  - {r['mall_name']} (评分:{r['score_market']}, 客流:{r['traffic_daily']}, 品牌:{r['brand_count']})")
