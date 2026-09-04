"""
RAG 公共工具：嵌入模型加载 + 查询拆分 + 多查询混合检索

统一嵌入模型（可用环境变量 EMBEDDING_MODEL 切换，默认 BAAI/bge-m3，1024 维）。
mars_agent / mars_api / create_vectordb_v6 共用此模块，保证模型与向量库维度一致。

说明：更换模型后必须重新运行 create_vectordb_v6.py 重建向量库，否则维度不一致会加载失败。
"""

import os
import re
from typing import List, Dict

# 加载 .env（可选依赖：python-dotenv）。未安装时静默跳过，
# 环境变量仍可通过 export / set 直接提供。
# 放在这里而非各入口：rag_utils 是所有脚本共同 import 的配置单点，
# 一处加载即可覆盖建库、评估与两个服务入口。
# 显式指定路径而非依赖 find_dotenv()：后者在交互环境（REPL / Jupyter /
# python -c / 打包后）会退化成从 CWD 查找，导致 .env 静默失效。
# override=False 是默认行为，shell 里已 set 的变量优先于 .env。
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _resolve_model_name() -> str:
    """若本地已下载该模型目录，优先用本地路径（离线可用，绕过 HF Hub 下载）"""
    model = EMBEDDING_MODEL
    if "/" in model:
        local = os.path.join(_MODEL_DIR, model.split("/")[-1])
        if os.path.isdir(local) and os.path.exists(os.path.join(local, "config.json")):
            return local
    return model

# 复合查询拆分分隔符：中文标点 / 空白 + 常见并列连词
_SPLIT_RE = re.compile(r"[,，、;；。．.\s]+|和|与|及|以及|还有|并且|而且|并且|同时")

_cache: Dict[str, object] = {}


def get_embeddings():
    """加载嵌入模型（带进程内缓存，新版/旧版导入回退）"""
    if "embeddings" in _cache:
        return _cache["embeddings"]

    model = _resolve_model_name()
    embeddings = None
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception:
        embeddings = None

    if embeddings is None:
        try:
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception:
            embeddings = None

    _cache["embeddings"] = embeddings
    return embeddings


def load_vectorstore(chroma_dir: str = "./chroma_db", embeddings=None):
    """加载 Chroma 向量库，失败返回 None（不影响主流程）"""
    if embeddings is None:
        embeddings = get_embeddings()
    if embeddings is None:
        return None

    try:
        from langchain_chroma import Chroma
    except Exception:
        try:
            from langchain_community.vectorstores import Chroma
        except Exception:
            return None

    try:
        return Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    except Exception as e:
        print(f"⚠️ 向量库加载失败(可忽略): {e}")
        return None


def split_query(query: str) -> List[str]:
    """把复合查询拆成若干子查询；单一概念则原样返回"""
    if not query or not query.strip():
        return []
    parts = [p.strip() for p in _SPLIT_RE.split(query) if p.strip()]
    if len(parts) <= 1:
        return [query.strip()]
    # 过滤过短碎片（如单个「的」），若过滤后为空则退回原查询
    parts = [p for p in parts if len(p) >= 2]
    return parts or [query.strip()]


def _doc_to_dict(doc, sim_0_100: float) -> Dict:
    meta = doc.metadata
    return {
        "mall_id": str(meta.get('mall_id', '')),
        "mall_name": meta.get('mall_name', ''),
        "district": meta.get('district', ''),
        "score_market": meta.get('score_market', 0) or 0,
        "traffic_daily": meta.get('traffic_daily', 0) or 0,
        "brand_count": meta.get('brand_count', 0) or 0,
        "similarity": round(sim_0_100, 1),
    }


def hybrid_vector_search(vectorstore, query: str, top_k: int = 5) -> List[Dict]:
    """多查询混合检索：拆分复合查询 → 各子查询分别检索 → 合并去重重排

    合并策略：取某商场在各子查询中的最高相似度，并给「同时命中多个子查询」
    的商场小幅加分，兼顾 OR 召回与 AND 精确。
    """
    if vectorstore is None:
        return [{"error": "向量库未加载，请安装 RAG 依赖(pip install -r requirements-rag.txt)后重启"}]

    sub_queries = split_query(query)
    if not sub_queries:
        return []

    per_k = max(top_k * 4, 20)

    best: Dict[str, dict] = {}
    for sq in sub_queries:
        try:
            results = vectorstore.similarity_search_with_score(sq, k=per_k)
        except Exception as e:
            print(f"⚠️ 子查询检索失败({sq}): {e}")
            continue
        for doc, dist in results:
            mid = str(doc.metadata.get('mall_id', ''))
            sim = max(0.0, (2 - dist) / 2 * 100)  # L2 距离 → 0-100 相似度
            if mid not in best:
                best[mid] = {"sim": sim, "count": 1, "doc": doc}
            else:
                best[mid]["count"] += 1
                if sim > best[mid]["sim"]:
                    best[mid]["sim"] = sim
                    best[mid]["doc"] = doc

    scored = []
    for mid, item in best.items():
        final = item["sim"] + 3.0 * (item["count"] - 1)  # 多子查询命中加分
        scored.append((final, item["doc"], item["sim"]))
    scored.sort(key=lambda x: -x[0])

    out = []
    for final, doc, _ in scored[:top_k]:
        d = _doc_to_dict(doc, min(final, 100.0))
        out.append(d)
    return out
