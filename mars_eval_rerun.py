"""重跑 medium 档里「完成=✗」的 6 条，验证是网络抖动还是真实能力问题。

背景：全量 50 条里 medium 完成率 0.538，是三档最低，且中途出现 3 次 DeepSeek
Connection error（约在第 30 条附近）。本脚本单独重跑这 6 条，排除/坐实网络因素。
"""

import sys

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from rag_utils import get_embeddings, load_vectorstore  # noqa: E402
from mars_agent import MARSAgent  # noqa: E402
from mars_eval_50 import build_queries  # noqa: E402
from mars_eval_retrieval import DATA_FILE, recall_at_k  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REQUIRED = ["score_customer_match", "analyze_brand_competition"]

# 全量 run 里「完成=✗」且不属于 medium 档的 4 条（easy×3 + hard×1，精确匹配）
RERUN = [
    "有名创优品的商场",
    "有古驰GUCCI的商场",
    "有FILA的商场",
    "白领商务人群的商场",
]


def precision_of(retrieved, relevant):
    return len(set(retrieved) & set(relevant)) / len(retrieved) if retrieved else 0.0


def main():
    df = pd.read_csv(DATA_FILE, dtype={"mall_id": str})
    targets = [(q, d, l) for q, d, l in build_queries(df) if q in RERUN]
    print(f"重跑 {len(targets)} 条（medium 档完成=✗）")

    embeddings = get_embeddings()
    vs = load_vectorstore("./chroma_db", embeddings)
    agent = MARSAgent(vs, df)
    status = agent.init_llm()
    print(status)
    if "成功" not in status:
        print("❌ LLM 未初始化成功，重跑无意义")
        return

    for q, d, l in targets:
        fa, traj = agent.react_agent.run(q, required_tools=REQUIRED)
        mids = agent._extract_mall_ids(fa)
        rejected = sum(1 for s in traj if "系统拒绝" in (s.get("observation") or ""))
        noact = sum(1 for s in traj if "未检测到 Action" in (s.get("observation") or ""))
        print(f"完成={'✓' if mids else '✗'} 步数={len(traj)} 一次={'✓' if rejected == 0 else '✗'} "
              f"废步={rejected + noact} 精度={precision_of(mids, l):.2f} "
              f"R@5={recall_at_k(mids, l, 5):.2f}  {q}")


if __name__ == "__main__":
    main()
