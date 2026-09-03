"""
MARS V7 FastAPI 服务
支持 SSE 流式输出 + RESTful API

注意：本文件已与 mars_agent_v77.py 对齐。
当前版本的 MARSV71Agent 不再包含 memory / reranker 子系统，
相关端点已做无状态降级处理。
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, AsyncGenerator
from contextlib import asynccontextmanager

# Windows 控制台默认编码可能是 cp1252/cp936，中文 print 会抛 UnicodeEncodeError，
# 这里统一把 stdout/stderr 切到 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from rag_utils import load_vectorstore, hybrid_vector_search

# ============================================================
# 数据模型 (Pydantic)
# ============================================================

class SearchRequest(BaseModel):
    """搜索请求"""
    query: str
    user_id: str = "default"
    stream: bool = False  # 是否流式输出
    top_k: int = 10
    min_score: Optional[float] = None
    category: Optional[str] = None


class FeedbackRequest(BaseModel):
    """反馈请求"""
    user_id: str = "default"
    feedback_type: str  # "accept" or "reject"
    mall_id: Optional[str] = None
    mall_name: Optional[str] = None
    reason: Optional[str] = None


class ToolCallRequest(BaseModel):
    """工具调用请求"""
    tool_name: str
    parameters: Dict


# ============================================================
# 全局变量（延迟加载）
# ============================================================

agent = None
df_malls = None


def get_agent():
    """获取或初始化 Agent"""
    global agent, df_malls

    if agent is not None:
        return agent

    print("[API] 初始化 MARS Agent...")

    import pandas as pd

    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

    df_malls = pd.read_csv("mall_wide_table_shanghai_final_v3.csv", dtype={'mall_id': str})

    # 向量库为可选依赖（模型与加载逻辑统一走 rag_utils）
    vectorstore = load_vectorstore("./chroma_db")

    # 导入 MARS Agent
    from mars_agent_v77 import MARSV71Agent
    agent = MARSV71Agent(vectorstore, df_malls)

    print(f"[API] Agent 初始化完成，商场数: {len(df_malls)}")

    return agent


# ============================================================
# FastAPI 应用
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] 服务启动中...")
    yield
    print("[API] 服务关闭")


app = FastAPI(
    title="MARS V7 API",
    description="Memory-Augmented Reasoning for Spatial Decisions",
    version="7.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 流式输出生成器
# ============================================================

async def stream_react_process(query: str, agent) -> AsyncGenerator[str, None]:
    """流式输出 ReAct 推理过程，SSE 格式: data: {json}\\n\\n"""

    yield f"data: {json.dumps({'type': 'start', 'query': query}, ensure_ascii=False)}\n\n"
    await asyncio.sleep(0.1)

    route_result = agent.router.run({"query": query})
    yield f"data: {json.dumps({'type': 'route', 'data': route_result}, ensure_ascii=False)}\n\n"
    await asyncio.sleep(0.1)

    if route_result.get('route_to') == 'react_agent':
        if agent.react_agent is None:
            from mars_agent_v77 import ReActAgent
            agent.react_agent = ReActAgent(agent.tool_registry, agent.llm_client)

        react_agent = agent.react_agent
        react_agent.trajectory = []
        history = ""

        for step in range(react_agent.max_steps):
            yield f"data: {json.dumps({'type': 'thinking', 'step': step + 1}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.3)

            response = react_agent._call_llm(query, history)
            parsed = react_agent._parse_response(response)

            step_data = {
                "step": step + 1,
                "thought": parsed.get("thought", ""),
                "action": parsed.get("action"),
                "action_input": parsed.get("action_input"),
            }
            react_agent.trajectory.append(step_data)

            yield f"data: {json.dumps({'type': 'thought', 'data': step_data}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.2)

            if parsed.get("final_answer"):
                yield f"data: {json.dumps({'type': 'final_answer', 'data': parsed['final_answer']}, ensure_ascii=False)}\n\n"
                break

            if parsed.get("action"):
                action = parsed["action"]
                action_input = parsed.get("action_input", {})

                yield f"data: {json.dumps({'type': 'tool_call', 'tool': action, 'input': action_input}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.2)

                result = react_agent.tools.execute(action, **action_input)
                observation = result.data if result.success else result.message
                step_data["observation"] = observation

                yield f"data: {json.dumps({'type': 'observation', 'data': observation}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.2)

                history += f"\nThought: {parsed.get('thought', '')}\n"
                history += f"Action: {action}\n"
                history += f"Action Input: {json.dumps(action_input, ensure_ascii=False)}\n"
                history += f"Observation: {json.dumps(observation, ensure_ascii=False)[:500]}\n"

    else:
        # 简单查询 → RAG（向量库未加载时降级为提示）
        yield f"data: {json.dumps({'type': 'rag_search', 'message': '执行向量检索...'}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.3)

        if agent.vectorstore is None:
            yield f"data: {json.dumps({'type': 'message', 'data': '向量库未加载，请使用 /api/search 或安装 RAG 依赖'}, ensure_ascii=False)}\n\n"
        else:
            candidates = hybrid_vector_search(agent.vectorstore, query, top_k=10)
            for i, mall in enumerate(candidates):
                yield f"data: {json.dumps({'type': 'result', 'rank': i + 1, 'data': mall}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.1)

    yield f"data: {json.dumps({'type': 'end', 'message': '推理完成'}, ensure_ascii=False)}\n\n"


# ============================================================
# API 端点
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "MARS V7 API",
        "status": "running",
        "version": "7.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/init")
async def init_llm():
    try:
        a = get_agent()
        result = a.init_llm()
        return {"success": True, "message": result}
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.post("/api/search")
async def search(request: SearchRequest):
    """搜索接口，支持普通响应和流式响应"""
    a = get_agent()

    if request.stream:
        return StreamingResponse(
            stream_react_process(request.query, a),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # 普通响应：执行完整推荐，返回 markdown 文本 + HTML 报告路径
    try:
        markdown_output, html_path = a.recommend(request.query)

        results = []
        malls = a.react_agent.collected_data.get("malls", []) if a.react_agent else []
        for r in malls[:request.top_k]:
            results.append({
                "mall_id": r.get('mall_id', ''),
                "mall_name": r.get('mall_name', ''),
                "district": r.get('district', ''),
                "score_market": r.get('score_market', 0),
                "traffic_daily": r.get('traffic_daily', 0),
                "match_score": r.get('match_score'),
                "match_level": r.get('match_level', ''),
            })

        return {
            "success": True,
            "query": request.query,
            "message": markdown_output,
            "html_report": html_path,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/search/stream")
async def search_stream(query: str):
    a = get_agent()
    return StreamingResponse(
        stream_react_process(query, a),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/api/feedback")
async def feedback(request: FeedbackRequest):
    """提交反馈（当前版本为无状态降级实现）"""
    return {
        "success": True,
        "message": "已收到反馈（当前版本未启用记忆系统）",
        "feedback": request.feedback_type,
    }


@app.post("/api/tool")
async def call_tool(request: ToolCallRequest):
    try:
        a = get_agent()
        result = a.tool_registry.execute(request.tool_name, **request.parameters)
        return {"success": result.success, "data": result.data, "message": result.message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tools")
async def list_tools():
    a = get_agent()
    tools = []
    for name, tool in a.tool_registry.tools.items():
        tools.append({"name": name, "description": tool.description, "parameters": tool.parameters})
    return {"tools": tools}


@app.get("/api/memory")
async def get_memory(user_id: str = "default"):
    """获取用户记忆状态（当前版本未启用记忆系统）"""
    return {
        "user_id": user_id,
        "note": "当前版本未启用记忆系统",
        "weights": {},
    }


@app.delete("/api/memory")
async def clear_memory(user_id: str = "default"):
    return {"success": True, "message": "记忆系统未启用，无需清空"}


# ============================================================
# 统计和监控
# ============================================================

@app.get("/api/stats")
async def get_stats():
    global df_malls
    if df_malls is None:
        get_agent()

    stats = {
        "total_malls": len(df_malls) if df_malls is not None else 0,
        "with_traffic": int(df_malls['traffic_daily'].notna().sum()) if df_malls is not None else 0,
        "timestamp": datetime.now().isoformat(),
    }
    if df_malls is not None and 'brand_count' in df_malls.columns:
        stats["with_brands"] = int((df_malls['brand_count'] > 0).sum())

    return stats


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MARS V7 API Server")
    print("=" * 60)
    print("\nEndpoints:")
    print("  POST /api/search      - 搜索（支持流式）")
    print("  GET  /api/search/stream?query=xxx - 流式搜索")
    print("  POST /api/feedback    - 提交反馈")
    print("  POST /api/tool        - 调用工具")
    print("  GET  /api/tools       - 工具列表")
    print("  GET  /api/memory      - 获取记忆")
    print("  GET  /api/stats       - 系统统计")
    print("=" * 60)
    print("\n启动服务...")

    uvicorn.run(
        "mars_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
