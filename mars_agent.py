"""
MARS v1.0: 详细报告版
生成专业的选址分析报告，包含完整的数据支撑和推荐理由

核心功能：
1. 客群匹配：基于TGI指数找到目标客户聚集的商场
2. 匹配度评分：计算精确的客群匹配度分数
3. 多维评估：评分、客流、客群画像综合分析
4. 分层采样：覆盖高端/中高端/中端商场，结果多样化
5. 竞品分析：基于知识库判断品牌入驻情况
6. 详细报告：生成包含对比表格、推荐理由的专业报告

工具列表（9个）：
- match_customer_profile: 客群匹配（支持top_k=10-50）
- score_customer_match: TGI匹配度评分
- analyze_customer_profile: 客群画像深度分析
- search_mall: 商场搜索（支持多种排序）
- get_traffic_ranking: 客流排名
- get_mall_detail: 商场详情
- compare_malls: 商场对比
- analyze_brand_competition: 竞品分析
- vector_search_malls: 向量语义检索（融合 hybrid + GraphRAG，三级回退）
"""

import os
import sys
import json
import re
import traceback
import requests
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import gradio as gr

from rag_utils import get_embeddings, load_vectorstore, hybrid_vector_search

try:
    from mars_hybrid_retrieval import HybridRetriever
except Exception:
    HybridRetriever = None

try:
    from mars_graphrag import MallKnowledgeGraph, GraphRAGRetriever
except Exception:
    MallKnowledgeGraph = None
    GraphRAGRetriever = None

# Windows 控制台默认编码可能是 cp1252/cp936，中文 print 会抛 UnicodeEncodeError，
# 这里统一把 stdout/stderr 切到 UTF-8，保证中文输出正常。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# 配置
# ============================================================

DATA_FILE = "mall_wide_table_shanghai_final_v3.csv"
CHROMA_DIR = "./chroma_db"
MEMORY_DIR = "./memory"

# LLM 配置
LLM_PROVIDER = "deepseek"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 设置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def to_json_serializable(obj):
    """将 pandas/numpy 类型转换为 JSON 可序列化类型"""
    if obj is None:
        return None
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return obj


# ============================================================
# 竞品分析 Agent（实验性 · LLM 辅助信号）
# ============================================================

class CompetitorSearchAgent:
    """
    竞品分析 Agent —— 实验性辅助信息（experimental enrichment），非事实数据源
    
    说明：
    - 外部搜索 API 当前环境不可用，暂用 LLM 训练知识作为实验性辅助信号（非事实数据源）
    - 生产环境应接入实时品牌 / POI / 商业地产数据源，LLM 仅作辅助信号
    - 会明确标注可信度，提醒用户数据可能非最新
    
    优点：
    - 不需要额外的API Key
    - 覆盖范围广（LLM知道的商场都能分析）
    - 有可信度标注，用户知道数据的可靠程度
    
    局限：
    - 数据基于训练截止日期，可能不是最新
    - 对于新开业商场或新入驻品牌可能不准确
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.cache = {}
    
    def search_competitors_in_mall(self, target_brand: str, category: str, 
                                   mall_name: str) -> Dict:
        """分析某商场内目标品牌的竞品情况"""
        cache_key = f"{target_brand}_{category}_{mall_name}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 使用LLM分析
        if self.llm_client:
            result = self._llm_analyze(target_brand, category, mall_name)
        else:
            result = self._fallback_response(target_brand, category, mall_name)
        
        self.cache[cache_key] = result
        return result
    
    def _llm_analyze(self, target_brand: str, category: str, mall_name: str) -> Dict:
        """使用LLM分析竞品情况"""
        prompt = f"""请根据你的知识，分析上海{mall_name}的{category}品牌入驻情况。

请回答以下问题：
1. {mall_name}有哪些知名{category}品牌入驻？列出你确定知道的品牌。
2. {target_brand}是否已入驻{mall_name}？
3. 对于{target_brand}来说，主要竞争品牌有哪些？

请严格按照以下JSON格式回答，不要有任何其他内容：
{{
    "mall": "{mall_name}",
    "category": "{category}",
    "target_brand": "{target_brand}",
    "target_brand_exists": true或false,
    "known_brands": ["品牌1", "品牌2"],
    "competitors": ["竞品1", "竞品2"],
    "confidence": "高"或"中"或"低",
    "note": "补充说明，如数据来源或不确定的地方"
}}

注意：
- 只列出你确实知道的品牌，不要猜测
- 如果不确定某品牌是否入驻，不要列入
- confidence填写你对这些信息的把握程度
- 如果对该商场了解很少，confidence填"低"
"""

        try:
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.1  # 降低温度，让回答更确定
            )
            content = response.choices[0].message.content.strip()
            
            # 解析JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
                
                # 计算竞争程度
                competitors = [c for c in data.get("competitors", []) 
                              if c.lower() != target_brand.lower()]
                
                if len(competitors) >= 5:
                    level = "🔴激烈竞争"
                elif len(competitors) >= 3:
                    level = "🟠中等竞争"
                elif len(competitors) >= 1:
                    level = "🟡少量竞争"
                else:
                    level = "🔵蓝海机会"
                
                confidence = data.get("confidence", "中")
                confidence_note = {
                    "高": "数据可信度高",
                    "中": "数据仅供参考，建议实地确认",
                    "低": "数据不确定，请务必实地考察"
                }.get(confidence, "建议实地确认")
                
                return {
                    "mall": mall_name,
                    "category": category,
                    "target_brand": target_brand,
                    "target_brand_exists": data.get("target_brand_exists"),
                    "known_brands": data.get("known_brands", []),
                    "competitors": competitors,
                    "competitor_count": len(competitors),
                    "competition_level": level,
                    "confidence": confidence,
                    "confidence_note": confidence_note,
                    "note": data.get("note", ""),
                    "data_source": "LLM辅助信号（实验性）",
                    "analysis": f"{mall_name}{category}品类：发现{len(competitors)}个竞品，{level}。{target_brand}{'已' if data.get('target_brand_exists') else '未'}入驻。[可信度:{confidence}]"
                }
                
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
        except Exception as e:
            print(f"LLM分析失败: {e}")
        
        return self._fallback_response(target_brand, category, mall_name)
    
    def _fallback_response(self, target_brand: str, category: str, mall_name: str) -> Dict:
        """备用响应（LLM不可用时使用静态知识库）"""
        # 上海主要商场的品牌数据（基于公开信息整理）
        mall_brands = {
            "前滩太古里": {
                "运动": {"Nike": True, "Adidas": True, "Lululemon": True, "始祖鸟": True, "Salomon": True, "On昂跑": True},
                "服装": {"COS": True, "Arket": True, "Theory": True, "Sandro": True, "Maje": True}
            },
            "港汇恒隆广场": {
                "运动": {"Nike": True, "Adidas": True, "Lululemon": True, "Puma": True},
                "服装": {"Zara": True, "H&M": True, "优衣库": True, "COS": True}
            },
            "上海IFC": {
                "运动": {"Nike": True, "Adidas": True, "Lululemon": True},
                "服装": {"Prada": True, "Burberry": True, "MaxMara": True}
            },
            "日月光中心": {
                "运动": {"Nike": True, "Adidas": True, "李宁": True, "迪卡侬": True},
                "服装": {"优衣库": True, "GU": True, "UR": True}
            },
            "静安嘉里中心": {
                "运动": {"Nike": True, "Adidas": True, "Lululemon": True, "始祖鸟": True},
                "服装": {"COS": True, "Theory": True, "Massimo Dutti": True}
            },
            "兴业太古汇": {
                "运动": {"Nike": True, "Adidas": True, "Lululemon": True},
                "服装": {"COS": True, "& Other Stories": True, "Acne Studios": True}
            },
            "环球港": {
                "运动": {"Nike": True, "Adidas": True, "李宁": True, "安踏": True, "特步": True},
                "服装": {"Zara": True, "H&M": True, "优衣库": True, "GAP": True}
            },
            "上海恒隆广场": {
                "运动": {"Nike": False, "Adidas": False},  # 奢侈品定位，运动品牌较少
                "服装": {"Gucci": True, "LV": True, "Dior": True, "Chanel": True, "Hermès": True}
            },
        }
        
        # 查找匹配的商场
        mall_data = None
        for key in mall_brands.keys():
            if key in mall_name or mall_name in key:
                mall_data = mall_brands[key]
                break
        
        if mall_data and category in mall_data:
            brands_in_category = mall_data[category]
            target_exists = brands_in_category.get(target_brand, None)
            competitors = [b for b, exists in brands_in_category.items() 
                          if exists and b.lower() != target_brand.lower()]
            
            if len(competitors) >= 5:
                level = "🔴激烈竞争"
            elif len(competitors) >= 3:
                level = "🟠中等竞争"
            elif len(competitors) >= 1:
                level = "🟡少量竞争"
            else:
                level = "🔵蓝海机会"
            
            return {
                "mall": mall_name,
                "category": category,
                "target_brand": target_brand,
                "target_brand_exists": target_exists,
                "known_brands": list(brands_in_category.keys()),
                "competitors": competitors,
                "competitor_count": len(competitors),
                "competition_level": level,
                "confidence": "中",
                "confidence_note": "基于公开信息整理，建议实地确认",
                "data_source": "静态知识库",
                "analysis": f"{mall_name}{category}品类：发现{len(competitors)}个竞品，{level}。{target_brand}{'已' if target_exists else '可能未' if target_exists is False else '不确定是否'}入驻。"
            }
        
        # 完全没有数据
        return {
            "mall": mall_name,
            "category": category,
            "target_brand": target_brand,
            "target_brand_exists": None,
            "competitors": [],
            "competitor_count": 0,
            "competition_level": "❓数据不足",
            "confidence": "低",
            "confidence_note": "暂无该商场数据，请实地考察",
            "data_source": "无",
            "analysis": f"暂无{mall_name}的{category}品牌数据，建议联系商场招商部门或实地考察。"
        }
    
    def batch_analyze_competition(self, target_brand: str, category: str,
                                  mall_names: List[str]) -> List[Dict]:
        """批量分析多个商场的竞品情况"""
        results = []
        for mall_name in mall_names[:3]:  # 最多分析3个商场
            try:
                competition = self.search_competitors_in_mall(target_brand, category, mall_name)
                results.append(competition)
            except Exception as e:
                results.append({
                    "mall": mall_name,
                    "category": category,
                    "error": str(e),
                    "competitors": [],
                    "competition_level": "❓分析失败"
                })
        return results


# ============================================================
# 工具定义
# ============================================================

@dataclass
class ToolResult:
    success: bool
    data: Any
    message: str


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, str]
    function: Callable
    
    def to_prompt(self) -> str:
        params_str = ", ".join([f"{k}: {v}" for k, v in self.parameters.items()])
        return f"- {self.name}({params_str}): {self.description}"


class ToolRegistry:
    """工具注册中心 v1.0"""
    
    def __init__(self, df_malls: pd.DataFrame, competitor_agent: CompetitorSearchAgent = None, vectorstore=None):
        self.df_malls = df_malls
        self.competitor_agent = competitor_agent or CompetitorSearchAgent()
        self.vectorstore = vectorstore
        self._hybrid_retriever = None  # 惰性构建的 BM25+稠密混合检索器
        self._graph_retriever = None   # 惰性构建的 GraphRAG 检索器（知识图谱 2-hop）
        self.tools: Dict[str, Tool] = {}
        self._register_all_tools()
    
    def _register_all_tools(self):
        """注册所有工具"""
        
        # 工具1: 搜索商场
        self.register(Tool(
            name="search_mall",
            description="根据关键词和条件搜索商场，返回基础信息（评分、客流）",
            parameters={
                "query": "搜索关键词(如'高端')",
                "min_score": "最低评分(可选,如85)",
                "district": "区域名称(可选,如'徐汇区')",
                "top_k": "返回数量(默认10)",
                "sort_by": "排序方式: score(评分)、traffic(客流)、balanced(综合,默认)"
            },
            function=self._search_mall
        ))
        
        # 工具2: 客流排名
        self.register(Tool(
            name="get_traffic_ranking",
            description="获取商场客流量排名",
            parameters={
                "district": "区域名称(可选,如'徐汇区')",
                "top_k": "返回数量(默认10，可设5-30)"
            },
            function=self._get_traffic_ranking
        ))
        
        # 工具3: 客群匹配
        self.register(Tool(
            name="match_customer_profile",
            description="根据目标客群特征匹配商场，返回多样化的候选商场列表",
            parameters={
                "age_group": "目标年龄段(如'25-35')",
                "consume_level": "消费水平(高/中高/中/低)",
                "keywords": "其他关键词(如'白领 运动')",
                "top_k": "返回数量(默认20，可设10-50)"
            },
            function=self._match_customer_profile
        ))
        
        # 工具4: 商场详情
        self.register(Tool(
            name="get_mall_detail",
            description="获取单个商场的详细信息",
            parameters={
                "mall_id": "商场ID"
            },
            function=self._get_mall_detail
        ))
        
        # 工具5: 对比商场
        self.register(Tool(
            name="compare_malls",
            description="对比多个商场的基础指标",
            parameters={
                "mall_ids": "商场ID列表(2-5个)"
            },
            function=self._compare_malls
        ))
        
        # 工具6: 竞品分析（实验性 · LLM辅助信号）
        self.register(Tool(
            name="analyze_brand_competition",
            description="分析商场内的品牌竞争情况（实验性LLM辅助信号，会标注可信度与来源）",
            parameters={
                "target_brand": "目标品牌(如'Nike')",
                "category": "品类(运动/服装/餐饮)",
                "mall_names": "候选商场名称列表(1-3个)"
            },
            function=self._analyze_brand_competition
        ))
        
        # 工具7: 客群画像深度分析
        self.register(Tool(
            name="analyze_customer_profile",
            description="深度分析商场客群画像，包含年龄分布、消费水平、TGI指数等详细数据",
            parameters={
                "mall_id": "商场ID",
                "focus": "关注维度(age/consume/all)，默认all"
            },
            function=self._analyze_customer_profile
        ))
        
        # 工具8: 客群匹配度评分
        self.register(Tool(
            name="score_customer_match",
            description="计算商场与目标客群的匹配度评分，基于TGI指数",
            parameters={
                "mall_ids": "商场ID列表",
                "target_age": "目标年龄段(如'18-35')",
                "target_consume": "目标消费水平(高/中高/中)"
            },
            function=self._score_customer_match
        ))

        # 工具9: 语义向量检索（RAG）
        self.register(Tool(
            name="vector_search_malls",
            description="基于向量库的语义检索：用自然语言描述(如'高端运动户外 高收入人群')匹配最相似的商场",
            parameters={
                "query": "自然语言描述",
                "top_k": "返回数量(默认5)"
            },
            function=self._vector_search_malls
        ))
    
    def register(self, tool: Tool):
        self.tools[tool.name] = tool
    
    def get_tools_prompt(self) -> str:
        lines = ["可用工具:"]
        for tool in self.tools.values():
            lines.append(tool.to_prompt())
        return "\n".join(lines)
    
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        if tool_name not in self.tools:
            return ToolResult(False, None, f"未知工具: {tool_name}")
        
        try:
            result = self.tools[tool_name].function(**kwargs)
            return ToolResult(True, result, "执行成功")
        except Exception as e:
            return ToolResult(False, None, f"执行失败: {str(e)}")
    
    # ===== 工具实现 =====
    
    def _search_mall(self, query: str = "", min_score: float = 0, 
                     district: str = None, top_k: int = 10,
                     sort_by: str = "balanced") -> List[Dict]:
        """
        搜索商场
        sort_by: score(评分), traffic(客流), balanced(综合-默认)
        """
        df = self.df_malls.copy()
        
        if min_score:
            df = df[df['score_market'] >= float(min_score)]
        
        if district:
            df = df[df['district'].str.contains(district, na=False)]
        
        if query:
            mask = (
                df['mall_name'].str.contains(query, na=False) |
                df['profile_text'].astype(str).str.contains(query, na=False) |
                df['district'].str.contains(query, na=False)
            )
            df = df[mask] if mask.any() else df
        
        # 根据不同策略排序
        if sort_by == "traffic":
            df = df[df['traffic_daily'].notna()]
            df = df.nlargest(top_k, 'traffic_daily')
        elif sort_by == "balanced":
            # 综合评分：评分归一化 * 0.4 + 客流归一化 * 0.6
            df = df.copy()
            df['score_norm'] = df['score_market'] / df['score_market'].max()
            df['traffic_norm'] = df['traffic_daily'].fillna(0) / (df['traffic_daily'].max() or 1)
            df['combined_score'] = df['score_norm'] * 0.4 + df['traffic_norm'] * 0.6
            df = df.nlargest(top_k, 'combined_score')
        else:  # score
            df = df.nlargest(top_k, 'score_market')
        
        return [{
            "mall_id": str(row['mall_id']),
            "mall_name": str(row['mall_name']),
            "district": str(row.get('district', '') or ''),
            "score_market": to_json_serializable(row.get('score_market', 0)) or 0,
            "traffic_daily": to_json_serializable(row.get('traffic_daily', 0)) or 0,
        } for _, row in df.iterrows()]
    
    def _get_traffic_ranking(self, district: str = None, top_k: int = 10) -> List[Dict]:
        df = self.df_malls[self.df_malls['traffic_daily'].notna()].copy()
        
        if district:
            df = df[df['district'].str.contains(district, na=False)]
        
        df = df.nlargest(top_k, 'traffic_daily')
        
        return [{
            "rank": rank,
            "mall_id": str(row['mall_id']),
            "mall_name": str(row['mall_name']),
            "district": str(row.get('district', '') or ''),
            "traffic_daily": int(to_json_serializable(row['traffic_daily']) or 0),
            "score_market": to_json_serializable(row.get('score_market', 0)) or 0,
        } for rank, (_, row) in enumerate(df.iterrows(), 1)]
    
    def _match_customer_profile(self, age_group: str = None, 
                                consume_level: str = None,
                                keywords: str = None,
                                top_k: int = 20) -> List[Dict]:
        """
        客群匹配 - 使用多样化采样策略
        返回更多样的商场，而不只是评分最高的几个
        """
        df = self.df_malls.copy()
        
        search_terms = []
        if age_group:
            search_terms.append(age_group)
        if consume_level:
            # 消费水平关键词映射
            consume_map = {
                "高": ["高消费", "高端"],
                "中高": ["中高", "次高"],
                "中": ["中等", "大众"],
                "低": ["低消费", "平价"]
            }
            search_terms.extend(consume_map.get(consume_level, [consume_level]))
        if keywords:
            search_terms.extend(keywords.split())
        
        if search_terms:
            mask = pd.Series([False] * len(df))
            for term in search_terms:
                # 优先搜索 embedding_text（覆盖率100%，内容更丰富）
                if 'embedding_text' in df.columns:
                    mask |= df['embedding_text'].astype(str).str.contains(term, na=False)
                # profile_text 作为补充
                mask |= df['profile_text'].astype(str).str.contains(term, na=False)
            df = df[mask] if mask.any() else df
        
        # 多样化采样策略：从不同区域、不同档次抽取商场
        df = df.copy()
        df['score_norm'] = df['score_market'] / df['score_market'].max()
        df['traffic_norm'] = df['traffic_daily'].fillna(0) / (df['traffic_daily'].max() or 1)
        df['combined_score'] = df['score_norm'] * 0.4 + df['traffic_norm'] * 0.6
        
        # 分层采样：高端(30%) + 中高端(40%) + 中端(30%)
        results_df = pd.DataFrame()
        
        # 高端商场（评分>=90）
        high_end = df[df['score_market'] >= 90].nlargest(int(top_k * 0.3) + 1, 'combined_score')
        results_df = pd.concat([results_df, high_end])
        
        # 中高端商场（评分80-90）
        mid_high = df[(df['score_market'] >= 80) & (df['score_market'] < 90)].nlargest(int(top_k * 0.4) + 1, 'combined_score')
        results_df = pd.concat([results_df, mid_high])
        
        # 中端商场（评分70-80，客流优先）
        mid = df[(df['score_market'] >= 70) & (df['score_market'] < 80)].nlargest(int(top_k * 0.3) + 1, 'traffic_norm')
        results_df = pd.concat([results_df, mid])
        
        # 去重并按综合分排序
        results_df = results_df.drop_duplicates(subset=['mall_id'])
        results_df = results_df.nlargest(top_k, 'combined_score')
        
        results = []
        for _, row in results_df.iterrows():
            # 从 embedding_text 提取客群关键信息
            profile_summary = ""
            if 'embedding_text' in row and pd.notna(row.get('embedding_text')):
                emb_text = str(row['embedding_text'])
                # 提取目标客群和消费能力
                import re
                target_match = re.search(r'目标客群：([^。]+)', emb_text)
                consume_match = re.search(r'消费能力：([^。]+)', emb_text)
                
                parts = []
                if target_match:
                    parts.append(f"👥 {target_match.group(1)}")
                if consume_match:
                    parts.append(f"💰 {consume_match.group(1)[:50]}")
                profile_summary = " | ".join(parts)
            
            # 补充 profile_text 的年龄分布
            if pd.notna(row.get('profile_text')):
                profile_text = str(row['profile_text'])
                import re
                age_match = re.search(r'age：([^；]+)', profile_text)
                if age_match and profile_summary:
                    profile_summary += f" | 年龄: {age_match.group(1)[:40]}"
            
            if not profile_summary:
                profile_summary = str(row.get('profile_text', '') or '')[:100]
            
            results.append({
                "mall_id": str(row['mall_id']),
                "mall_name": str(row['mall_name']),
                "district": str(row.get('district', '') or ''),
                "score_market": to_json_serializable(row.get('score_market', 0)) or 0,
                "traffic_daily": to_json_serializable(row.get('traffic_daily', 0)) or 0,
                "profile_summary": profile_summary[:200],
            })
        
        return results
    
    def _get_mall_detail(self, mall_id: str) -> Dict:
        """获取商场详情 - 包含 embedding_text 的丰富信息"""
        row = self.df_malls[self.df_malls['mall_id'].astype(str) == str(mall_id)]
        if row.empty:
            return {"error": "商场不存在"}
        
        row = row.iloc[0]
        
        # 从 embedding_text 提取关键信息
        location_feature = ""
        target_customer = ""
        consume_ability = ""
        
        if 'embedding_text' in row and pd.notna(row.get('embedding_text')):
            emb_text = str(row['embedding_text'])
            import re
            
            loc_match = re.search(r'位置特征：([^。]+)', emb_text)
            target_match = re.search(r'目标客群：([^。]+)', emb_text)
            consume_match = re.search(r'消费能力：([^。]+)', emb_text)
            
            if loc_match:
                location_feature = loc_match.group(1)
            if target_match:
                target_customer = target_match.group(1)
            if consume_match:
                consume_ability = consume_match.group(1)
        
        return {
            "mall_id": str(row['mall_id']),
            "mall_name": str(row['mall_name']),
            "district": str(row.get('district', '') or ''),
            "address": str(row.get('address', '') or ''),
            "score_market": to_json_serializable(row.get('score_market', 0)) or 0,
            "traffic_daily": to_json_serializable(row.get('traffic_daily', 0)) or 0,
            "mall_area": to_json_serializable(row.get('mall_area', 0)) or 0,
            # 新增：从 embedding_text 提取的结构化信息
            "location_feature": location_feature,
            "target_customer": target_customer,
            "consume_ability": consume_ability,
            "profile_text": str(row.get('profile_text', '') or '')[:150],
        }
    
    def _compare_malls(self, mall_ids: List[str]) -> List[Dict]:
        return [self._get_mall_detail(mid) for mid in mall_ids[:5] 
                if "error" not in self._get_mall_detail(mid)]
    
    def _analyze_brand_competition(self, target_brand: str, category: str,
                                   mall_names: List[str]) -> List[Dict]:
        """【实验性辅助信号】分析商场内的品牌竞争情况"""
        return self.competitor_agent.batch_analyze_competition(
            target_brand, category, mall_names[:3]  # 限制最多3个商场，避免LLM调用过多
        )
    
    def _analyze_customer_profile(self, mall_id: str, focus: str = "all") -> Dict:
        """
        深度分析商场客群画像
        
        解析 customer_profile_json 提取：
        - 年龄分布及TGI
        - 消费水平及TGI
        - 到访偏好
        - 商场忠诚度（到店频次）
        """
        row = self.df_malls[self.df_malls['mall_id'].astype(str) == str(mall_id)]
        if row.empty:
            return {"error": "商场不存在"}
        
        row = row.iloc[0]
        result = {
            "mall_id": str(mall_id),
            "mall_name": str(row['mall_name']),
        }
        
        # 解析 customer_profile_json
        if 'customer_profile_json' not in row or pd.isna(row.get('customer_profile_json')):
            result["error"] = "该商场暂无详细客群数据"
            # 降级使用 embedding_text
            if pd.notna(row.get('embedding_text')):
                result["fallback_profile"] = str(row['embedding_text'])[:300]
            return result
        
        try:
            profile = json.loads(row['customer_profile_json'])
        except:
            result["error"] = "客群数据解析失败"
            return result
        
        # 提取年龄分布
        if focus in ["all", "age"]:
            age_dist = []
            for key, val in profile.items():
                if key.startswith('age|'):
                    age_range = key.split('|')[1]
                    age_dist.append({
                        "age_range": age_range,
                        "ratio": round(val['ratio'] * 100, 1),
                        "tgi": val['tgi'],
                        "interpretation": "高浓度" if val['tgi'] > 120 else "中等" if val['tgi'] > 80 else "低浓度"
                    })
            # 按比例排序
            age_dist.sort(key=lambda x: x['ratio'], reverse=True)
            result["age_distribution"] = age_dist
            
            # 计算年轻人占比（18-35岁）
            young_ratio = sum(a['ratio'] for a in age_dist if any(y in a['age_range'] for y in ['18-24', '25-30', '31-35']))
            result["young_ratio"] = f"{young_ratio:.1f}%"
        
        # 提取消费水平
        if focus in ["all", "consume"]:
            consume_dist = []
            for key, val in profile.items():
                if key.startswith('consume|'):
                    level = key.split('|')[1]
                    consume_dist.append({
                        "level": level,
                        "ratio": round(val['ratio'] * 100, 1),
                        "tgi": val['tgi'],
                        "interpretation": "高浓度" if val['tgi'] > 120 else "中等" if val['tgi'] > 80 else "低浓度"
                    })
            consume_dist.sort(key=lambda x: x['ratio'], reverse=True)
            result["consume_distribution"] = consume_dist
            
            # 计算高消费占比
            high_consume_ratio = sum(c['ratio'] for c in consume_dist if c['level'] in ['高', '次高'])
            result["high_consume_ratio"] = f"{high_consume_ratio:.1f}%"
        
        # 提取到访偏好（了解客群兴趣）
        if focus == "all":
            visit_prefs = []
            for key, val in profile.items():
                if key.startswith('到访偏好大类|') and val['tgi'] > 100:
                    pref = key.split('|')[1]
                    visit_prefs.append({
                        "preference": pref,
                        "tgi": val['tgi']
                    })
            visit_prefs.sort(key=lambda x: x['tgi'], reverse=True)
            result["visit_preferences"] = visit_prefs[:5]
        
        # 提取忠诚度（到店频次）
        if focus == "all":
            freq_dist = []
            for key, val in profile.items():
                if key.startswith('商场到店频次|'):
                    freq = key.split('|')[1]
                    freq_dist.append({
                        "frequency": freq,
                        "ratio": round(val['ratio'] * 100, 1),
                        "tgi": val['tgi']
                    })
            result["visit_frequency"] = freq_dist
            
            # 高频用户占比（5次以上）
            high_freq = sum(f['ratio'] for f in freq_dist if '次' in f['frequency'] and 
                          any(n in f['frequency'] for n in ['5-6', '7-8', '9-10', '11']))
            result["loyal_customer_ratio"] = f"{high_freq:.1f}%"
        
        return result
    
    def _score_customer_match(self, mall_ids: List[str], 
                              target_age: str = "18-35",
                              target_consume: str = "高") -> List[Dict]:
        """
        计算商场与目标客群的匹配度评分
        
        基于 TGI 指数计算：
        - TGI > 120: 目标人群高度聚集
        - TGI 80-120: 接近平均水平
        - TGI < 80: 目标人群较少
        """
        # 解析目标年龄段
        target_ages = []
        if "18" in target_age or "24" in target_age:
            target_ages.append("18-24")
        if "25" in target_age or "30" in target_age:
            target_ages.append("25-30")
        if "31" in target_age or "35" in target_age:
            target_ages.append("31-35")
        if "36" in target_age or "40" in target_age:
            target_ages.append("36-40")
        if not target_ages:
            target_ages = ["18-24", "25-30", "31-35"]  # 默认年轻人
        
        # 解析目标消费水平
        target_consumes = []
        if "高" in target_consume:
            target_consumes.extend(["高", "次高"])
        elif "中" in target_consume:
            target_consumes.extend(["中", "次高"])
        else:
            target_consumes = ["中"]
        
        results = []
        
        for mall_id in mall_ids[:10]:
            row = self.df_malls[self.df_malls['mall_id'].astype(str) == str(mall_id)]
            if row.empty:
                continue
            
            row = row.iloc[0]
            result = {
                "mall_id": str(mall_id),
                "mall_name": str(row['mall_name']),
                "district": str(row.get('district', '')),
            }
            
            if pd.isna(row.get('customer_profile_json')):
                result["match_score"] = 50  # 无数据给中等分
                result["note"] = "缺少详细客群数据"
                results.append(result)
                continue
            
            try:
                profile = json.loads(row['customer_profile_json'])
            except:
                result["match_score"] = 50
                result["note"] = "数据解析失败"
                results.append(result)
                continue
            
            # 计算年龄匹配分（0-50分）
            age_score = 0
            age_details = []
            for key, val in profile.items():
                if key.startswith('age|'):
                    age_range = key.split('|')[1]
                    if age_range in target_ages:
                        # TGI > 100 表示高于平均，越高越好
                        tgi = val['tgi']
                        contribution = min(15, (tgi - 50) / 5)  # TGI 100 -> 10分, TGI 150 -> 20分
                        age_score += max(0, contribution)
                        age_details.append(f"{age_range}(TGI={tgi})")
            
            age_score = min(50, age_score)
            result["age_score"] = round(age_score, 1)
            result["age_match"] = ", ".join(age_details)
            
            # 计算消费匹配分（0-50分）
            consume_score = 0
            consume_details = []
            for key, val in profile.items():
                if key.startswith('consume|'):
                    level = key.split('|')[1]
                    if level in target_consumes:
                        tgi = val['tgi']
                        contribution = min(20, (tgi - 50) / 4)
                        consume_score += max(0, contribution)
                        consume_details.append(f"{level}(TGI={tgi})")
            
            consume_score = min(50, consume_score)
            result["consume_score"] = round(consume_score, 1)
            result["consume_match"] = ", ".join(consume_details)
            
            # 总分
            total_score = age_score + consume_score
            result["match_score"] = round(total_score, 1)
            
            # 评级
            if total_score >= 70:
                result["match_level"] = "⭐⭐⭐ 高度匹配"
            elif total_score >= 50:
                result["match_level"] = "⭐⭐ 中等匹配"
            else:
                result["match_level"] = "⭐ 匹配度较低"
            
            results.append(result)
        
        # 按匹配分排序
        results.sort(key=lambda x: x.get('match_score', 0), reverse=True)

        return results

    def _get_hybrid_retriever(self):
        """惰性构建 BM25+稠密混合检索器（首次调用时用向量库 page_content 建 BM25 索引）。

        失败或不可用时返回 None/False，_vector_search_malls 会回退到纯向量检索。
        """
        if self._hybrid_retriever is None and self.vectorstore is not None and HybridRetriever is not None:
            try:
                self._hybrid_retriever = HybridRetriever(self.vectorstore)
                print("✅ 混合检索(BM25+稠密)已启用")
            except Exception as e:
                print(f"⚠️ 混合检索初始化失败，回退纯向量: {e}")
                self._hybrid_retriever = False
        return self._hybrid_retriever

    def _get_graph_retriever(self):
        """惰性构建 GraphRAG 检索器（知识图谱 2-hop 扩展）。

        首次调用需构建图谱（同区 O(n²) 相似度，约 1~10s）；失败返回 None/False，
        _vector_search_malls 会回退到混合检索或纯向量。
        """
        if self._graph_retriever is None and self.vectorstore is not None and GraphRAGRetriever is not None:
            try:
                kg = MallKnowledgeGraph()
                kg.build_from_dataframe(self.df_malls)
                self._graph_retriever = GraphRAGRetriever(self.vectorstore, kg, self.df_malls)
                print("✅ 图谱检索(GraphRAG 2-hop)已启用")
            except Exception as e:
                print(f"⚠️ 图谱检索初始化失败，回退混合检索: {e}")
                self._graph_retriever = False
        return self._graph_retriever

    def _vector_search_malls(self, query: str, top_k: int = 5) -> List[Dict]:
        """基于向量库的语义检索（RAG）——优先 hybrid+GraphRAG 融合，逐级回退混合/纯向量"""
        hr = self._get_hybrid_retriever()
        gr = self._get_graph_retriever()
        if hr and gr:
            try:
                return gr.hybrid_graph_search(hr, query, top_k=top_k)
            except Exception as e:
                print(f"⚠️ 融合检索失败，回退混合检索: {e}")
        if hr:
            try:
                return hr.search(query, top_k=top_k)
            except Exception as e:
                print(f"⚠️ 混合检索失败，回退纯向量: {e}")
        return hybrid_vector_search(self.vectorstore, query, top_k=top_k)


# ============================================================
# ReAct Agent（更新版）
# ============================================================

REACT_SYSTEM_PROMPT = """你是一个商业选址专家Agent。你需要通过 Thought-Action-Observation 循环来解决用户的选址问题。

【重要规则】
1. 禁止编造任何数字或事实！所有数据必须来自工具返回的Observation
2. 匹配度评分必须来自 score_customer_match 工具
3. 竞品信息必须来自 analyze_brand_competition 工具，并注明其可信度(confidence)
4. 至少执行3个工具调用后才能给出Final Answer

【选址逻辑】（按重要性排序）
1. **客群匹配** ⭐⭐⭐⭐⭐（核心）:
   - 第1步：match_customer_profile 或 vector_search_malls 找候选商场（top_k=20-30）
   - 第2步：score_customer_match 计算TGI匹配度
   
2. **商场评分** ⭐⭐⭐⭐: 从match结果已包含

3. **客流量** ⭐⭐⭐: get_traffic_ranking 验证排名

4. **竞品分析** ⭐⭐（可选）: analyze_brand_competition 分析品牌竞争
   - 返回结果会标注可信度，低可信度需提醒用户实地确认

【top_k参数】快速:15-20 | 全面:30-40

{tools}

【输出格式】
Thought: [分析]
Action: [工具]
Action Input: {{"参数": "值"}}

完成3+步后：
Final Answer: 

**推荐Top3商场：**
1. XX商场 - TGI匹配度XX分，评分XX，客流XX/天
2. ...
3. ...

**推荐理由：**（50字内）

**竞品情况：**（如有调用，说明竞争程度和可信度）
"""


class ReActAgent:
    """ReAct Agent v1.0"""
    
    def __init__(self, tool_registry: ToolRegistry, llm_client=None):
        self.tools = tool_registry
        self.llm_client = llm_client
        self.max_steps = 8
        self.trajectory = []
        self.collected_data = {}
        self.report_generator = HTMLReportGenerator()  # 添加报告生成器


    def run(self, query: str) -> Tuple[str, List[Dict]]:
        self.trajectory = []
        self.collected_data = {"malls": [], "competition": []}
        history = ""
        min_steps = 3  # 最少执行3步工具调用
        tool_calls = 0  # 记录工具调用次数
        
        for step in range(self.max_steps):
            response = self._call_llm(query, history)
            parsed = self._parse_response(response)
            
            self.trajectory.append({
                "step": step + 1,
                "thought": parsed.get("thought", ""),
                "action": parsed.get("action"),
                "action_input": parsed.get("action_input"),
            })
            
            # 检查是否过早结束
            if parsed.get("final_answer"):
                if tool_calls < min_steps:
                    # 强制继续，提示LLM需要更多工具调用
                    history += f"\nThought: {parsed.get('thought', '')}\n"
                    history += f"Observation: 【系统提示】你只调用了{tool_calls}次工具，还需要调用更多工具验证数据。请继续使用 score_customer_match 计算TGI匹配度，或使用其他工具获取更多信息。\n"
                    self.trajectory[-1]["observation"] = f"系统要求继续调用工具（已调用{tool_calls}次，最少{min_steps}次）"
                    continue
                else:
                    self.trajectory[-1]["final_answer"] = parsed["final_answer"]
                    return parsed["final_answer"], self.trajectory
            
            if parsed.get("action"):
                action = parsed["action"]
                action_input = parsed.get("action_input", {})
                tool_calls += 1  # 记录工具调用
                
                result = self.tools.execute(action, **action_input)
                observation = json.dumps(result.data, ensure_ascii=False, indent=2) if result.success else result.message
                
                # 收集数据
                if result.success and result.data:
                    self._collect_data(action, result.data)
                
                self.trajectory[-1]["observation"] = observation
                
                # 增加截断长度，让LLM看到更多结果
                obs_truncated = observation[:3000] + "..." if len(observation) > 3000 else observation
                history += f"\nThought: {parsed.get('thought', '')}\n"
                history += f"Action: {action}\n"
                history += f"Action Input: {json.dumps(action_input, ensure_ascii=False)}\n"
                history += f"Observation: {obs_truncated}\n"
            else:
                history += f"\nThought: {parsed.get('thought', '')}\n"
                history += "Observation: 请输出 Action 或 Final Answer\n"
        
        return self._force_summarize(query), self.trajectory
    
    def _collect_data(self, action: str, data: Any):
        """收集工具返回的数据，用于生成最终报告"""
        if action in ["search_mall", "get_traffic_ranking", "match_customer_profile", "vector_search_malls"]:
            if isinstance(data, list):
                # 收集商场基础数据
                for item in data[:15]:
                    self.collected_data["malls"].append(item)
        
        elif action == "score_customer_match":
            # 收集匹配度评分数据
            if isinstance(data, list):
                if "match_scores" not in self.collected_data:
                    self.collected_data["match_scores"] = {}
                for item in data:
                    mall_id = item.get("mall_id", "")
                    if mall_id:
                        self.collected_data["match_scores"][mall_id] = {
                            "match_score": item.get("match_score", 0),
                            "match_level": item.get("match_level", ""),
                            "age_match": item.get("age_match", ""),
                            "consume_match": item.get("consume_match", ""),
                        }
                    # 同时更新malls中的数据
                    mall_name = item.get("mall_name", "")
                    for m in self.collected_data.get("malls", []):
                        if m.get("mall_name") == mall_name or m.get("mall_id") == mall_id:
                            m["match_score"] = item.get("match_score", 0)
                            m["match_level"] = item.get("match_level", "")
                            m["age_match"] = item.get("age_match", "")
                            m["consume_match"] = item.get("consume_match", "")
        
        elif action in ["search_brand_competition", "analyze_brand_competition"]:
            if isinstance(data, list):
                self.collected_data["competition"].extend(data)
    
    def _call_llm(self, query: str, history: str) -> str:
        if not self.llm_client:
            return self._mock_llm_response(query, history)
        
        try:
            from openai import OpenAI
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": REACT_SYSTEM_PROMPT.format(
                        tools=self.tools.get_tools_prompt()
                    )},
                    {"role": "user", "content": f"用户问题: {query}\n\n历史:\n{history}\n\n请继续:"}
                ],
                max_tokens=2000  # 增加到2000，避免Final Answer被截断
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Thought: LLM调用失败: {e}\nFinal Answer: 抱歉，系统出现错误"
    
    def _mock_llm_response(self, query: str, history: str) -> str:
        """模拟LLM响应 - 客群匹配优先"""
        step_count = history.count("Action:")
        
        # 提取品牌和品类信息
        brand = "该品牌"
        category = "运动"
        for b in ["耐克", "Nike", "始祖鸟", "Adidas", "李宁", "安踏", "Lululemon"]:
            if b in query:
                brand = b
                break
        if "服装" in query or "服饰" in query:
            category = "服装"
        elif "餐饮" in query or "餐厅" in query:
            category = "餐饮"
        
        # 根据品牌推断目标客群
        brand_profiles = {
            "Nike": {"age": "18-35", "consume": "中高", "keywords": "年轻 运动 时尚"},
            "耐克": {"age": "18-35", "consume": "中高", "keywords": "年轻 运动 时尚"},
            "始祖鸟": {"age": "25-45", "consume": "高", "keywords": "高端 户外 白领"},
            "Lululemon": {"age": "25-40", "consume": "高", "keywords": "女性 瑜伽 白领"},
            "李宁": {"age": "18-35", "consume": "中", "keywords": "年轻 国潮 学生"},
            "安踏": {"age": "18-40", "consume": "中", "keywords": "大众 运动 家庭"},
        }
        profile = brand_profiles.get(brand, {"age": "25-35", "consume": "中高", "keywords": "白领"})
        
        # 提取区域条件
        district = None
        for d in ["徐汇", "静安", "黄浦", "浦东", "长宁", "虹口", "杨浦"]:
            if d in query:
                district = d + "区"
                break
        
        min_score = 85 if "高端" in query or "中高端" in query else 75
        
        if step_count == 0:
            # 第1步：客群匹配（最重要！）
            return f"""Thought: 用户需要为{brand}选址。选址的核心是找到目标客户聚集的地方。{brand}的目标客群是{profile['age']}岁、{profile['consume']}消费水平的人群。首先用客群匹配找到符合条件的商场。
Action: match_customer_profile
Action Input: {{"age_group": "{profile['age']}", "consume_level": "{profile['consume']}", "keywords": "{profile['keywords']}", "top_k": 25}}"""
        
        elif step_count == 1:
            # 第2步：TGI匹配度评分（核心！）
            mall_ids = []
            for m in self.collected_data.get("malls", [])[:10]:
                mid = m.get("mall_id", "")
                if mid and mid not in mall_ids:
                    mall_ids.append(mid)
            
            if not mall_ids:
                mall_ids = ["7360353142875267072", "7360353133333225472", "7360353191545970688"]
            
            return f"""Thought: 已找到客群匹配的候选商场。现在计算这些商场与{brand}目标客群的TGI匹配度评分，这是选址的核心依据。
Action: score_customer_match
Action Input: {{"mall_ids": {json.dumps(mall_ids[:8])}, "target_age": "{profile['age']}", "target_consume": "{profile['consume']}"}}"""
        
        elif step_count == 2:
            # 第3步：客流验证
            district_param = f'"district": "{district}", ' if district else ''
            return f"""Thought: 匹配度评分完成。现在验证这些商场的客流量排名，确保有足够的人流支撑门店经营。
Action: get_traffic_ranking
Action Input: {{{district_param}"top_k": 15}}"""
        
        elif step_count == 3:
            # 第4步：竞品分析（补充）
            mall_names = []
            for m in self.collected_data.get("malls", [])[:5]:
                name = m.get("mall_name", "")
                if name and name not in mall_names:
                    mall_names.append(name)
            
            if not mall_names:
                mall_names = ["上海港汇恒隆广场", "上海前滩太古里", "上海日月光中心"]
            
            return f"""Thought: 核心筛选完成（客群匹配度+客流）。最后补充了解一下候选商场的{category}竞品情况作为参考。
Action: analyze_brand_competition
Action Input: {{"target_brand": "{brand}", "category": "{category}", "mall_names": {json.dumps(mall_names[:3], ensure_ascii=False)}}}"""
        
        elif step_count >= 4:
            return self._generate_final_answer_v2(query, brand, category, profile)
        
        return "Thought: 信息收集完成\nFinal Answer: 请查看以上分析结果"
    
    def _generate_final_answer_v2(self, query: str, brand: str, category: str, profile: Dict = None) -> str:
        """生成详细的选址分析报告"""
        malls_data = self.collected_data.get("malls", [])
        competition_data = self.collected_data.get("competition", [])
        match_scores = self.collected_data.get("match_scores", {})
        
        if profile is None:
            profile = {"age": "25-35", "consume": "中高", "keywords": ""}
        
        # 整合数据
        mall_dict = {}
        for m in malls_data:
            name = m.get("mall_name", "")
            if name and name not in mall_dict:
                mall_dict[name] = {
                    "name": name,
                    "mall_id": m.get("mall_id", ""),
                    "district": m.get("district", ""),
                    "traffic": m.get("traffic_daily", 0),
                    "score": m.get("score_market", 0),
                    "profile": m.get("profile_summary", ""),
                    "match_score": m.get("match_score", 0),
                    "age_match": m.get("age_match", ""),
                    "consume_match": m.get("consume_match", ""),
                }
        
        # 补充匹配度评分
        for mall_id, score_data in match_scores.items():
            for name, data in mall_dict.items():
                if data.get("mall_id") == mall_id:
                    data["match_score"] = score_data.get("match_score", 0)
                    data["match_level"] = score_data.get("match_level", "")
                    data["age_match"] = score_data.get("age_match", "")
                    data["consume_match"] = score_data.get("consume_match", "")
        
        # 补充竞品信息
        for c in competition_data:
            name = c.get("mall", "")
            for mall_name in mall_dict:
                if name in mall_name or mall_name in name:
                    mall_dict[mall_name]["competition"] = c.get("competition_level", "")
                    mall_dict[mall_name]["competitors"] = c.get("competitors", [])
                    mall_dict[mall_name]["target_exists"] = c.get("target_brand_exists")
                    mall_dict[mall_name]["confidence"] = c.get("confidence", "")
        
        # 排序：匹配度 > 评分 > 客流
        def sort_key(x):
            match = x.get("match_score", 0) or 0
            score = x.get("score", 0) or 0
            traffic = x.get("traffic", 0) or 0
            return (match, score, traffic)
        
        sorted_malls = sorted(mall_dict.values(), key=sort_key, reverse=True)[:5]
        
        # ========== 生成详细报告 ==========
        lines = []
        
        # 报告头部
        lines.append("=" * 50)
        lines.append(f"# 📊 {brand} 上海选址分析报告")
        lines.append("=" * 50)
        lines.append("")
        
        # 一、需求概述
        lines.append("## 一、选址需求")
        lines.append(f"- **品牌**: {brand}")
        lines.append(f"- **品类**: {category}")
        lines.append(f"- **目标客群**: {profile['age']}岁 | {profile['consume']}消费 | {profile.get('keywords', '')}")
        lines.append(f"- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        
        # 二、推荐商场排名
        lines.append("## 二、推荐商场 TOP 5")
        lines.append("")
        
        for i, m in enumerate(sorted_malls, 1):
            # 商场标题
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1]
            lines.append(f"### {medal} 第{i}名：{m['name']}")
            lines.append(f"📍 **{m.get('district', '上海')}**")
            lines.append("")
            
            # 核心指标表格
            lines.append("| 指标 | 数值 | 说明 |")
            lines.append("|------|------|------|")
            
            # TGI匹配度
            match_score = m.get('match_score', 0)
            match_level = m.get('match_level', '')
            if match_score:
                lines.append(f"| 🎯 客群匹配度 | **{match_score}分** | {match_level} |")
            
            # 商场评分
            score = m.get('score', 0)
            if score:
                score_level = "顶级" if score >= 90 else "优秀" if score >= 85 else "良好" if score >= 80 else "一般"
                lines.append(f"| ⭐ 商场评分 | **{score}分** | {score_level}商场 |")
            
            # 日均客流
            traffic = m.get('traffic', 0)
            if traffic:
                traffic_level = "超高" if traffic >= 40000 else "高" if traffic >= 25000 else "中等" if traffic >= 15000 else "一般"
                lines.append(f"| 👥 日均客流 | **{int(traffic):,}人** | {traffic_level}客流 |")
            
            lines.append("")
            
            # 客群画像详情
            if m.get('age_match') or m.get('consume_match'):
                lines.append("**📈 客群匹配详情：**")
                if m.get('age_match'):
                    lines.append(f"- 年龄匹配: {m['age_match']}")
                if m.get('consume_match'):
                    lines.append(f"- 消费匹配: {m['consume_match']}")
                lines.append("")
            
            # 客群画像摘要
            if m.get('profile'):
                lines.append(f"**👤 客群画像**: {m['profile'][:100]}...")
                lines.append("")
            
            # 竞品情况
            if m.get('competition'):
                target_status = ""
                if m.get('target_exists') is True:
                    target_status = f"（⚠️ {brand}已入驻）"
                elif m.get('target_exists') is False:
                    target_status = f"（✅ {brand}未入驻，有机会）"
                
                lines.append(f"**🏪 竞品情况**: {m['competition']} {target_status}")
                if m.get('competitors'):
                    lines.append(f"- 主要竞品: {', '.join(m['competitors'][:5])}")
                if m.get('confidence'):
                    lines.append(f"- 数据可信度: {m['confidence']}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        # 三、综合对比
        lines.append("## 三、商场对比一览")
        lines.append("")
        lines.append("| 排名 | 商场 | 区域 | 匹配度 | 评分 | 日客流 | 竞争 |")
        lines.append("|:----:|------|------|:------:|:----:|-------:|------|")
        
        for i, m in enumerate(sorted_malls, 1):
            match = m.get('match_score', '-')
            score = m.get('score', '-')
            traffic = f"{int(m.get('traffic', 0)):,}" if m.get('traffic') else '-'
            comp = m.get('competition', '-')
            lines.append(f"| {i} | {m['name'][:12]} | {m.get('district', '-')[:4]} | {match} | {score} | {traffic} | {comp[:6] if comp else '-'} |")
        
        lines.append("")
        
        # 四、选址建议
        lines.append("## 四、选址建议")
        lines.append("")
        
        if sorted_malls:
            top = sorted_malls[0]
            lines.append(f"### 🏆 首选推荐：{top['name']}")
            lines.append("")
            lines.append("**推荐理由：**")
            
            reasons = []
            if top.get('match_score') and top['match_score'] >= 50:
                reasons.append(f"客群匹配度高达{top['match_score']}分，目标客户浓度高")
            if top.get('score') and top['score'] >= 85:
                reasons.append(f"商场评分{top['score']}分，品牌形象有保障")
            if top.get('traffic') and top['traffic'] >= 20000:
                reasons.append(f"日均客流{int(top['traffic']):,}人，曝光量充足")
            if top.get('target_exists') is False:
                reasons.append(f"{brand}尚未入驻，市场空白机会")
            
            for j, reason in enumerate(reasons, 1):
                lines.append(f"{j}. {reason}")
            
            lines.append("")
            
            # 备选推荐
            if len(sorted_malls) >= 2:
                lines.append(f"### 🥈 备选推荐：{sorted_malls[1]['name']}")
                alt = sorted_malls[1]
                alt_reasons = []
                if alt.get('traffic', 0) > top.get('traffic', 0):
                    alt_reasons.append("客流量更大")
                if alt.get('score', 0) > top.get('score', 0):
                    alt_reasons.append("商场评分更高")
                if alt.get('target_exists') is False:
                    alt_reasons.append("品牌未入驻")
                if alt_reasons:
                    lines.append(f"- 优势: {', '.join(alt_reasons)}")
                lines.append("")
        
        # 五、注意事项
        lines.append("## 五、注意事项")
        lines.append("")
        lines.append("1. **数据时效性**: 客流数据基于历史统计，实际可能有波动")
        lines.append("2. **竞品信息**: 品牌入驻情况可能有变化，建议实地确认")
        lines.append("3. **租金成本**: 本报告未包含租金数据，需另行询价")
        lines.append("4. **位置点位**: 具体铺位选择需实地考察")
        lines.append("")
        
        # 报告结尾
        lines.append("---")
        lines.append(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append("*数据来源: MARS智能选址系统 v1.0*")
        
        return "Thought: 已完成详细的选址分析报告\nFinal Answer:\n" + "\n".join(lines)
    
    def _parse_response(self, response: str) -> Dict:
        result = {}
        
        thought_match = re.search(r'Thought:\s*(.+?)(?=\n(?:Action|Final Answer)|$)', response, re.DOTALL)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()
        
        final_match = re.search(r'Final Answer:\s*(.+)', response, re.DOTALL)
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            return result
        
        action_match = re.search(r'Action:\s*(\w+)', response)
        if action_match:
            result["action"] = action_match.group(1).strip()
        
        input_match = re.search(r'Action Input:\s*(\{.+?\})', response, re.DOTALL)
        if input_match:
            try:
                result["action_input"] = json.loads(input_match.group(1))
            except:
                result["action_input"] = {}
        
        return result
    
    def _force_summarize(self, query: str) -> str:
        """强制生成总结"""
        malls = self.collected_data.get("malls", [])
        competition = self.collected_data.get("competition", [])
        
        output = [f"## 📊 选址分析报告\n\n**需求**: {query}\n"]
        
        if competition:
            output.append("\n### 🏆 推荐商场（按竞争程度排序）\n")
            
            # 按竞争程度排序
            sorted_comp = sorted(competition, 
                key=lambda x: {"🔵蓝海": 5, "🟢较少": 4, "🟡中等": 3, "🟠中等偏高": 2, "🔴激烈": 1}.get(x.get("competition_level", ""), 0),
                reverse=True)
            
            for i, comp in enumerate(sorted_comp[:5], 1):
                output.append(f"**{i}. {comp.get('mall', '')}**")
                output.append(f"   - 竞争: {comp.get('competition_level', '未知')}")
                if comp.get('competitors'):
                    output.append(f"   - 竞品: {', '.join(comp['competitors'])}")
                output.append("")
        
        elif malls:
            output.append("\n### 候选商场\n")
            for i, m in enumerate(malls[:5], 1):
                output.append(f"{i}. {m.get('mall_name', '')} - 评分{m.get('score_market', 0)}")
        
        return "\n".join(output)
    
    def format_trajectory(self) -> str:
        output = ["### 🧠 ReAct 推理过程\n"]
        
        for step in self.trajectory:
            output.append(f"**Step {step['step']}**\n")
            output.append(f"💭 Thought: {step.get('thought', 'N/A')}\n")
            
            if step.get('action'):
                output.append(f"🔧 Action: `{step['action']}`\n")
                output.append(f"📥 Input: `{json.dumps(step.get('action_input', {}), ensure_ascii=False)}`\n")
            
            if step.get('observation'):
                obs = step['observation']
                if len(obs) > 500:
                    obs = obs[:500] + "..."
                output.append(f"👁 Observation:\n```json\n{obs}\n```\n")
            
            if step.get('final_answer'):
                output.append(f"✅ Final Answer: {step['final_answer']}\n")
            
            output.append("\n---\n")
        
        return "".join(output)
class HTMLReportGenerator:
    """
    生成专业的HTML选址分析报告
    包含：散点图、柱状图、雷达图等可视化图表
    """
    
    def generate_report(self, brand: str, category: str, profile: Dict, 
                        malls_data: List[Dict], competition_data: List[Dict],
                        match_scores: Dict) -> str:
        """生成完整的HTML报告"""
        
        # 整合数据
        report_data = self._prepare_data(malls_data, competition_data, match_scores)
        
        # 生成HTML
        html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{brand} 选址分析报告 - MARS</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: #fff;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .header .subtitle {{
            color: #a0a0a0;
            font-size: 14px;
        }}
        .header .brand-tag {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 8px 20px;
            border-radius: 20px;
            margin-top: 15px;
            font-size: 16px;
        }}
        .section {{
            padding: 30px 40px;
            border-bottom: 1px solid #eee;
        }}
        .section:last-child {{
            border-bottom: none;
        }}
        .section-title {{
            font-size: 18px;
            color: #1a1a2e;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section-title::before {{
            content: '';
            width: 4px;
            height: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 2px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .info-card {{
            background: #f8f9fa;
            padding: 15px 20px;
            border-radius: 12px;
            border-left: 4px solid #667eea;
        }}
        .info-card .label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        .info-card .value {{
            font-size: 18px;
            font-weight: 600;
            color: #1a1a2e;
        }}
        .chart-container {{
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .chart-title {{
            font-size: 14px;
            color: #666;
            margin-bottom: 15px;
            text-align: center;
        }}
        .chart-wrapper {{
            position: relative;
            height: 300px;
        }}
        .ranking-list {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .ranking-item {{
            display: flex;
            align-items: center;
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .ranking-item:hover {{
            transform: translateX(5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        .ranking-item.top1 {{
            background: linear-gradient(135deg, #fff9e6 0%, #fff3cd 100%);
            border: 2px solid #ffc107;
        }}
        .ranking-item.top2 {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 2px solid #adb5bd;
        }}
        .ranking-item.top3 {{
            background: linear-gradient(135deg, #fff5f5 0%, #ffe8e8 100%);
            border: 2px solid #cd7f32;
        }}
        .rank-badge {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 18px;
            margin-right: 20px;
            color: white;
        }}
        .rank-1 {{ background: linear-gradient(135deg, #ffd700 0%, #ffb300 100%); }}
        .rank-2 {{ background: linear-gradient(135deg, #c0c0c0 0%, #a0a0a0 100%); }}
        .rank-3 {{ background: linear-gradient(135deg, #cd7f32 0%, #b87333 100%); }}
        .rank-other {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .mall-info {{
            flex: 1;
        }}
        .mall-name {{
            font-size: 18px;
            font-weight: 600;
            color: #1a1a2e;
            margin-bottom: 5px;
        }}
        .mall-district {{
            font-size: 13px;
            color: #666;
        }}
        .mall-metrics {{
            display: flex;
            gap: 20px;
            align-items: center;
        }}
        .metric {{
            text-align: center;
        }}
        .metric-value {{
            font-size: 20px;
            font-weight: 700;
            color: #667eea;
        }}
        .metric-label {{
            font-size: 11px;
            color: #999;
        }}
        .competition-badge {{
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: 500;
        }}
        .competition-red {{ background: #ffe5e5; color: #dc3545; }}
        .competition-orange {{ background: #fff3e0; color: #ff9800; }}
        .competition-yellow {{ background: #fffde7; color: #ffc107; }}
        .competition-green {{ background: #e8f5e9; color: #4caf50; }}
        .competition-blue {{ background: #e3f2fd; color: #2196f3; }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .comparison-table th {{
            background: #1a1a2e;
            color: white;
            padding: 12px;
            text-align: left;
            font-size: 13px;
        }}
        .comparison-table td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
            font-size: 14px;
        }}
        .comparison-table tr:hover {{
            background: #f8f9fa;
        }}
        .recommendation-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
            padding: 25px;
            margin-top: 20px;
        }}
        .recommendation-box h3 {{
            font-size: 18px;
            margin-bottom: 15px;
        }}
        .recommendation-box ul {{
            list-style: none;
            padding: 0;
        }}
        .recommendation-box li {{
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
        }}
        .recommendation-box li::before {{
            content: '✓';
            position: absolute;
            left: 0;
            color: #90EE90;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 12px;
            background: #f8f9fa;
        }}
        .scatter-note {{
            font-size: 12px;
            color: #888;
            text-align: center;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- 头部 -->
        <div class="header">
            <h1>📊 选址分析报告</h1>
            <p class="subtitle">MARS智能选址系统 · {datetime.now().year}年{datetime.now().month}月{datetime.now().day}日</p>
            <span class="brand-tag">🏷️ {brand}</span>
        </div>
        
        <!-- 需求概述 -->
        <div class="section">
            <h2 class="section-title">选址需求</h2>
            <div class="info-grid">
                <div class="info-card">
                    <div class="label">品牌名称</div>
                    <div class="value">{brand}</div>
                </div>
                <div class="info-card">
                    <div class="label">品类</div>
                    <div class="value">{category}</div>
                </div>
                <div class="info-card">
                    <div class="label">目标年龄</div>
                    <div class="value">{profile.get('age', '25-35')}岁</div>
                </div>
                <div class="info-card">
                    <div class="label">消费水平</div>
                    <div class="value">{profile.get('consume', '中高')}</div>
                </div>
            </div>
        </div>
        
        <!-- 核心指标散点图 -->
        <div class="section">
            <h2 class="section-title">客群匹配度 vs 日客流量</h2>
            <div class="chart-container">
                <div class="chart-title">横轴: 日均客流量 | 纵轴: 客群匹配度评分 | 气泡大小: 商场评分</div>
                <div class="chart-wrapper">
                    <canvas id="scatterChart"></canvas>
                </div>
                <p class="scatter-note">💡 右上角的商场是最佳选择：客流大且客群匹配度高</p>
            </div>
        </div>
        
        <!-- 商场评分对比 -->
        <div class="section">
            <h2 class="section-title">商场评分对比</h2>
            <div class="chart-container">
                <div class="chart-wrapper">
                    <canvas id="barChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- 推荐排名 -->
        <div class="section">
            <h2 class="section-title">推荐商场 TOP 5</h2>
            <div class="ranking-list">
                {self._generate_ranking_html(report_data)}
            </div>
        </div>
        
        <!-- 对比表格 -->
        <div class="section">
            <h2 class="section-title">商场对比一览</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>商场名称</th>
                        <th>区域</th>
                        <th>匹配度</th>
                        <th>评分</th>
                        <th>日客流</th>
                        <th>竞争情况</th>
                    </tr>
                </thead>
                <tbody>
                    {self._generate_table_rows(report_data)}
                </tbody>
            </table>
        </div>
        
        <!-- 选址建议 -->
        <div class="section">
            <h2 class="section-title">选址建议</h2>
            {self._generate_recommendation_html(report_data, brand, profile)}
        </div>
        
        <!-- 页脚 -->
        <div class="footer">
            报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            数据来源: MARS智能选址系统 v1.0 | 
            © MARS
        </div>
    </div>
    
    <script>
        // 散点图数据
        const scatterData = {self._generate_scatter_data(report_data)};
        
        // 柱状图数据
        const barLabels = {self._generate_bar_labels(report_data)};
        const barScores = {self._generate_bar_scores(report_data)};
        const barMatches = {self._generate_bar_matches(report_data)};
        
        // 散点图
        new Chart(document.getElementById('scatterChart'), {{
            type: 'bubble',
            data: {{
                datasets: [{{
                    label: '商场',
                    data: scatterData,
                    backgroundColor: scatterData.map((d, i) => {{
                        const colors = ['rgba(255, 99, 132, 0.7)', 'rgba(54, 162, 235, 0.7)', 
                                       'rgba(255, 206, 86, 0.7)', 'rgba(75, 192, 192, 0.7)', 
                                       'rgba(153, 102, 255, 0.7)'];
                        return colors[i % colors.length];
                    }}),
                    borderColor: scatterData.map((d, i) => {{
                        const colors = ['rgb(255, 99, 132)', 'rgb(54, 162, 235)', 
                                       'rgb(255, 206, 86)', 'rgb(75, 192, 192)', 
                                       'rgb(153, 102, 255)'];
                        return colors[i % colors.length];
                    }}),
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                const d = context.raw;
                                return [
                                    d.label,
                                    '客流: ' + d.x.toLocaleString() + '/天',
                                    '匹配度: ' + d.y + '分',
                                    '评分: ' + d.r + '分'
                                ];
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{ display: true, text: '日均客流量（人/天）' }},
                        min: 0
                    }},
                    y: {{
                        title: {{ display: true, text: '客群匹配度（分）' }},
                        min: 0,
                        max: 100
                    }}
                }}
            }}
        }});
        
        // 柱状图
        new Chart(document.getElementById('barChart'), {{
            type: 'bar',
            data: {{
                labels: barLabels,
                datasets: [
                    {{
                        label: '商场评分',
                        data: barScores,
                        backgroundColor: 'rgba(102, 126, 234, 0.8)',
                        borderColor: 'rgb(102, 126, 234)',
                        borderWidth: 1,
                        borderRadius: 5,
                        barPercentage: 0.6
                    }},
                    {{
                        label: '客群匹配度',
                        data: barMatches,
                        backgroundColor: 'rgba(118, 75, 162, 0.8)',
                        borderColor: 'rgb(118, 75, 162)',
                        borderWidth: 1,
                        borderRadius: 5,
                        barPercentage: 0.6
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'top' }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>'''
        
        return html
    
    def _prepare_data(self, malls_data: List[Dict], competition_data: List[Dict], 
                      match_scores: Dict) -> List[Dict]:
        """整理报告数据"""
        mall_dict = {}
        
        for m in malls_data:
            name = m.get("mall_name", "")
            if name and name not in mall_dict:
                mall_dict[name] = {
                    "name": name,
                    "mall_id": m.get("mall_id", ""),
                    "district": m.get("district", ""),
                    "traffic": m.get("traffic_daily", 0) or 0,
                    "score": m.get("score_market", 0) or 0,
                    "match_score": m.get("match_score", 0) or 0,
                    "match_level": m.get("match_level", ""),
                    "age_match": m.get("age_match", ""),
                    "consume_match": m.get("consume_match", ""),
                }
        
        # 补充匹配度
        for mall_id, score_data in match_scores.items():
            for name, data in mall_dict.items():
                if data.get("mall_id") == mall_id:
                    data["match_score"] = score_data.get("match_score", 0) or data["match_score"]
                    data["match_level"] = score_data.get("match_level", "") or data["match_level"]
        
        # 补充竞品
        for c in competition_data:
            comp_name = c.get("mall", "")
            for name in mall_dict:
                if comp_name in name or name in comp_name:
                    mall_dict[name]["competition"] = c.get("competition_level", "")
                    mall_dict[name]["competitors"] = c.get("competitors", [])
                    mall_dict[name]["target_exists"] = c.get("target_brand_exists")
        
        # 排序
        sorted_data = sorted(mall_dict.values(), 
                            key=lambda x: (x.get("match_score", 0), x.get("score", 0), x.get("traffic", 0)),
                            reverse=True)
        
        return sorted_data[:5]
    
    def _generate_ranking_html(self, data: List[Dict]) -> str:
        """生成排名列表HTML"""
        html_parts = []
        for i, m in enumerate(data, 1):
            rank_class = f"top{i}" if i <= 3 else ""
            badge_class = f"rank-{i}" if i <= 3 else "rank-other"
            
            traffic = int(m.get('traffic', 0))
            traffic_str = f"{traffic:,}" if traffic else "-"
            
            score = m.get('score', 0)
            match_score = m.get('match_score', 0)
            
            competition = m.get('competition', '')
            comp_class = 'competition-blue'
            if '激烈' in competition or '🔴' in competition:
                comp_class = 'competition-red'
            elif '中等' in competition or '🟠' in competition:
                comp_class = 'competition-orange'
            elif '🟡' in competition:
                comp_class = 'competition-yellow'
            elif '🟢' in competition or '较少' in competition:
                comp_class = 'competition-green'
            
            html_parts.append(f'''
                <div class="ranking-item {rank_class}">
                    <div class="rank-badge {badge_class}">{i}</div>
                    <div class="mall-info">
                        <div class="mall-name">{m.get('name', '')}</div>
                        <div class="mall-district">📍 {m.get('district', '上海')}</div>
                    </div>
                    <div class="mall-metrics">
                        <div class="metric">
                            <div class="metric-value">{match_score}</div>
                            <div class="metric-label">匹配度</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{score}</div>
                            <div class="metric-label">评分</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{traffic_str}</div>
                            <div class="metric-label">日客流</div>
                        </div>
                        {f'<span class="competition-badge {comp_class}">{competition[:6] if competition else ""}</span>' if competition else ''}
                    </div>
                </div>
            ''')
        
        return ''.join(html_parts)
    
    def _generate_table_rows(self, data: List[Dict]) -> str:
        """生成对比表格行"""
        rows = []
        for i, m in enumerate(data, 1):
            traffic = int(m.get('traffic', 0))
            traffic_str = f"{traffic:,}" if traffic else "-"
            competition = m.get('competition', '-')[:8] if m.get('competition') else '-'
            
            rows.append(f'''
                <tr>
                    <td><strong>{i}</strong></td>
                    <td>{m.get('name', '')[:15]}</td>
                    <td>{m.get('district', '-')[:5]}</td>
                    <td><strong>{m.get('match_score', '-')}</strong></td>
                    <td>{m.get('score', '-')}</td>
                    <td>{traffic_str}</td>
                    <td>{competition}</td>
                </tr>
            ''')
        
        return ''.join(rows)
    
    def _generate_recommendation_html(self, data: List[Dict], brand: str, profile: Dict) -> str:
        """生成推荐建议HTML"""
        if not data:
            return '<p>暂无推荐数据</p>'
        
        top = data[0]
        reasons = []
        
        if top.get('match_score', 0) >= 50:
            reasons.append(f"客群匹配度达到{top['match_score']}分，目标客户浓度高")
        if top.get('score', 0) >= 85:
            reasons.append(f"商场评分{top['score']}分，品牌形象有保障")
        if top.get('traffic', 0) >= 20000:
            reasons.append(f"日均客流{int(top['traffic']):,}人，曝光量充足")
        if top.get('target_exists') is False:
            reasons.append(f"{brand}尚未入驻，存在市场空白机会")
        
        if not reasons:
            reasons = ["综合指标表现优秀", "适合品牌定位"]
        
        reasons_html = ''.join([f'<li>{r}</li>' for r in reasons])
        
        html = f'''
            <div class="recommendation-box">
                <h3>🏆 首选推荐：{top.get('name', '')}</h3>
                <ul>
                    {reasons_html}
                </ul>
            </div>
        '''
        
        if len(data) >= 2:
            alt = data[1]
            html += f'''
                <div style="margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 12px;">
                    <strong>🥈 备选推荐：{alt.get('name', '')}</strong>
                    <p style="color: #666; margin-top: 5px; font-size: 14px;">
                        匹配度{alt.get('match_score', '-')}分 | 评分{alt.get('score', '-')} | 日客流{int(alt.get('traffic', 0)):,}
                    </p>
                </div>
            '''
        
        return html
    
    def _generate_scatter_data(self, data: List[Dict]) -> str:
        """生成散点图数据"""
        points = []
        for m in data:
            traffic = m.get('traffic', 0) or 0
            match_score = m.get('match_score', 0) or 50
            score = m.get('score', 0) or 80
            # r是气泡大小，用评分除以5使气泡大小适中
            points.append({
                'x': traffic,
                'y': match_score,
                'r': score / 5,
                'label': m.get('name', '')[:10]
            })
        
        return json.dumps(points, ensure_ascii=False)
    
    def _generate_bar_labels(self, data: List[Dict]) -> str:
        """生成柱状图标签"""
        labels = [m.get('name', '')[:8] for m in data]
        return json.dumps(labels, ensure_ascii=False)
    
    def _generate_bar_scores(self, data: List[Dict]) -> str:
        """生成柱状图评分数据"""
        scores = [m.get('score', 0) or 0 for m in data]
        return json.dumps(scores)
    
    def _generate_bar_matches(self, data: List[Dict]) -> str:
        """生成柱状图匹配度数据"""
        matches = [m.get('match_score', 0) or 0 for m in data]
        return json.dumps(matches)
    


# ============================================================
# Router Agent
# ============================================================

class RouterAgent:
    def __init__(self):
        self.name = "Router"
    
    def run(self, input_data: Dict) -> Dict:
        query = input_data.get("query", "")
        intent = self._classify_intent(query)
        
        return {
            "query": query,
            "intent": intent["type"],
            "confidence": intent["confidence"],
            "route_to": intent["route_to"]
        }
    
    def _classify_intent(self, query: str) -> Dict:
        if any(kw in query for kw in ["对比", "比较", "vs"]):
            return {"type": "comparison", "confidence": 0.9, "route_to": "react_agent"}
        
        if any(kw in query for kw in ["选址", "开店", "入驻"]) or len(query) > 20:
            return {"type": "site_selection", "confidence": 0.8, "route_to": "react_agent"}
        
        if any(kw in query for kw in ["客流", "人流"]):
            return {"type": "traffic", "confidence": 0.85, "route_to": "react_agent"}
        
        return {"type": "simple_query", "confidence": 0.7, "route_to": "rag_agent"}


# ============================================================
# 主 Agent
# ============================================================

def _trunc(s, n: int = 80) -> str:
    """截断字符串，超出加省略号"""
    s = str(s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def _rx_first(pattern: str, text) -> str:
    """在文本里取第一个正则捕获组，取不到返回空串"""
    if not text:
        return ""
    m = re.search(pattern, str(text))
    return m.group(1).strip() if m else ""


def _safe_json_load(s):
    """安全解析 JSON 字符串为 dict，失败返回 {}"""
    if not s or (isinstance(s, float) and pd.isna(s)) or not str(s).strip():
        return {}
    try:
        d = json.loads(s)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


class MARSAgent:
    """MARS v1.0 主Agent"""
    
    def __init__(self, vectorstore, df_malls: pd.DataFrame, user_id: str = "default"):
        self.vectorstore = vectorstore
        self.df_malls = df_malls
        
        # 创建竞品搜索Agent
        self.competitor_agent = CompetitorSearchAgent()
        
        # 创建工具注册中心
        self.tool_registry = ToolRegistry(df_malls, self.competitor_agent, vectorstore)
        
        # 创建其他Agent
        self.router = RouterAgent()
        self.react_agent = None
        self.llm_client = None
        
        print(f"[MARS v1.0] 初始化完成")
        print(f"  - 商场数据: {len(df_malls)}")
        print(f"  - 可用工具: {list(self.tool_registry.tools.keys())}")
    
    def init_llm(self) -> str:
        try:
            from openai import OpenAI
            self.llm_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
            self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            
            self.react_agent = ReActAgent(self.tool_registry, self.llm_client)
            self.competitor_agent.llm_client = self.llm_client
            
            return "✅ LLM 初始化成功"
        except Exception as e:
            self.react_agent = ReActAgent(self.tool_registry, None)
            return f"⚠️ LLM 初始化失败 ({e})，使用模拟模式"
    
    def _build_evidence_block(self, malls: List[Dict]) -> str:
        """根据收集到的商场，生成「推荐依据（数据溯源）」段落。

        每条依据都从 df_malls 宽表真实字段取值（评分/客流/品牌/客群/区位），
        保证可逐条核对、非模型杜撰，用于降低幻觉、提升用户信任度。
        """
        if not malls:
            return ""

        # 按 mall_id 去重，保留出现顺序，最多 5 家
        seen, uniq = set(), []
        for m in malls:
            mid = str(m.get("mall_id", "") or "")
            if mid:
                if mid in seen:
                    continue
                seen.add(mid)
            uniq.append(m)
            if len(uniq) >= 5:
                break
        if not uniq:
            return ""

        lines = ["\n### 📎 推荐依据（数据溯源）\n",
                 "> 以下每条依据均取自商场宽表真实字段，可逐条核对，非模型杜撰。\n"]

        for i, m in enumerate(uniq, 1):
            mid = str(m.get("mall_id", "") or "")
            row = self.df_malls[self.df_malls["mall_id"].astype(str) == mid]
            r = row.iloc[0] if len(row) else None

            name = m.get("mall_name") or (r["mall_name"] if r is not None else "未知")
            district = m.get("district") or (str(r.get("district", "")) if r is not None else "")
            score = m.get("score_market") or (r["score_market"] if r is not None else 0)
            traffic = m.get("traffic_daily") or (r["traffic_daily"] if r is not None else 0)

            head = f"**{i}. {name}**"
            if district:
                head += f"（{district}）"
            head += f" · 评分 {score}"
            head += f" · 客流 {int(traffic):,}/天" if traffic else " · 客流无"
            lines.append(head)

            bullets = []
            ms = m.get("match_score")
            if ms is not None:
                ml = m.get("match_level", "")
                bullets.append(f"- 匹配度：{ml}（TGI {ms} 分）" if ml else f"- 匹配度：{ms} 分")

            # 消费能力 / 客单价（position_text）
            pos = str(r.get("position_text", "") or "") if r is not None else ""
            consume = _rx_first(r"消费能力[：:]([^。；]+)", pos)
            price = _rx_first(r"客单价([0-9]+[-~][0-9]+元)", pos)
            if consume:
                bullets.append(f"- 消费能力：{consume}")
            elif price:
                bullets.append(f"- 客单价：{price}")

            # 客群（profile_text）
            pt = str(r.get("profile_text", "") or "") if r is not None else ""
            age = _rx_first(r"age[：:]([^；;]+)", pt)
            cons = _rx_first(r"consume[：:]([^；;]+)", pt)
            if age or cons:
                seg = "、".join(x for x in [age, cons] if x)
                bullets.append(f"- 客群：{seg}")

            # 代表品牌（brand_data_json.top_brands）
            if r is not None:
                bd = _safe_json_load(r.get("brand_data_json"))
                brands = bd.get("top_brands", []) or []
                if brands:
                    bullets.append(f"- 代表品牌：{_trunc('、'.join(str(b) for b in brands[:6]), 60)}")

            # 品类结构（门店数前 2 的品类）
            if r is not None:
                cat_counts = []
                for cat in ["餐饮", "服装", "娱乐服务", "运动", "珠宝", "护肤化妆品"]:
                    cnt = r.get(f"{cat}_brand_count", 0) or 0
                    if cnt:
                        cat_counts.append((cat, int(cnt)))
                if cat_counts:
                    cat_counts.sort(key=lambda x: -x[1])
                    top2 = "、".join(f"{c}({n})" for c, n in cat_counts[:2])
                    bullets.append(f"- 品类结构：{top2}")

            if not bullets:
                bullets.append("- （仅基础评分/客流，无额外字段）")

            lines.extend(bullets)
            lines.append("")

        lines.append("> 数据来源：`score_market` 评分 / `traffic_daily` 客流 / `brand_data_json` 品牌 / `customer_profile_json` 客群画像 / `position_text` 区位。\n")
        return "\n".join(lines)

    def recommend(self, user_input: str) -> Tuple[str, str]:
        """
        执行选址推荐
        返回: (markdown输出, html报告路径)
        """
        if not user_input.strip():
            return "❌ 请输入需求", None
        
        output = []
        
        # 路由
        output.append("### 🚦 Step 1: 意图路由\n\n")
        route_result = self.router.run({"query": user_input})
        output.append(f"- 意图类型: **{route_result['intent']}**\n")
        output.append(f"- 路由到: `{route_result['route_to']}`\n\n")
        
        # ReAct 推理
        if self.react_agent is None:
            self.react_agent = ReActAgent(self.tool_registry, self.llm_client)
        
        output.append("### 🤖 Step 2: ReAct 推理\n\n")
        
        final_answer, trajectory = self.react_agent.run(user_input)
        
        # 显示推理过程
        output.append(self.react_agent.format_trajectory())
        
        # 显示最终答案
        if final_answer:
            output.append("\n### 📋 最终推荐\n\n")
            output.append(final_answer)

        # 推荐依据（数据溯源）
        evidence = self._build_evidence_block(self.react_agent.collected_data.get("malls", []))
        if evidence:
            output.append(evidence)

        # 生成HTML报告
        html_path = self._generate_html_report(user_input)

        return "".join(output), html_path
    
    def _generate_html_report(self, query: str) -> str:
        """生成HTML报告并保存"""
        try:
            # 提取品牌和品类
            brand = "品牌"
            category = "运动"
            for b in ["耐克", "Nike", "始祖鸟", "Adidas", "李宁", "安踏", "Lululemon", "Olay"]:
                if b in query:
                    brand = b
                    break
            
            if "服装" in query:
                category = "服装"
            elif "餐饮" in query:
                category = "餐饮"
            elif "护肤" in query or "美妆" in query:
                category = "美妆"
            
            # 获取客群配置
            brand_profiles = {
                "Nike": {"age": "18-35", "consume": "中高", "keywords": "年轻 运动 时尚"},
                "耐克": {"age": "18-35", "consume": "中高", "keywords": "年轻 运动 时尚"},
                "始祖鸟": {"age": "25-45", "consume": "高", "keywords": "高端 户外 白领"},
                "Lululemon": {"age": "25-40", "consume": "高", "keywords": "女性 瑜伽 白领"},
                "李宁": {"age": "18-35", "consume": "中", "keywords": "年轻 国潮 学生"},
                "安踏": {"age": "18-40", "consume": "中", "keywords": "大众 运动 家庭"},
            }
            profile = brand_profiles.get(brand, {"age": "25-35", "consume": "中高", "keywords": "白领"})
            
            # 获取收集的数据
            malls_data = self.react_agent.collected_data.get("malls", [])
            competition_data = self.react_agent.collected_data.get("competition", [])
            match_scores = self.react_agent.collected_data.get("match_scores", {})
            
            # 生成HTML
            html_content = self.react_agent.report_generator.generate_report(
                brand=brand,
                category=category,
                profile=profile,
                malls_data=malls_data,
                competition_data=competition_data,
                match_scores=match_scores
            )
            
            # 保存到文件
            report_dir = Path("./reports")
            report_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"选址报告_{brand}_{timestamp}.html"
            filepath = report_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(filepath)
            
        except Exception as e:
            print(f"生成HTML报告失败: {e}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================
# Gradio 界面
# ============================================================

print("=" * 60)
print("MARS v1.0: 智能选址分析系统（含可视化报告）")
print("=" * 60)

print("\n加载数据...")

# 先读取商场数据（核心，必须成功）
df_malls = pd.read_csv(DATA_FILE, dtype={'mall_id': str})
print(f"✅ 商场数据加载完成: {len(df_malls)} 个商场")


def _load_vectorstore():
    """尝试加载嵌入模型和 Chroma 向量库，失败时返回 None。

    向量库仅用于 RAG 检索，主流程(ReAct 工具)只依赖 df_malls；
    未安装 sentence-transformers / chromadb 时优雅降级，不影响选址推荐。
    """
    embeddings = get_embeddings()
    if embeddings is None:
        print("⚠️ 嵌入模型加载失败（缺少 sentence-transformers 等依赖），跳过向量库")
        return None
    return load_vectorstore(CHROMA_DIR, embeddings)


vectorstore = _load_vectorstore()
if vectorstore is not None:
    print("✅ 向量库加载完成")
else:
    print("⚠️ 向量库未加载（缺少依赖或模型未下载），选址推荐功能不受影响")

# 创建 Agent
agent = MARSAgent(vectorstore, df_malls)

# Gradio 示例
EXAMPLES = [
    ["耐克想在上海找中高端商场开店"],
    ["始祖鸟想找运动品牌竞争少的高端商场"],
    ["李宁想进入客流量大的商场"],
    ["帮我分析徐汇区适合运动品牌的商场"],
]

with gr.Blocks(title="🧠 MARS v1.0") as demo:
    
    gr.Markdown("""
    # 🧠 MARS v1.0 - 智能选址分析系统
    
    **核心功能**: 基于TGI客群匹配度的智能选址推荐，生成专业分析报告
    
    **新增特性**: 📊 可视化HTML报告（含散点图、柱状图、对比表格）
    """)
    
    with gr.Tab("🤖 智能选址"):
        with gr.Row():
            with gr.Column(scale=3):
                user_input = gr.Textbox(
                    label="描述您的选址需求",
                    lines=3,
                    placeholder="例如：耐克想在上海找中高端商场开店"
                )
                with gr.Row():
                    search_btn = gr.Button("🚀 智能推荐", variant="primary", size="lg")
                    init_btn = gr.Button("🔧 初始化 LLM")
                status = gr.Textbox(label="状态", value="请先初始化 LLM（可选）", interactive=False)
        
        gr.Examples(examples=EXAMPLES, inputs=user_input)
        
        result_output = gr.Markdown(label="推荐结果")
        
        with gr.Row():
            report_file = gr.File(label="📊 下载HTML报告", visible=True)
            report_html = gr.HTML(label="报告预览", visible=False)
        
        def do_search(user_input):
            try:
                markdown_output, html_path = agent.recommend(user_input)
                return markdown_output, html_path
            except Exception as e:
                return f"❌ 错误: {str(e)}\n\n```\n{traceback.format_exc()}\n```", None
        
        def do_init():
            return agent.init_llm()
        
        search_btn.click(do_search, inputs=[user_input], outputs=[result_output, report_file])
        init_btn.click(do_init, outputs=[status])
    
    with gr.Tab("🔧 工具测试"):
        gr.Markdown("### 直接测试工具调用\n")
        
        tool_select = gr.Dropdown(
            choices=list(agent.tool_registry.tools.keys()),
            label="选择工具",
            value="search_mall"
        )
        tool_params = gr.Textbox(
            label="参数 (JSON格式)",
            value='{"query": "高端", "min_score": 85, "top_k": 5}'
        )
        tool_btn = gr.Button("执行工具")
        tool_output = gr.Markdown()
        
        def test_tool(tool_name, params_str):
            try:
                params = json.loads(params_str)
                result = agent.tool_registry.execute(tool_name, **params)
                if result.success:
                    return f"✅ 成功\n\n```json\n{json.dumps(result.data, ensure_ascii=False, indent=2)}\n```"
                else:
                    return f"❌ 失败: {result.message}"
            except Exception as e:
                return f"❌ 错误: {str(e)}"
        
        tool_btn.click(test_tool, inputs=[tool_select, tool_params], outputs=[tool_output])
        
        # 竞品搜索测试
        gr.Markdown("### 竞品搜索测试\n")
        
        with gr.Row():
            brand_input = gr.Textbox(label="目标品牌", value="Nike")
            category_input = gr.Dropdown(choices=["运动", "服装", "餐饮"], label="品类", value="运动")
        
        malls_input = gr.Textbox(
            label="商场列表 (逗号分隔)",
            value="上海港汇恒隆广场, 上海前滩太古里, 上海环球港"
        )
        comp_btn = gr.Button("搜索竞品")
        comp_output = gr.Markdown()
        
        def test_competition(brand, category, malls_str):
            try:
                mall_names = [m.strip() for m in malls_str.split(",")]
                result = agent.tool_registry.execute(
                    "analyze_brand_competition",
                    target_brand=brand,
                    category=category,
                    mall_names=mall_names
                )
                if result.success:
                    return f"✅ 成功\n\n```json\n{json.dumps(result.data, ensure_ascii=False, indent=2)}\n```"
                else:
                    return f"❌ 失败: {result.message}"
            except Exception as e:
                return f"❌ 错误: {str(e)}"
        
        comp_btn.click(test_competition, inputs=[brand_input, category_input, malls_input], outputs=[comp_output])
    
    with gr.Tab("📊 系统信息"):
        gr.Markdown(f"""
        ### 数据统计
        - 商场总数: **{len(df_malls)}**
        - 有客流数据: **{df_malls['traffic_daily'].notna().sum()}**
        
        ### 检索与推理
        - 三级混合检索：BM25 + 稠密 RRF + 知识图谱 2-hop（逐级回退）
        - 多智能体协作：Router → ReAct → CompetitorSearchAgent
        - 竞品分析为实验性 LLM 辅助信号（标注置信度与来源，非事实数据源）
        
        ### 可用工具（9 类）
        - `search_mall`: 根据关键词和条件搜索商场，返回基础信息（评分、客流）
        - `get_traffic_ranking`: 获取商场客流量排名
        - `match_customer_profile`: 根据目标客群特征匹配商场
        - `get_mall_detail`: 获取单个商场的详细信息
        - `compare_malls`: 对比多个商场的基础指标
        - `analyze_brand_competition`: 品牌竞争分析（实验性 LLM 辅助信号）
        - `analyze_customer_profile`: 客群画像分析（年龄分布 / 消费水平 / TGI）
        - `score_customer_match`: 客群匹配度评分（基于 TGI 指数）
        - `vector_search_malls`: 语义检索（自然语言 → 相似商场）
        
        ### 优雅降级
        - LLM Key 缺失时进入模拟模式，不影响界面演示
        - 向量库未加载时，检索降级为提示、选址推荐不受影响
        """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 启动 MARS v1.0")
    print("=" * 60)
    print("访问: http://127.0.0.1:7860")
    print("=" * 60 + "\n")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False, theme=gr.themes.Soft())
