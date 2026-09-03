"""
MARS V7 GraphRAG 模块
基于商场-品牌-区域数据构建知识图谱

面试考点:
1. 为什么要用GraphRAG？解决传统RAG的什么问题？
2. 知识图谱怎么构建？三元组怎么设计？
3. 图谱查询和向量检索怎么结合？
"""

import pandas as pd
import numpy as np
import json
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import networkx as nx

from rag_utils import hybrid_vector_search


@dataclass
class Triple:
    """知识图谱三元组"""
    head: str           # 头实体
    relation: str       # 关系
    tail: str           # 尾实体
    head_type: str      # 头实体类型
    tail_type: str      # 尾实体类型
    properties: Dict = None  # 附加属性


class MallKnowledgeGraph:
    """
    商场知识图谱
    
    实体类型:
    - Mall: 商场
    - Brand: 品牌
    - District: 区域
    - Category: 品类
    - CustomerProfile: 客群画像
    
    关系类型:
    - LOCATED_IN: 商场 → 区域
    - HAS_BRAND: 商场 → 品牌
    - BELONGS_TO: 品牌 → 品类
    - COMPETES_WITH: 品牌 → 品牌
    - TARGETS: 商场 → 客群画像
    - SIMILAR_TO: 商场 → 商场
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_index = defaultdict(set)  # type -> entity_ids
        self.triples: List[Triple] = []
        
    def build_from_dataframe(self, df_malls: pd.DataFrame, df_brands: pd.DataFrame = None):
        """
        从数据构建知识图谱
        
        Args:
            df_malls: 商场宽表
            df_brands: 品牌入驻数据（可选）
        """
        print("[GraphRAG] 开始构建知识图谱...")
        
        # 1. 添加商场实体和基本关系
        self._add_mall_entities(df_malls)
        
        # 2. 添加区域实体和关系
        self._add_district_relations(df_malls)
        
        # 3. 添加品类竞争关系
        self._add_category_relations(df_malls)
        
        # 4. 添加客群画像关系
        self._add_profile_relations(df_malls)
        
        # 5. 计算商场相似度关系
        self._add_similarity_relations(df_malls)
        
        # 6. 如果有品牌数据，添加品牌关系
        if df_brands is not None:
            self._add_brand_relations(df_brands)
        
        print(f"[GraphRAG] 构建完成:")
        print(f"  - 节点数: {self.graph.number_of_nodes()}")
        print(f"  - 边数: {self.graph.number_of_edges()}")
        print(f"  - 三元组数: {len(self.triples)}")
        
    def _add_mall_entities(self, df: pd.DataFrame):
        """添加商场实体"""
        for _, row in df.iterrows():
            mall_id = f"mall_{row['mall_id']}"
            
            # 添加节点（NaN 是 truthy，`or 0` 挡不住 traffic_daily 的 353 个 NaN，需显式判空）
            _sm = row.get('score_market', 0)
            _td = row.get('traffic_daily', 0)
            _bc = row.get('brand_count', 0)
            self.graph.add_node(
                mall_id,
                type="Mall",
                name=row['mall_name'],
                score_market=0.0 if pd.isna(_sm) else float(_sm),
                traffic_daily=0.0 if pd.isna(_td) else float(_td),
                brand_count=0.0 if pd.isna(_bc) else float(_bc),
            )
            
            self.entity_index["Mall"].add(mall_id)
    
    def _add_district_relations(self, df: pd.DataFrame):
        """添加区域关系"""
        districts = df['district'].dropna().unique()
        
        for district in districts:
            district_id = f"district_{district}"
            
            # 添加区域节点
            if district_id not in self.graph:
                self.graph.add_node(district_id, type="District", name=district)
                self.entity_index["District"].add(district_id)
            
            # 添加商场-区域关系
            malls_in_district = df[df['district'] == district]
            for _, row in malls_in_district.iterrows():
                mall_id = f"mall_{row['mall_id']}"
                
                self.graph.add_edge(mall_id, district_id, relation="LOCATED_IN")
                
                self.triples.append(Triple(
                    head=row['mall_name'],
                    relation="位于",
                    tail=district,
                    head_type="Mall",
                    tail_type="District"
                ))
    
    def _add_category_relations(self, df: pd.DataFrame):
        """添加品类关系"""
        categories = ["服装", "餐饮", "运动", "珠宝", "护肤化妆品", "娱乐服务"]
        
        for category in categories:
            cat_id = f"category_{category}"
            density_col = f"{category}_density"
            count_col = f"{category}_brand_count"
            
            if density_col not in df.columns:
                continue
            
            # 添加品类节点
            self.graph.add_node(cat_id, type="Category", name=category)
            self.entity_index["Category"].add(cat_id)
            
            # 添加商场-品类关系（带权重）
            for _, row in df.iterrows():
                mall_id = f"mall_{row['mall_id']}"
                density = row.get(density_col, 0) or 0
                count = row.get(count_col, 0) or 0
                
                if count > 0:
                    # 判断竞争级别
                    if density < 0.08:
                        level = "蓝海"
                    elif density < 0.15:
                        level = "中等"
                    elif density < 0.25:
                        level = "激烈"
                    else:
                        level = "红海"
                    
                    self.graph.add_edge(
                        mall_id, cat_id,
                        relation="HAS_CATEGORY",
                        density=density,
                        count=count,
                        level=level
                    )
                    
                    self.triples.append(Triple(
                        head=row['mall_name'],
                        relation=f"{category}竞争{level}",
                        tail=category,
                        head_type="Mall",
                        tail_type="Category",
                        properties={"density": density, "count": count}
                    ))
    
    def _add_profile_relations(self, df: pd.DataFrame):
        """添加客群画像关系"""
        # 提取主要客群特征
        profile_features = ["高消费", "年轻白领", "家庭客群", "学生群体", "商务人群"]
        
        for feature in profile_features:
            feature_id = f"profile_{feature}"
            self.graph.add_node(feature_id, type="CustomerProfile", name=feature)
            self.entity_index["CustomerProfile"].add(feature_id)
        
        # 根据 profile_text 匹配
        for _, row in df.iterrows():
            mall_id = f"mall_{row['mall_id']}"
            profile_text = str(row.get('profile_text', ''))
            
            # 简单关键词匹配
            if "高" in profile_text and "消费" in profile_text:
                self.graph.add_edge(mall_id, "profile_高消费", relation="TARGETS")
            if any(kw in profile_text for kw in ["25-30", "25-35", "白领"]):
                self.graph.add_edge(mall_id, "profile_年轻白领", relation="TARGETS")
            if any(kw in profile_text for kw in ["儿童", "亲子", "家庭"]):
                self.graph.add_edge(mall_id, "profile_家庭客群", relation="TARGETS")
    
    def _add_similarity_relations(self, df: pd.DataFrame, top_k: int = 5):
        """计算并添加商场相似度关系"""
        # 基于多维特征计算相似度
        features = ['score_market', 'score_pop', 'score_consume', 'traffic_daily']
        
        # 计算两两相似度（简化：只取同区域的）
        for district in df['district'].dropna().unique():
            district_malls = df[df['district'] == district]
            
            if len(district_malls) < 2:
                continue
            
            for i, row1 in district_malls.iterrows():
                mall1_id = f"mall_{row1['mall_id']}"
                
                similarities = []
                for j, row2 in district_malls.iterrows():
                    if i >= j:
                        continue
                    
                    # 计算欧氏距离的倒数作为相似度
                    dist = 0
                    for f in features:
                        if f in df.columns:
                            v1 = row1.get(f, 0)
                            v2 = row2.get(f, 0)
                            # NaN 是 truthy，`or 0` 挡不住（NaN or 0 == NaN），需显式判空
                            v1 = 0.0 if pd.isna(v1) else float(v1)
                            v2 = 0.0 if pd.isna(v2) else float(v2)
                            dist += (v1 - v2) ** 2
                    
                    sim = 1 / (1 + np.sqrt(dist))
                    similarities.append((row2['mall_id'], row2['mall_name'], sim))
                
                # 取 Top-K 相似商场
                similarities.sort(key=lambda x: x[2], reverse=True)
                for mall2_id, mall2_name, sim in similarities[:top_k]:
                    if sim > 0.7:  # 只保留高相似度
                        mall2_node = f"mall_{mall2_id}"
                        self.graph.add_edge(
                            mall1_id, mall2_node,
                            relation="SIMILAR_TO",
                            similarity=sim
                        )
                        
                        self.triples.append(Triple(
                            head=row1['mall_name'],
                            relation="相似于",
                            tail=mall2_name,
                            head_type="Mall",
                            tail_type="Mall",
                            properties={"similarity": round(sim, 3)}
                        ))
    
    def _add_brand_relations(self, df_brands: pd.DataFrame):
        """添加品牌关系"""
        if df_brands is None or df_brands.empty:
            return
        
        # 假设品牌数据有: brand_name, category, mall_id
        for _, row in df_brands.iterrows():
            brand_name = row.get('brand_name', '')
            mall_id = row.get('mall_id', '')
            category = row.get('category_1', '') or row.get('category', '')
            
            if not brand_name or not mall_id:
                continue
            
            brand_id = f"brand_{brand_name}"
            mall_node = f"mall_{mall_id}"
            
            # 添加品牌节点
            if brand_id not in self.graph:
                self.graph.add_node(brand_id, type="Brand", name=brand_name, category=category)
                self.entity_index["Brand"].add(brand_id)
            
            # 添加入驻关系
            if mall_node in self.graph:
                self.graph.add_edge(mall_node, brand_id, relation="HAS_BRAND")
                
                self.triples.append(Triple(
                    head=self.graph.nodes[mall_node].get('name', mall_id),
                    relation="入驻品牌",
                    tail=brand_name,
                    head_type="Mall",
                    tail_type="Brand"
                ))
    
    # ============================================================
    # 图谱查询方法
    # ============================================================
    
    def query_by_relation(self, entity: str, relation: str, direction: str = "out") -> List[Dict]:
        """
        根据关系查询
        
        Args:
            entity: 实体名称或ID
            relation: 关系类型
            direction: "out" 出边, "in" 入边
        """
        results = []
        
        # 找到实体节点
        entity_node = self._find_entity(entity)
        if not entity_node:
            return results
        
        if direction == "out":
            edges = self.graph.out_edges(entity_node, data=True)
            for _, target, data in edges:
                if data.get('relation') == relation:
                    target_data = self.graph.nodes[target]
                    results.append({
                        "entity": target_data.get('name', target),
                        "type": target_data.get('type'),
                        "properties": data
                    })
        else:
            edges = self.graph.in_edges(entity_node, data=True)
            for source, _, data in edges:
                if data.get('relation') == relation:
                    source_data = self.graph.nodes[source]
                    results.append({
                        "entity": source_data.get('name', source),
                        "type": source_data.get('type'),
                        "properties": data
                    })
        
        return results
    
    def find_blue_ocean_by_graph(self, category: str, district: str = None) -> List[Dict]:
        """
        用图谱查询蓝海商场
        
        比传统查询的优势：可以考虑关联关系
        """
        cat_node = f"category_{category}"
        if cat_node not in self.graph:
            return []
        
        results = []
        
        # 找所有有该品类的商场
        for source, target, data in self.graph.in_edges(cat_node, data=True):
            if data.get('relation') != "HAS_CATEGORY":
                continue
            
            level = data.get('level', '')
            if level not in ["蓝海", "中等"]:
                continue
            
            mall_data = self.graph.nodes[source]
            
            # 如果指定区域，检查是否在该区域
            if district:
                in_district = False
                for _, d_target, d_data in self.graph.out_edges(source, data=True):
                    if d_data.get('relation') == "LOCATED_IN":
                        if district in self.graph.nodes[d_target].get('name', ''):
                            in_district = True
                            break
                if not in_district:
                    continue
            
            results.append({
                "mall_id": source.replace("mall_", ""),
                "mall_name": mall_data.get('name'),
                "category": category,
                "density": data.get('density', 0),
                "level": level,
                "score_market": mall_data.get('score_market', 0),
                "traffic_daily": mall_data.get('traffic_daily', 0),
            })
        
        # 按密度排序
        results.sort(key=lambda x: x['density'])
        return results
    
    def find_similar_malls(self, mall_name: str, top_k: int = 5) -> List[Dict]:
        """查找相似商场"""
        mall_node = self._find_entity(mall_name)
        if not mall_node:
            return []
        
        results = []
        for _, target, data in self.graph.out_edges(mall_node, data=True):
            if data.get('relation') == "SIMILAR_TO":
                target_data = self.graph.nodes[target]
                results.append({
                    "mall_name": target_data.get('name'),
                    "similarity": data.get('similarity', 0),
                    "score_market": target_data.get('score_market', 0),
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def get_mall_context(self, mall_name: str) -> Dict:
        """
        获取商场的完整上下文（用于增强RAG）
        
        返回商场的所有关联信息，可以拼接到RAG检索结果中
        """
        mall_node = self._find_entity(mall_name)
        if not mall_node:
            return {}
        
        mall_data = self.graph.nodes[mall_node]
        
        context = {
            "basic_info": {
                "name": mall_data.get('name'),
                "score_market": mall_data.get('score_market'),
                "traffic_daily": mall_data.get('traffic_daily'),
                "brand_count": mall_data.get('brand_count'),
            },
            "location": None,
            "categories": [],
            "customer_profiles": [],
            "similar_malls": [],
        }
        
        # 遍历出边
        for _, target, data in self.graph.out_edges(mall_node, data=True):
            relation = data.get('relation')
            target_data = self.graph.nodes[target]
            
            if relation == "LOCATED_IN":
                context["location"] = target_data.get('name')
            
            elif relation == "HAS_CATEGORY":
                context["categories"].append({
                    "name": target_data.get('name'),
                    "density": data.get('density'),
                    "level": data.get('level'),
                })
            
            elif relation == "TARGETS":
                context["customer_profiles"].append(target_data.get('name'))
            
            elif relation == "SIMILAR_TO":
                context["similar_malls"].append({
                    "name": target_data.get('name'),
                    "similarity": data.get('similarity'),
                })
        
        return context
    
    def graph_enhanced_search(self, query_entities: List[str], hop: int = 2) -> Set[str]:
        """
        图增强搜索：从查询实体出发，扩展相关实体
        
        这是 GraphRAG 的核心：通过图谱扩展召回
        
        Args:
            query_entities: 查询中提取的实体
            hop: 跳数
        
        Returns:
            扩展后的相关商场ID集合
        """
        related_malls = set()
        
        for entity in query_entities:
            node = self._find_entity(entity)
            if not node:
                continue
            
            # BFS 扩展
            visited = {node}
            current_level = {node}
            
            for _ in range(hop):
                next_level = set()
                for n in current_level:
                    # 出边
                    for _, target, _ in self.graph.out_edges(n, data=True):
                        if target not in visited:
                            visited.add(target)
                            next_level.add(target)
                            
                            # 如果是商场节点，加入结果
                            if self.graph.nodes[target].get('type') == "Mall":
                                related_malls.add(target.replace("mall_", ""))
                    
                    # 入边
                    for source, _, _ in self.graph.in_edges(n, data=True):
                        if source not in visited:
                            visited.add(source)
                            next_level.add(source)
                            
                            if self.graph.nodes[source].get('type') == "Mall":
                                related_malls.add(source.replace("mall_", ""))
                
                current_level = next_level
        
        return related_malls
    
    def _find_entity(self, name: str) -> Optional[str]:
        """查找实体节点"""
        # 直接匹配节点ID
        if name in self.graph:
            return name
        
        # 按名称搜索
        for node, data in self.graph.nodes(data=True):
            if data.get('name') == name:
                return node
        
        # 模糊匹配
        for node, data in self.graph.nodes(data=True):
            if name in str(data.get('name', '')):
                return node
        
        return None
    
    # ============================================================
    # 导出和可视化
    # ============================================================
    
    def export_triples(self, filepath: str):
        """导出三元组"""
        data = []
        for t in self.triples:
            data.append({
                "head": t.head,
                "relation": t.relation,
                "tail": t.tail,
                "head_type": t.head_type,
                "tail_type": t.tail_type,
                "properties": t.properties
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"[GraphRAG] 导出 {len(data)} 个三元组到 {filepath}")
    
    def get_statistics(self) -> Dict:
        """获取图谱统计信息"""
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "total_triples": len(self.triples),
            "entity_types": {},
            "relation_types": defaultdict(int),
        }
        
        # 统计实体类型
        for entity_type, entities in self.entity_index.items():
            stats["entity_types"][entity_type] = len(entities)
        
        # 统计关系类型
        for _, _, data in self.graph.edges(data=True):
            rel = data.get('relation', 'UNKNOWN')
            stats["relation_types"][rel] += 1
        
        stats["relation_types"] = dict(stats["relation_types"])
        
        return stats


# ============================================================
# GraphRAG 检索器
# ============================================================

class GraphRAGRetriever:
    """
    GraphRAG 检索器
    
    结合向量检索和图谱检索的混合检索
    
    面试考点：向量检索和图谱检索各自的优缺点？怎么结合？
    """
    
    def __init__(self, vectorstore, knowledge_graph: MallKnowledgeGraph, df_malls: pd.DataFrame):
        self.vectorstore = vectorstore
        self.kg = knowledge_graph
        self.df_malls = df_malls
    
    def hybrid_search(self, query: str, top_k: int = 10, 
                     vector_weight: float = 0.6,
                     graph_weight: float = 0.4) -> List[Dict]:
        """
        混合检索
        
        1. 向量检索：语义相似度
        2. 图谱检索：关系扩展
        3. 融合排序
        
        Args:
            query: 查询文本
            top_k: 返回数量
            vector_weight: 向量检索权重
            graph_weight: 图谱检索权重
        """
        # 1. 向量检索
        vector_results = self._vector_search(query, top_k=top_k * 2)
        
        # 2. 提取查询实体
        query_entities = self._extract_entities(query)
        
        # 3. 图谱扩展
        graph_expanded = self.kg.graph_enhanced_search(query_entities, hop=2)
        
        # 4. 融合结果
        mall_scores = defaultdict(float)
        
        # 向量分数
        for mall_id, score in vector_results.items():
            mall_scores[mall_id] += vector_weight * score
        
        # 图谱分数（被图谱扩展命中的加分）
        for mall_id in graph_expanded:
            mall_scores[mall_id] += graph_weight * 0.5  # 基础加分
            
            # 如果同时被向量检索命中，额外加分
            if mall_id in vector_results:
                mall_scores[mall_id] += graph_weight * 0.3
        
        # 5. 排序并获取详情
        sorted_malls = sorted(mall_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for mall_id, score in sorted_malls[:top_k]:
            mall_row = self.df_malls[self.df_malls['mall_id'].astype(str) == str(mall_id)]
            if mall_row.empty:
                continue
            
            mall = mall_row.iloc[0]
            
            # 获取图谱上下文
            context = self.kg.get_mall_context(mall['mall_name'])
            
            results.append({
                "mall_id": str(mall_id),
                "mall_name": mall['mall_name'],
                "district": mall.get('district', ''),
                "score_market": mall.get('score_market', 0),
                "traffic_daily": mall.get('traffic_daily', 0) or 0,
                "hybrid_score": round(score, 4),
                "in_vector_results": mall_id in vector_results,
                "in_graph_expansion": mall_id in graph_expanded,
                "graph_context": context,  # 图谱上下文
            })
        
        return results
    
    def hybrid_graph_search(self, retriever, query: str, top_k: int = 10,
                            candidate_k: int = 50, graph_weight: float = 0.35) -> List[Dict]:
        """hybrid(BM25+稠密) 之上，把图谱 2-hop 扩展作为第三路召回并进 RRF。

        实测（eval_graphrag_fusion.py，50 条）：单独 GraphRAG ≈ 基线（品牌查询抽不到实体、
        图谱空转）；此融合比纯 hybrid 净胜（MRR 0.694→0.744、R@5 0.225→0.237），
        且补 hard/medium 短板、不拖累 easy。

        Args:
            retriever: HybridRetriever 实例（提供 vs / _bm25_rank / _signal / _rrf / _meta_by_id）
        """
        dense_res = hybrid_vector_search(self.vectorstore, query, top_k=candidate_k)
        dense_rank = [str(r["mall_id"]) for r in dense_res if r.get("mall_id")]
        dense_by_mid = {str(r["mall_id"]): r for r in dense_res if r.get("mall_id")}

        bm25_rank = retriever._bm25_rank(query, candidate_k)
        sig = retriever._signal(query)
        w_bm25 = float(np.clip((sig - 1.2) / 3.0, 0.0, 1.0))

        entities = self._extract_entities(query)
        graph_expanded = self.kg.graph_enhanced_search(entities, hop=2)
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

    def _vector_search(self, query: str, top_k: int = 20) -> Dict[str, float]:
        """向量检索"""
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        mall_scores = {}
        for doc, dist in results:
            mall_id = str(doc.metadata.get('mall_id', ''))
            score = max(0, (2 - dist) / 2)  # 归一化到 0-1
            mall_scores[mall_id] = score
        
        return mall_scores
    
    def _extract_entities(self, query: str) -> List[str]:
        """从查询中提取实体（简化版）"""
        entities = []
        
        # 提取区域：从图谱真实 District 节点名解析，返回完整名（如"徐汇区"/"浦东新区"），
        # 避免区名关键字（徐汇/浦东）撞上名字里带区名的商场、被 _find_entity 误解析成商场
        for nid in self.kg.entity_index.get("District", set()):
            name = self.kg.graph.nodes[nid].get("name", "")
            if not name:
                continue
            root = name.replace("新区", "").replace("区", "")  # 浦东新区→浦东, 徐汇区→徐汇
            if root and (name in query or root in query):
                entities.append(name)
        
        # 提取品类
        categories = ["服装", "餐饮", "运动", "珠宝", "化妆品", "娱乐", "亲子"]
        for c in categories:
            if c in query:
                entities.append(c)
        
        # 提取客群特征
        profiles = ["高消费", "年轻", "白领", "家庭", "学生"]
        for p in profiles:
            if p in query:
                entities.append(p)
        
        # 提取商场名（如果提到具体商场）
        # 这里简化处理，实际应该用NER
        
        return entities


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 加载数据
    df_malls = pd.read_csv("mall_wide_table_shanghai_final_v3.csv", dtype={'mall_id': str})
    
    # 构建知识图谱
    kg = MallKnowledgeGraph()
    kg.build_from_dataframe(df_malls)
    
    # 打印统计
    stats = kg.get_statistics()
    print(f"\n图谱统计:")
    print(f"  节点: {stats['total_nodes']}")
    print(f"  边: {stats['total_edges']}")
    print(f"  实体类型: {stats['entity_types']}")
    print(f"  关系类型: {stats['relation_types']}")
    
    # 测试查询
    print("\n=== 测试查询 ===")
    
    # 1. 查找蓝海商场
    blue_oceans = kg.find_blue_ocean_by_graph("运动", district="徐汇")
    print(f"\n徐汇区运动品类蓝海商场:")
    for bo in blue_oceans[:5]:
        print(f"  - {bo['mall_name']}: 密度{bo['density']:.1%}, {bo['level']}")
    
    # 2. 查找相似商场
    similar = kg.find_similar_malls("上海港汇恒隆广场")
    print(f"\n与港汇恒隆相似的商场:")
    for s in similar:
        print(f"  - {s['mall_name']}: 相似度{s['similarity']:.2f}")
    
    # 3. 获取商场上下文
    context = kg.get_mall_context("上海前滩太古里")
    print(f"\n前滩太古里的图谱上下文:")
    print(f"  位置: {context['location']}")
    print(f"  品类: {context['categories']}")
    print(f"  客群: {context['customer_profiles']}")
    
    # 导出三元组
    kg.export_triples("mall_knowledge_graph.json")
