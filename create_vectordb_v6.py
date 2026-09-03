"""
MARS V6 向量数据库生成脚本
使用 mall_wide_table_shanghai_final_v3.csv 生成向量数据库

包含：
- 基础信息
- 客群画像
- 5维评分
- 客流数据
- 品牌竞品数据
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm

from rag_utils import get_embeddings, EMBEDDING_MODEL, hybrid_vector_search

# ============================================================
# 配置
# ============================================================

DATA_FILE = "mall_wide_table_shanghai_final_v3.csv"
CHROMA_DIR = "./chroma_db"
BATCH_SIZE = 50

# 使用镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# ============================================================
# 加载数据
# ============================================================

print("=" * 60)
print("MARS V6 向量数据库生成")
print("=" * 60)

print(f"\n加载数据: {DATA_FILE}")
df = pd.read_csv(DATA_FILE, dtype={'mall_id': str})

print(f"总商场数: {len(df)}")
print(f"有客流数据: {df['traffic_daily'].notna().sum()}")
print(f"有品牌数据: {(df['brand_count'] > 0).sum()}")

# ============================================================
# 生成嵌入文本
# ============================================================

def _parse_json(s):
    """安全解析 JSON 字符串为 dict"""
    if s is None or (isinstance(s, float) and pd.isna(s)) or not str(s).strip():
        return {}
    try:
        d = json.loads(s)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _top_by(d, prefix, key='ratio', min_val=0.0, limit=1):
    """在 customer_profile_json 中，取 prefix|value 里 key(ratio/tgi) 排名靠前的取值"""
    items = []
    for k, v in d.items():
        if isinstance(v, dict) and k.startswith(prefix + '|'):
            val = v.get(key, 0) or 0
            if val >= min_val:
                items.append((val, k.split('|', 1)[1]))
    items.sort(key=lambda x: -x[0])
    return [name for _, name in items[:limit]]


def _tags_from_profile(d) -> str:
    """从 TGI 客群画像抽取可读标签"""
    tags = []

    # 消费水平
    consume = _top_by(d, 'consume', 'ratio', limit=2)
    if consume:
        if any(c in ('高', '次高') for c in consume):
            tags.append('高消费客群')
        elif consume[0] == '中':
            tags.append('中等消费客群')
        elif consume[0] in ('次低', '低'):
            tags.append('大众消费客群')

    # 年龄
    age = _top_by(d, 'age', 'ratio', limit=1)
    if age:
        a = age[0]
        if a == '0-17':
            tags.append('亲子少儿客群')
        elif a in ('18-24', '25-30'):
            tags.append('年轻客群')
        elif a in ('31-35', '36-40'):
            tags.append('青年客群')
        elif a in ('41-45', '46-60'):
            tags.append('中年客群')
        elif a == '61以上':
            tags.append('银发客群')

    # 子女/家庭
    kids = _top_by(d, '子女年龄', 'ratio', limit=1)
    if kids and kids[0] != '无子女':
        tags.append('家庭亲子客群')

    # 有车
    if _top_by(d, '是否有车', 'ratio', limit=1) == ['是']:
        tags.append('有车客群')

    # 居住社区房价 → 高净值
    home = _top_by(d, '居住社区房价', 'ratio', limit=1)
    if home:
        h = home[0]
        if h in ('80000-99999', '>100000'):
            tags.append('高净值人群')
        elif h in ('60000-79999', '40000-59999'):
            tags.append('中高收入人群')

    # 手机价格 → 高消费力
    phone = _top_by(d, '手机价格', 'ratio', limit=1)
    if phone and phone[0] in ('8001-9000', '9001-10000', '>10000', '7001-8000', '6001-7000', '5001-6000'):
        tags.append('高消费力人群')

    # 学历
    edu = _top_by(d, '学历', 'ratio', limit=1)
    if edu and edu[0] in ('硕士', '本科'):
        tags.append('高学历人群')

    # 到访偏好大类（用 tgi 找显著偏好的活动）
    pref_map = {
        '购物': '购物爱好者', '运动健身': '运动健身人群', '美食': '美食爱好者',
        '娱乐休闲': '休闲娱乐人群', '旅游景点': '旅游人群', '教育学校': '教育相关客群',
        '医疗保健': '健康养生人群', '汽车': '汽车相关客群',
    }
    prefs = _top_by(d, '到访偏好大类', 'tgi', min_val=115, limit=3)
    for p in prefs:
        tag = pref_map.get(p)
        if tag and tag not in tags:
            tags.append(tag)

    # 商场到店频次 → 高粘性
    freq = _top_by(d, '商场到店频次', 'ratio', limit=1)
    if freq and freq[0] in ('11次及以上', '9-10次', '7-8次'):
        tags.append('高频到店客群')

    return '、'.join(tags) if tags else ''


def _brand_names(row) -> str:
    """从 brand_data_json 抽取真实品牌名，利于「运动户外/奢侈/咖啡」类查询"""
    bd = _parse_json(row.get('brand_data_json'))
    brands = bd.get('top_brands', []) or []
    if brands:
        return '、'.join(str(b) for b in brands[:50])
    return ''


def _category_tiers(row) -> str:
    """品类结构：补齐中间档（丰富/较多/一般/少）"""
    tiers = []
    for cat in ['服装', '餐饮', '运动', '珠宝', '护肤化妆品', '娱乐服务']:
        density = row.get(f'{cat}_density', 0) or 0
        count = row.get(f'{cat}_brand_count', 0) or 0
        if count <= 0:
            continue
        if density >= 0.25:
            tier = f'{cat}丰富'
        elif density >= 0.12:
            tier = f'{cat}较多'
        elif density >= 0.06:
            tier = f'{cat}一般'
        else:
            tier = f'{cat}少'
        tiers.append(f'{tier}({int(count)})')
    return '、'.join(tiers)


def generate_embedding_text(row) -> str:
    """生成用于向量化的文本（关键词密集、贴合自然查询习惯）"""
    parts = []

    # 1. 名称 + 位置
    parts.append(str(row['mall_name']).strip())
    if pd.notna(row.get('district')):
        parts.append(f"上海{row['district']}")

    # 2. 商场档次（由 score_market 派生）
    score = row.get('score_market', 0) or 0
    if score >= 90:
        parts.append("顶级高端商场")
    elif score >= 85:
        parts.append("高端商场")
    elif score >= 75:
        parts.append("中高端商场")
    elif score >= 65:
        parts.append("中端商场")
    else:
        parts.append("大众商场")

    # 3. 客流
    traffic = row.get('traffic_daily', 0) or 0
    if traffic >= 30000:
        parts.append("超高人气")
    elif traffic >= 15000:
        parts.append("高人气")
    elif traffic >= 8000:
        parts.append("中等人气")

    # 4. 定位文本（高信号自然语言：目标客群 / 消费能力 / 区位）
    pos = row.get('position_text', '')
    if pd.notna(pos) and str(pos).strip():
        parts.append(str(pos).strip())

    # 5. 真实品牌名
    brands = _brand_names(row)
    if brands:
        parts.append(f"品牌：{brands}")

    # 6. 品类结构
    tiers = _category_tiers(row)
    if tiers:
        parts.append(f"品类：{tiers}")

    # 7. 客群标签（TGI 显式标签）
    tags = _tags_from_profile(_parse_json(row.get('customer_profile_json')))
    if tags:
        parts.append(f"客群：{tags}")

    return "\n".join(parts)


print("\n生成嵌入文本...")
df['embedding_text_v6'] = df.apply(generate_embedding_text, axis=1)

# 显示样例
print("\n样例文本:")
for i in range(3):
    print(f"\n[{i+1}] {df.iloc[i]['mall_name']}")
    print(f"    {df.iloc[i]['embedding_text_v6'][:200]}...")

# ============================================================
# 加载嵌入模型
# ============================================================

print(f"\n加载嵌入模型 ({EMBEDDING_MODEL})...")
embeddings = get_embeddings()
if embeddings is None:
    raise SystemExit("❌ 嵌入模型加载失败，请先安装依赖: pip install -r requirements-rag.txt")
print("✅ 嵌入模型加载完成")

# ============================================================
# 创建向量数据库
# ============================================================

print(f"\n创建向量数据库: {CHROMA_DIR}")

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 删除旧数据库
import shutil
if Path(CHROMA_DIR).exists():
    shutil.rmtree(CHROMA_DIR)
    print("已删除旧数据库")

# 准备文档
documents = []
for idx, row in df.iterrows():
    # 元数据
    metadata = {
        'mall_id': str(row['mall_id']),
        'mall_name': row['mall_name'],
        'district': row.get('district', ''),
        'address': row.get('address', '')[:100] if pd.notna(row.get('address')) else '',
        'latitude': float(row['latitude']) if pd.notna(row.get('latitude')) else 0,
        'longitude': float(row['longitude']) if pd.notna(row.get('longitude')) else 0,
        'score_market': float(row.get('score_market', 0) or 0),
        'score_pop': float(row.get('score_pop', 0) or 0),
        'score_consume': float(row.get('score_consume', 0) or 0),
        'score_area': float(row.get('score_area', 0) or 0),
        'score_operation': float(row.get('score_operation', 0) or 0),
        'traffic_daily': float(row.get('traffic_daily', 0) or 0),
        'brand_count': int(row.get('brand_count', 0) or 0),
        'mall_area': float(row.get('mall_area', 0) or 0),
        'floor_count': int(row.get('floor_count', 0) or 0),
    }
    
    # 品类密度
    for cat in ['服装', '餐饮', '运动', '珠宝', '护肤化妆品', '娱乐服务']:
        density_col = f'{cat}_density'
        count_col = f'{cat}_brand_count'
        metadata[density_col] = float(row.get(density_col, 0) or 0)
        metadata[count_col] = int(row.get(count_col, 0) or 0)
    
    doc = Document(
        page_content=row['embedding_text_v6'],
        metadata=metadata
    )
    documents.append(doc)

print(f"准备文档: {len(documents)} 个")

# 分批创建
print("\n开始向量化...")
for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="进度"):
    batch = documents[i:i+BATCH_SIZE]
    
    if i == 0:
        # 第一批创建新数据库
        vectorstore = Chroma.from_documents(
            documents=batch,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
    else:
        # 后续批次添加
        vectorstore.add_documents(batch)

print(f"\n✅ 向量数据库创建完成: {CHROMA_DIR}")

# ============================================================
# 验证
# ============================================================

print("\n" + "=" * 60)
print("验证测试")
print("=" * 60)

# 重新加载
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

test_queries = [
    "高端运动户外品牌，高收入人群",
    "亲子餐饮，家庭客群多",
    "年轻白领，咖啡茶饮",
    "高客流量商场",
    "服装品牌少的蓝海商场",
]

for query in test_queries:
    print(f"\n查询: {query}")
    results = hybrid_vector_search(vectorstore, query, top_k=3)
    for r in results:
        if "error" in r:
            print(f"  - {r['error']}")
            continue
        traffic = r.get('traffic_daily', 0) or 0
        brand = r.get('brand_count', 0) or 0
        traffic_str = f"{traffic:,.0f}/天" if traffic > 0 else "无"
        brand_str = f"{brand}家" if brand > 0 else "无"
        print(f"  - {r.get('mall_name', 'Unknown')} (评分:{r.get('score_market', 0)}, 客流:{traffic_str}, 品牌:{brand_str}) [相似度:{r.get('similarity', 0)}%]")

print("\n" + "=" * 60)
print("✅ 向量数据库生成完成！")
print(f"   路径: {CHROMA_DIR}")
print(f"   文档数: {len(documents)}")
print("=" * 60)

# ============================================================
# 保存统计信息
# ============================================================

stats = {
    "total_malls": len(df),
    "with_traffic": int(df['traffic_daily'].notna().sum()),
    "with_brands": int((df['brand_count'] > 0).sum()),
    "data_file": DATA_FILE,
    "chroma_dir": CHROMA_DIR,
}

with open(Path(CHROMA_DIR) / "stats.json", 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n统计信息已保存到 {CHROMA_DIR}/stats.json")
