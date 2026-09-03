"""
修复品牌合并两处问题：
  1) top_brands 只保留前 20 → 放宽到 TOP_N（默认 50）
  2) brand_count 存的是「门店行数」→ 改成「去重品牌数」（同一品牌多铺位不重复计）

重算范围：brand_count / brand_data_json / 6 个品类列（服装/餐饮/运动/珠宝/护肤化妆品/娱乐服务 的 count+density）。
其余列（评分、客流、position_text、customer_profile_json、embedding_text 等）一律不动。
品类 density 语义不变：品类门店数 / 该商场门店总行数（与原来一致）。

用法:
    python fix_brand_merge.py
"""

import os
import json
import shutil

import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))          # data3/
PARENT = os.path.dirname(DATA_DIR)                             # data/
WIDE = os.path.join(DATA_DIR, "mall_wide_table_shanghai_final_v3.csv")
STORE = os.path.join(PARENT, "biz_store.csv")

TOP_N = int(os.environ.get("TOP_N", "50"))

# 明确非品牌的设施/功能区名（精确匹配，零误伤）。从门店表实测筛出。
JUNK_BRANDS = frozenset({
    "母婴室", "休息区", "服务台", "垃圾房", "吸烟区", "自助寄存",
    "车库-出口", "亲子洗手间", "母婴室|无障碍卫生间", "魔方柜自助寄存",
})

# brand_data_json 里的 8 个品类（含 数码电器/其他），与源表 category_2 取值一一对应
ALL_CATS = ["服装", "餐饮", "娱乐服务", "其他", "珠宝", "数码电器", "护肤化妆品", "运动"]
# 展平成宽表列的 6 个品类（与现有 final_v3 列一致）
WIDE_CATS = ["服装", "餐饮", "运动", "珠宝", "护肤化妆品", "娱乐服务"]


def build(mall_id, g):
    """对单个商场的门店分组，计算品牌统计"""
    # 剔除明确非品牌（母婴室/休息区等设施名），不参与品牌数/品类统计
    g = g[~g["brand_name"].astype(str).isin(JUNK_BRANDS)]
    total = len(g)
    names = g["brand_name"].dropna()
    brand_count = int(names.nunique())                       # 去重品牌数（关键修复）
    top = names.value_counts().head(TOP_N).index.tolist()    # top N 品牌（按出现次数）

    cat_counts = g["category_2"].dropna().value_counts().to_dict()
    cat_dist = {c: int(cat_counts.get(c, 0)) for c in ALL_CATS}

    d = {
        "brand_count": brand_count,
        "category_distribution": cat_dist,
        "top_brands": top,
    }
    for c in ALL_CATS:
        cnt = int(cat_counts.get(c, 0))
        d[f"{c}_count"] = cnt
        d[f"{c}_density"] = round(cnt / total, 4) if total else 0.0

    return {
        "brand_count": brand_count,
        "brand_data_json": json.dumps(d, ensure_ascii=False),
        **{f"{c}_brand_count": int(cat_counts.get(c, 0)) for c in WIDE_CATS},
        **{f"{c}_density": round(cat_counts.get(c, 0) / total, 4) if total else 0.0 for c in WIDE_CATS},
    }


def main():
    print("=" * 60)
    print("品牌合并修复")
    print("=" * 60)

    wide = pd.read_csv(WIDE, dtype={"mall_id": str})
    st = pd.read_csv(STORE, dtype={"mall_id": str}, on_bad_lines="skip")

    # 备份
    bak = WIDE.replace(".csv", ".bak.csv")
    shutil.copy(WIDE, bak)
    print(f"已备份原表 → {os.path.basename(bak)}")

    # 按 mall_id 预计算
    lookup = {str(mid): build(str(mid), g) for mid, g in st.groupby("mall_id")}

    # 回填宽表（只动品牌相关列，保持列顺序不变）
    brand_cols = ["brand_count", "brand_data_json"] + \
                 [f"{c}_{s}" for c in WIDE_CATS for s in ("brand_count", "density")]
    n_hit = 0
    for idx, row in wide.iterrows():
        mid = str(row["mall_id"])
        if mid in lookup:
            for col in brand_cols:
                wide.at[idx, col] = lookup[mid][col]
            n_hit += 1
        else:
            wide.at[idx, "brand_count"] = 0
            wide.at[idx, "brand_data_json"] = "{}"
            for c in WIDE_CATS:
                wide.at[idx, f"{c}_brand_count"] = 0
                wide.at[idx, f"{c}_density"] = 0.0

    wide.to_csv(WIDE, index=False)

    print(f"\n完成！TOP_N={TOP_N}，brand_count 改为去重品牌数")
    print(f"命中品牌数据的商场数: {n_hit}")
    print(f"有品牌数据(brand_count>0): {int((wide['brand_count'] > 0).sum())}")

    # 抽查
    print("\n抽查（大商场）:")
    for name in ["江桥万达", "中山公园龙之梦", "月星环球港", "新世界城", "久光百货"]:
        r = wide[wide["mall_name"].str.contains(name, na=False)]
        if len(r):
            r = r.iloc[0]
            bd = json.loads(r["brand_data_json"])
            print(f"  {r['mall_name']}: brand_count={r['brand_count']} (去重), "
                  f"top_brands={len(bd.get('top_brands', []))} 条, "
                  f"餐饮={bd.get('餐饮_count')}")


if __name__ == "__main__":
    main()
