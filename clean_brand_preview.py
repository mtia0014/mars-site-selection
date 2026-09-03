"""
品牌名清洗 —— 只做预览，不改任何数据、不重建向量库。

输出：
  1. 全局统计（原始去重 vs 清洗后去重，品牌数会降多少）
  2. Top 高频原始品牌名 → 清洗结果 对照表（人工审查映射质量）
  3. 3 个大商场的 top 品牌 清洗前后对比
  4. 疑似误伤清单（清洗后变短很多 / 被丢弃的）

清洗规则（见 clean_brand 内注释）：
  A. 丢弃纯垃圾词（休息区/服务台/卫生间/电梯/扶梯…）
  B. 去掉括号及其内容（楼层/位置/分店号）
  C. 去掉尾部门店后缀（店/门店/旗舰店/专卖店/专柜…）
  D. 去掉头部「上海(市)+行政区」前缀
  E. 去掉尾部位置词（路/街/道/号/广场/万达/金街/大悦城/万象城/来福士…）
"""

import re
from collections import Counter

import pandas as pd

STORE = r"D:\Work\agent\data\biz_store.csv"

# ---- 纯垃圾 / 非品牌，整条丢弃 ----
DROP_RE = re.compile(
    r"休息区|服务台|客服|咨询台|卫生间|洗手间|电梯|扶梯|通道|入口|出口|"
    r"售票|存包|寄存|前台|收银|母婴室|吸烟|垃圾|中庭|广场$|^广场|金街$|^金街"
)

# ---- 尾部门店后缀（可叠加，重复剥离） ----
STORE_SUFFIX_RE = re.compile(r"(店中店|旗舰店|专卖店|专营店|专柜|门店|店铺|店)+$")

# ---- 头部城市/行政区前缀 ----
CITY_DISTRICT_RE = re.compile(
    r"^(上海市|上海|北京市|北京|广州市|广州|深圳市|深圳|杭州市|杭州|成都市|成都|"
    r"浦东新区|静安区|徐汇区|长宁区|普陀区|虹口区|杨浦区|黄浦区|闵行区|宝山区|"
    r"嘉定区|松江区|青浦区|奉贤区|金山区|崇明区|浦东|徐汇|静安|长宁|普陀|虹口|杨浦|黄浦|闵行|宝山|嘉定|松江|青浦|奉贤|金山|崇明)+"
)

# ---- 尾部位置词（含商场/商圈/地名） ----
LOC_TAIL_RE = re.compile(
    r"(路|街|道|号|广场|万达茂|万达|金街|大悦城|万象城|万象汇|来福士|龙之梦|新天地|"
    r"太古里|悦荟|正大|百联|久光|龙湖|天街|印象城|奥莱|奥莱斯|商城|购物中心)+$"
)


def clean_brand(s):
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None

    # A. 纯垃圾
    if DROP_RE.search(s):
        return None

    # B. 括号及其内容
    s = re.sub(r"[\(\（][^\)\）]*[\)\）]", "", s)

    # C. 尾部门店后缀（重复剥离）
    prev = None
    while prev != s:
        prev = s
        s = STORE_SUFFIX_RE.sub("", s)

    # D. 头部城市/行政区（重复剥离）
    prev = None
    while prev != s:
        prev = s
        s = CITY_DISTRICT_RE.sub("", s)

    # E. 尾部位置词（重复剥离）
    prev = None
    while prev != s:
        prev = s
        s = LOC_TAIL_RE.sub("", s)

    s = s.strip(" -_—·、/（）()")
    if not s or len(s) < 2:
        return None
    if DROP_RE.search(s):
        return None
    return s


def main():
    st = pd.read_csv(STORE, dtype={"mall_id": str}, on_bad_lines="skip")

    raw = st["brand_name"].dropna().astype(str)
    cleaned = raw.map(clean_brand)

    raw_uniq = raw.nunique()
    cln_uniq = cleaned.dropna().nunique()
    kept = cleaned.notna().sum()
    dropped = cleaned.isna().sum()

    print("=" * 64)
    print("品牌名清洗 —— 预览（未改任何数据）")
    print("=" * 64)
    print(f"门店行数            : {len(raw)}")
    print(f"原始去重字符串      : {raw_uniq}")
    print(f"清洗后去重品牌      : {cln_uniq}")
    print(f"  降幅              : {(raw_uniq - cln_uniq) / raw_uniq * 100:.1f}%")
    print(f"被判定为垃圾丢弃    : {dropped} 条门店（{dropped/len(raw)*100:.1f}%）")

    # ---- 高频原始名 → 清洗结果 ----
    print("\n" + "=" * 64)
    print("Top 40 高频原始名 → 清洗结果（审查映射质量）")
    print("=" * 64)
    top_raw = raw.value_counts().head(40)
    for name, cnt in top_raw.items():
        c = clean_brand(name)
        flag = "" if c else "  [丢弃]"
        if c and c == name:
            flag += "  [未变]"
        print(f"  {name[:38]:<38} → {str(c):<22} (门店{int(cnt)}){flag}")

    # ---- 3 个大商场 前后对比（mall_name 从宽表取，门店表 mall_name 为空） ----
    print("\n" + "=" * 64)
    print("大商场 top 品牌 清洗前后对比")
    print("=" * 64)
    wide = pd.read_csv(
        r"D:\Work\agent\data\data3\mall_wide_table_shanghai_final_v3.csv", dtype={"mall_id": str}
    )
    for mall_kw in ["江桥万达", "中山公园龙之梦", "月星环球港"]:
        ids = wide[wide["mall_name"].str.contains(mall_kw, na=False)]["mall_id"].astype(str).unique()
        sub = st[st["mall_id"].astype(str).isin(ids)]
        if sub.empty:
            continue
        sub_cln = sub["brand_name"].dropna().astype(str).map(clean_brand)
        before = sub["brand_name"].value_counts().head(10).index.tolist()
        after = sub_cln.dropna().value_counts().head(10).index.tolist()
        print(f"\n[{mall_kw}]")
        print(f"  前: " + " | ".join(b[:12] for b in before))
        print(f"  后: " + " | ".join(a[:12] for a in after))

    # ---- 疑似误伤：清洗后比原始短很多（>60%），可能是把真实品牌剥掉了 ----
    print("\n" + "=" * 64)
    print("疑似误伤清单（清洗后长度骤减 >60%，可能剥掉了真实品牌）")
    print("=" * 64)
    seen = set()
    n = 0
    for name, cnt in raw.value_counts().items():
        c = clean_brand(name)
        if c and c != name and len(c) <= len(name) * 0.4:
            if c in seen:
                continue
            seen.add(c)
            print(f"  {name[:36]:<36} → {c[:20]}")
            n += 1
            if n >= 30:
                break
    if n == 0:
        print("  （无）")


if __name__ == "__main__":
    main()
