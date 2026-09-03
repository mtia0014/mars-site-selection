"""
生成合成样例宽表 sample_data.csv（20 家虚构商场 × 40 列）。

真实宽表 mall_wide_table_shanghai_final_v3.csv 因数据授权未随仓库分发，
本脚本生成一份同结构、字段完整、检索有意义的合成样例，供克隆者跑通全链路：

    python generate_sample_data.py
    copy /Y sample_data.csv mall_wide_table_shanghai_final_v3.csv
    python create_vectordb_v6.py
    run_gradio.bat

所有 mall_name / mall_id / address 均为虚构；品牌、品类、客群画像为通用公开词，
仅用于演示检索效果，不涉及任何真实商业数据。
"""

import json

import pandas as pd

OUT = "sample_data.csv"

# ============ 品牌目录（品类 → 公开品牌名，仅演示用）============
BRANDS = {
    "服装": ["优衣库", "ZARA", "H&M", "UR", "太平鸟", "波司登", "海澜之家", "江南布衣"],
    "餐饮": ["星巴克", "瑞幸咖啡", "Manner", "喜茶", "蜜雪冰城", "海底捞", "西贝莜面村", "外婆家", "肯德基", "麦当劳", "太二酸菜鱼"],
    "运动": ["耐克", "阿迪达斯", "李宁", "安踏", "lululemon", "迪卡侬", "斯凯奇", "斐乐"],
    "珠宝": ["周大福", "周生生", "六福珠宝", "老凤祥", "卡地亚", "蒂芙尼", "宝格丽", "I Do"],
    "护肤化妆品": ["兰蔻", "雅诗兰黛", "资生堂", "丝芙兰", "SK-II", "悦木之源", "完美日记", "花西子"],
    "娱乐服务": ["万达影城", "金逸影城", "星聚会KTV", "威尔士健身", "超级猩猩", "大玩家", "泡泡玛特"],
}
CATS = list(BRANDS.keys())

# ============ 客群画像维度 ============
CONSUME = ["高", "次高", "中", "次低", "低"]
AGE = ["0-17", "18-24", "25-30", "31-35", "36-40", "41-45", "46-60", "61以上"]
HOME_PRICE = [">100000", "80000-99999", "60000-79999", "40000-59999", "20000-39999", "<20000"]
PHONE_PRICE = [">10000", "9001-10000", "8001-9000", "7001-8000", "5001-6000", "<5000"]
EDU = ["硕士", "本科", "大专", "高中及以下"]
FREQ = ["11次及以上", "9-10次", "7-8次", "4-6次", "1-3次"]
VISIT = ["购物", "运动健身", "美食", "娱乐休闲", "旅游景点", "教育学校", "医疗保健", "汽车"]


def _dist(dominant, vals):
    """dominant 值给最高 ratio/tgi，其余按序递减，返回 {value: {'ratio':.., 'tgi':..}}"""
    ordered = [dominant] + [v for v in vals if v != dominant]
    out = {}
    for i, v in enumerate(ordered):
        ratio = round(max(0.03, 0.45 - 0.11 * i), 2)
        out[v] = {"ratio": ratio, "tgi": round(max(45, 165 - 40 * i), 0)}
    return out


def _fmt(d, top=3):
    """把 {value: {'ratio':..}} 格式化为 '值(r%), 值(r%)'"""
    items = sorted(d.items(), key=lambda kv: -kv[1]["ratio"])
    return ", ".join(f"{v}({int(x['ratio'] * 100)}%)" for v, x in items[:top])


def _profile_json(p):
    """由 persona 生成 customer_profile_json（覆盖 _tags_from_profile 读取的所有键）"""
    d = {}
    for v, x in _dist(p["consume"], CONSUME).items():
        d[f"consume|{v}"] = x
    for v, x in _dist(p["age"], AGE).items():
        d[f"age|{v}"] = x
    d[f"子女年龄|{p['kids']}"] = {"ratio": 0.5, "tgi": 130}
    d[f"是否有车|{p['car']}"] = {"ratio": 0.55 if p["car"] == "是" else 0.45, "tgi": 120}
    for v, x in _dist(p["home"], HOME_PRICE).items():
        d[f"居住社区房价|{v}"] = x
    for v, x in _dist(p["phone"], PHONE_PRICE).items():
        d[f"手机价格|{v}"] = x
    for v, x in _dist(p["edu"], EDU).items():
        d[f"学历|{v}"] = x
    for v, x in _dist(p["freq"], FREQ).items():
        d[f"商场到店频次|{v}"] = x
    for i, pref in enumerate(p["prefs"]):
        d[f"到访偏好大类|{pref}"] = {"ratio": round(max(0.2, 0.6 - 0.15 * i), 2), "tgi": round(175 - 20 * i, 0)}
    return d


def _profile_text(p):
    """profile_text：可读文本，含 age：段（analyze_customer_profile 的正则依赖）与图谱关键词"""
    segs = []
    segs.append("APP偏好大类：" + _fmt({"社交": {"ratio": 1.0}, "系统": {"ratio": 0.99}, "工具": {"ratio": 0.98}}))
    segs.append("age：" + _fmt(_dist(p["age"], AGE)))
    segs.append("consume：" + _fmt(_dist(p["consume"], CONSUME)))
    segs.append("residence：2000-5000米(35%), 0-2000米(28%), 5000-10000米(20%)")
    segs.append("workplace：0-500米(40%), 2000-5000米(18%), 5000-10000米(12%)")
    if p.get("kw"):
        segs.append("关键词：" + "、".join(p["kw"]))
    return "；".join(segs)


def _position_text(p):
    """position_text：目标客群 / 商业配套 / 消费能力 / 综合评价 四段式"""
    return (
        f"位置特征：{p['pos']}。"
        f"目标客群：{p['target']}。"
        f"商业配套：{p['biz']}。"
        f"消费能力：{p['spend']}。"
        f"综合评价：{p['summary']}"
    )


def _brand_payload(brands):
    """由品类→品牌 dict 计算 brand_count / 品类计数密度 / top_brands / brand_data_json"""
    top = [b for lst in brands.values() for b in lst]
    total = len(top)
    counts = {c: len(brands.get(c, [])) for c in CATS}
    density = {c: round(counts[c] / total, 3) if total else 0.0 for c in CATS}
    bd = {
        "brand_count": total,
        "top_brands": top,
        "category_distribution": counts,
    }
    return total, counts, density, bd


# ============ 20 家虚构商场 ============
# persona 字段：consume/age/kids/car/home/phone/edu/prefs/freq + pos/target/biz/spend/summary + kw
MALLS = [
    {
        "name": "上海星澜天地", "district": "徐汇区", "address": "徐汇区云锦路88号",
        "lat": 31.182, "lng": 121.452, "area": 85000, "floors": 6, "open": "2021-06-18",
        "score_market": 91, "traffic": 32000, "sales": 820000,
        "persona": {"consume": "高", "age": "25-30", "kids": "无子女", "car": "是", "home": "80000-99999",
                    "phone": "8001-9000", "edu": "本科", "prefs": ["购物", "运动健身", "美食"], "freq": "7-8次",
                    "pos": "徐汇滨江核心，人工智能集聚区", "target": "科技工作者、高端消费者、年轻白领",
                    "biz": "科技艺术主题，智能体验业态", "spend": "高消费能力，客单价250-400元，周边房价12-14万/㎡",
                    "summary": "科技艺术商业地标，定位独特", "kw": ["白领", "高消费", "科技"]},
        "brands": {"服装": ["优衣库", "ZARA", "UR"], "餐饮": ["星巴克", "喜茶", "Manner", "海底捞"],
                   "运动": ["耐克", "lululemon", "李宁"], "护肤化妆品": ["丝芙兰", "兰蔻"], "娱乐服务": ["威尔士健身", "金逸影城"]},
    },
    {
        "name": "上海云栖里", "district": "静安区", "address": "静安区南京西路1225号",
        "lat": 31.231, "lng": 121.451, "area": 62000, "floors": 5, "open": "2019-11-01",
        "score_market": 88, "traffic": 28000, "sales": 700000,
        "persona": {"consume": "次高", "age": "25-30", "kids": "无子女", "car": "是", "home": "60000-79999",
                    "phone": "7001-8000", "edu": "本科", "prefs": ["美食", "购物", "娱乐休闲"], "freq": "7-8次",
                    "pos": "南京西路商圈次级位置", "target": "都市白领、年轻消费群体",
                    "biz": "咖啡茶饮与轻食业态密集", "spend": "中高消费，客单价180-300元",
                    "summary": "白领咖啡社交地标", "kw": ["白领", "咖啡", "年轻"]},
        "brands": {"餐饮": ["星巴克", "瑞幸咖啡", "Manner", "喜茶", "蜜雪冰城", "太二酸菜鱼"],
                   "服装": ["优衣库", "H&M"], "运动": ["阿迪达斯", "李宁"], "娱乐服务": ["金逸影城", "星聚会KTV"]},
    },
    {
        "name": "上海浦江汇", "district": "黄浦区", "address": "黄浦区中山东一路500号",
        "lat": 31.240, "lng": 121.491, "area": 110000, "floors": 8, "open": "2017-09-15",
        "score_market": 93, "traffic": 15000, "sales": 1200000,
        "persona": {"consume": "高", "age": "31-35", "kids": "无子女", "car": "是", "home": ">100000",
                    "phone": ">10000", "edu": "硕士", "prefs": ["购物", "旅游景点", "美食"], "freq": "4-6次",
                    "pos": "外滩金融集聚带，顶级商业位置", "target": "金融精英、高净值人群、国际客群",
                    "biz": "高端商业配套，奢侈品集群", "spend": "极高消费能力，客单价350-600元，周边房价15-20万/㎡",
                    "summary": "顶级金融区商业，稀缺性突出", "kw": ["高净值", "金融", "奢侈品"]},
        "brands": {"珠宝": ["卡地亚", "蒂芙尼", "宝格丽", "周大福"], "护肤化妆品": ["兰蔻", "雅诗兰黛", "SK-II", "资生堂"],
                   "服装": ["ZARA", "H&M"], "餐饮": ["星巴克", "喜茶"], "娱乐服务": ["威尔士健身"]},
    },
    {
        "name": "上海虹桥里", "district": "长宁区", "address": "长宁区虹桥路1665号",
        "lat": 31.205, "lng": 121.402, "area": 70000, "floors": 5, "open": "2020-05-20",
        "score_market": 85, "traffic": 35000, "sales": 650000,
        "persona": {"consume": "次高", "age": "18-24", "kids": "无子女", "car": "否", "home": "60000-79999",
                    "phone": "7001-8000", "edu": "本科", "prefs": ["运动健身", "购物", "美食"], "freq": "7-8次",
                    "pos": "虹桥枢纽辐射，交通便利", "target": "年轻运动人群、通勤白领",
                    "biz": "运动品牌集群，健身业态丰富", "spend": "中高消费，客单价200-350元",
                    "summary": "运动户外聚集地", "kw": ["年轻", "运动", "白领"]},
        "brands": {"运动": ["耐克", "阿迪达斯", "李宁", "安踏", "lululemon", "迪卡侬", "斯凯奇"],
                   "餐饮": ["星巴克", "瑞幸咖啡", "蜜雪冰城", "肯德基"], "服装": ["优衣库", "UR"],
                   "娱乐服务": ["超级猩猩", "威尔士健身", "万达影城"]},
    },
    {
        "name": "上海静安屿", "district": "静安区", "address": "静安区愚园路68号",
        "lat": 31.224, "lng": 121.445, "area": 30000, "floors": 4, "open": "2018-03-10",
        "score_market": 78, "traffic": 18000, "sales": 400000,
        "persona": {"consume": "中", "age": "25-30", "kids": "无子女", "car": "否", "home": "60000-79999",
                    "phone": "5001-6000", "edu": "本科", "prefs": ["美食", "娱乐休闲", "购物"], "freq": "4-6次",
                    "pos": "愚园路文艺街区", "target": "文艺青年、社区白领",
                    "biz": "精品咖啡与生活方式业态", "spend": "中等消费，客单价120-200元",
                    "summary": "社区型精品商场", "kw": ["白领", "文艺", "咖啡"]},
        "brands": {"餐饮": ["星巴克", "Manner", "喜茶", "西贝莜面村", "外婆家"],
                   "服装": ["优衣库"], "护肤化妆品": ["完美日记", "花西子"], "娱乐服务": ["泡泡玛特"]},
    },
    {
        "name": "上海东岸中心", "district": "浦东新区", "address": "浦东新区世纪大道100号",
        "lat": 31.236, "lng": 121.506, "area": 95000, "floors": 7, "open": "2016-12-24",
        "score_market": 82, "traffic": 24000, "sales": 750000,
        "persona": {"consume": "高", "age": "31-35", "kids": "无子女", "car": "是", "home": "80000-99999",
                    "phone": "8001-9000", "edu": "本科", "prefs": ["购物", "美食", "娱乐休闲"], "freq": "4-6次",
                    "pos": "陆家嘴金融区东扩", "target": "金融从业者、中高收入家庭",
                    "biz": "珠宝钟表与商务餐饮", "spend": "高消费，客单价300-500元",
                    "summary": "金融区商务商业", "kw": ["高收入", "金融", "珠宝"]},
        "brands": {"珠宝": ["周大福", "周生生", "六福珠宝", "老凤祥", "I Do"],
                   "护肤化妆品": ["兰蔻", "丝芙兰", "资生堂"], "餐饮": ["星巴克", "海底捞", "西贝莜面村"],
                   "服装": ["ZARA", "太平鸟"], "娱乐服务": ["金逸影城", "威尔士健身"]},
    },
    {
        "name": "上海长风里", "district": "普陀区", "address": "普陀区大渡河路196号",
        "lat": 31.230, "lng": 121.395, "area": 88000, "floors": 6, "open": "2015-08-08",
        "score_market": 72, "traffic": 20000, "sales": 500000,
        "persona": {"consume": "中", "age": "31-35", "kids": "有学龄前儿童", "car": "是", "home": "40000-59999",
                    "phone": "5001-6000", "edu": "大专", "prefs": ["美食", "娱乐休闲", "教育学校"], "freq": "9-10次",
                    "pos": "长风公园板块，居住密集", "target": "家庭亲子客群、周边居民",
                    "biz": "家庭餐饮与儿童业态", "spend": "中等消费，客单价100-180元",
                    "summary": "家庭亲子社区商业", "kw": ["家庭", "亲子", "儿童"]},
        "brands": {"餐饮": ["海底捞", "西贝莜面村", "外婆家", "肯德基", "麦当劳", "星巴克", "蜜雪冰城"],
                   "娱乐服务": ["万达影城", "大玩家", "泡泡玛特", "威尔士健身"],
                   "服装": ["优衣库", "海澜之家"], "运动": ["安踏", "李宁"]},
    },
    {
        "name": "上海南桥百汇", "district": "奉贤区", "address": "奉贤区南桥镇望园南路1号",
        "lat": 30.915, "lng": 121.475, "area": 55000, "floors": 4, "open": "2013-05-01",
        "score_market": 60, "traffic": 6000, "sales": 150000,
        "persona": {"consume": "次低", "age": "46-60", "kids": "无子女", "car": "否", "home": "20000-39999",
                    "phone": "<5000", "edu": "高中及以下", "prefs": ["美食", "娱乐休闲"], "freq": "4-6次",
                    "pos": "奉贤南桥传统商圈", "target": "本地居民、中老年客群",
                    "biz": "传统百货业态，零售为主", "spend": "大众消费，客单价80-120元",
                    "summary": "远郊传统百货", "kw": ["中老年", "大众", "传统"]},
        "brands": {"餐饮": ["肯德基", "麦当劳", "蜜雪冰城", "外婆家"], "服装": ["海澜之家", "波司登"],
                   "娱乐服务": ["金逸影城", "星聚会KTV"]},
    },
    {
        "name": "上海莘庄荟", "district": "闵行区", "address": "闵行区莘庄地铁站南广场",
        "lat": 31.111, "lng": 121.382, "area": 68000, "floors": 5, "open": "2017-06-30",
        "score_market": 74, "traffic": 26000, "sales": 550000,
        "persona": {"consume": "中", "age": "25-30", "kids": "无子女", "car": "否", "home": "40000-59999",
                    "phone": "5001-6000", "edu": "本科", "prefs": ["运动健身", "美食", "购物"], "freq": "7-8次",
                    "pos": "莘庄交通枢纽上盖", "target": "通勤白领、年轻家庭",
                    "biz": "运动与快时尚业态", "spend": "中等消费，客单价120-220元",
                    "summary": "枢纽型社区商业", "kw": ["白领", "年轻", "运动"]},
        "brands": {"运动": ["耐克", "阿迪达斯", "李宁", "迪卡侬"], "餐饮": ["星巴克", "瑞幸咖啡", "喜茶", "肯德基", "太二酸菜鱼"],
                   "服装": ["优衣库", "UR", "太平鸟"], "娱乐服务": ["超级猩猩", "金逸影城"]},
    },
    {
        "name": "上海北外滩云集", "district": "虹口区", "address": "虹口区东大名路588号",
        "lat": 31.254, "lng": 121.503, "area": 42000, "floors": 5, "open": "2019-04-18",
        "score_market": 80, "traffic": 16000, "sales": 420000,
        "persona": {"consume": "高", "age": "31-35", "kids": "无子女", "car": "是", "home": "80000-99999",
                    "phone": "8001-9000", "edu": "本科", "prefs": ["购物", "美食", "旅游景点"], "freq": "4-6次",
                    "pos": "北外滩滨江，新兴商务区", "target": "商务人士、高收入白领",
                    "biz": "珠宝美妆与滨江餐饮", "spend": "高消费，客单价250-400元",
                    "summary": "滨江商务商业", "kw": ["高收入", "白领", "珠宝"]},
        "brands": {"珠宝": ["周生生", "六福珠宝", "老凤祥"], "护肤化妆品": ["雅诗兰黛", "兰蔻", "SK-II", "丝芙兰"],
                   "餐饮": ["星巴克", "喜茶", "海底捞"], "服装": ["ZARA"], "娱乐服务": ["金逸影城"]},
    },
    {
        "name": "上海五角场里", "district": "杨浦区", "address": "杨浦区淞沪路151号",
        "lat": 31.301, "lng": 121.515, "area": 50000, "floors": 5, "open": "2014-09-28",
        "score_market": 76, "traffic": 30000, "sales": 600000,
        "persona": {"consume": "中", "age": "18-24", "kids": "无子女", "car": "否", "home": "40000-59999",
                    "phone": "5001-6000", "edu": "本科", "prefs": ["运动健身", "美食", "娱乐休闲"], "freq": "11次及以上",
                    "pos": "五角场商圈，高校环绕", "target": "高校学生、年轻白领",
                    "biz": "运动与餐饮业态密集", "spend": "中等消费，客单价100-180元",
                    "summary": "高校商圈人气商业", "kw": ["学生", "年轻", "运动"]},
        "brands": {"运动": ["耐克", "阿迪达斯", "李宁", "安踏", "迪卡侬", "斐乐"],
                   "餐饮": ["瑞幸咖啡", "蜜雪冰城", "肯德基", "麦当劳", "太二酸菜鱼", "外婆家"],
                   "娱乐服务": ["万达影城", "超级猩猩", "大玩家"], "服装": ["优衣库", "H&M"]},
    },
    {
        "name": "上海宝山星港城", "district": "宝山区", "address": "宝山区牡丹江路1558号",
        "lat": 31.405, "lng": 121.489, "area": 60000, "floors": 4, "open": "2012-11-11",
        "score_market": 62, "traffic": 9000, "sales": 180000,
        "persona": {"consume": "中", "age": "36-40", "kids": "有学龄儿童", "car": "是", "home": "40000-59999",
                    "phone": "5001-6000", "edu": "大专", "prefs": ["美食", "娱乐休闲", "教育学校"], "freq": "9-10次",
                    "pos": "宝山老城居住区", "target": "家庭客群、周边居民",
                    "biz": "家庭餐饮与儿童娱乐", "spend": "大众消费，客单价90-150元",
                    "summary": "家庭社区商业", "kw": ["家庭", "亲子", "儿童"]},
        "brands": {"餐饮": ["海底捞", "外婆家", "肯德基", "麦当劳", "蜜雪冰城", "西贝莜面村"],
                   "娱乐服务": ["万达影城", "大玩家", "泡泡玛特"], "服装": ["海澜之家", "优衣库"]},
    },
    {
        "name": "上海松江云间里", "district": "松江区", "address": "松江区中山中路77号",
        "lat": 31.031, "lng": 121.227, "area": 38000, "floors": 3, "open": "2011-04-01",
        "score_market": 66, "traffic": 5000, "sales": 90000,
        "persona": {"consume": "次低", "age": "41-45", "kids": "无子女", "car": "否", "home": "20000-39999",
                    "phone": "<5000", "edu": "高中及以下", "prefs": ["美食", "娱乐休闲"], "freq": "1-3次",
                    "pos": "松江老城传统商圈", "target": "本地居民",
                    "biz": "传统零售，品牌较少", "spend": "大众消费，客单价70-100元",
                    "summary": "品牌较少的蓝海商场", "kw": ["大众", "蓝海", "传统"]},
        "brands": {"餐饮": ["肯德基", "蜜雪冰城"], "娱乐服务": ["金逸影城"]},
    },
    {
        "name": "上海嘉定新城汇", "district": "嘉定区", "address": "嘉定区白银路288号",
        "lat": 31.382, "lng": 121.256, "area": 45000, "floors": 4, "open": "2016-10-01",
        "score_market": 68, "traffic": 12000, "sales": 220000,
        "persona": {"consume": "中", "age": "31-35", "kids": "有学龄前儿童", "car": "是", "home": "40000-59999",
                    "phone": "5001-6000", "edu": "本科", "prefs": ["美食", "娱乐休闲", "教育学校"], "freq": "7-8次",
                    "pos": "嘉定新城居住核心", "target": "年轻家庭、新城居民",
                    "biz": "家庭餐饮与儿童娱乐", "spend": "中等消费，客单价100-180元",
                    "summary": "新城家庭商业", "kw": ["家庭", "亲子", "新城"]},
        "brands": {"餐饮": ["海底捞", "西贝莜面村", "星巴克", "瑞幸咖啡", "肯德基", "外婆家"],
                   "娱乐服务": ["万达影城", "大玩家", "泡泡玛特", "威尔士健身"],
                   "服装": ["优衣库", "海澜之家"], "运动": ["李宁", "安踏"]},
    },
    {
        "name": "上海徐汇滨江one", "district": "徐汇区", "address": "徐汇区龙腾大道2888号",
        "lat": 31.182, "lng": 121.461, "area": 90000, "floors": 6, "open": "2022-09-30",
        "score_market": 89, "traffic": 22000, "sales": 680000,
        "persona": {"consume": "高", "age": "25-30", "kids": "无子女", "car": "是", "home": "80000-99999",
                    "phone": "8001-9000", "edu": "硕士", "prefs": ["运动健身", "购物", "美食"], "freq": "7-8次",
                    "pos": "徐汇滨江西岸艺术带", "target": "高消费年轻人群、运动爱好者",
                    "biz": "高端运动与艺术商业", "spend": "高消费，客单价250-450元",
                    "summary": "滨江运动艺术地标", "kw": ["高消费", "年轻", "运动", "白领"]},
        "brands": {"运动": ["lululemon", "耐克", "阿迪达斯", "迪卡侬", "斐乐"],
                   "护肤化妆品": ["兰蔻", "丝芙兰", "完美日记"], "餐饮": ["星巴克", "Manner", "喜茶", "太二酸菜鱼"],
                   "服装": ["优衣库", "UR"], "娱乐服务": ["超级猩猩", "威尔士健身"]},
    },
    {
        "name": "上海陆家嘴中庭", "district": "浦东新区", "address": "浦东新区陆家嘴环路1000号",
        "lat": 31.239, "lng": 121.500, "area": 120000, "floors": 9, "open": "2015-01-01",
        "score_market": 95, "traffic": 18000, "sales": 1500000,
        "persona": {"consume": "高", "age": "31-35", "kids": "无子女", "car": "是", "home": ">100000",
                    "phone": ">10000", "edu": "硕士", "prefs": ["购物", "旅游景点", "美食"], "freq": "1-3次",
                    "pos": "陆家嘴核心，顶级商业位置", "target": "高净值人群、国际商务客群",
                    "biz": "顶级奢侈品集群", "spend": "极高消费能力，客单价400-800元，周边房价15-20万/㎡",
                    "summary": "顶级奢侈品旗舰商业", "kw": ["高净值", "奢侈品", "金融"]},
        "brands": {"珠宝": ["卡地亚", "蒂芙尼", "宝格丽", "周大福", "周生生"],
                   "护肤化妆品": ["兰蔻", "雅诗兰黛", "SK-II", "资生堂", "丝芙兰"],
                   "服装": ["ZARA", "H&M"], "餐饮": ["星巴克", "喜茶", "Manner"], "娱乐服务": ["威尔士健身"]},
    },
    {
        "name": "上海古北里", "district": "长宁区", "address": "长宁区黄金城道688号",
        "lat": 31.213, "lng": 121.398, "area": 40000, "floors": 4, "open": "2018-11-18",
        "score_market": 83, "traffic": 11000, "sales": 350000,
        "persona": {"consume": "高", "age": "36-40", "kids": "有学龄儿童", "car": "是", "home": "80000-99999",
                    "phone": "8001-9000", "edu": "硕士", "prefs": ["购物", "美食", "教育学校"], "freq": "4-6次",
                    "pos": "古北国际社区", "target": "高收入家庭、外籍人士",
                    "biz": "珠宝美妆与精品餐饮", "spend": "高消费，客单价250-400元",
                    "summary": "国际社区精品商业", "kw": ["高收入", "家庭", "国际"]},
        "brands": {"珠宝": ["周大福", "周生生", "卡地亚"], "护肤化妆品": ["雅诗兰黛", "资生堂", "兰蔻", "丝芙兰"],
                   "餐饮": ["星巴克", "喜茶", "西贝莜面村"], "服装": ["优衣库", "ZARA"], "娱乐服务": ["威尔士健身"]},
    },
    {
        "name": "上海七宝万科汇", "district": "闵行区", "address": "闵行区漕宝路3366号",
        "lat": 31.159, "lng": 121.349, "area": 100000, "floors": 6, "open": "2016-12-16",
        "score_market": 70, "traffic": 32000, "sales": 580000,
        "persona": {"consume": "中", "age": "31-35", "kids": "有学龄前儿童", "car": "是", "home": "40000-59999",
                    "phone": "5001-6000", "edu": "本科", "prefs": ["美食", "娱乐休闲", "教育学校"], "freq": "9-10次",
                    "pos": "七宝商圈核心", "target": "家庭客群、周边居民",
                    "biz": "家庭餐饮与亲子娱乐", "spend": "中等消费，客单价120-200元",
                    "summary": "人气家庭购物中心", "kw": ["家庭", "亲子", "儿童"]},
        "brands": {"餐饮": ["海底捞", "西贝莜面村", "外婆家", "肯德基", "麦当劳", "星巴克", "喜茶", "蜜雪冰城"],
                   "娱乐服务": ["万达影城", "大玩家", "泡泡玛特", "威尔士健身", "金逸影城"],
                   "服装": ["优衣库", "UR", "太平鸟", "海澜之家"], "运动": ["耐克", "李宁", "安踏"]},
    },
    {
        "name": "上海青浦吾悦广场", "district": "青浦区", "address": "青浦区淀山湖大道851号",
        "lat": 31.150, "lng": 121.122, "area": 72000, "floors": 4, "open": "2017-01-21",
        "score_market": 64, "traffic": 8000, "sales": 160000,
        "persona": {"consume": "次低", "age": "36-40", "kids": "无子女", "car": "否", "home": "20000-39999",
                    "phone": "<5000", "edu": "大专", "prefs": ["美食", "娱乐休闲"], "freq": "4-6次",
                    "pos": "青浦城区居住板块", "target": "本地居民",
                    "biz": "大众零售，品牌较少", "spend": "大众消费，客单价80-130元",
                    "summary": "品牌较少的蓝海商场", "kw": ["大众", "蓝海"]},
        "brands": {"餐饮": ["肯德基", "麦当劳", "蜜雪冰城"], "服装": ["海澜之家"], "娱乐服务": ["金逸影城", "星聚会KTV"]},
    },
    {
        "name": "上海临港新城里", "district": "浦东新区", "address": "浦东新区南汇新城镇申港大道200号",
        "lat": 30.894, "lng": 121.914, "area": 65000, "floors": 4, "open": "2019-06-06",
        "score_market": 58, "traffic": 3500, "sales": 80000,
        "persona": {"consume": "低", "age": "31-35", "kids": "无子女", "car": "否", "home": "<20000",
                    "phone": "<5000", "edu": "大专", "prefs": ["美食", "娱乐休闲"], "freq": "1-3次",
                    "pos": "临港新城，远郊新兴区域", "target": "新城居民、产业工人",
                    "biz": "基础零售，品牌较少", "spend": "大众消费，客单价60-100元",
                    "summary": "远郊蓝海商场", "kw": ["大众", "蓝海", "远郊"]},
        "brands": {"餐饮": ["肯德基", "蜜雪冰城"], "娱乐服务": ["金逸影城"]},
    },
]


def _make_row(spec):
    p = spec["persona"]
    brand_count, counts, density, bd = _brand_payload(spec["brands"])

    position_text = _position_text(p)
    profile_json = _profile_json(p)
    profile_text = _profile_text(p)
    embedding_text = (
        f"{spec['name']}，上海城区。{position_text}。客群画像：{profile_text}"
    )

    # 五维评分：以 score_market 为锚，其余四维做小幅派生（仅演示，不影响检索主路径）
    sm = spec["score_market"]
    return {
        "mall_id": f"SYN{spec['idx']:04d}",
        "mall_name": spec["name"],
        "manage_group": "示例商业集团",
        "unit_type": "购物中心",
        "address": spec["address"],
        "province": "上海",
        "city": "上海",
        "district": spec["district"],
        "latitude": spec["lat"],
        "longitude": spec["lng"],
        "mall_area": spec["area"],
        "floor_count": spec["floors"],
        "open_date": spec["open"],
        "operation_status": "运营中",
        "city_rank_name": "一线",
        "position_text": position_text,
        "score_area": min(99, sm + 2),
        "score_consume": min(99, sm + 4),
        "score_market": sm,
        "score_operation": max(60, sm - 5),
        "score_pop": min(99, round(55 + spec["traffic"] / 1500)),
        "embedding_text": embedding_text,
        "customer_profile_json": json.dumps(profile_json, ensure_ascii=False),
        "profile_text": profile_text,
        "brand_count": brand_count,
        "brand_data_json": json.dumps(bd, ensure_ascii=False),
        **{f"{c}_brand_count": counts[c] for c in CATS},
        **{f"{c}_density": density[c] for c in CATS},
        "traffic_daily": spec["traffic"],
        "avg_daily_sales": spec["sales"],
    }


def main():
    for i, m in enumerate(MALLS, start=1):
        m["idx"] = i
    rows = [_make_row(m) for m in MALLS]

    # 列顺序与真实宽表表头一致（40 列）
    columns = [
        "mall_id", "mall_name", "manage_group", "unit_type", "address", "province",
        "city", "district", "latitude", "longitude", "mall_area", "floor_count",
        "open_date", "operation_status", "city_rank_name", "position_text",
        "score_area", "score_consume", "score_market", "score_operation", "score_pop",
        "embedding_text", "customer_profile_json", "profile_text", "brand_count",
        "brand_data_json", "服装_brand_count", "服装_density", "餐饮_brand_count",
        "餐饮_density", "娱乐服务_brand_count", "娱乐服务_density", "珠宝_brand_count",
        "珠宝_density", "护肤化妆品_brand_count", "护肤化妆品_density", "运动_brand_count",
        "运动_density", "traffic_daily", "avg_daily_sales",
    ]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"✅ 已生成 {OUT}：{len(df)} 家虚构商场 × {len(columns)} 列")
    print(f"   下一步：copy /Y {OUT} mall_wide_table_shanghai_final_v3.csv")
    print(f"   再跑：python create_vectordb_v6.py")


if __name__ == "__main__":
    main()
