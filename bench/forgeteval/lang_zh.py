"""ForgetEval — Chinese (zh) template module.

Parallel structure to the English templates in generate.py: same five
families, same four sub-templates per family, same case anatomy.  Only
the entity pools and phrasing are native Chinese.

The multilingual embedder
(paraphrase-multilingual-MiniLM-L12-v2, 384-dim) is required at
evaluation time; run.py auto-selects it for --lang zh|ja.
"""
from __future__ import annotations

import random

from bench.forgeteval.generate import GeneratedCase, _scatter


# ─── entity pools ──────────────────────────────────────────────────

NAMES = [
    "王芳", "李伟", "张敏", "刘洋", "陈强", "赵静", "杨磊", "黄丽",
    "周勇", "吴娟", "徐涛", "孙娜", "朱明", "胡英", "林杰", "何丹",
    "高翔", "罗梅", "郑超", "梁雪", "谢辉", "韩静", "唐磊", "冯娟",
    "邓伟", "曹丽", "彭明", "曾敏", "蔡杰", "田芳",
]

COMPANIES = [
    "阿里巴巴", "字节跳动", "腾讯", "美团", "京东", "百度", "滴滴",
    "拼多多", "小米", "华为", "网易", "携程", "哔哩哔哩", "快手", "知乎",
    "蚂蚁集团",
]

CITIES = [
    "北京", "上海", "深圳", "杭州", "广州", "成都", "西安", "重庆",
    "苏州", "武汉", "南京", "天津", "厦门", "长沙", "青岛",
]

COLORS = [
    "蓝色", "绿色", "红色", "黑色", "白色", "紫色", "黄色", "粉色",
    "灰色", "棕色", "橙色", "青色",
]

DIETS = [
    "素食", "全素", "海鲜素", "生酮", "地中海", "无麸质", "杂食", "低碳水",
]

THEMES = ["深色", "浅色", "棕褐", "单色"]

LANGUAGES = [
    "英语", "法语", "西班牙语", "日语", "韩语", "德语",
    "意大利语", "葡萄牙语", "俄语", "阿拉伯语",
]

BOOKS = [
    "三体", "活着", "围城", "边城", "白鹿原", "茶馆", "红高粱", "京华烟云",
]

HOBBIES = [
    "围棋", "跑步", "油画", "吉他", "骑行", "摄影",
    "陶艺", "攀岩", "编织", "滑雪",
]

STREETS = [
    "长安街", "中山路", "解放路", "健康路", "朝阳街",
    "复兴路", "文化路", "胜利街",
]

EMAIL_DOMAINS = ["example.com", "mail.cn", "post.org", "service.io"]
# Use ASCII-Latin prefixes for emails so identifier-precision tests work
# the same way as in English (purge sees the email as a single token cluster).
EMAIL_PREFIXES = [
    "alice", "bob", "carol", "dan", "eve", "frank",
    "grace", "ivan", "judy", "ken", "lily", "max",
    "nina", "owen", "paul",
]

CONDITIONS = [
    "高血压", "糖尿病", "哮喘", "偏头痛", "焦虑症", "心律不齐", "贫血", "银屑病",
]

FOODS = [
    "麻辣火锅", "北京烤鸭", "水饺", "麻婆豆腐", "宫保鸡丁",
    "小笼包", "兰州拉面", "鱼香肉丝", "糖醋里脊",
]

# Office-life trivia; deliberately disjoint from every entity pool above.
FILLER = [
    "会议室搬到了4楼。",
    "咖啡机周二维修。",
    "季度回顾邮件已发送。",
    "办公时间是9点到5点,周五弹性。",
    "停车证在前台办理。",
    "图书馆周末晚9点关门。",
    "年假申请需要提前两周。",
    "团建午餐每隔一周一次。",
    "VPN证书将在三月续期。",
    "消防演习定在下个月。",
    "打印机正在维护中。",
    "年会安排在12月15日。",
    "本季度的咖啡预算增加了。",
    "自行车停放区上周重新粉刷。",
    "全员会议从周四改到周三。",
    "立式办公桌可以申请。",
    "食堂改用可降解餐具。",
    "访客证在当天结束失效。",
    "楼宇空调系统在四月检修。",
    "零食柜补货日是周一。",
    "灭火器检查已过期。",
    "回收站每周清理两次。",
    "会议室白板上周清洁过。",
    "卫生间洗手液已更换。",
    "邮件室4点关门。",
    "晚间7点后需要刷卡进楼。",
    "窗户清洁每月第三个周五。",
    "楼梯间的饮水机已更换。",
    "电梯检修昨天完成。",
    "失物招领在前台。",
]


def _s(facts, rng):
    """Local helper: scatter using zh FILLER + the runner's _DISTRACTORS."""
    return _scatter(facts, rng, fillers=FILLER)


# ─── SUPERSESSION ──────────────────────────────────────────────────

def _supersession_job(rng):
    name = rng.choice(NAMES)
    old, new = rng.sample(COMPANIES, 2)
    return GeneratedCase(
        id="", family="supersession",
        setup_facts=_s([f"{name}在{old}工作。"], rng),
        mutations=[("supersede", f"{name}的工作单位",
                    f"{name}现在在{new}工作,2026年入职。")],
        final_query=f"{name}在哪里工作?",
        must_contain=[new], must_not_contain=[old],
    )


def _supersession_theme(rng):
    name = rng.choice(NAMES)
    old, new = rng.sample(THEMES, 2)
    return GeneratedCase(
        id="", family="supersession",
        setup_facts=_s([f"{name}在所有应用中偏好{old}模式。"], rng),
        mutations=[("supersede", f"{name}的主题偏好",
                    f"{name}已永久切换到{new}模式。")],
        final_query=f"{name}使用什么主题?",
        must_contain=[new], must_not_contain=[old],
    )


def _supersession_diet(rng):
    name = rng.choice(NAMES)
    old, new = rng.sample(DIETS, 2)
    return GeneratedCase(
        id="", family="supersession",
        setup_facts=_s([f"{name}的饮食是{old},严格遵守。"], rng),
        mutations=[("supersede", f"{name}的饮食",
                    f"{name}今年改成了{new}饮食。")],
        final_query=f"{name}的饮食是什么?",
        must_contain=[new], must_not_contain=[old],
    )


def _supersession_long_form(rng):
    name = rng.choice(NAMES)
    old, new = rng.sample(COMPANIES, 2)
    city = rng.choice(CITIES)
    lang = rng.choice(LANGUAGES)
    hobby = rng.choice(HOBBIES)
    return GeneratedCase(
        id="", family="supersession",
        setup_facts=_s([
            f"{name}是{old}的高级工程师,常驻{city},精通{lang},"
            f"周末喜欢{hobby}。"
        ], rng),
        mutations=[("supersede", f"{name}的工作单位",
                    f"{name}是{new}的高级工程师,常驻{city},精通{lang},"
                    f"周末喜欢{hobby}。")],
        final_query=f"{name}在哪里工作?",
        must_contain=[new], must_not_contain=[old],
    )


SUPERSESSION = [_supersession_job, _supersession_theme,
                _supersession_diet, _supersession_long_form]


# ─── DECAY ─────────────────────────────────────────────────────────

def _decay_otp(rng):
    code = "".join(rng.choices("0123456789", k=6))
    name = rng.choice(NAMES)
    return GeneratedCase(
        id="", family="decay",
        setup_facts=_s([
            f"{name}的会话OTP是{code}。",
            f"{name}下午喜欢喝冰咖啡。",
        ], rng),
        mutations=[("release", f"OTP登录会话码{code}")],
        final_query=f"{name}有没有一次性密码?",
        must_contain=[], must_not_contain=[code],
    )


def _decay_verification(rng):
    code = "".join(rng.choices("0123456789", k=5))
    return GeneratedCase(
        id="", family="decay",
        setup_facts=_s([f"临时验证码:{code}。"], rng),
        mutations=[("release", f"验证码{code}")],
        final_query="验证码是什么?",
        must_contain=[], must_not_contain=[code],
    )


def _decay_flight_cancelled(rng):
    name = rng.choice(NAMES)
    dest = rng.choice(CITIES)
    return GeneratedCase(
        id="", family="decay",
        setup_facts=_s([f"{name}下周五订了去{dest}的机票。"], rng),
        mutations=[("release", f"飞往{dest}的航班")],
        final_query=f"{name}的出行计划是什么?",
        must_contain=[], must_not_contain=[dest],
    )


def _decay_one_of_many(rng):
    services = rng.sample(
        ["AWS", "GitHub", "Stripe", "Notion", "Linear", "Vercel", "Figma"], 4
    )
    codes = ["".join(rng.choices("0123456789", k=6)) for _ in range(4)]
    target_service = services[0]
    target_code = codes[0]
    facts = [f"{svc}登录的OTP:{code}。" for svc, code in zip(services, codes)]
    return GeneratedCase(
        id="", family="decay",
        setup_facts=_s(facts, rng),
        mutations=[("release", f"OTP {target_service} {target_code}")],
        final_query=f"{target_service}的OTP是什么?",
        must_contain=[], must_not_contain=[target_code],
    )


DECAY = [_decay_otp, _decay_verification, _decay_flight_cancelled,
         _decay_one_of_many]


# ─── AMNESIA ───────────────────────────────────────────────────────

def _amnesia_person(rng):
    target, other = rng.sample(NAMES, 2)
    f1, f2 = rng.sample(FOODS, 2)
    return GeneratedCase(
        id="", family="amnesia",
        setup_facts=_s([
            f"{target}喜欢{f1}。",
            f"{other}喜欢{f2}。",
        ], rng),
        mutations=[("release", f"{target} {f1} 喜好")],
        final_query="大家喜欢吃什么?",
        must_contain=[other], must_not_contain=[target],
    )


def _amnesia_multi(rng):
    target, other = rng.sample(NAMES, 2)
    job_t, job_o = rng.sample(COMPANIES, 2)
    city = rng.choice(CITIES)
    lang = rng.choice(LANGUAGES)
    return GeneratedCase(
        id="", family="amnesia",
        setup_facts=_s([
            f"{target}在{job_t}工作。",
            f"{target}住在{city}。",
            f"{target}会说{lang}。",
            f"{other}在{job_o}工作。",
        ], rng),
        mutations=[("release", f"{target}的个人信息")],
        final_query="讲讲这些人。",
        must_contain=[other], must_not_contain=[target],
    )


def _amnesia_topic(rng):
    name = rng.choice(NAMES)
    book = rng.choice(BOOKS)
    hobby = rng.choice(HOBBIES)
    return GeneratedCase(
        id="", family="amnesia",
        setup_facts=_s([
            f"{name}读完了《{book}》。",
            f"{name}十月份完成了第一次{hobby}。",
        ], rng),
        mutations=[("release", f"读书 {book}")],
        final_query=f"{name}最近完成了什么?",
        must_contain=[hobby], must_not_contain=[book],
    )


def _amnesia_many_facts(rng):
    target, other = rng.sample(NAMES, 2)
    job_t, job_o = rng.sample(COMPANIES, 2)
    city = rng.choice(CITIES)
    lang = rng.choice(LANGUAGES)
    hobby = rng.choice(HOBBIES)
    book = rng.choice(BOOKS)
    return GeneratedCase(
        id="", family="amnesia",
        setup_facts=_s([
            f"{target}在{job_t}工作。",
            f"{target}住在{city}。",
            f"{target}会说{lang}。",
            f"{target}喜欢{hobby}。",
            f"{target}上个月读完了《{book}》。",
            f"{other}在{job_o}工作。",
        ], rng),
        mutations=[("release", f"{target}的个人信息")],
        final_query="讲讲这些人。",
        must_contain=[other], must_not_contain=[target],
    )


AMNESIA = [_amnesia_person, _amnesia_multi, _amnesia_topic, _amnesia_many_facts]


# ─── PURGE ─────────────────────────────────────────────────────────

def _purge_api_key(rng):
    suffix = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=10))
    key = f"sk-{suffix}"
    return GeneratedCase(
        id="", family="purge",
        setup_facts=_s([f"运维使用的API key:{key}-secret。"], rng),
        mutations=[("purge", f"API key {key}")],
        final_query="API key是什么?",
        must_contain=[], must_not_contain=[key],
    )


def _purge_phi(rng):
    target = rng.choice(NAMES)
    year = rng.randint(1950, 2005)
    cond = rng.choice(CONDITIONS)
    return GeneratedCase(
        id="", family="purge",
        setup_facts=_s([
            f"患者{target},出生于{year}-04-12,确诊{cond}。",
        ], rng),
        mutations=[("purge", f"{target}的病历")],
        final_query=f"讲讲{target}。",
        must_contain=[], must_not_contain=[target, cond],
    )


def _purge_gdpr(rng):
    tgt_prefix, oth_prefix = rng.sample(EMAIL_PREFIXES, 2)
    domain = rng.choice(EMAIL_DOMAINS)
    tgt = f"{tgt_prefix}@{domain}"
    oth = f"{oth_prefix}@{domain}"
    return GeneratedCase(
        id="", family="purge",
        setup_facts=_s([
            f"客户{tgt}上周购买了3件商品。",
            f"客户{oth}昨天下了订单。",
        ], rng),
        mutations=[("purge", f"{tgt}的客户记录")],
        final_query=f"显示客户{tgt}的数据。",
        must_contain=[oth], must_not_contain=[tgt],
    )


def _purge_many_similar(rng):
    suffixes = ["".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=10))
                for _ in range(5)]
    target = suffixes[0]
    facts = [f"内部服务的API key:sk-{s}-secret。" for s in suffixes]
    return GeneratedCase(
        id="", family="purge",
        setup_facts=_s(facts, rng),
        mutations=[("purge", f"API key sk-{target}")],
        final_query=f"sk-{target}是什么?",
        must_contain=[], must_not_contain=[f"sk-{target}"],
    )


PURGE = [_purge_api_key, _purge_phi, _purge_gdpr, _purge_many_similar]


# ─── DRIFT ─────────────────────────────────────────────────────────

def _drift_jobs(rng):
    name = rng.choice(NAMES)
    c1, c2, c3 = rng.sample(COMPANIES, 3)
    return GeneratedCase(
        id="", family="drift",
        setup_facts=_s([f"{name}2020年加入{c1}。"], rng),
        mutations=[
            ("supersede", f"{name}的工作", f"{name}2022年跳槽到{c2}。"),
            ("supersede", f"{name}的工作", f"{name}从2025年起在{c3}。"),
        ],
        final_query=f"{name}在哪里工作?",
        must_contain=[c3], must_not_contain=[c1, c2],
    )


def _drift_address(rng):
    name = rng.choice(NAMES)
    s1, s2, s3 = rng.sample(STREETS, 3)
    n1, n2, n3 = (rng.randint(1, 99) for _ in range(3))
    return GeneratedCase(
        id="", family="drift",
        setup_facts=_s([f"{name}住在{s1}{n1}号。"], rng),
        mutations=[
            ("supersede", f"{name}的地址", f"{name}搬到了{s2}{n2}号。"),
            ("supersede", f"{name}的地址", f"{name}现在住在{s3}{n3}号。"),
        ],
        final_query=f"{name}的地址在哪?",
        must_contain=[s3], must_not_contain=[s1, s2],
    )


def _drift_color(rng):
    name = rng.choice(NAMES)
    c1, c2, c3 = rng.sample(COLORS, 3)
    return GeneratedCase(
        id="", family="drift",
        setup_facts=_s([f"{name}最喜欢的颜色是{c1}。"], rng),
        mutations=[
            ("supersede", f"{name}最喜欢的颜色", f"{name}改成了{c2}。"),
            ("supersede", f"{name}最喜欢的颜色", f"现在{name}最喜欢{c3}。"),
        ],
        final_query=f"{name}最喜欢什么颜色?",
        must_contain=[c3], must_not_contain=[c1, c2],
    )


def _drift_long_chain(rng):
    name = rng.choice(NAMES)
    companies = rng.sample(COMPANIES, 6)
    years = [2018, 2020, 2021, 2023, 2025]
    setup = [f"{name}2017年加入{companies[0]}。"]
    muts = [("supersede", f"{name}的工作",
             f"{name}{y}年跳槽到{c}。")
            for c, y in zip(companies[1:], years)]
    final = companies[-1]
    return GeneratedCase(
        id="", family="drift",
        setup_facts=_s(setup, rng),
        mutations=muts,
        final_query=f"{name}在哪里工作?",
        must_contain=[final],
        must_not_contain=list(companies[:-1]),
    )


DRIFT = [_drift_jobs, _drift_address, _drift_color, _drift_long_chain]


# ─── manifest ──────────────────────────────────────────────────────

FAMILIES: dict[str, list] = {
    "supersession": SUPERSESSION,
    "decay": DECAY,
    "amnesia": AMNESIA,
    "purge": PURGE,
    "drift": DRIFT,
}
