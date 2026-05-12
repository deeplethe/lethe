"""ForgetEval — Japanese (ja) template module.

Parallel structure to the English templates in generate.py: same five
families, same four sub-templates per family, same case anatomy.  Only
the entity pools and phrasing are native Japanese.

Requires a multilingual embedder
(paraphrase-multilingual-MiniLM-L12-v2 is the default for --lang ja).
"""
from __future__ import annotations

import random

from bench.forgeteval.generate import GeneratedCase, _scatter


# ─── entity pools ──────────────────────────────────────────────────

NAMES = [
    "田中", "山本", "佐藤", "鈴木", "高橋", "渡辺", "中村", "小林",
    "加藤", "吉田", "山田", "木村", "斎藤", "清水", "山口", "池田",
    "阿部", "橋本", "石川", "中島", "前田", "藤田", "後藤", "岡田",
    "小川", "長谷川", "村上", "近藤", "石井",
]

COMPANIES = [
    "ソニー", "トヨタ", "任天堂", "メルカリ", "ホンダ", "楽天",
    "リクルート", "サイバーエージェント", "ソフトバンク", "パナソニック",
    "日産", "富士通", "ヤフー", "GMO", "グリー", "DeNA",
]

CITIES = [
    "東京", "大阪", "京都", "札幌", "福岡", "横浜", "名古屋", "神戸",
    "仙台", "広島", "那覇", "金沢", "高松", "鎌倉", "奈良",
]

COLORS = [
    "青色", "緑色", "赤色", "黒色", "白色", "紫色", "黄色",
    "ピンク", "灰色", "茶色", "オレンジ色", "ターコイズ",
]

DIETS = [
    "ベジタリアン", "ヴィーガン", "ペスカタリアン", "ケトジェニック",
    "地中海食", "グルテンフリー", "肉食", "雑食",
]

THEMES = ["ダーク", "ライト", "セピア", "モノクロ"]

LANGUAGES = [
    "英語", "フランス語", "スペイン語", "中国語", "韓国語",
    "ドイツ語", "イタリア語", "ポルトガル語", "ロシア語", "アラビア語",
]

BOOKS = [
    "ノルウェイの森", "雪国", "風立ちぬ", "走れメロス",
    "こころ", "1Q84", "海辺のカフカ", "蹴りたい背中",
]

HOBBIES = [
    "将棋", "ランニング", "絵画", "ギター", "サイクリング",
    "写真", "陶芸", "クライミング", "編み物", "スキー",
]

STREETS = [
    "銀座", "表参道", "渋谷", "新宿", "池袋",
    "秋葉原", "六本木", "麻布",
]

EMAIL_DOMAINS = ["example.com", "mail.jp", "post.org", "service.io"]
EMAIL_PREFIXES = [
    "alice", "bob", "carol", "dan", "eve", "frank",
    "grace", "ivan", "judy", "ken", "lily", "max",
    "nina", "owen", "paul",
]

CONDITIONS = [
    "高血圧", "糖尿病", "喘息", "偏頭痛", "不安症",
    "不整脈", "貧血", "乾癬",
]

FOODS = [
    "寿司", "ラーメン", "天ぷら", "うどん", "そば",
    "餃子", "焼き鳥", "牛丼", "カレーライス", "おにぎり",
]

FILLER = [
    "会議室は4階に移動しました。",
    "コーヒーマシンは火曜日にメンテナンス中です。",
    "四半期レビューのスケジュールが先週メールで送られました。",
    "営業時間は9時から5時、金曜日はフレックスです。",
    "駐車許可証は受付で発行されます。",
    "図書室は週末21時に閉まります。",
    "有給休暇の申請は2週間前までに必要です。",
    "社員ランチは隔週金曜日に行われます。",
    "VPN証明書は3月に更新されます。",
    "避難訓練が来月予定されています。",
    "プリンターは現在メンテナンス中です。",
    "忘年会は12月15日に開催されます。",
    "今四半期のコーヒー予算が増額されました。",
    "自転車置き場が先週塗り直されました。",
    "全社ミーティングが木曜から水曜に変更されました。",
    "スタンディングデスクは申請可能です。",
    "社員食堂は再生可能カトラリーに切り替えました。",
    "ビジターパスは当日中に失効します。",
    "建物の空調は4月に点検されます。",
    "スナック棚の補充日は月曜日です。",
    "消火器の点検期限が切れています。",
    "リサイクル回収は週2回です。",
    "会議室のホワイトボードは先週清掃されました。",
    "洗面所のソープディスペンサーが交換されました。",
    "郵便室は16時に閉まります。",
    "夜7時以降は入館にバッジが必要です。",
    "窓の清掃は毎月第3金曜日に行われます。",
    "階段の給水機が交換されました。",
    "エレベーターのメンテナンスは昨日完了しました。",
    "落とし物は受付にあります。",
]


def _s(facts, rng):
    return _scatter(facts, rng, fillers=FILLER)


# ─── SUPERSESSION ──────────────────────────────────────────────────

def _supersession_job(rng):
    name = rng.choice(NAMES)
    old, new = rng.sample(COMPANIES, 2)
    return GeneratedCase(
        id="", family="supersession",
        setup_facts=_s([f"{name}は{old}で働いている。"], rng),
        mutations=[("supersede", f"{name}の勤務先",
                    f"{name}は2026年から{new}で働いている。")],
        final_query=f"{name}はどこで働いていますか?",
        must_contain=[new], must_not_contain=[old],
    )


def _supersession_theme(rng):
    name = rng.choice(NAMES)
    old, new = rng.sample(THEMES, 2)
    return GeneratedCase(
        id="", family="supersession",
        setup_facts=_s([f"{name}は全てのアプリで{old}モードを好む。"], rng),
        mutations=[("supersede", f"{name}のテーマ設定",
                    f"{name}は{new}モードに完全に切り替えた。")],
        final_query=f"{name}が使っているテーマは?",
        must_contain=[new], must_not_contain=[old],
    )


def _supersession_diet(rng):
    name = rng.choice(NAMES)
    old, new = rng.sample(DIETS, 2)
    return GeneratedCase(
        id="", family="supersession",
        setup_facts=_s([f"{name}は{old}を厳格に実践している。"], rng),
        mutations=[("supersede", f"{name}の食生活",
                    f"{name}は今年から{new}に変更した。")],
        final_query=f"{name}の食生活は?",
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
            f"{name}は{old}のシニアエンジニアで、{city}在住、"
            f"{lang}が堪能、週末は{hobby}を楽しむ。"
        ], rng),
        mutations=[("supersede", f"{name}の勤務先",
                    f"{name}は{new}のシニアエンジニアで、{city}在住、"
                    f"{lang}が堪能、週末は{hobby}を楽しむ。")],
        final_query=f"{name}はどこで働いていますか?",
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
            f"{name}のセッションOTPは{code}。",
            f"{name}は午後にアイスコーヒーが好き。",
        ], rng),
        mutations=[("release", f"OTPログインセッションコード{code}")],
        final_query=f"{name}にワンタイムパスワードはありますか?",
        must_contain=[], must_not_contain=[code],
    )


def _decay_verification(rng):
    code = "".join(rng.choices("0123456789", k=5))
    return GeneratedCase(
        id="", family="decay",
        setup_facts=_s([f"一時認証コード:{code}。"], rng),
        mutations=[("release", f"認証コード{code}")],
        final_query="認証コードは何ですか?",
        must_contain=[], must_not_contain=[code],
    )


def _decay_flight_cancelled(rng):
    name = rng.choice(NAMES)
    dest = rng.choice(CITIES)
    return GeneratedCase(
        id="", family="decay",
        setup_facts=_s([f"{name}は来週金曜日に{dest}行きの航空券を予約した。"], rng),
        mutations=[("release", f"{dest}行きのフライト")],
        final_query=f"{name}の旅行予定は?",
        must_contain=[], must_not_contain=[dest],
    )


def _decay_one_of_many(rng):
    services = rng.sample(
        ["AWS", "GitHub", "Stripe", "Notion", "Linear", "Vercel", "Figma"], 4
    )
    codes = ["".join(rng.choices("0123456789", k=6)) for _ in range(4)]
    target_service = services[0]
    target_code = codes[0]
    facts = [f"{svc}ログイン用OTP:{code}。" for svc, code in zip(services, codes)]
    return GeneratedCase(
        id="", family="decay",
        setup_facts=_s(facts, rng),
        mutations=[("release", f"OTP {target_service} {target_code}")],
        final_query=f"{target_service}のOTPは?",
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
            f"{target}は{f1}が好き。",
            f"{other}は{f2}が好き。",
        ], rng),
        mutations=[("release", f"{target} {f1} 嗜好")],
        final_query="みんなの好物は?",
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
            f"{target}は{job_t}で働いている。",
            f"{target}は{city}に住んでいる。",
            f"{target}は{lang}を話す。",
            f"{other}は{job_o}で働いている。",
        ], rng),
        mutations=[("release", f"{target}のプロフィール情報")],
        final_query="この人たちについて教えて。",
        must_contain=[other], must_not_contain=[target],
    )


def _amnesia_topic(rng):
    name = rng.choice(NAMES)
    book = rng.choice(BOOKS)
    hobby = rng.choice(HOBBIES)
    return GeneratedCase(
        id="", family="amnesia",
        setup_facts=_s([
            f"{name}は『{book}』を読み終えた。",
            f"{name}は10月に初めての{hobby}を完了した。",
        ], rng),
        mutations=[("release", f"読書 {book}")],
        final_query=f"{name}が最近達成したことは?",
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
            f"{target}は{job_t}で働いている。",
            f"{target}は{city}に住んでいる。",
            f"{target}は{lang}を話す。",
            f"{target}は{hobby}が好き。",
            f"{target}は先月『{book}』を読み終えた。",
            f"{other}は{job_o}で働いている。",
        ], rng),
        mutations=[("release", f"{target}のプロフィール情報")],
        final_query="この人たちについて教えて。",
        must_contain=[other], must_not_contain=[target],
    )


AMNESIA = [_amnesia_person, _amnesia_multi, _amnesia_topic, _amnesia_many_facts]


# ─── PURGE ─────────────────────────────────────────────────────────

def _purge_api_key(rng):
    suffix = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=10))
    key = f"sk-{suffix}"
    return GeneratedCase(
        id="", family="purge",
        setup_facts=_s([f"運用チームのAPIキー:{key}-secret。"], rng),
        mutations=[("purge", f"APIキー {key}")],
        final_query="APIキーは何ですか?",
        must_contain=[], must_not_contain=[key],
    )


def _purge_phi(rng):
    target = rng.choice(NAMES)
    year = rng.randint(1950, 2005)
    cond = rng.choice(CONDITIONS)
    return GeneratedCase(
        id="", family="purge",
        setup_facts=_s([
            f"患者{target}、生年月日{year}-04-12、{cond}と診断。",
        ], rng),
        mutations=[("purge", f"{target}の医療記録")],
        final_query=f"{target}について教えて。",
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
            f"顧客{tgt}は先週3点購入した。",
            f"顧客{oth}は昨日注文した。",
        ], rng),
        mutations=[("purge", f"{tgt}の顧客記録")],
        final_query=f"顧客{tgt}のデータを表示。",
        must_contain=[oth], must_not_contain=[tgt],
    )


def _purge_many_similar(rng):
    suffixes = ["".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=10))
                for _ in range(5)]
    target = suffixes[0]
    facts = [f"内部サービスのAPIキー:sk-{s}-secret。" for s in suffixes]
    return GeneratedCase(
        id="", family="purge",
        setup_facts=_s(facts, rng),
        mutations=[("purge", f"APIキー sk-{target}")],
        final_query=f"sk-{target}は何ですか?",
        must_contain=[], must_not_contain=[f"sk-{target}"],
    )


PURGE = [_purge_api_key, _purge_phi, _purge_gdpr, _purge_many_similar]


# ─── DRIFT ─────────────────────────────────────────────────────────

def _drift_jobs(rng):
    name = rng.choice(NAMES)
    c1, c2, c3 = rng.sample(COMPANIES, 3)
    return GeneratedCase(
        id="", family="drift",
        setup_facts=_s([f"{name}は2020年に{c1}に入社した。"], rng),
        mutations=[
            ("supersede", f"{name}の勤務先", f"{name}は2022年に{c2}へ転職した。"),
            ("supersede", f"{name}の勤務先", f"{name}は2025年から{c3}にいる。"),
        ],
        final_query=f"{name}はどこで働いていますか?",
        must_contain=[c3], must_not_contain=[c1, c2],
    )


def _drift_address(rng):
    name = rng.choice(NAMES)
    s1, s2, s3 = rng.sample(STREETS, 3)
    n1, n2, n3 = (rng.randint(1, 99) for _ in range(3))
    return GeneratedCase(
        id="", family="drift",
        setup_facts=_s([f"{name}は{s1}{n1}番地に住んでいる。"], rng),
        mutations=[
            ("supersede", f"{name}の住所", f"{name}は{s2}{n2}番地へ引越した。"),
            ("supersede", f"{name}の住所", f"{name}は今{s3}{n3}番地に住んでいる。"),
        ],
        final_query=f"{name}の住所は?",
        must_contain=[s3], must_not_contain=[s1, s2],
    )


def _drift_color(rng):
    name = rng.choice(NAMES)
    c1, c2, c3 = rng.sample(COLORS, 3)
    return GeneratedCase(
        id="", family="drift",
        setup_facts=_s([f"{name}の好きな色は{c1}。"], rng),
        mutations=[
            ("supersede", f"{name}の好きな色", f"{name}は{c2}に変えた。"),
            ("supersede", f"{name}の好きな色", f"今{name}が好きなのは{c3}。"),
        ],
        final_query=f"{name}の好きな色は?",
        must_contain=[c3], must_not_contain=[c1, c2],
    )


def _drift_long_chain(rng):
    name = rng.choice(NAMES)
    companies = rng.sample(COMPANIES, 6)
    years = [2018, 2020, 2021, 2023, 2025]
    setup = [f"{name}は2017年に{companies[0]}に入社した。"]
    muts = [("supersede", f"{name}の勤務先",
             f"{name}は{y}年に{c}へ転職した。")
            for c, y in zip(companies[1:], years)]
    final = companies[-1]
    return GeneratedCase(
        id="", family="drift",
        setup_facts=_s(setup, rng),
        mutations=muts,
        final_query=f"{name}はどこで働いていますか?",
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
