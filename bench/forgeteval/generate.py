"""ForgetEval — template-based test generation.

The hand-crafted 15 in tests.py are a smoke set; this module produces
hundreds of cases from compact templates + entity pools.  Each case is
deterministic given (template, seed) and includes background distractor
facts so retrieval isn't trivial.

Design rule (same as the rest of Lethe): no magic constants, no regex
detectors. Each generator is a pure function rng -> GeneratedCase.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


# ─── entity pools ──────────────────────────────────────────────────

NAMES = [
    "Alice", "Bob", "Charlie", "Dana", "Eve", "Frank", "Grace", "Hannah",
    "Ivan", "Judy", "Kim", "Liam", "Mia", "Noah", "Olivia", "Peter",
    "Quinn", "Riya", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zoe", "Aiden", "Bella", "Caleb", "Diana",
]

COMPANIES = [
    "Google", "Anthropic", "Meta", "OpenAI", "Stripe", "Notion", "Apple",
    "Microsoft", "Amazon", "Netflix", "Spotify", "Shopify", "Airbnb",
    "Uber", "LinkedIn", "Figma", "Snowflake", "Databricks",
]

CITIES = [
    "Berlin", "Tokyo", "Paris", "London", "Madrid", "Lima", "Boston",
    "Seattle", "Vienna", "Sydney", "Oslo", "Toronto", "Singapore",
    "Mumbai", "Dubai", "Cairo",
]

COLORS = [
    "blue", "green", "crimson", "black", "white", "purple", "amber",
    "yellow", "magenta", "brown", "silver", "teal",
]

DIETS = [
    "vegetarian", "vegan", "pescatarian", "keto", "paleo", "gluten-free",
    "carnivore", "Mediterranean",
]

THEMES = ["dark", "light", "sepia", "monochrome"]

LANGUAGES = [
    "Portuguese", "Mandarin", "French", "Spanish", "Japanese", "German",
    "Korean", "Arabic", "Italian", "Russian",
]

BOOKS = [
    "Dune", "Foundation", "Neuromancer", "Snow Crash", "Anathem",
    "Permutation City", "Hyperion", "Stranger in a Strange Land",
]

HOBBIES = [
    "chess", "running", "painting", "guitar", "cycling", "photography",
    "pottery", "rock climbing", "knitting", "skiing",
]

STREETS = ["Elm", "Oak", "Pine", "Maple", "Birch", "Cedar", "Willow", "Aspen"]

EMAIL_DOMAINS = ["example.com", "mailbox.io", "acme.org", "post.dev"]

CONDITIONS = [
    "hypertension", "diabetes", "asthma", "migraine", "anxiety",
    "arrhythmia", "anemia", "psoriasis",
]

# Filler facts: unrelated to any topic above so they don't bleed into
# must_contain/must_not_contain checks.  Used to make retrieval harder.
FILLER = [
    "The conference room moved to the 4th floor.",
    "Coffee machine is offline on Tuesdays.",
    "Quarterly review schedule was emailed last week.",
    "Office hours are 9 to 5 with flexible Fridays.",
    "Parking validation is at the front desk.",
    "The library closes at 9pm on weekdays.",
    "Annual leave requests need approval two weeks ahead.",
    "Catered lunch happens every other Friday.",
    "VPN renewal is due in March.",
    "Fire drill is scheduled for next month.",
    "The print queue is down for maintenance.",
    "Holiday party falls on December 15th.",
    "Coffee budget was increased this quarter.",
    "The bicycle rack near the lobby was repainted.",
    "All-hands moved from Thursday to Wednesday.",
    "Standing desks are available on request.",
    "The cafeteria switched to compostable cutlery.",
    "Visitor badges expire at end of day.",
    "Building HVAC inspection happens in April.",
    "Snack bar restock day is Monday.",
]


# ─── case + runner ─────────────────────────────────────────────────

@dataclass
class GeneratedCase:
    id: str
    family: str
    setup_facts: list[str]
    mutations: list[tuple]            # (op, *args) — supersede/release/purge
    final_query: str
    must_contain: list[str]
    must_not_contain: list[str]

    def run(self, adapter) -> bool:
        adapter.reset()
        for f in self.setup_facts:
            adapter.inscribe(f)
        for m in self.mutations:
            op = m[0]
            if op == "supersede":
                adapter.supersede(m[1], m[2])
            elif op == "release":
                adapter.release(m[1])
            elif op == "purge":
                adapter.purge(m[1])
            else:
                raise ValueError(op)
        top = adapter.recall_texts(self.final_query, k=10)
        blob = " ".join(top).lower()
        for s in self.must_contain:
            if s.lower() not in blob:
                return False
        for s in self.must_not_contain:
            if s.lower() in blob:
                return False
        return True


def _scatter(facts: list[str], rng: random.Random, n: int = 4) -> list[str]:
    """Sprinkle n distractor facts into a setup list at random positions."""
    fillers = rng.sample(FILLER, min(n, len(FILLER)))
    out = list(facts)
    for f in fillers:
        out.insert(rng.randint(0, len(out)), f)
    return out


# ─── SUPERSESSION ──────────────────────────────────────────────────

def _supersession_job(rng):
    name = rng.choice(NAMES)
    old, new = rng.sample(COMPANIES, 2)
    return GeneratedCase(
        id="", family="supersession",
        setup_facts=_scatter([f"{name} works at {old}."], rng),
        mutations=[("supersede", f"{name} job employer",
                    f"{name} now works at {new} as of 2026.")],
        final_query=f"Where does {name} work?",
        must_contain=[new], must_not_contain=[old],
    )


def _supersession_theme(rng):
    name = rng.choice(NAMES)
    old, new = rng.sample(THEMES, 2)
    return GeneratedCase(
        id="", family="supersession",
        setup_facts=_scatter([f"{name} prefers {old} mode in all apps."], rng),
        mutations=[("supersede", f"{name} theme preference",
                    f"{name} switched to {new} mode permanently.")],
        final_query=f"What theme does {name} use?",
        must_contain=[new], must_not_contain=[old],
    )


def _supersession_diet(rng):
    name = rng.choice(NAMES)
    old, new = rng.sample(DIETS, 2)
    return GeneratedCase(
        id="", family="supersession",
        setup_facts=_scatter([f"{name} is {old} and follows that strictly."], rng),
        mutations=[("supersede", f"{name} diet",
                    f"{name} now eats {new} style as of this year.")],
        final_query=f"What is {name}'s diet?",
        must_contain=[new], must_not_contain=[old],
    )


SUPERSESSION = [_supersession_job, _supersession_theme, _supersession_diet]


# ─── DECAY ─────────────────────────────────────────────────────────

def _decay_otp(rng):
    code = "".join(rng.choices("0123456789", k=6))
    name = rng.choice(NAMES)
    return GeneratedCase(
        id="", family="decay",
        setup_facts=_scatter(
            [f"Session OTP for {name}: {code}.",
             f"{name} likes iced coffee in the afternoon."], rng),
        mutations=[("release", f"OTP login session code {code}")],
        final_query=f"Any one-time codes for {name}?",
        must_contain=[], must_not_contain=[code],
    )


def _decay_verification(rng):
    code = "".join(rng.choices("0123456789", k=5))
    return GeneratedCase(
        id="", family="decay",
        setup_facts=_scatter([f"Temporary verification code: {code}."], rng),
        mutations=[("release", f"verification code {code}")],
        final_query="What was the verification code?",
        must_contain=[], must_not_contain=[code],
    )


def _decay_flight_cancelled(rng):
    name = rng.choice(NAMES)
    dest = rng.choice(CITIES)
    return GeneratedCase(
        id="", family="decay",
        setup_facts=_scatter(
            [f"{name} booked a flight to {dest} for next Friday."], rng),
        mutations=[("release", f"flight to {dest}")],
        final_query=f"What are {name}'s travel plans?",
        must_contain=[], must_not_contain=[dest],
    )


DECAY = [_decay_otp, _decay_verification, _decay_flight_cancelled]


# ─── AMNESIA ───────────────────────────────────────────────────────

FOODS = ["anchovy pizza", "spicy ramen", "plain rice", "sushi", "tacos",
         "curry", "pho", "dumplings", "paella"]


def _amnesia_person(rng):
    target, other = rng.sample(NAMES, 2)
    f1, f2 = rng.sample(FOODS, 2)
    return GeneratedCase(
        id="", family="amnesia",
        setup_facts=_scatter(
            [f"{target} likes {f1}.", f"{other} likes {f2}."], rng),
        mutations=[("release", f"{target} {f1} preferences")],
        final_query="What do people like to eat?",
        must_contain=[other], must_not_contain=[target],
    )


def _amnesia_multi(rng):
    target, other = rng.sample(NAMES, 2)
    job_t, job_o = rng.sample(COMPANIES, 2)
    city = rng.choice(CITIES)
    lang = rng.choice(LANGUAGES)
    return GeneratedCase(
        id="", family="amnesia",
        setup_facts=_scatter([
            f"{target} works at {job_t}.",
            f"{target} lives in {city}.",
            f"{target} speaks {lang}.",
            f"{other} works at {job_o}.",
        ], rng),
        mutations=[("release", f"{target} profile information")],
        final_query="Tell me about people.",
        must_contain=[other], must_not_contain=[target],
    )


def _amnesia_topic(rng):
    name = rng.choice(NAMES)
    book = rng.choice(BOOKS)
    hobby = rng.choice(HOBBIES)
    return GeneratedCase(
        id="", family="amnesia",
        setup_facts=_scatter([
            f"{name} finished reading '{book}'.",
            f"{name} completed their first {hobby} session in October.",
        ], rng),
        mutations=[("release", f"books reading {book}")],
        final_query=f"What has {name} accomplished?",
        must_contain=[hobby], must_not_contain=[book],
    )


AMNESIA = [_amnesia_person, _amnesia_multi, _amnesia_topic]


# ─── PURGE ─────────────────────────────────────────────────────────

def _purge_api_key(rng):
    suffix = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=10))
    key = f"sk-{suffix}"
    return GeneratedCase(
        id="", family="purge",
        setup_facts=_scatter([f"API key issued to ops: {key}-secret."], rng),
        mutations=[("purge", f"API key {key}")],
        final_query="What is the API key?",
        must_contain=[], must_not_contain=[key],
    )


def _purge_phi(rng):
    first, last = rng.sample(NAMES, 2)
    year = rng.randint(1950, 2005)
    cond = rng.choice(CONDITIONS)
    return GeneratedCase(
        id="", family="purge",
        setup_facts=_scatter([
            f"Patient {first} {last}, DOB {year}-04-12, diagnosed with {cond}.",
        ], rng),
        mutations=[("purge", f"{first} {last} medical record")],
        final_query=f"Tell me about {first} {last}.",
        must_contain=[], must_not_contain=[f"{first} {last}", cond],
    )


def _purge_gdpr(rng):
    target, other = rng.sample(NAMES, 2)
    domain = rng.choice(EMAIL_DOMAINS)
    tgt = f"{target.lower()}@{domain}"
    oth = f"{other.lower()}@{domain}"
    return GeneratedCase(
        id="", family="purge",
        setup_facts=_scatter([
            f"Customer {tgt} bought 3 items last week.",
            f"Customer {oth} placed an order yesterday.",
        ], rng),
        mutations=[("purge", f"{tgt} customer history")],
        final_query=f"Show me customer {tgt} data.",
        must_contain=[oth], must_not_contain=[tgt],
    )


PURGE = [_purge_api_key, _purge_phi, _purge_gdpr]


# ─── DRIFT ─────────────────────────────────────────────────────────

def _drift_jobs(rng):
    name = rng.choice(NAMES)
    c1, c2, c3 = rng.sample(COMPANIES, 3)
    return GeneratedCase(
        id="", family="drift",
        setup_facts=_scatter([f"{name} started at {c1} in 2020."], rng),
        mutations=[
            ("supersede", f"{name} employer", f"{name} moved to {c2} in 2022."),
            ("supersede", f"{name} employer", f"{name} is at {c3} as of 2025."),
        ],
        final_query=f"Where does {name} work?",
        must_contain=[c3], must_not_contain=[c1, c2],
    )


def _drift_address(rng):
    name = rng.choice(NAMES)
    s1, s2, s3 = rng.sample(STREETS, 3)
    n1, n2, n3 = (rng.randint(1, 99) for _ in range(3))
    return GeneratedCase(
        id="", family="drift",
        setup_facts=_scatter([f"{name} lives at {n1} {s1} Street."], rng),
        mutations=[
            ("supersede", f"{name} address",
             f"{name} moved to {n2} {s2} Avenue."),
            ("supersede", f"{name} address",
             f"{name} now lives at {n3} {s3} Road."),
        ],
        final_query=f"What is {name}'s address?",
        must_contain=[s3], must_not_contain=[s1, s2],
    )


def _drift_color(rng):
    name = rng.choice(NAMES)
    c1, c2, c3 = rng.sample(COLORS, 3)
    return GeneratedCase(
        id="", family="drift",
        setup_facts=_scatter([f"{name}'s favorite color is {c1}."], rng),
        mutations=[
            ("supersede", f"{name} favorite color", f"{name} switched to {c2}."),
            ("supersede", f"{name} favorite color", f"Now {name} prefers {c3}."),
        ],
        final_query=f"What is {name}'s favorite color?",
        must_contain=[c3], must_not_contain=[c1, c2],
    )


DRIFT = [_drift_jobs, _drift_address, _drift_color]


# ─── full generator ────────────────────────────────────────────────

FAMILIES: dict[str, list] = {
    "supersession": SUPERSESSION,
    "decay": DECAY,
    "amnesia": AMNESIA,
    "purge": PURGE,
    "drift": DRIFT,
}


def generate(n_per_family: int, seed: int = 42) -> list[GeneratedCase]:
    """Round-robin through each family's sub-templates to give variety."""
    rng = random.Random(seed)
    out: list[GeneratedCase] = []
    for family, gens in FAMILIES.items():
        for i in range(n_per_family):
            gen = gens[i % len(gens)]
            case = gen(rng)
            case.id = f"{family}__{gen.__name__.lstrip('_')}__{i:03d}"
            out.append(case)
    return out
