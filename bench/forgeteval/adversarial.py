"""ForgetEval-Adv — hand-crafted adversarial layer.

64 cases across 8 attack categories, 8 cases each.  Each case is a
`GeneratedCase` (same dataclass as the template-generated suite), so
the runner, adapter Protocol, and scoring logic in `run.py` apply
unchanged.

Design philosophy: each case targets a specific architectural weakness
the template-generated suite cannot reliably probe.  See
`docs/forgeteval_adversarial.md` for the full design rationale,
annotation protocol, and reporting format.

Attack categories (8 each, 64 total):
  1. substring_trap         — must-not substring appears in a distractor
  2. prefix_collision       — identifiers share long common prefix
  3. paraphrase_supersession — new fact lexically distant from old
  4. negation_trap          — negated fact must not be confused
  5. temporal_qualifier     — date-stamped supersession chains
  6. shared_attribute       — two entities share one attribute
  7. compound_fact          — single sentence carries two facts
  8. identifier_obfuscation — same identifier, different surface forms

Author intent comments are preserved in source for the IAA protocol.
"""
from __future__ import annotations

from bench.forgeteval.generate import GeneratedCase


# ─── 1. substring_trap ─────────────────────────────────────────────
# Family: any.  Attack: a must_not substring is innocently embedded in
# a distractor fact.  Tests whether the scoring is robust to false
# substring matches.  These probe the *benchmark* as much as the
# system: a system that passes here is fine; a system that fails
# might be fine but the *case* is broken.  Cases here are written so
# that the must_not substring is rare enough that the surviving
# distractor cannot accidentally satisfy it.

ADV_SUBSTRING_TRAP = [
    GeneratedCase(
        id="adv_substring_trap_01",
        family="supersession",
        # intent: must_not "Vienna" is the city; distractor mentions
        # "Vienna sausage" (food), the substring "Vienna" appears
        # but the system should still pass because the supersede
        # works on the city fact, not the sausage trivia.
        setup_facts=[
            "User currently lives in Vienna.",
            "The office cafeteria serves Vienna sausage on Tuesdays.",
            "Hallway plants were repotted last week.",
            "Conference room reservation system is offline.",
        ],
        mutations=[("supersede", "user residence city", "User moved to Oslo last month.")],
        final_query="Where does the user live?",
        must_contain=["Oslo"],
        must_not_contain=["lives in Vienna", "currently lives"],
    ),
    GeneratedCase(
        id="adv_substring_trap_02",
        family="purge",
        # intent: deleting password "raven42" must remove it from
        # recall.  Distractor "raven" (a bird-watching club) shares
        # the prefix but is not the password.
        setup_facts=[
            "Account password for backup: raven42#secure.",
            "The bird-watching club logo is a raven silhouette.",
            "Annual hackathon prizes were sponsored by an angel investor.",
            "The HVAC system was inspected and approved.",
        ],
        mutations=[("purge", "account password backup")],
        final_query="What is the backup account password?",
        must_contain=[],
        must_not_contain=["raven42", "raven42#secure"],
    ),
    GeneratedCase(
        id="adv_substring_trap_03",
        family="amnesia",
        # intent: forget "Frank" the person.  Distractor about
        # "frank discussion" (adjective use) shares the substring.
        setup_facts=[
            "Frank works at Notion as a backend engineer.",
            "Frank's favorite color is teal.",
            "Last quarter's all-hands had a frank discussion about layoffs.",
            "Grace organizes the office quiz nights.",
        ],
        mutations=[("release", "everything about Frank the engineer")],
        final_query="Tell me about the people we discussed.",
        must_contain=["Grace"],
        must_not_contain=["Frank works", "Frank's favorite"],
    ),
    GeneratedCase(
        id="adv_substring_trap_04",
        family="decay",
        # intent: release session token "kappa7"; distractor mentions
        # "kappa coefficient" (statistics term) which contains the
        # substring but is not the token.
        setup_facts=[
            "Session token issued: kappa7-2025-08.",
            "Inter-annotator agreement is measured with Cohen's kappa coefficient.",
            "The deployment pipeline runs on Tuesdays at 02:00 UTC.",
            "Document templates were standardized in 2024.",
        ],
        mutations=[("release", "session token kappa7")],
        final_query="What is the current session token?",
        must_contain=[],
        must_not_contain=["kappa7-2025", "kappa7-"],
    ),
    GeneratedCase(
        id="adv_substring_trap_05",
        family="drift",
        # intent: chain of job changes ending at Anthropic.  Distractor
        # mentions "anthropic principle" (cosmology) which contains
        # the substring "Anthropic" but is not an employer.
        setup_facts=[
            "User started at Google in 2020.",
            "A reading-group discussed the anthropic principle last week.",
            "Office printers were replaced with networked models.",
        ],
        mutations=[
            ("supersede", "user employer Google", "User moved to Meta in 2022."),
            ("supersede", "user employer Meta", "User joined Anthropic in 2025."),
        ],
        final_query="Where does the user work now?",
        must_contain=["joined Anthropic"],
        must_not_contain=["started at Google", "moved to Meta"],
    ),
    GeneratedCase(
        id="adv_substring_trap_06",
        family="supersession",
        # intent: supersede diet "vegan" → "pescatarian".  Distractor
        # mentions "vegan restaurant" (location, not the user's diet).
        setup_facts=[
            "User follows a vegan diet strictly.",
            "There is a popular vegan restaurant on the corner.",
            "Last team offsite was held in Lisbon.",
        ],
        mutations=[("supersede", "user dietary preference", "User now eats pescatarian, including fish.")],
        final_query="What is the user's diet?",
        must_contain=["pescatarian"],
        must_not_contain=["follows a vegan", "vegan diet strictly"],
    ),
    GeneratedCase(
        id="adv_substring_trap_07",
        family="purge",
        # intent: purge customer "alice@beta.io"; distractor mentions
        # "beta release notes" (product term) which shares "beta".
        setup_facts=[
            "Customer alice@beta.io has 3 active subscriptions.",
            "Beta release notes were published last Friday.",
            "Quarterly board meeting is in October.",
        ],
        mutations=[("purge", "alice@beta.io")],
        final_query="Show me data for customer alice@beta.io.",
        must_contain=[],
        must_not_contain=["alice@beta.io has", "alice@beta.io's"],
    ),
    GeneratedCase(
        id="adv_substring_trap_08",
        family="amnesia",
        # intent: forget "Charlie" the engineer.  Distractor mentions
        # "Charlie horse" idiom in a chat snippet.
        setup_facts=[
            "Charlie joined the platform team in March.",
            "Charlie speaks German and Italian fluently.",
            "Someone in chat joked about getting a charlie horse during the standup.",
            "Diana submitted the patent application last quarter.",
        ],
        mutations=[("release", "everything about Charlie on the team")],
        final_query="Who is on the team?",
        must_contain=["Diana"],
        must_not_contain=["Charlie joined", "Charlie speaks"],
    ),
]


# ─── 2. prefix_collision ───────────────────────────────────────────
# Family: purge.  Attack: two identifiers share a long common prefix.
# Deleting one must not also delete the other.  Tests whether the
# purge path uses precise identifier match (good) or fuzzy/semantic
# match (bad).

ADV_PREFIX_COLLISION = [
    GeneratedCase(
        id="adv_prefix_collision_01",
        family="purge",
        # intent: alice@acme.io vs alice.smith@acme.io
        setup_facts=[
            "Customer alice@acme.io subscribed in 2024.",
            "Customer alice.smith@acme.io subscribed in 2023.",
            "Quarterly all-hands was rescheduled.",
        ],
        mutations=[("purge", "alice@acme.io")],
        final_query="Which alice customers are still active?",
        must_contain=["alice.smith@acme.io"],
        must_not_contain=["alice@acme.io subscribed"],
    ),
    GeneratedCase(
        id="adv_prefix_collision_02",
        family="purge",
        # intent: api_key_v1 vs api_key_v2
        setup_facts=[
            "Production key: api_key_v1_abc123.",
            "Staging key: api_key_v2_def456.",
            "Office WiFi password was rotated.",
        ],
        mutations=[("purge", "api_key_v1_abc123")],
        final_query="Show me the staging API key.",
        must_contain=["api_key_v2_def456"],
        must_not_contain=["api_key_v1_abc123"],
    ),
    GeneratedCase(
        id="adv_prefix_collision_03",
        family="purge",
        # intent: patient John Smith vs patient John Smithson
        setup_facts=[
            "Patient John Smith was prescribed metformin 500mg.",
            "Patient John Smithson is allergic to penicillin.",
            "The pharmacy received a shipment of insulin last week.",
        ],
        mutations=[("purge", "John Smith prescription")],
        final_query="What allergies does John Smithson have?",
        must_contain=["penicillin"],
        must_not_contain=["John Smith was prescribed", "metformin 500mg"],
    ),
    GeneratedCase(
        id="adv_prefix_collision_04",
        family="purge",
        # intent: order_2024_001 vs order_2024_0010
        setup_facts=[
            "Order order_2024_001 was for 5 units at $200 each.",
            "Order order_2024_0010 was for 12 units at $150 each.",
            "Warehouse 3 inventory was audited Friday.",
        ],
        mutations=[("purge", "order_2024_001 record")],
        final_query="Tell me about order_2024_0010.",
        must_contain=["12 units"],
        must_not_contain=["order_2024_001 was for 5"],
    ),
    GeneratedCase(
        id="adv_prefix_collision_05",
        family="purge",
        # intent: bob@example.com vs bob@example.com.uk (different TLD)
        setup_facts=[
            "User bob@example.com signed the NDA on 2025-03-01.",
            "User bob@example.com.uk submitted a separate NDA on 2025-04-15.",
            "Legal team rotated their PGP keys recently.",
        ],
        mutations=[("purge", "bob@example.com NDA")],
        final_query="Whose 2025-04-15 NDA is still on file?",
        must_contain=["bob@example.com.uk"],
        must_not_contain=["bob@example.com signed"],
    ),
    GeneratedCase(
        id="adv_prefix_collision_06",
        family="purge",
        # intent: sk-abc-prod vs sk-abc-dev
        setup_facts=[
            "Production secret token: sk-abc-prod-9f3a.",
            "Development secret token: sk-abc-dev-2c7b.",
            "Office plants were watered on schedule.",
        ],
        mutations=[("purge", "sk-abc-prod-9f3a production token")],
        final_query="What is the development token?",
        must_contain=["sk-abc-dev-2c7b"],
        must_not_contain=["sk-abc-prod-9f3a"],
    ),
    GeneratedCase(
        id="adv_prefix_collision_07",
        family="purge",
        # intent: case_id 12345 vs case_id 123456 (substring prefix)
        setup_facts=[
            "Legal case_id 12345 was filed in district court.",
            "Legal case_id 123456 was filed in appellate court.",
            "The HR portal was migrated to a new vendor.",
        ],
        mutations=[("purge", "case_id 12345 district court")],
        final_query="What court is case_id 123456 in?",
        must_contain=["appellate"],
        must_not_contain=["case_id 12345 was filed"],
    ),
    GeneratedCase(
        id="adv_prefix_collision_08",
        family="purge",
        # intent: dana@startup.com vs dana@startup.com.au
        setup_facts=[
            "Customer dana@startup.com purchased Plan A.",
            "Customer dana@startup.com.au purchased Plan B.",
            "Marketing emails are sent on Wednesdays.",
        ],
        mutations=[("purge", "dana@startup.com purchase")],
        final_query="Which plan did dana@startup.com.au buy?",
        must_contain=["Plan B"],
        must_not_contain=["dana@startup.com purchased Plan A"],
    ),
]


# ─── 3. paraphrase_supersession ────────────────────────────────────
# Family: supersession, drift.  Attack: the new fact and the old fact
# share few surface tokens.  Lexical recall cannot find the old fact
# from a query phrased like the new fact, so supersession requires
# semantic alignment.

ADV_PARAPHRASE = [
    GeneratedCase(
        id="adv_paraphrase_supersession_01",
        family="supersession",
        # intent: "works at Stripe" → "quit fintech, joined an AI safety lab"
        setup_facts=[
            "User works at Stripe as a senior backend engineer.",
            "User commutes by bike to the SF office.",
        ],
        mutations=[("supersede", "user employer", "User quit the payments industry and joined an AI safety lab in San Francisco.")],
        final_query="What does the user do for work now?",
        must_contain=["AI safety", "lab"],
        must_not_contain=["Stripe", "backend engineer"],
    ),
    GeneratedCase(
        id="adv_paraphrase_supersession_02",
        family="supersession",
        # intent: "vegan" → "started eating fish only for omega-3"
        setup_facts=[
            "User has been strictly vegan since 2020.",
            "User runs 10km every weekend.",
        ],
        mutations=[("supersede", "user dietary lifestyle", "User now consumes fish as the only animal product, citing omega-3 needs.")],
        final_query="Is the user vegan?",
        must_contain=["fish", "omega-3"],
        must_not_contain=["strictly vegan", "vegan since 2020"],
    ),
    GeneratedCase(
        id="adv_paraphrase_supersession_03",
        family="supersession",
        # intent: "lives in Berlin" → "relocated to Lisbon to escape winter"
        setup_facts=[
            "User has lived in Berlin since graduating in 2018.",
            "User plays chess on Tuesdays.",
        ],
        mutations=[("supersede", "user city of residence", "User relocated to Lisbon last spring to escape the German winter.")],
        final_query="Where does the user live?",
        must_contain=["Lisbon"],
        must_not_contain=["lived in Berlin", "since graduating"],
    ),
    GeneratedCase(
        id="adv_paraphrase_supersession_04",
        family="supersession",
        # intent: "Python is the only language" → "switched to writing services in Rust"
        setup_facts=[
            "User writes all backend code in Python and rejects other languages.",
            "User uses an ergonomic keyboard.",
        ],
        mutations=[("supersede", "user programming language preference", "User has rebuilt the team's microservices in Rust over the last six months.")],
        final_query="What language does the user write services in?",
        must_contain=["Rust"],
        must_not_contain=["writes all backend code in Python", "rejects other languages"],
    ),
    GeneratedCase(
        id="adv_paraphrase_supersession_05",
        family="supersession",
        # intent: "married" → "filed for divorce earlier this year"
        setup_facts=[
            "User is married and lives with their spouse in Toronto.",
            "User collects vintage cameras.",
        ],
        mutations=[("supersede", "user marital status", "User filed for divorce earlier this year and now lives alone.")],
        final_query="Is the user married?",
        must_contain=["divorce", "alone"],
        must_not_contain=["is married and lives", "lives with their spouse"],
    ),
    GeneratedCase(
        id="adv_paraphrase_supersession_06",
        family="drift",
        # intent: 4-step chain with growing paraphrase distance
        setup_facts=[
            "User joined Google in 2018 as a software engineer.",
        ],
        mutations=[
            ("supersede", "user employer 2018", "Departed Mountain View, took a role at a hedge fund in NYC in 2020."),
            ("supersede", "user employer 2020 finance", "Left finance entirely in 2022 to teach mathematics at a small college in Vermont."),
            ("supersede", "user employer 2022 teaching", "In 2025 returned to industry, joining a robotics startup in Boston as VP of engineering."),
        ],
        final_query="What is the user's current job?",
        must_contain=["robotics", "Boston"],
        must_not_contain=["joined Google in 2018", "hedge fund in NYC", "teach mathematics"],
    ),
    GeneratedCase(
        id="adv_paraphrase_supersession_07",
        family="supersession",
        # intent: "owns a Tesla" → "switched to a bicycle and never drives"
        setup_facts=[
            "User drives a Tesla Model 3, purchased new in 2022.",
            "User listens to audiobooks during the commute.",
        ],
        mutations=[("supersede", "user vehicle", "User sold the car and now commutes exclusively by bicycle, refusing to drive.")],
        final_query="What does the user drive?",
        must_contain=["bicycle"],
        must_not_contain=["drives a Tesla", "Tesla Model 3"],
    ),
    GeneratedCase(
        id="adv_paraphrase_supersession_08",
        family="supersession",
        # intent: "introvert who avoids events" → "running a 200-person community"
        setup_facts=[
            "User is a quiet introvert who avoids large gatherings.",
            "User prefers email over phone calls.",
        ],
        mutations=[("supersede", "user social temperament", "User is now organizing a 200-person developer meetup every month and hosts a weekly podcast.")],
        final_query="What is the user's social style?",
        must_contain=["organizing", "200-person"],
        must_not_contain=["quiet introvert", "avoids large gatherings"],
    ),
]


# ─── 4. negation_trap ──────────────────────────────────────────────
# Family: supersession, decay.  Attack: vector embeddings often blur
# "X" and "not X."  Releasing or superseding a *negated* fact should
# not affect recall of the affirmative form.

ADV_NEGATION = [
    GeneratedCase(
        id="adv_negation_trap_01",
        family="supersession",
        # intent: "User does NOT work at Anthropic" is the stored fact;
        # supersede to "User does work at Anthropic" — negation flips.
        setup_facts=[
            "User does NOT work at Anthropic and has never interviewed there.",
            "User's coffee preference is espresso.",
        ],
        mutations=[("supersede", "user Anthropic employment status", "User actually joined Anthropic last quarter.")],
        final_query="Does the user work at Anthropic?",
        must_contain=["joined Anthropic"],
        must_not_contain=["does NOT work at Anthropic", "never interviewed"],
    ),
    GeneratedCase(
        id="adv_negation_trap_02",
        family="decay",
        # intent: release a negated fact "doesn't drink coffee" should
        # not affect the unrelated "drinks tea" fact.
        setup_facts=[
            "User does not drink coffee under any circumstances.",
            "User drinks green tea every afternoon.",
        ],
        mutations=[("release", "user does not drink coffee")],
        final_query="What does the user drink?",
        must_contain=["green tea"],
        must_not_contain=["does not drink coffee"],
    ),
    GeneratedCase(
        id="adv_negation_trap_03",
        family="supersession",
        # intent: "doesn't have a dog" → "adopted a golden retriever"
        setup_facts=[
            "User does not have any pets and has stated this firmly.",
            "User's apartment has a small balcony.",
        ],
        mutations=[("supersede", "user pet ownership", "User adopted a golden retriever named Mochi last month.")],
        final_query="Does the user have a pet?",
        must_contain=["golden retriever", "Mochi"],
        must_not_contain=["does not have any pets", "stated this firmly"],
    ),
    GeneratedCase(
        id="adv_negation_trap_04",
        family="supersession",
        # intent: negated allergy → affirmative
        setup_facts=[
            "Patient has no known peanut allergy.",
            "Patient is currently on lisinopril for hypertension.",
        ],
        mutations=[("supersede", "patient peanut allergy status", "Patient was admitted with severe peanut anaphylaxis last week.")],
        final_query="Does the patient have a peanut allergy?",
        must_contain=["anaphylaxis"],
        must_not_contain=["no known peanut allergy"],
    ),
    GeneratedCase(
        id="adv_negation_trap_05",
        family="decay",
        # intent: release "doesn't speak French" — must not affect
        # the affirmative "speaks Spanish"
        setup_facts=[
            "User does not speak French at all.",
            "User speaks Spanish at a conversational level.",
        ],
        mutations=[("release", "user French language status")],
        final_query="What languages does the user speak?",
        must_contain=["Spanish"],
        must_not_contain=["does not speak French"],
    ),
    GeneratedCase(
        id="adv_negation_trap_06",
        family="supersession",
        # intent: "User doesn't use Slack" → "User now uses Slack daily"
        setup_facts=[
            "User refuses to use Slack and only communicates via email.",
            "User's email signature includes a Mastodon handle.",
        ],
        mutations=[("supersede", "user Slack usage", "User started using Slack daily after team-wide rollout.")],
        final_query="Does the user use Slack?",
        must_contain=["using Slack daily"],
        must_not_contain=["refuses to use Slack"],
    ),
    GeneratedCase(
        id="adv_negation_trap_07",
        family="decay",
        # intent: release a "did not attend" fact, should not affect
        # the unrelated "attended XYZ" fact.
        setup_facts=[
            "User did not attend the 2024 holiday party.",
            "User attended the 2025 onboarding session.",
        ],
        mutations=[("release", "user holiday party 2024 attendance")],
        final_query="Which events has the user attended?",
        must_contain=["onboarding"],
        must_not_contain=["did not attend the 2024 holiday party"],
    ),
    GeneratedCase(
        id="adv_negation_trap_08",
        family="supersession",
        # intent: "doesn't own a car" → "leases a Tesla now"
        setup_facts=[
            "User does not own a car and uses public transit exclusively.",
            "User has a monthly transit pass.",
        ],
        mutations=[("supersede", "user vehicle ownership", "User started leasing a Tesla Model Y this summer.")],
        final_query="What vehicle does the user have?",
        must_contain=["Tesla", "leasing"],
        must_not_contain=["does not own a car", "uses public transit exclusively"],
    ),
]


# ─── 5. temporal_qualifier ─────────────────────────────────────────
# Family: supersession, drift.  Attack: facts with embedded dates.
# Supersession must respect the temporal sequence (latest wins), not
# get confused by date tokens.

ADV_TEMPORAL = [
    GeneratedCase(
        id="adv_temporal_qualifier_01",
        family="supersession",
        # intent: "joined in 2020" → "moved in 2022" — temporal supersede
        setup_facts=[
            "User joined Google in March 2020 as a software engineer.",
            "User's hobby is competitive chess on weekends.",
        ],
        mutations=[("supersede", "user current employer Google", "User moved to Meta in June 2022.")],
        final_query="Where does the user currently work?",
        must_contain=["Meta", "June 2022"],
        must_not_contain=["joined Google in March 2020"],
    ),
    GeneratedCase(
        id="adv_temporal_qualifier_02",
        family="drift",
        # intent: 3-step temporal chain, latest wins
        setup_facts=[
            "User lived in Tokyo from 2015 to 2018.",
            "User's main language is English.",
        ],
        mutations=[
            ("supersede", "user residence Tokyo 2015", "User lived in Berlin from 2018 to 2022."),
            ("supersede", "user residence Berlin 2018", "User has lived in Lisbon since 2022."),
        ],
        final_query="Where does the user live now?",
        must_contain=["Lisbon", "since 2022"],
        must_not_contain=["lived in Tokyo", "Berlin from 2018"],
    ),
    GeneratedCase(
        id="adv_temporal_qualifier_03",
        family="supersession",
        # intent: trap — "visited Google in 2024" is NOT supersession
        # of "joined Google in 2020"; the supersession should be by
        # the explicit Meta event, not the visit.
        setup_facts=[
            "User joined Google in 2020 as their first job.",
            "User visited Google's campus in 2024 for a conference.",
            "User started a chocolate-making side project.",
        ],
        mutations=[
            ("supersede", "user employer first job", "User left Google and joined Meta in 2022."),
        ],
        final_query="Where does the user work now?",
        must_contain=["Meta"],
        must_not_contain=["joined Google in 2020 as their first job"],
    ),
    GeneratedCase(
        id="adv_temporal_qualifier_04",
        family="supersession",
        # intent: dated diet change
        setup_facts=[
            "Throughout 2022 user was strictly keto.",
            "User's preferred gym is a CrossFit box downtown.",
        ],
        mutations=[("supersede", "user diet 2022 keto", "Starting January 2024, user switched to a Mediterranean diet for cardiovascular reasons.")],
        final_query="What is the user's current diet?",
        must_contain=["Mediterranean"],
        must_not_contain=["strictly keto", "Throughout 2022"],
    ),
    GeneratedCase(
        id="adv_temporal_qualifier_05",
        family="drift",
        # intent: 4-step political affiliation chain
        setup_facts=[
            "User was a registered Libertarian in 2016.",
        ],
        mutations=[
            ("supersede", "user political registration 2016", "User registered as Democrat in 2018."),
            ("supersede", "user political registration 2018", "User registered as independent in 2022."),
            ("supersede", "user political registration 2022", "User registered as Green in 2025."),
        ],
        final_query="What is the user's current political registration?",
        must_contain=["Green", "2025"],
        must_not_contain=["registered Libertarian", "registered as Democrat", "registered as independent"],
    ),
    GeneratedCase(
        id="adv_temporal_qualifier_06",
        family="supersession",
        # intent: ownership change with dates
        setup_facts=[
            "User has owned the rental property at 14 Birch Street since 2010.",
            "User maintains a small vegetable garden.",
        ],
        mutations=[("supersede", "user property 14 Birch Street", "User sold 14 Birch Street in October 2024 and now rents downtown.")],
        final_query="Does the user own 14 Birch Street?",
        must_contain=["sold", "October 2024"],
        must_not_contain=["owned the rental property at 14 Birch Street since 2010"],
    ),
    GeneratedCase(
        id="adv_temporal_qualifier_07",
        family="supersession",
        # intent: subscription with renewal dates
        setup_facts=[
            "Customer subscribed to Pro plan on 2023-04-15.",
            "Customer is in the European Union jurisdiction.",
        ],
        mutations=[("supersede", "customer subscription Pro plan", "Customer downgraded to Free plan on 2025-02-10 after cancellation.")],
        final_query="What plan is the customer on?",
        must_contain=["Free plan", "2025-02-10"],
        must_not_contain=["subscribed to Pro plan on 2023-04-15"],
    ),
    GeneratedCase(
        id="adv_temporal_qualifier_08",
        family="drift",
        # intent: marital status drift over years
        setup_facts=[
            "User was single in 2018.",
        ],
        mutations=[
            ("supersede", "user marital status 2018 single", "User got engaged in 2020."),
            ("supersede", "user marital status engaged", "User married in 2021."),
            ("supersede", "user marital status married", "User finalized divorce in 2024."),
        ],
        final_query="What is the user's current marital status?",
        must_contain=["divorce", "2024"],
        must_not_contain=["was single in 2018", "got engaged in 2020", "married in 2021"],
    ),
]


# ─── 6. shared_attribute ───────────────────────────────────────────
# Family: amnesia.  Attack: two entities share one attribute.  Forget
# one entity must not collapse the other's link to the shared attribute.

ADV_SHARED_ATTR = [
    GeneratedCase(
        id="adv_shared_attribute_01",
        family="amnesia",
        # intent: Dana and Eve both in Berlin; forget Dana
        setup_facts=[
            "Dana lives in Berlin and works at Stripe.",
            "Eve lives in Berlin and works at Notion.",
            "Frank lives in Tokyo and works at Spotify.",
        ],
        mutations=[("release", "everything about Dana")],
        final_query="Who lives in Berlin?",
        must_contain=["Eve", "Berlin"],
        must_not_contain=["Dana lives in Berlin"],
    ),
    GeneratedCase(
        id="adv_shared_attribute_02",
        family="amnesia",
        # intent: Alice and Bob both like chess; forget Alice
        setup_facts=[
            "Alice's favorite hobby is chess and competitive bridge.",
            "Bob's favorite hobby is chess and rock climbing.",
            "Carla's favorite hobby is pottery.",
        ],
        mutations=[("release", "everything about Alice")],
        final_query="Who likes chess?",
        must_contain=["Bob"],
        must_not_contain=["Alice's favorite hobby is chess"],
    ),
    GeneratedCase(
        id="adv_shared_attribute_03",
        family="amnesia",
        # intent: two patients share a condition; forget one
        setup_facts=[
            "Patient John has type 2 diabetes and is on metformin.",
            "Patient Mary has type 2 diabetes and is on insulin therapy.",
            "Patient Tom has hypertension.",
        ],
        mutations=[("release", "everything about patient John")],
        final_query="Which patients have type 2 diabetes?",
        must_contain=["Mary", "diabetes"],
        must_not_contain=["John has type 2 diabetes"],
    ),
    GeneratedCase(
        id="adv_shared_attribute_04",
        family="amnesia",
        # intent: two employees share an office; forget one
        setup_facts=[
            "Hannah and Ivan share the 4th floor office near the elevator.",
            "Hannah leads the backend team.",
            "Ivan leads the design team.",
            "Judy works remotely from Madrid.",
        ],
        mutations=[("release", "everything about Hannah on the team")],
        final_query="Who works on the 4th floor?",
        must_contain=["Ivan"],
        must_not_contain=["Hannah and Ivan share"],
    ),
    GeneratedCase(
        id="adv_shared_attribute_05",
        family="amnesia",
        # intent: two customers share a coupon code; purge one customer's
        # data, the coupon attribute itself on the other survives.
        setup_facts=[
            "Customer alice@example.com used coupon SAVE20 in March.",
            "Customer bob@example.com used coupon SAVE20 in April.",
            "Marketing team rotates coupons quarterly.",
        ],
        mutations=[("release", "everything about alice@example.com")],
        final_query="Who used coupon SAVE20?",
        must_contain=["bob@example.com"],
        must_not_contain=["alice@example.com used coupon"],
    ),
    GeneratedCase(
        id="adv_shared_attribute_06",
        family="amnesia",
        # intent: two students share a class; forget one
        setup_facts=[
            "Liam is enrolled in CS 224N and CS 231N this term.",
            "Mia is enrolled in CS 224N and CS 230 this term.",
            "Noah is enrolled in MATH 51 only.",
        ],
        mutations=[("release", "everything about student Liam this term")],
        final_query="Who is enrolled in CS 224N?",
        must_contain=["Mia"],
        must_not_contain=["Liam is enrolled in CS 224N"],
    ),
    GeneratedCase(
        id="adv_shared_attribute_07",
        family="amnesia",
        # intent: two team members both speak Mandarin; forget one
        setup_facts=[
            "Olivia speaks English and Mandarin natively.",
            "Peter speaks French and Mandarin.",
            "Quinn speaks German only.",
        ],
        mutations=[("release", "everything about Olivia's languages")],
        final_query="Who on the team speaks Mandarin?",
        must_contain=["Peter"],
        must_not_contain=["Olivia speaks English and Mandarin"],
    ),
    GeneratedCase(
        id="adv_shared_attribute_08",
        family="amnesia",
        # intent: two clients share a project; forget one
        setup_facts=[
            "Client Riya works on the Atlas project and pays monthly.",
            "Client Sam works on the Atlas project and pays annually.",
            "Client Tara works on a different project entirely.",
        ],
        mutations=[("release", "everything about client Riya")],
        final_query="Who else works on the Atlas project?",
        must_contain=["Sam"],
        must_not_contain=["Riya works on the Atlas"],
    ),
]


# ─── 7. compound_fact ──────────────────────────────────────────────
# Family: supersession.  Attack: a single sentence carries two facts.
# Superseding only one must preserve the other.

ADV_COMPOUND = [
    GeneratedCase(
        id="adv_compound_fact_01",
        family="supersession",
        # intent: "lives in Berlin AND works at Stripe" — change only location
        setup_facts=[
            "User lives in Berlin and works at Stripe as a backend engineer.",
            "User uses an ergonomic mechanical keyboard.",
        ],
        mutations=[("supersede", "user city of residence", "User relocated to Madrid and continues working remotely.")],
        final_query="Where does the user work?",
        must_contain=["Stripe"],
        must_not_contain=["lives in Berlin"],
    ),
    GeneratedCase(
        id="adv_compound_fact_02",
        family="supersession",
        # intent: marital + work — change only work
        setup_facts=[
            "User is married to Jamie and works at Google.",
            "User has two cats named Pixel and Bit.",
        ],
        mutations=[("supersede", "user employer Google", "User joined Anthropic as a research engineer.")],
        final_query="Is the user still married?",
        must_contain=["married to Jamie"],
        must_not_contain=["works at Google"],
    ),
    GeneratedCase(
        id="adv_compound_fact_03",
        family="supersession",
        # intent: language + diet — change only diet
        setup_facts=[
            "User speaks Italian fluently and follows a Mediterranean diet.",
            "User runs a small wine-tasting club in their neighborhood.",
        ],
        mutations=[("supersede", "user dietary preference Mediterranean", "User switched to a strict vegan diet for ethical reasons.")],
        final_query="What languages does the user speak?",
        must_contain=["Italian"],
        must_not_contain=["Mediterranean diet"],
    ),
    GeneratedCase(
        id="adv_compound_fact_04",
        family="supersession",
        # intent: hobby + city — change only city
        setup_facts=[
            "User plays the cello and lives in Vienna.",
            "User's day job is database administration.",
        ],
        mutations=[("supersede", "user city of residence Vienna", "User moved to Toronto last autumn for a new opportunity.")],
        final_query="Does the user still play the cello?",
        must_contain=["cello"],
        must_not_contain=["lives in Vienna"],
    ),
    GeneratedCase(
        id="adv_compound_fact_05",
        family="supersession",
        # intent: car + pet — change only car
        setup_facts=[
            "User drives a 2018 Subaru and owns a Border Collie named Rex.",
            "User's preferred reading genre is hard sci-fi.",
        ],
        mutations=[("supersede", "user vehicle Subaru", "User upgraded to a new electric pickup truck.")],
        final_query="What pet does the user have?",
        must_contain=["Border Collie", "Rex"],
        must_not_contain=["drives a 2018 Subaru"],
    ),
    GeneratedCase(
        id="adv_compound_fact_06",
        family="supersession",
        # intent: degree + employer — change only employer
        setup_facts=[
            "User has a PhD in physics from MIT and works as a quant at Renaissance.",
            "User's morning routine includes a 5km run.",
        ],
        mutations=[("supersede", "user employer quant Renaissance", "User left finance entirely and joined a nonprofit climate research lab.")],
        final_query="What is the user's educational background?",
        must_contain=["PhD in physics", "MIT"],
        must_not_contain=["works as a quant at Renaissance"],
    ),
    GeneratedCase(
        id="adv_compound_fact_07",
        family="supersession",
        # intent: subscription + payment method — change only payment
        setup_facts=[
            "Customer is on the Enterprise plan and pays via wire transfer.",
            "Customer's account is in good standing.",
        ],
        mutations=[("supersede", "customer payment method wire transfer", "Customer updated payment method to credit card on file.")],
        final_query="What plan is the customer on?",
        must_contain=["Enterprise"],
        must_not_contain=["pays via wire transfer"],
    ),
    GeneratedCase(
        id="adv_compound_fact_08",
        family="supersession",
        # intent: project + role — change only role
        setup_facts=[
            "User leads the Atlas project and reports to the VP of Engineering.",
            "User's preferred meeting time is afternoons.",
        ],
        mutations=[("supersede", "user reporting line VP Engineering", "User now reports directly to the CTO after a reorg.")],
        final_query="What project does the user lead?",
        must_contain=["Atlas"],
        must_not_contain=["reports to the VP of Engineering"],
    ),
]


# ─── 8. identifier_obfuscation ─────────────────────────────────────
# Family: purge.  Attack: same identifier in different surface forms
# (case, whitespace, encoding).  A correct purge by one form should
# also purge the others, OR the system should be honest that it only
# handles canonical form.

ADV_IDENTIFIER_OBFUSCATION = [
    GeneratedCase(
        id="adv_identifier_obfuscation_01",
        family="purge",
        # intent: same email, different case
        setup_facts=[
            "Customer ALICE@ACME.IO subscribed to Plan A.",
            "Customer alice@acme.io is the same person (case-normalized).",
            "Office lunches are catered on Wednesdays.",
        ],
        mutations=[("purge", "alice@acme.io")],
        final_query="What plan does alice@acme.io have?",
        must_contain=[],
        must_not_contain=["ALICE@ACME.IO subscribed", "alice@acme.io is the same"],
    ),
    GeneratedCase(
        id="adv_identifier_obfuscation_02",
        family="purge",
        # intent: same email with trailing space
        setup_facts=[
            "Customer bob@example.com bought 5 licenses.",
            "Customer bob@example.com  (note trailing whitespace) is the same account.",
            "Quarterly OKR review is in two weeks.",
        ],
        mutations=[("purge", "bob@example.com")],
        final_query="What did bob@example.com buy?",
        must_contain=[],
        must_not_contain=["bob@example.com bought 5", "is the same account"],
    ),
    GeneratedCase(
        id="adv_identifier_obfuscation_03",
        family="purge",
        # intent: phone number with/without country code
        setup_facts=[
            "Patient phone on file: +1-415-555-0123.",
            "Patient phone alternate format: 415-555-0123 (no country code).",
            "Clinic moved to a new address last month.",
        ],
        mutations=[("purge", "phone 415-555-0123")],
        final_query="What is the patient's phone number?",
        must_contain=[],
        must_not_contain=["+1-415-555-0123", "415-555-0123 (no country code)"],
    ),
    GeneratedCase(
        id="adv_identifier_obfuscation_04",
        family="purge",
        # intent: UUID with/without hyphens
        setup_facts=[
            "User token: 550e8400-e29b-41d4-a716-446655440000.",
            "Same user, compact form: 550e8400e29b41d4a716446655440000.",
            "Server maintenance window is Tuesdays.",
        ],
        mutations=[("purge", "user token 550e8400")],
        final_query="What is the user token?",
        must_contain=[],
        must_not_contain=["550e8400-e29b-41d4", "550e8400e29b41d4"],
    ),
    GeneratedCase(
        id="adv_identifier_obfuscation_05",
        family="purge",
        # intent: SSN with/without dashes
        setup_facts=[
            "Customer SSN on file: 123-45-6789.",
            "Customer alternate record: 123456789 (no dashes).",
            "Tax filings are due in April.",
        ],
        mutations=[("purge", "SSN 123-45-6789")],
        final_query="What is the customer's SSN?",
        must_contain=[],
        must_not_contain=["123-45-6789", "123456789 (no dashes)"],
    ),
    GeneratedCase(
        id="adv_identifier_obfuscation_06",
        family="purge",
        # intent: credit card with/without spaces
        setup_facts=[
            "Card on file: 4111-1111-1111-1111.",
            "Same card, no separators: 4111111111111111.",
            "Office holiday party will be online.",
        ],
        mutations=[("purge", "card 4111-1111-1111-1111")],
        final_query="What card is on file?",
        must_contain=[],
        must_not_contain=["4111-1111-1111-1111", "4111111111111111"],
    ),
    GeneratedCase(
        id="adv_identifier_obfuscation_07",
        family="purge",
        # intent: email with quoted local part
        setup_facts=[
            "Customer \"john.doe\"@company.com subscribed to Pro.",
            "Same customer in canonical form: john.doe@company.com.",
            "Annual leave policy updates take effect next quarter.",
        ],
        mutations=[("purge", "john.doe@company.com")],
        final_query="What plan does john.doe@company.com have?",
        must_contain=[],
        must_not_contain=["\"john.doe\"@company.com subscribed", "Same customer in canonical"],
    ),
    GeneratedCase(
        id="adv_identifier_obfuscation_08",
        family="purge",
        # intent: GitHub handle with/without leading @
        setup_facts=[
            "Contractor @deeplethe-engineer is on the access list.",
            "Same contractor without @: deeplethe-engineer.",
            "All-hands is the third Friday of each month.",
        ],
        mutations=[("purge", "@deeplethe-engineer contractor")],
        final_query="Is @deeplethe-engineer still on the access list?",
        must_contain=[],
        must_not_contain=["@deeplethe-engineer is on", "deeplethe-engineer."],
    ),
]


# ─── master list ───────────────────────────────────────────────────

ATTACK_CATEGORIES: dict[str, list[GeneratedCase]] = {
    "substring_trap":          ADV_SUBSTRING_TRAP,
    "prefix_collision":        ADV_PREFIX_COLLISION,
    "paraphrase_supersession": ADV_PARAPHRASE,
    "negation_trap":           ADV_NEGATION,
    "temporal_qualifier":      ADV_TEMPORAL,
    "shared_attribute":        ADV_SHARED_ATTR,
    "compound_fact":           ADV_COMPOUND,
    "identifier_obfuscation":  ADV_IDENTIFIER_OBFUSCATION,
}

ADVERSARIAL_TESTS: list[GeneratedCase] = [
    case for cases in ATTACK_CATEGORIES.values() for case in cases
]


def case_to_attack_category(case_id: str) -> str:
    """Recover the attack category from a case id (used in reporting)."""
    for category in ATTACK_CATEGORIES:
        if case_id.startswith(f"adv_{category}_"):
            return category
    return "unknown"
