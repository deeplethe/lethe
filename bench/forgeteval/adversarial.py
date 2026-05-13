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
    # ─── v0.4 expansion: 8 more prefix-collision cases ──────────────
    GeneratedCase(
        id="adv_prefix_collision_09",
        family="purge",
        # intent: usernames sharing prefix "admin" vs "admin1"
        setup_facts=[
            "User admin has full root access.",
            "User admin1 has read-only dashboard access.",
            "Audit log was archived last week.",
        ],
        mutations=[("purge", "user admin root access")],
        final_query="What access does admin1 have?",
        must_contain=["read-only"],
        must_not_contain=["admin has full root"],
    ),
    GeneratedCase(
        id="adv_prefix_collision_10",
        family="purge",
        # intent: project_2024 vs project_2024_v2
        setup_facts=[
            "project_2024 budget is $500K.",
            "project_2024_v2 budget is $750K.",
            "Coffee inventory was restocked.",
        ],
        mutations=[("purge", "project_2024 budget record")],
        final_query="What is the project_2024_v2 budget?",
        must_contain=["$750K"],
        must_not_contain=["project_2024 budget is $500K"],
    ),
    GeneratedCase(
        id="adv_prefix_collision_11",
        family="purge",
        # intent: shared department prefix
        setup_facts=[
            "Department engineering reports to CTO.",
            "Department engineering-platform reports to VP Platform.",
            "All-hands meeting moved to Wednesday.",
        ],
        mutations=[("purge", "department engineering reporting line")],
        final_query="Who does engineering-platform report to?",
        must_contain=["VP Platform"],
        must_not_contain=["engineering reports to CTO"],
    ),
    GeneratedCase(
        id="adv_prefix_collision_12",
        family="purge",
        # intent: ticket numbers sharing prefix
        setup_facts=[
            "Ticket TKT-100 was resolved in March.",
            "Ticket TKT-1000 is still open with priority high.",
            "Quarterly metrics will be presented Thursday.",
        ],
        mutations=[("purge", "ticket TKT-100 record")],
        final_query="What is the status of TKT-1000?",
        must_contain=["still open"],
        must_not_contain=["TKT-100 was resolved"],
    ),
    GeneratedCase(
        id="adv_prefix_collision_13",
        family="purge",
        # intent: file paths sharing prefix
        setup_facts=[
            "Config file /etc/app/conf is currently in use.",
            "Config file /etc/app/conf.bak holds the previous version.",
            "Server uptime is 47 days.",
        ],
        mutations=[("purge", "/etc/app/conf active configuration")],
        final_query="What does the .bak file contain?",
        must_contain=["previous version"],
        must_not_contain=["/etc/app/conf is currently"],
    ),
    GeneratedCase(
        id="adv_prefix_collision_14",
        family="purge",
        # intent: domain prefix collision
        setup_facts=[
            "Domain mycompany.com expires in 2027.",
            "Domain mycompany.com.br is owned by a different entity.",
            "DNS provider was changed last month.",
        ],
        mutations=[("purge", "mycompany.com domain registration")],
        final_query="Who owns mycompany.com.br?",
        must_contain=["different entity"],
        must_not_contain=["mycompany.com expires"],
    ),
    GeneratedCase(
        id="adv_prefix_collision_15",
        family="purge",
        # intent: hash prefix collision
        setup_facts=[
            "Commit hash abc1234 reverted the payment bug.",
            "Commit hash abc12345 fixed the email template typo.",
            "CI pipeline was migrated to a new runner.",
        ],
        mutations=[("purge", "commit abc1234 record")],
        final_query="What did commit abc12345 do?",
        must_contain=["email template typo"],
        must_not_contain=["abc1234 reverted the payment"],
    ),
    GeneratedCase(
        id="adv_prefix_collision_16",
        family="purge",
        # intent: phone country-code prefix collision
        setup_facts=[
            "Phone +1-555-0100 is the main office line.",
            "Phone +1-555-01000 is the fax machine extension.",
            "Building cleaning is on Fridays.",
        ],
        mutations=[("purge", "phone +1-555-0100 office line")],
        final_query="What is the fax extension?",
        must_contain=["+1-555-01000"],
        must_not_contain=["+1-555-0100 is the main"],
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
    # ─── v0.4 expansion: 8 more shared-attribute cases ───────────────
    GeneratedCase(
        id="adv_shared_attribute_09",
        family="amnesia",
        # intent: two employees same dept; forget one
        setup_facts=[
            "Engineer Uma works in the platform team and ships Go services.",
            "Engineer Victor works in the platform team and writes Rust services.",
            "Engineer Wendy works in the data team.",
        ],
        mutations=[("release", "everything about Uma")],
        final_query="Who is in the platform team?",
        must_contain=["Victor"],
        must_not_contain=["Uma works in the platform"],
    ),
    GeneratedCase(
        id="adv_shared_attribute_10",
        family="amnesia",
        # intent: two devices same model; forget one user's
        setup_facts=[
            "Xavier owns a MacBook Pro 16 M3 for video editing.",
            "Yara owns a MacBook Pro 16 M3 for music production.",
            "Zoe owns a desktop PC for gaming.",
        ],
        mutations=[("release", "everything about Xavier's hardware")],
        final_query="Who owns a MacBook Pro 16?",
        must_contain=["Yara"],
        must_not_contain=["Xavier owns a MacBook"],
    ),
    GeneratedCase(
        id="adv_shared_attribute_11",
        family="amnesia",
        # intent: two members same allergy; forget one
        setup_facts=[
            "Aiden has a severe shellfish allergy and carries EpiPen.",
            "Bella has a severe shellfish allergy noted in the chart.",
            "Caleb has no known allergies.",
        ],
        mutations=[("release", "everything about Aiden the patient")],
        final_query="Who else has a shellfish allergy?",
        must_contain=["Bella"],
        must_not_contain=["Aiden has a severe shellfish"],
    ),
    GeneratedCase(
        id="adv_shared_attribute_12",
        family="amnesia",
        # intent: two collaborators on same paper; forget one
        setup_facts=[
            "Diana co-authored the 2025 latency paper at SIGMOD.",
            "Ethan co-authored the 2025 latency paper at SIGMOD.",
            "Felix co-authored a different paper at VLDB.",
        ],
        mutations=[("release", "everything about Diana's research")],
        final_query="Who else worked on the 2025 SIGMOD latency paper?",
        must_contain=["Ethan"],
        must_not_contain=["Diana co-authored the 2025"],
    ),
    GeneratedCase(
        id="adv_shared_attribute_13",
        family="amnesia",
        # intent: two subscribers share coupon code (different from earlier)
        setup_facts=[
            "Subscriber gina@x.com applied coupon AUTUMN15 at checkout.",
            "Subscriber hugo@y.com applied coupon AUTUMN15 at checkout.",
            "Subscriber iris@z.com purchased a separate plan.",
        ],
        mutations=[("release", "everything about gina@x.com")],
        final_query="Who else applied AUTUMN15?",
        must_contain=["hugo@y.com"],
        must_not_contain=["gina@x.com applied coupon"],
    ),
    GeneratedCase(
        id="adv_shared_attribute_14",
        family="amnesia",
        # intent: two streets share neighborhood; forget one resident
        setup_facts=[
            "Resident Jin lives on Oak Street in the Mission district.",
            "Resident Kai lives on Pine Street in the Mission district.",
            "Resident Lily lives in the Sunset district entirely.",
        ],
        mutations=[("release", "everything about resident Jin")],
        final_query="Who else lives in the Mission district?",
        must_contain=["Kai"],
        must_not_contain=["Jin lives on Oak Street"],
    ),
    GeneratedCase(
        id="adv_shared_attribute_15",
        family="amnesia",
        # intent: two students same advisor; forget one
        setup_facts=[
            "Student Maya is advised by Professor Chen in the NLP group.",
            "Student Noah is advised by Professor Chen in the NLP group.",
            "Student Owen has a different advisor in the systems group.",
        ],
        mutations=[("release", "everything about student Maya")],
        final_query="Who else is advised by Professor Chen?",
        must_contain=["Noah"],
        must_not_contain=["Maya is advised by Professor Chen"],
    ),
    GeneratedCase(
        id="adv_shared_attribute_16",
        family="amnesia",
        # intent: two athletes same sport + team; forget one
        setup_facts=[
            "Athlete Paula plays striker for FC Lisbon.",
            "Athlete Quentin plays striker for FC Lisbon.",
            "Athlete Ruth plays goalkeeper for Sporting Madrid.",
        ],
        mutations=[("release", "everything about athlete Paula")],
        final_query="Who else plays striker for FC Lisbon?",
        must_contain=["Quentin"],
        must_not_contain=["Paula plays striker"],
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
    # ─── v0.4 expansion: 8 more identifier-obfuscation cases ─────────
    GeneratedCase(
        id="adv_identifier_obfuscation_09",
        family="purge",
        # intent: IP address with/without zero padding
        setup_facts=[
            "Server IP 192.168.001.005 hosts the staging environment.",
            "Same server short form 192.168.1.5 (no padding) was logged.",
            "Office WiFi password rotates monthly.",
        ],
        mutations=[("purge", "server 192.168.1.5 record")],
        final_query="What does 192.168.1.5 host?",
        must_contain=[],
        must_not_contain=["192.168.001.005 hosts", "192.168.1.5 (no padding)"],
    ),
    GeneratedCase(
        id="adv_identifier_obfuscation_10",
        family="purge",
        # intent: URL with/without trailing slash + with/without https
        setup_facts=[
            "Webhook endpoint https://api.acme.io/v2/events fires on push.",
            "Same endpoint http-form api.acme.io/v2/events/ (trailing slash) is in legacy docs.",
            "Slack channel was renamed last week.",
        ],
        mutations=[("purge", "api.acme.io/v2/events webhook")],
        final_query="Where does the webhook fire?",
        must_contain=[],
        must_not_contain=["https://api.acme.io/v2/events fires",
                          "api.acme.io/v2/events/ (trailing slash)"],
    ),
    GeneratedCase(
        id="adv_identifier_obfuscation_11",
        family="purge",
        # intent: same name with Mr. / Dr. / Prof. prefix
        setup_facts=[
            "Mr. Alan Turing logged into the lab system.",
            "Dr. Alan Turing accessed the same project repository.",
            "Office plants were watered this morning.",
        ],
        mutations=[("purge", "Alan Turing user account")],
        final_query="Does Alan Turing have any sessions on file?",
        must_contain=[],
        must_not_contain=["Mr. Alan Turing logged", "Dr. Alan Turing accessed"],
    ),
    GeneratedCase(
        id="adv_identifier_obfuscation_12",
        family="purge",
        # intent: email with + tag variations
        setup_facts=[
            "Customer ada@example.com signed up in January.",
            "Same customer email-with-tag: ada+promo@example.com used a coupon.",
            "Print queue was migrated to AirPrint.",
        ],
        mutations=[("purge", "ada@example.com customer")],
        final_query="Does ada+promo@example.com still have coupon history?",
        must_contain=[],
        must_not_contain=["ada@example.com signed up",
                          "ada+promo@example.com used a coupon"],
    ),
    GeneratedCase(
        id="adv_identifier_obfuscation_13",
        family="purge",
        # intent: nick name vs full name
        setup_facts=[
            "Robert Smith was hired on 2023-01-15.",
            "Same employee, nickname Bob Smith, was promoted in 2024.",
            "Annual review window opens next month.",
        ],
        mutations=[("purge", "Robert Smith / Bob Smith employee record")],
        final_query="Show me Bob Smith's promotion record.",
        must_contain=[],
        must_not_contain=["Robert Smith was hired",
                          "Bob Smith, was promoted"],
    ),
    GeneratedCase(
        id="adv_identifier_obfuscation_14",
        family="purge",
        # intent: same project ID with/without prefix abbreviation
        setup_facts=[
            "Project ID PRJ-ATLAS-2024 has 5 milestones.",
            "Same project shortcut form ATLAS-2024 (no PRJ prefix) is in slides.",
            "Office snacks were restocked.",
        ],
        mutations=[("purge", "project ATLAS-2024 PRJ-ATLAS-2024 record")],
        final_query="How many milestones in ATLAS-2024?",
        must_contain=[],
        must_not_contain=["PRJ-ATLAS-2024 has 5",
                          "ATLAS-2024 (no PRJ prefix)"],
    ),
    GeneratedCase(
        id="adv_identifier_obfuscation_15",
        family="purge",
        # intent: date format variations YYYY-MM-DD vs MM/DD/YYYY
        setup_facts=[
            "Event scheduled for 2025-12-25 is the company gala.",
            "Same event in US-format: 12/25/2025 will be at the convention center.",
            "Holiday schedule was published.",
        ],
        mutations=[("purge", "event 2025-12-25 gala")],
        final_query="Where is the 12/25/2025 event held?",
        must_contain=[],
        must_not_contain=["2025-12-25 is the company gala",
                          "12/25/2025 will be at"],
    ),
    GeneratedCase(
        id="adv_identifier_obfuscation_16",
        family="purge",
        # intent: account ID with currency suffix
        setup_facts=[
            "Account 1234-USD holds the dollar balance.",
            "Same account 1234 (without currency suffix) appears in the ledger.",
            "Audit was completed last quarter.",
        ],
        mutations=[("purge", "account 1234 / 1234-USD record")],
        final_query="What is the balance on account 1234?",
        must_contain=[],
        must_not_contain=["1234-USD holds the dollar",
                          "1234 (without currency suffix)"],
    ),
]


# ─── 9. cross_lingual_identifier (v0.3) ────────────────────────────
# Family: purge.  Attack: the same logical entity is stored in
# multiple language scripts or romanizations.  Purge by one form
# should reach the others (GDPR / multilingual deployments).

ADV_CROSS_LINGUAL_IDENTIFIER = [
    GeneratedCase(
        id="adv_cross_lingual_identifier_01",
        family="purge",
        # intent: Chinese name + Latin romanization refer to same person.
        setup_facts=[
            "Customer 张伟 placed an order in March.",
            "The same customer in pinyin: Zhang Wei subscribed to Plan A.",
            "Marketing team rotated coupons last quarter.",
        ],
        mutations=[("purge", "customer Zhang Wei aka 张伟")],
        final_query="What orders does Zhang Wei have?",
        must_contain=[],
        must_not_contain=["张伟", "Zhang Wei subscribed"],
    ),
    GeneratedCase(
        id="adv_cross_lingual_identifier_02",
        family="purge",
        # intent: Japanese name in kanji vs romaji.
        setup_facts=[
            "Patient 田中太郎 was prescribed aspirin.",
            "Same patient romaji form: Tanaka Taro has hypertension.",
            "Pharmacy was inspected on Tuesday.",
        ],
        mutations=[("purge", "patient Tanaka Taro / 田中太郎")],
        final_query="What medication did Tanaka Taro receive?",
        must_contain=[],
        must_not_contain=["田中太郎 was prescribed", "Tanaka Taro has"],
    ),
    GeneratedCase(
        id="adv_cross_lingual_identifier_03",
        family="purge",
        # intent: Korean hangul vs romanization.
        setup_facts=[
            "User 김민지 logged in from Seoul.",
            "Same user as Kim Minji has 2FA enabled.",
            "Office cafeteria is closed on Sundays.",
        ],
        mutations=[("purge", "Kim Minji 김민지 account")],
        final_query="Show me Kim Minji's login activity.",
        must_contain=[],
        must_not_contain=["김민지 logged", "Kim Minji has"],
    ),
    GeneratedCase(
        id="adv_cross_lingual_identifier_04",
        family="purge",
        # intent: Russian Cyrillic vs Latin transliteration.
        setup_facts=[
            "Customer Иван Петров pays via wire.",
            "Same customer (Ivan Petrov, transliterated) has Premium tier.",
            "Quarterly sales meeting is in November.",
        ],
        mutations=[("purge", "Ivan Petrov / Иван Петров")],
        final_query="What tier is Ivan Petrov on?",
        must_contain=[],
        must_not_contain=["Иван Петров pays", "Ivan Petrov, transliterated"],
    ),
    GeneratedCase(
        id="adv_cross_lingual_identifier_05",
        family="purge",
        # intent: Arabic name in Arabic script vs Latin transliteration.
        setup_facts=[
            "Patient محمد علي is allergic to penicillin.",
            "Same patient as Mohammed Ali had surgery last year.",
            "Lab equipment was upgraded recently.",
        ],
        mutations=[("purge", "Mohammed Ali / محمد علي patient record")],
        final_query="Is Mohammed Ali allergic to anything?",
        must_contain=[],
        must_not_contain=["محمد علي is allergic", "Mohammed Ali had surgery"],
    ),
    GeneratedCase(
        id="adv_cross_lingual_identifier_06",
        family="purge",
        # intent: company in English vs Chinese.
        setup_facts=[
            "Vendor 阿里巴巴 shipped 50 units.",
            "Same vendor as Alibaba sent invoice for $5000.",
            "Annual conference is in Las Vegas this year.",
        ],
        mutations=[("purge", "vendor Alibaba 阿里巴巴")],
        final_query="What did Alibaba ship?",
        must_contain=[],
        must_not_contain=["阿里巴巴 shipped", "Alibaba sent invoice"],
    ),
    GeneratedCase(
        id="adv_cross_lingual_identifier_07",
        family="purge",
        # intent: Spanish name with accents vs without.
        setup_facts=[
            "Customer José García placed order #42.",
            "Same customer as Jose Garcia (no accents) has Plan B.",
            "Office supply order was approved.",
        ],
        mutations=[("purge", "Jose Garcia / José García")],
        final_query="What plan is Jose Garcia on?",
        must_contain=[],
        must_not_contain=["José García placed", "Jose Garcia (no accents)"],
    ),
    GeneratedCase(
        id="adv_cross_lingual_identifier_08",
        family="purge",
        # intent: German Eszett (ß) vs ss.
        setup_facts=[
            "Vendor Großmann GmbH delivered components.",
            "Same vendor as Grossmann GmbH (without ß) sent the invoice.",
            "Holiday schedule was distributed.",
        ],
        mutations=[("purge", "Grossmann / Großmann GmbH")],
        final_query="Show me Grossmann's recent activity.",
        must_contain=[],
        must_not_contain=["Großmann GmbH delivered", "Grossmann GmbH (without"],
    ),
    # ─── v0.4 expansion: 8 more cross-lingual cases ──────────────────
    GeneratedCase(
        id="adv_cross_lingual_identifier_09",
        family="purge",
        # intent: Hindi devanagari vs Latin transliteration
        setup_facts=[
            "Customer राज शर्मा placed an order.",
            "Same customer as Raj Sharma (transliterated) has a refund pending.",
            "Holiday calendar was updated.",
        ],
        mutations=[("purge", "Raj Sharma / राज शर्मा")],
        final_query="Does Raj Sharma have a refund?",
        must_contain=[],
        must_not_contain=["राज शर्मा placed", "Raj Sharma (transliterated)"],
    ),
    GeneratedCase(
        id="adv_cross_lingual_identifier_10",
        family="purge",
        # intent: Thai script vs romanization
        setup_facts=[
            "Vendor สมชาย ใจดี delivered the parts.",
            "Same vendor Somchai Jaidee (romanized) issued an invoice.",
            "Warehouse layout was reorganized.",
        ],
        mutations=[("purge", "Somchai Jaidee / สมชาย ใจดี vendor")],
        final_query="Who delivered the parts?",
        must_contain=[],
        must_not_contain=["สมชาย ใจดี delivered",
                          "Somchai Jaidee (romanized)"],
    ),
    GeneratedCase(
        id="adv_cross_lingual_identifier_11",
        family="purge",
        # intent: Greek vs Latin transliteration
        setup_facts=[
            "Customer Νίκος Παπαδόπουλος is on the renewal list.",
            "Same customer Nikos Papadopoulos (Latin) paid by card.",
            "DNS migration finished overnight.",
        ],
        mutations=[("purge", "Nikos Papadopoulos / Νίκος Παπαδόπουλος")],
        final_query="How did Nikos Papadopoulos pay?",
        must_contain=[],
        must_not_contain=["Νίκος Παπαδόπουλος is on",
                          "Nikos Papadopoulos (Latin)"],
    ),
    GeneratedCase(
        id="adv_cross_lingual_identifier_12",
        family="purge",
        # intent: Hebrew vs Latin transliteration
        setup_facts=[
            "Patient דוד כהן has an upcoming appointment.",
            "Same patient David Cohen (transliterated) is on medication.",
            "Office printer toner was replaced.",
        ],
        mutations=[("purge", "David Cohen / דוד כהן patient")],
        final_query="What medication is David Cohen taking?",
        must_contain=[],
        must_not_contain=["דוד כהן has an upcoming",
                          "David Cohen (transliterated)"],
    ),
    GeneratedCase(
        id="adv_cross_lingual_identifier_13",
        family="purge",
        # intent: city in Chinese vs English
        setup_facts=[
            "Branch office 北京 reports to corporate quarterly.",
            "Same branch office Beijing (English) hosts 120 employees.",
            "Travel policy was revised in October.",
        ],
        mutations=[("purge", "Beijing / 北京 branch office")],
        final_query="How many employees in the Beijing branch?",
        must_contain=[],
        must_not_contain=["北京 reports to", "Beijing (English) hosts"],
    ),
    GeneratedCase(
        id="adv_cross_lingual_identifier_14",
        family="purge",
        # intent: Vietnamese diacritics vs without
        setup_facts=[
            "Engineer Nguyễn Văn Hùng filed a security report.",
            "Same engineer Nguyen Van Hung (no diacritics) was promoted last cycle.",
            "Quarterly OKR review is next Monday.",
        ],
        mutations=[("purge", "Nguyen Van Hung / Nguyễn Văn Hùng record")],
        final_query="Was Nguyen Van Hung promoted?",
        must_contain=[],
        must_not_contain=["Nguyễn Văn Hùng filed",
                          "Nguyen Van Hung (no diacritics) was"],
    ),
    GeneratedCase(
        id="adv_cross_lingual_identifier_15",
        family="purge",
        # intent: French name with vs without accents
        setup_facts=[
            "Client François Müller paid the September invoice.",
            "Same client Francois Muller (no accents) requested a refund.",
            "Conference room schedule was updated.",
        ],
        mutations=[("purge", "Francois Muller / François Müller client")],
        final_query="Did Francois Muller request a refund?",
        must_contain=[],
        must_not_contain=["François Müller paid",
                          "Francois Muller (no accents) requested"],
    ),
    GeneratedCase(
        id="adv_cross_lingual_identifier_16",
        family="purge",
        # intent: emoji-containing handle vs plain ascii
        setup_facts=[
            "Reviewer @dev🚀 left feedback on PR #42.",
            "Same reviewer plain form: @dev (no emoji) approved PR #45.",
            "CI runner pool was scaled up.",
        ],
        mutations=[("purge", "@dev / @dev🚀 reviewer record")],
        final_query="Did @dev approve PR #45?",
        must_contain=[],
        must_not_contain=["@dev🚀 left feedback",
                          "@dev (no emoji) approved"],
    ),
]


# ─── 10. recursive_supersession (v0.3) ─────────────────────────────
# Family: drift.  Attack: a supersession chain where the LATEST state
# matches an earlier-superseded state.  Tests whether the system can
# correctly identify the current state when "now back to X" returns
# to an old value.  Common in real preferences: "tried Y, didn't like
# it, switched back to X."

ADV_RECURSIVE_SUPERSESSION = [
    GeneratedCase(
        id="adv_recursive_supersession_01",
        family="drift",
        # intent: Chrome → Brave → Chrome.  Final state = Chrome again.
        setup_facts=[
            "User has been using Chrome as primary browser since 2020.",
        ],
        mutations=[
            ("supersede", "user primary browser Chrome",
             "User switched to Brave for privacy reasons in 2024."),
            ("supersede", "user primary browser Brave",
             "User switched back to Chrome in 2025 after Brave broke an extension."),
        ],
        final_query="What is the user's current browser?",
        must_contain=["Chrome"],
        must_not_contain=["switched to Brave"],
    ),
    GeneratedCase(
        id="adv_recursive_supersession_02",
        family="drift",
        # intent: vegetarian → omnivore → vegetarian again.
        setup_facts=[
            "User was vegetarian throughout 2019.",
        ],
        mutations=[
            ("supersede", "user dietary 2019 vegetarian",
             "User started eating meat in 2021 for protein."),
            ("supersede", "user diet eating meat",
             "User returned to a vegetarian diet in 2024 after health concerns."),
        ],
        final_query="Does the user eat meat now?",
        must_contain=["vegetarian"],
        must_not_contain=["started eating meat in 2021"],
    ),
    GeneratedCase(
        id="adv_recursive_supersession_03",
        family="drift",
        # intent: Apple → Android → Apple.
        setup_facts=[
            "User has an iPhone 12 as their primary device.",
        ],
        mutations=[
            ("supersede", "user phone iPhone",
             "User switched to a Samsung Galaxy in 2023 for the camera."),
            ("supersede", "user phone Samsung",
             "User came back to iPhone 15 Pro in 2025 for ecosystem reasons."),
        ],
        final_query="What phone does the user have?",
        must_contain=["iPhone 15 Pro"],
        must_not_contain=["Samsung Galaxy"],
    ),
    GeneratedCase(
        id="adv_recursive_supersession_04",
        family="drift",
        # intent: Berlin → Lisbon → Berlin.
        setup_facts=[
            "User lived in Berlin from 2015 to 2020.",
        ],
        mutations=[
            ("supersede", "user residence Berlin 2015",
             "User moved to Lisbon in 2020 for the climate."),
            ("supersede", "user residence Lisbon",
             "User returned to Berlin in 2024 after their family relocated."),
        ],
        final_query="Where does the user live now?",
        must_contain=["Berlin"],
        must_not_contain=["moved to Lisbon in 2020"],
    ),
    GeneratedCase(
        id="adv_recursive_supersession_05",
        family="drift",
        # intent: Mac → Linux → Mac (developer workstation).
        setup_facts=[
            "Developer uses macOS on their primary workstation.",
        ],
        mutations=[
            ("supersede", "developer workstation OS",
             "Developer switched to Ubuntu Linux in 2023 for kernel work."),
            ("supersede", "developer workstation Linux",
             "Developer switched back to macOS in 2025 for video editing."),
        ],
        final_query="What OS does the developer use?",
        must_contain=["macOS"],
        must_not_contain=["Ubuntu Linux in 2023"],
    ),
    GeneratedCase(
        id="adv_recursive_supersession_06",
        family="drift",
        # intent: dog → cat → dog.
        setup_facts=[
            "User owned a Golden Retriever named Buddy until 2020.",
        ],
        mutations=[
            ("supersede", "user pet Golden Retriever",
             "User adopted two cats in 2021 after Buddy passed away."),
            ("supersede", "user pets cats",
             "User adopted a Border Collie puppy in 2025; the cats live with relatives."),
        ],
        final_query="What kind of pet does the user have now?",
        must_contain=["Border Collie"],
        must_not_contain=["adopted two cats in 2021"],
    ),
    GeneratedCase(
        id="adv_recursive_supersession_07",
        family="drift",
        # intent: married → divorced → married (different spouse).
        setup_facts=[
            "User was married to Alex in 2015.",
        ],
        mutations=[
            ("supersede", "user marital status Alex",
             "User got divorced from Alex in 2020."),
            ("supersede", "user marital status divorced",
             "User married Jordan in 2024 after a long engagement."),
        ],
        final_query="Is the user married?",
        must_contain=["married Jordan"],
        must_not_contain=["divorced from Alex"],
    ),
    GeneratedCase(
        id="adv_recursive_supersession_08",
        family="drift",
        # intent: Slack → Discord → Slack at workplace.
        setup_facts=[
            "Team uses Slack as primary communication tool since 2018.",
        ],
        mutations=[
            ("supersede", "team primary communication Slack",
             "Team migrated to Discord in 2023 to save on costs."),
            ("supersede", "team communication Discord",
             "Team moved back to Slack in 2025 after Discord lacked compliance features."),
        ],
        final_query="What tool does the team use for communication?",
        must_contain=["Slack"],
        must_not_contain=["migrated to Discord in 2023"],
    ),
]


# ─── master list ───────────────────────────────────────────────────

ATTACK_CATEGORIES: dict[str, list[GeneratedCase]] = {
    "substring_trap":           ADV_SUBSTRING_TRAP,
    "prefix_collision":         ADV_PREFIX_COLLISION,
    "paraphrase_supersession":  ADV_PARAPHRASE,
    "negation_trap":            ADV_NEGATION,
    "temporal_qualifier":       ADV_TEMPORAL,
    "shared_attribute":         ADV_SHARED_ATTR,
    "compound_fact":            ADV_COMPOUND,
    "identifier_obfuscation":   ADV_IDENTIFIER_OBFUSCATION,
    "cross_lingual_identifier": ADV_CROSS_LINGUAL_IDENTIFIER,
    "recursive_supersession":   ADV_RECURSIVE_SUPERSESSION,
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
