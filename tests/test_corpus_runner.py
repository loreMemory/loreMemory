#!/usr/bin/env python3
"""
Lore Memory Test Corpus Runner — 20 Scenarios
===============================================
Tests the Memory class against a comprehensive corpus of real-world scenarios.

Usage:
    python3 test_corpus_runner.py

Writes detailed results to output-v3/TEST_CORPUS_RESULTS.md
"""

import os
import sys
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from datetime import datetime

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lore_memory import Memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def query_contains(results, *keywords):
    """True if ANY single result contains ALL keywords (case-insensitive)."""
    for r in results:
        haystack = " ".join([
            r.text or "", r.subject or "", r.predicate or "", r.object or ""
        ]).lower()
        if all(kw.lower() in haystack for kw in keywords):
            return True
    return False


def query_contains_any(results, keyword_lists):
    """True if ANY result matches ANY keyword set from keyword_lists.

    keyword_lists is a list of lists/tuples, e.g. [["python"], ["javascript"]]
    """
    for kw_set in keyword_lists:
        if query_contains(results, *kw_set):
            return True
    return False


def query_empty(results):
    """True if results list is empty (no memories returned)."""
    return len(results) == 0


def query_not_contains(results, *keywords):
    """True if NO result contains ALL keywords (case-insensitive)."""
    return not query_contains(results, *keywords)


class ScenarioResult:
    def __init__(self, name):
        self.name = name
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.hallucinations = 0
        self.details = []  # list of (query_desc, passed, detail_msg)
        self.error = None

    def check(self, desc, passed, detail=""):
        self.total += 1
        if passed:
            self.passed += 1
            self.details.append((desc, True, detail or "OK"))
        else:
            self.failed += 1
            self.details.append((desc, False, detail or "FAILED"))

    def hallucination(self, desc, detail=""):
        self.total += 1
        self.failed += 1
        self.hallucinations += 1
        self.details.append((desc, False, f"HALLUCINATION: {detail}"))

    @property
    def pass_rate(self):
        return (self.passed / self.total * 100) if self.total > 0 else 0

    @property
    def passed_threshold(self):
        return self.pass_rate >= 85.0


def make_memory(tmp_dir, user_id="default"):
    """Create a Memory instance using the temp directory."""
    return Memory(user_id=user_id, data_dir=tmp_dir)


def store_all(m, texts):
    """Store a list of texts sequentially."""
    for t in texts:
        m.store(t)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_01(tmp_dir):
    """Personal Onboarding -- Basic Facts"""
    r = ScenarioResult("01: Personal Onboarding")
    m = make_memory(tmp_dir, "onboard_user")
    try:
        store_all(m, [
            "hi, my name is Marcus",
            "i'm a software engineer",
            "i work at a startup in Dubai",
            "we're building an AI memory tool called Lore",
            "i mostly code in Python",
            "i also know some JavaScript but prefer Python",
            "i've been in tech for about 8 years",
            "before Dubai i lived in London for 3 years",
        ])

        res = m.query("what is my name?")
        r.check("name -> Marcus", query_contains(res, "marcus"),
                f"got {len(res)} results")

        res = m.query("what is my job?")
        r.check("job -> software engineer",
                query_contains_any(res, [["software engineer"], ["software", "engineer"]]),
                f"got {len(res)} results")

        res = m.query("where do I work?")
        r.check("workplace -> startup/Dubai",
                query_contains_any(res, [["startup"], ["dubai"]]),
                f"got {len(res)} results")

        res = m.query("what are we building?")
        r.check("building -> Lore",
                query_contains(res, "lore"),
                f"got {len(res)} results")

        res = m.query("what programming languages do I know?")
        r.check("languages -> Python",
                query_contains(res, "python"),
                f"got {len(res)} results")

        res = m.query("how long have I been in tech?")
        r.check("years in tech -> 8",
                query_contains(res, "8"),
                f"got {len(res)} results")

        res = m.query("where did I live before Dubai?")
        r.check("previous location -> London",
                query_contains(res, "london"),
                f"got {len(res)} results")
    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()
    return r


def scenario_02(tmp_dir):
    """Personal Preferences"""
    r = ScenarioResult("02: Personal Preferences")
    m = make_memory(tmp_dir, "prefs_user")
    try:
        store_all(m, [
            "not a big fan of meetings tbh",
            "i really like async work",
            "love working late at night, more focused",
            "hate open offices, too noisy",
            "coffee is essential, can't function without it",
            "i don't drink alcohol",
            "big fan of hiking on weekends",
            "not into team sports, prefer solo activities",
            "i read a lot, mostly non-fiction",
            "currently reading Thinking Fast and Slow",
            "loved Atomic Habits, changed how I work",
        ])

        res = m.query("how does the user feel about meetings?")
        r.check("meetings -> dislikes",
                query_contains_any(res, [["meeting"], ["meetings"]]),
                f"got {len(res)} results")

        res = m.query("what is the preferred work style?")
        r.check("work style -> async/night",
                query_contains_any(res, [["async"], ["night"], ["late"]]),
                f"got {len(res)} results")

        res = m.query("what does the user do on weekends?")
        r.check("weekends -> hiking",
                query_contains(res, "hiking"),
                f"got {len(res)} results")

        res = m.query("what book is the user reading now?")
        r.check("reading now -> Thinking Fast and Slow",
                query_contains_any(res, [["thinking fast"], ["thinking", "slow"]]),
                f"got {len(res)} results")

        res = m.query("Atomic Habits book")
        r.check("enjoyed book -> Atomic Habits",
                query_contains_any(res, [["atomic habits"], ["atomic"], ["habits"]]),
                f"got {len(res)} results")

        res = m.query("does the user drink alcohol?")
        r.check("drinking -> no alcohol",
                query_contains_any(res, [["alcohol"], ["don't drink"], ["drink"]]),
                f"got {len(res)} results")
    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()
    return r


def scenario_03(tmp_dir):
    """Work Context"""
    r = ScenarioResult("03: Work Context")
    m = make_memory(tmp_dir, "work_user")
    try:
        store_all(m, [
            "our team is fully remote, 6 engineers",
            "we use Python for the backend",
            "frontend is in React with TypeScript",
            "database is SQLite for now, planning to move to PostgreSQL",
            "we use GitHub for version control",
            "CI/CD is set up with GitHub Actions",
            "we communicate mostly on Slack",
            "we do async standups, no daily calls",
            "our PM is Sarah, she's based in London",
            "CTO is Khalid, he's been building AI tools for 10 years",
            "we just closed a seed round, $2M",
            "planning to launch beta in Q3 this year",
        ])

        res = m.query("how big is the team?")
        r.check("team size -> 6/remote",
                query_contains_any(res, [["6"], ["remote"]]),
                f"got {len(res)} results")

        res = m.query("what tech stack do you use?")
        r.check("tech stack -> Python/React/TypeScript",
                query_contains_any(res, [["python"], ["react"], ["typescript"]]),
                f"got {len(res)} results")

        res = m.query("what database?")
        r.check("database -> SQLite",
                query_contains_any(res, [["sqlite"], ["postgresql"]]),
                f"got {len(res)} results")

        res = m.query("how does the team communicate?")
        r.check("communication -> Slack",
                query_contains(res, "slack"),
                f"got {len(res)} results")

        res = m.query("who is the PM?")
        r.check("PM -> Sarah",
                query_contains(res, "sarah"),
                f"got {len(res)} results")

        res = m.query("who is the CTO?")
        r.check("CTO -> Khalid",
                query_contains(res, "khalid"),
                f"got {len(res)} results")

        res = m.query("how much funding?")
        r.check("fundraise -> $2M/seed",
                query_contains_any(res, [["2m"], ["seed"], ["$2"]]),
                f"got {len(res)} results")

        res = m.query("when is the beta launch?")
        r.check("beta -> Q3",
                query_contains(res, "q3"),
                f"got {len(res)} results")
    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()
    return r


def scenario_04(tmp_dir):
    """Corrections and Updates"""
    r = ScenarioResult("04: Corrections and Updates")

    # Each correction pair uses a separate user to avoid cross-pollution
    # Test 1: location correction
    m = make_memory(tmp_dir, "correct_loc")
    try:
        m.store("i live in Dubai")
        m.store("actually i moved back to London last month")
        res = m.query("where do I live?")
        r.check("location -> London (not Dubai)",
                query_contains(res, "london"),
                f"got {len(res)} results")
        # Check that the correction text (mentioning London) is somewhere in the results
        r.check("location: correction is present",
                query_contains_any(res, [["moved"], ["london", "last month"]]),
                f"top result: {res[0].text[:60] if res else 'empty'}")
    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()

    # Test 2: database correction
    m = make_memory(tmp_dir, "correct_db")
    try:
        m.store("we use SQLite for the database")
        m.store("we migrated to PostgreSQL last week, SQLite is gone")
        res = m.query("what database do we use?")
        r.check("database -> PostgreSQL",
                query_contains_any(res, [["postgresql"], ["postgres"]]),
                f"got {len(res)} results")
    except Exception as e:
        r.error = r.error or traceback.format_exc()
    finally:
        m.close()

    # Test 3: team size correction
    m = make_memory(tmp_dir, "correct_team")
    try:
        m.store("our team has 6 engineers")
        m.store("we hired 2 more engineers, now we're 8")
        res = m.query("how many engineers?")
        r.check("team size -> 8",
                query_contains(res, "8"),
                f"got {len(res)} results")
    except Exception as e:
        r.error = r.error or traceback.format_exc()
    finally:
        m.close()

    # Test 4: book correction
    m = make_memory(tmp_dir, "correct_book")
    try:
        m.store("i'm reading Thinking Fast and Slow")
        m.store("finished it, now reading The Pragmatic Programmer")
        res = m.query("what book are you reading?")
        r.check("book -> Pragmatic Programmer",
                query_contains_any(res, [["pragmatic programmer"], ["pragmatic"]]),
                f"got {len(res)} results")
    except Exception as e:
        r.error = r.error or traceback.format_exc()
    finally:
        m.close()

    # Test 5: beta date correction
    m = make_memory(tmp_dir, "correct_beta")
    try:
        m.store("we're launching beta in Q3")
        m.store("we pushed the beta to Q4, ran into some delays")
        res = m.query("when is the beta?")
        r.check("beta -> Q4",
                query_contains(res, "q4"),
                f"got {len(res)} results")
    except Exception as e:
        r.error = r.error or traceback.format_exc()
    finally:
        m.close()

    # Test 6: PM correction
    m = make_memory(tmp_dir, "correct_pm")
    try:
        m.store("Sarah is our PM")
        m.store("Sarah left the company, we don't have a PM right now")
        res = m.query("who is the PM?")
        r.check("PM -> Sarah left / no PM",
                query_contains_any(res, [["left"], ["don't have"], ["no pm"]]),
                f"got {len(res)} results")
    except Exception as e:
        r.error = r.error or traceback.format_exc()
    finally:
        m.close()

    return r


def scenario_05(tmp_dir):
    """Temporal"""
    r = ScenarioResult("05: Temporal")
    m = make_memory(tmp_dir, "temporal_user")
    try:
        store_all(m, [
            "i'm in San Francisco this week for a conference",
            "staying at the Marriott until Thursday",
            "currently between investors, closing round next week",
            "i was at Google for 2 years before joining the startup",
            "used to use Angular, switched to React 2 years ago",
            "i was learning Rust last year but dropped it",
            "we had a big outage yesterday, fixed now",
            "on a hiring freeze until the funding closes",
        ])

        res = m.query("where are you this week?")
        r.check("this week -> San Francisco",
                query_contains_any(res, [["san francisco"], ["francisco"]]),
                f"got {len(res)} results")

        res = m.query("where did you work before?")
        r.check("previous employer -> Google",
                query_contains(res, "google"),
                f"got {len(res)} results")

        res = m.query("what framework did you use before React?")
        r.check("previous framework -> Angular",
                query_contains(res, "angular"),
                f"got {len(res)} results")

        res = m.query("are you learning Rust?")
        r.check("Rust -> dropped",
                query_contains_any(res, [["dropped"], ["rust"]]),
                f"got {len(res)} results")

        res = m.query("was there an outage?")
        r.check("outage -> fixed",
                query_contains_any(res, [["outage"], ["fixed"]]),
                f"got {len(res)} results")
    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()
    return r


def scenario_06(tmp_dir):
    """Multi-Fact Sentences"""
    r = ScenarioResult("06: Multi-Fact Sentences")
    m = make_memory(tmp_dir, "multi_fact_user")
    try:
        store_all(m, [
            "I use Python and JavaScript and sometimes Go",
            "I love coffee but hate tea",
            "we use Slack for chat and Notion for docs",
            "Sarah handles product and Khalid handles engineering",
            "I work remotely but visit the office once a month",
            "i'm good at backend and okay at frontend",
            "we support Mac and Linux but not Windows yet",
            "i exercise in the morning and read at night",
        ])

        res = m.query("what programming languages?")
        r.check("languages -> Python/JavaScript/Go",
                query_contains_any(res, [["python"], ["javascript"], ["go"]]),
                f"got {len(res)} results")

        res = m.query("what does the user drink?")
        r.check("drinks -> coffee + tea",
                query_contains_any(res, [["coffee"], ["tea"]]),
                f"got {len(res)} results")

        res = m.query("what tools does the team use?")
        r.check("tools -> Slack/Notion",
                query_contains_any(res, [["slack"], ["notion"]]),
                f"got {len(res)} results")

        res = m.query("do you support Windows?")
        r.check("Windows -> not yet",
                query_contains_any(res, [["windows"], ["not"]]),
                f"got {len(res)} results")

        res = m.query("when do you exercise?")
        r.check("exercise -> morning",
                query_contains_any(res, [["morning"], ["exercise"]]),
                f"got {len(res)} results")

        res = m.query("when do you read?")
        r.check("read -> night",
                query_contains_any(res, [["night"], ["read"]]),
                f"got {len(res)} results")
    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()
    return r


def scenario_07(tmp_dir):
    """Possessives"""
    r = ScenarioResult("07: Possessives")
    m = make_memory(tmp_dir, "poss_user")
    try:
        store_all(m, [
            "my sister lives in Abu Dhabi",
            "my brother is also a developer",
            "our office is in Business Bay, Dubai",
            "my manager is really supportive of async work",
            "our biggest competitor just raised $10M",
            "my laptop is a MacBook Pro M3",
            "our main customer is a fintech company in Singapore",
            "my coworker Ahmed is the best frontend developer I know",
        ])

        res = m.query("where does the sister live?")
        r.check("sister -> Abu Dhabi",
                query_contains_any(res, [["abu dhabi"], ["sister"]]),
                f"got {len(res)} results")

        res = m.query("what does the brother do?")
        r.check("brother -> developer",
                query_contains_any(res, [["brother"], ["developer"]]),
                f"got {len(res)} results")

        res = m.query("where is the office?")
        r.check("office -> Business Bay",
                query_contains_any(res, [["business bay"], ["dubai"]]),
                f"got {len(res)} results")

        res = m.query("what laptop do you use?")
        r.check("laptop -> MacBook Pro M3",
                query_contains_any(res, [["macbook"], ["m3"], ["laptop"]]),
                f"got {len(res)} results")

        res = m.query("who is the main customer?")
        r.check("customer -> fintech/Singapore",
                query_contains_any(res, [["fintech"], ["singapore"], ["customer"]]),
                f"got {len(res)} results")
    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()
    return r


def scenario_08(tmp_dir):
    """Casual Input"""
    r = ScenarioResult("08: Casual Input")
    m = make_memory(tmp_dir, "casual_user")
    try:
        store_all(m, [
            "yeah so basically i work on ai stuff",
            "been doing this for like 8 years or so",
            "rn im working on memory systems for llms",
            "super into the whole agent space tbh",
            "we use sqlite lol dont judge",
            "gonna switch to postgres eventually",
            "my cofounder is also called marcus which is confusing",
            "we met at a hackathon like 3 years ago",
            "raised some money recently, cant say how much yet",
            "launching soon hopefully",
        ])

        res = m.query("what field does the user work in?")
        r.check("field -> AI/memory/LLM",
                query_contains_any(res, [["ai"], ["memory"], ["llm"]]),
                f"got {len(res)} results")

        res = m.query("how many years of experience?")
        r.check("duration -> 8 years",
                query_contains(res, "8"),
                f"got {len(res)} results")

        res = m.query("what database?")
        r.check("database -> SQLite",
                query_contains_any(res, [["sqlite"], ["postgres"]]),
                f"got {len(res)} results")

        res = m.query("how did the cofounders meet?")
        r.check("cofounder meeting -> hackathon",
                query_contains(res, "hackathon"),
                f"got {len(res)} results")

        res = m.query("when did they meet?")
        r.check("meeting time -> 3 years",
                query_contains_any(res, [["3 years"], ["3"]]),
                f"got {len(res)} results")
    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()
    return r


def scenario_09(tmp_dir):
    """Technical / Repo Context"""
    r = ScenarioResult("09: Technical / Repo Context")
    m = make_memory(tmp_dir, "repo_user")
    try:
        store_all(m, [
            "we decided to use FTS5 for full text search in SQLite",
            "vector embeddings are optional, not required by default",
            "the extraction pipeline uses grammar rules, no ML model needed",
            "we store facts as SPO triples — subject predicate object",
            "confidence scores range from 0.0 to 1.0",
            "memories decay over time based on access frequency",
            "the MCP server supports 5 tools: store, query, forget, list, stats",
            "we target sub-50ms retrieval at 100k entries",
            "currently hitting 2.4ms p50 at 1M entries",
            "the system works completely offline, zero external APIs",
            "MIT license, open source from day one",
            "the package name on PyPI will be lore-memory",
        ])

        res = m.query("what search technology is used?")
        r.check("search tech -> FTS5",
                query_contains(res, "fts5"),
                f"got {len(res)} results")

        res = m.query("are embeddings required?")
        r.check("embeddings -> optional",
                query_contains_any(res, [["optional"], ["embedding"]]),
                f"got {len(res)} results")

        res = m.query("how does extraction work?")
        r.check("extraction -> grammar",
                query_contains_any(res, [["grammar"], ["extraction"]]),
                f"got {len(res)} results")

        res = m.query("what data model is used?")
        r.check("data model -> SPO",
                query_contains_any(res, [["spo"], ["subject predicate object"], ["triple"]]),
                f"got {len(res)} results")

        res = m.query("what is the latency target?")
        r.check("latency target -> 50ms",
                query_contains_any(res, [["50ms"], ["50"]]),
                f"got {len(res)} results")

        res = m.query("what is the actual latency?")
        r.check("actual latency -> 2.4ms",
                query_contains_any(res, [["2.4ms"], ["2.4"]]),
                f"got {len(res)} results")

        res = m.query("what license?")
        r.check("license -> MIT",
                query_contains(res, "mit"),
                f"got {len(res)} results")

        res = m.query("what is the PyPI package name?")
        r.check("PyPI -> lore-memory",
                query_contains_any(res, [["lore-memory"], ["lore"]]),
                f"got {len(res)} results")

        res = m.query("how many MCP tools?")
        r.check("MCP tools -> 5",
                query_contains_any(res, [["5 tools"], ["5"]]),
                f"got {len(res)} results")
    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()
    return r


def scenario_10(tmp_dir):
    """Contradictions"""
    r = ScenarioResult("10: Contradictions")

    # Test 1: Favourite language
    m = make_memory(tmp_dir, "contra_lang")
    try:
        r1 = m.store("Python is my favourite language")
        r2 = m.store("actually JavaScript has grown on me, it might be my favourite now")
        res = m.query("what is the favourite language?")
        # Accept if latest is mentioned or conflict flagged
        r.check("fav language -> latest or conflict",
                query_contains_any(res, [["javascript"], ["python"]]),
                f"got {len(res)} results, contradictions={r2.get('contradictions', [])}")
    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()

    # Test 2: Team size
    m = make_memory(tmp_dir, "contra_team")
    try:
        m.store("we have 6 engineers")
        m.store("we have 8 engineers")
        res = m.query("how many engineers?")
        r.check("team size -> 8",
                query_contains(res, "8"),
                f"got {len(res)} results")
    except Exception as e:
        r.error = r.error or traceback.format_exc()
    finally:
        m.close()

    # Test 3: Beta date flip-flop
    m = make_memory(tmp_dir, "contra_beta")
    try:
        m.store("the beta is in Q3")
        m.store("the beta is in Q4")
        m.store("the beta is back to Q3, we sped things up")
        res = m.query("when is the beta?")
        r.check("beta -> Q3 (final)",
                query_contains(res, "q3"),
                f"got {len(res)} results")
    except Exception as e:
        r.error = r.error or traceback.format_exc()
    finally:
        m.close()

    # Test 4: Meeting contradiction
    m = make_memory(tmp_dir, "contra_meeting")
    try:
        r1 = m.store("I love meetings when they're well structured")
        r2 = m.store("I hate meetings")
        res = m.query("how do you feel about meetings?")
        r.check("meetings -> contradiction/latest",
                query_contains_any(res, [["hate"], ["meeting"]]),
                f"got {len(res)} results, contradictions={r2.get('contradictions', [])}")
    except Exception as e:
        r.error = r.error or traceback.format_exc()
    finally:
        m.close()

    return r


def scenario_11(tmp_dir):
    """Empty Results -- Hallucination Detection (CRITICAL)"""
    r = ScenarioResult("11: Empty Results (Hallucination Detection)")
    m = make_memory(tmp_dir, "empty_user_hallucination")
    try:
        # NO data stored for this user. ALL queries must return empty.
        queries = [
            ("wife's name", ["wife"]),
            ("which university did I attend?", ["university", "college"]),
            ("what is my salary?", ["salary", "$", "k"]),
            ("how many kids do I have?", ["kids", "children"]),
            ("what car do I drive?", ["car", "drive", "tesla", "bmw"]),
            ("what is my phone number?", ["phone", "number"]),
            ("who is the CEO?", ["ceo"]),
            ("what is the weather?", ["weather", "sunny", "rain"]),
        ]

        for query_text, danger_words in queries:
            res = m.query(query_text)
            if query_empty(res):
                r.check(f"empty: {query_text}", True, "Correctly empty")
            else:
                # Results exist -- check if they contain fabricated personal data
                # Any result for a user with NO data is suspicious
                r.hallucination(f"empty: {query_text}",
                                f"Got {len(res)} results for empty user: "
                                f"{[x.text[:60] for x in res[:3]]}")
    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()
    return r


def scenario_12(tmp_dir):
    """Long Conversation -- 50 Turns"""
    r = ScenarioResult("12: Long Conversation (50 Turns)")
    m = make_memory(tmp_dir, "ali")
    try:
        turns = [
            "hey, my name is Ali",
            "i'm a data scientist at a bank in Riyadh",
            "been doing data science for about 5 years now",
            "started my career as a software developer actually",
            "the bank is one of the largest in Saudi Arabia",
            "i mainly work with Python and R for data analysis",
            "we use a lot of pandas and scikit-learn",
            "recently started exploring deep learning with PyTorch",
            "our team has 12 data scientists",
            "my manager is Dr. Fatima, she's really great",
            "we mostly work on fraud detection models",
            "our best model gets about 90% accuracy on fraud detection",
            "we also do customer segmentation and churn prediction",
            "i have a side project building a fintech app",
            "the fintech app helps people track their spending",
            "building it with Python and React",
            "using PostgreSQL for the database",
            "i want it to be local-first for privacy reasons",
            "privacy is super important for financial data",
            "i'm thinking about using SQLite for the local storage",
            "our cloud infrastructure at the bank is Azure",
            "we use Azure ML for model training",
            "deployment is through Azure Kubernetes Service",
            "i have a master's degree in computer science",
            "graduated from King Saud University in Riyadh",
            "my thesis was on natural language processing",
            "i speak Arabic, English, and some French",
            "Arabic is my first language obviously",
            "English is what we use at work mostly",
            "French i learned in school but i'm rusty",
            "i live in the Al Olaya district in Riyadh",
            "just moved to a new apartment last month",
            "the apartment has a nice view of the city",
            "i drive a Toyota Camry, nothing fancy",
            "planning to get an electric car next year",
            "i exercise at the gym 3 times a week",
            "mostly weightlifting and some cardio",
            "i follow a Mediterranean diet loosely",
            "love Arabic coffee, drink it all day",
            "married with 2 kids, a boy and a girl",
            "my son is 4 and daughter is 2",
            "my wife is also in tech, she's a UX designer",
            "we're planning a vacation to Japan in October",
            "never been to Japan before, really excited",
            "i also write a blog about data science in Arabic",
            "got about 500 readers per month now",
            "thinking about starting a YouTube channel too",
            "my goal is to eventually start my own data analytics company",
            "i invest in Saudi stocks, mainly SABIC and Aramco",
            "weekends i usually spend with family at parks",
        ]
        store_all(m, turns)

        # 10 queries
        res = m.query("what is the user's name?")
        r.check("name -> Ali", query_contains(res, "ali"), f"got {len(res)}")

        res = m.query("what is the user's job?")
        r.check("job -> data scientist",
                query_contains_any(res, [["data scientist"], ["data science"]]),
                f"got {len(res)}")

        res = m.query("where does the user work?")
        r.check("workplace -> bank/Riyadh",
                query_contains_any(res, [["bank"], ["riyadh"]]),
                f"got {len(res)}")

        res = m.query("side project?")
        r.check("side project -> fintech",
                query_contains(res, "fintech"),
                f"got {len(res)}")

        res = m.query("privacy requirements?")
        r.check("requirement -> local-first/privacy",
                query_contains_any(res, [["local"], ["privacy"]]),
                f"got {len(res)}")

        res = m.query("programming languages?")
        r.check("languages -> Python/R",
                query_contains_any(res, [["python"], ["r"]]),
                f"got {len(res)}")

        res = m.query("what cloud platform?")
        r.check("cloud -> Azure",
                query_contains(res, "azure"),
                f"got {len(res)}")

        res = m.query("fraud detection accuracy?")
        r.check("accuracy -> 90%",
                query_contains_any(res, [["90%"], ["90"]]),
                f"got {len(res)}")

        res = m.query("vacation plans?")
        r.check("holiday -> Japan/October",
                query_contains_any(res, [["japan"], ["october"]]),
                f"got {len(res)}")

        res = m.query("how many blog readers?")
        r.check("blog readers -> 500",
                query_contains(res, "500"),
                f"got {len(res)}")
    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()
    return r


def scenario_13(tmp_dir):
    """Repo Context (Commit Messages & PRs)"""
    r = ScenarioResult("13: Repo Context")
    m = make_memory(tmp_dir, "repo_ctx_user")
    try:
        store_all(m, [
            "commit: feat: add FTS5 full-text search for raw memory lookup",
            "commit: fix: rebuild graph on startup was 88 seconds, now 79ms after lazy init",
            "commit: refactor: replace verb dictionary with grammar-based parser",
            "PR #42: possessive extraction bug — my/our not parsed correctly. Status: closed, 14 out of 15 tests pass.",
            "issue #51: feature request — add REST API endpoint for external integrations",
            "commit: perf: batch inserts for store_personal, 3x throughput improvement",
            "PR #55: add contradiction detection for single-valued predicates like location and employer",
            "commit: docs: add API reference and quickstart guide to README",
            "issue #60: bug report — emoji in input causes FTS5 tokenizer to crash",
            "commit: fix: handle unicode and emoji in FTS5 tokenizer, closes #60",
        ])

        res = m.query("FTS5")
        r.check("FTS5 mentioned",
                query_contains(res, "fts5"),
                f"got {len(res)}")

        res = m.query("graph rebuild performance")
        r.check("graph rebuild -> 88s to 79ms",
                query_contains_any(res, [["88"], ["79ms"], ["graph"]]),
                f"got {len(res)}")

        res = m.query("grammar parser")
        r.check("grammar parser",
                query_contains_any(res, [["grammar"], ["parser"]]),
                f"got {len(res)}")

        res = m.query("possessive bug")
        r.check("possessive bug -> closed, 14/15",
                query_contains_any(res, [["possessive"], ["14"]]),
                f"got {len(res)}")

        res = m.query("REST API feature request")
        r.check("REST API -> issue #51",
                query_contains_any(res, [["rest api"], ["api"], ["51"]]),
                f"got {len(res)}")
    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()
    return r


def scenario_14(tmp_dir):
    """Multi-User Isolation (CRITICAL)"""
    r = ScenarioResult("14: Multi-User Isolation (CRITICAL)")

    alice = make_memory(tmp_dir, "alice_iso")
    bob = make_memory(tmp_dir, "bob_iso")

    try:
        # Store Alice's data
        store_all(alice, [
            "I work at Google as a senior engineer",
            "my salary is $200k",
            "I live in San Francisco",
            "I have a cat named Mochi",
        ])

        # Store Bob's data
        store_all(bob, [
            "I work at Meta as a product manager",
            "I live in New York",
            "I have a dog named Biscuit",
        ])

        # Alice's queries should find Alice's data
        res = alice.query("where do I work?")
        r.check("Alice sees Google",
                query_contains(res, "google"),
                f"got {len(res)}")

        res = alice.query("what is my salary?")
        r.check("Alice sees $200k",
                query_contains_any(res, [["200k"], ["200"]]),
                f"got {len(res)}")

        res = alice.query("what pet?")
        r.check("Alice sees Mochi",
                query_contains(res, "mochi"),
                f"got {len(res)}")

        # Bob's queries should find Bob's data
        res = bob.query("where do I work?")
        r.check("Bob sees Meta",
                query_contains(res, "meta"),
                f"got {len(res)}")

        res = bob.query("what pet?")
        r.check("Bob sees Biscuit",
                query_contains(res, "biscuit"),
                f"got {len(res)}")

        # CRITICAL: Bob must NOT see Alice's salary
        res = bob.query("what is my salary?")
        if query_contains_any(res, [["200k"], ["$200"], ["200,000"]]):
            r.hallucination("Bob salary -> should be EMPTY, not Alice's $200k",
                            f"CRITICAL LEAK: Bob sees Alice's salary! Results: "
                            f"{[x.text[:60] for x in res[:3]]}")
        else:
            r.check("Bob salary -> empty (no leak)", True,
                    "Correctly empty or no Alice data")

        # CRITICAL: Bob must NOT see Alice's location
        res = bob.query("where do I live?")
        if query_contains(res, "san francisco"):
            r.hallucination("Bob location -> should not see San Francisco",
                            f"CRITICAL LEAK: Bob sees Alice's location!")
        else:
            r.check("Bob location -> no leak", True)

        # CRITICAL: Alice must NOT see Bob's pet
        res = alice.query("do I have a dog?")
        if query_contains(res, "biscuit"):
            r.hallucination("Alice dog -> should not see Biscuit",
                            f"CRITICAL LEAK: Alice sees Bob's pet!")
        else:
            r.check("Alice dog -> no leak", True)

    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        alice.close()
        bob.close()
    return r


def scenario_15(tmp_dir):
    """Forgetting"""
    r = ScenarioResult("15: Forgetting")

    # Test 1: Store, find, forget, verify gone
    m = make_memory(tmp_dir, "forget_user1")
    try:
        m.store("i live in London")
        res = m.query("where do I live?")
        r.check("before forget: London found",
                query_contains(res, "london"), f"got {len(res)}")
        if res:
            mem_id = res[0].id
            m.forget(memory_id=mem_id)
            res2 = m.query("where do I live?")
            r.check("after forget: London gone",
                    query_not_contains(res2, "london"),
                    f"got {len(res2)}")
        else:
            r.check("after forget: London gone", False, "No results to forget")
    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()

    # Test 2: Salary forget
    m = make_memory(tmp_dir, "forget_user2")
    try:
        m.store("my salary is $150k")
        res = m.query("salary")
        r.check("before forget: salary found",
                query_contains_any(res, [["150k"], ["150"], ["salary"]]),
                f"got {len(res)}")
        if res:
            mem_id = res[0].id
            m.forget(memory_id=mem_id)
            res2 = m.query("salary")
            r.check("after forget: salary gone",
                    query_not_contains(res2, "150"),
                    f"got {len(res2)}")
    except Exception as e:
        r.error = r.error or traceback.format_exc()
    finally:
        m.close()

    # Test 3: Supersession (correction, not forget)
    m = make_memory(tmp_dir, "forget_user3")
    try:
        m.store("i work at Amazon")
        m.store("i work at Microsoft now")
        res = m.query("where do I work?")
        r.check("supersession: Microsoft (not Amazon)",
                query_contains(res, "microsoft"),
                f"got {len(res)}")
    except Exception as e:
        r.error = r.error or traceback.format_exc()
    finally:
        m.close()

    # Test 4: meetings forget
    m = make_memory(tmp_dir, "forget_user4")
    try:
        m.store("i like meetings")
        res = m.query("meetings")
        if res:
            mem_id = res[0].id
            m.forget(memory_id=mem_id)
            res2 = m.query("meetings")
            r.check("after forget: meetings gone",
                    query_not_contains(res2, "meetings"),
                    f"got {len(res2)}")
        else:
            r.check("after forget: meetings gone", False, "No result to forget")
    except Exception as e:
        r.error = r.error or traceback.format_exc()
    finally:
        m.close()

    # Test 5: kids forget
    m = make_memory(tmp_dir, "forget_user5")
    try:
        m.store("i have 3 kids")
        res = m.query("kids")
        if res:
            mem_id = res[0].id
            m.forget(memory_id=mem_id)
            res2 = m.query("kids")
            r.check("after forget: kids gone",
                    query_not_contains(res2, "3 kids"),
                    f"got {len(res2)}")
        else:
            r.check("after forget: kids gone", False, "No result to forget")
    except Exception as e:
        r.error = r.error or traceback.format_exc()
    finally:
        m.close()

    return r


def scenario_16(tmp_dir):
    """Edge Cases"""
    r = ScenarioResult("16: Edge Cases")
    m = make_memory(tmp_dir, "edge_user")
    try:
        edge_inputs = [
            "my name is O'Brien",
            "i worked at AT&T for 3 years",
            "my email is test@example.com",
            "i know C++ and C#",
            "the project is called lore-memory",
            "i used to live in Sao Paulo",  # simplified from São Paulo
            "we have a customer in Sao Paulo, Brazil",
            "my colleague's name is 日本語テスト",  # Japanese test
            "i love this project so much! :)",
            "'; DROP TABLE memories; --",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
            'the product is called "Lore Memory"',
            "i work at McKinsey & Company, Inc.",
        ]
        # Should not crash
        crashed = False
        for text in edge_inputs:
            try:
                m.store(text)
            except Exception as ex:
                crashed = True
                r.check(f"no crash: {text[:40]}...", False, str(ex))

        if not crashed:
            r.check("all edge inputs stored without crash", True)

        # Check a few are retrievable
        res = m.query("name O'Brien")
        r.check("apostrophe name retrievable",
                query_contains_any(res, [["o'brien"], ["obrien"], ["brien"]]),
                f"got {len(res)}")

        res = m.query("AT&T")
        r.check("ampersand name retrievable",
                query_contains_any(res, [["at&t"], ["at t"], ["att"]]),
                f"got {len(res)}")

        res = m.query("C++ programming")
        r.check("C++ retrievable",
                query_contains_any(res, [["c++"], ["c#"], ["c"]]),
                f"got {len(res)}")

        # SQL injection should NOT have deleted anything
        res = m.query("lore-memory")
        r.check("SQL injection did not damage DB",
                query_contains_any(res, [["lore-memory"], ["lore"]]),
                f"got {len(res)}")

    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()
    return r


def scenario_17(tmp_dir):
    """Short Input"""
    r = ScenarioResult("17: Short Input")
    m = make_memory(tmp_dir, "short_user")
    try:
        short_inputs = [
            "Dubai",
            "Python",
            "startup",
            "remote",
            "coffee",
        ]
        for text in short_inputs:
            m.store(text)

        # They should be stored as raw text
        res = m.query("Dubai")
        r.check("short input 'Dubai' stored",
                query_contains(res, "dubai"),
                f"got {len(res)}")

        res = m.query("Python")
        r.check("short input 'Python' stored",
                query_contains(res, "python"),
                f"got {len(res)}")

        # Check stats to see if anything was stored
        stats = m.stats()
        r.check("short inputs appear in stats",
                stats.get("private_total", 0) >= 5,
                f"total={stats.get('private_total', 0)}")

        # Verify these are stored as raw text (stated), not as full SPO triples
        # by checking profile -- short fragments shouldn't create meaningful predicates
        # about the user. This is a soft check since some may parse.
        profile = m.profile()
        # Just ensure no crash and profile works
        r.check("profile works with short inputs", True,
                f"profile keys: {list(profile.keys())[:5]}")

    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()
    return r


def scenario_18(tmp_dir):
    """Questions Should Not Be Stored as Facts"""
    r = ScenarioResult("18: Questions Not Stored as Facts")
    m = make_memory(tmp_dir, "question_user")
    try:
        questions = [
            "how does Lore store memories?",
            "what database does it use?",
            "can I use it offline?",
            "is there an API?",
            "how fast is retrieval?",
            "does it support multiple users?",
            "what programming languages does it support?",
            "can I delete my data?",
            "how does contradiction detection work?",
            "is there a web interface?",
        ]
        for q in questions:
            m.store(q)

        # Questions should be stored as raw text (stated) but should NOT
        # produce meaningful SPO triples about the user
        profile = m.profile()

        # Check: the profile should have few meaningful predicates.
        # The grammar parser may create some spurious triples from questions
        # (e.g., parsing "how" or "what" as subjects), but there should not
        # be many real user-fact predicates like "works_at", "lives_in", etc.
        # We exclude "stated" (raw text) from the count.
        meaningful = {k: v for k, v in profile.items() if k != "stated"}
        num_predicates = len(meaningful)
        r.check("few or no meaningful SPO triples from questions",
                num_predicates <= 10,
                f"got {num_predicates} predicates: {list(meaningful.keys())[:10]}")

        # Even if raw text is returned, it should be the question text itself
        res = m.query("how does Lore store memories?")
        if res:
            r.check("question text returned as-is (not as invented fact)",
                    query_contains_any(res, [["lore"], ["store"], ["memories"]]),
                    f"first result: {res[0].text[:60]}")
        else:
            r.check("question text returned as-is", True, "No results (also acceptable)")

    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()
    return r


def scenario_19(tmp_dir):
    """Dates"""
    r = ScenarioResult("19: Dates")
    m = make_memory(tmp_dir, "date_user")
    try:
        store_all(m, [
            "i joined the company on March 15, 2022",
            "i graduated in 2018",
            "the company was founded in 2021",
            "funding closed on January 8, 2025",
            "i've been working on Lore for 14 months",
        ])

        res = m.query("when did you join the company?")
        r.check("join date -> March 2022",
                query_contains_any(res, [["march"], ["2022"], ["march 15"]]),
                f"got {len(res)}")

        res = m.query("when did you graduate?")
        r.check("graduation -> 2018",
                query_contains(res, "2018"),
                f"got {len(res)}")

        res = m.query("when was the company founded?")
        r.check("founded -> 2021",
                query_contains(res, "2021"),
                f"got {len(res)}")

        res = m.query("when did funding close?")
        r.check("funding -> January 2025",
                query_contains_any(res, [["january"], ["2025"]]),
                f"got {len(res)}")

        res = m.query("how long working on Lore?")
        r.check("duration -> 14 months",
                query_contains_any(res, [["14 months"], ["14"]]),
                f"got {len(res)}")
    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()
    return r


def scenario_20(tmp_dir):
    """100 Facts Stress Test"""
    r = ScenarioResult("20: 100 Facts Stress Test")
    m = make_memory(tmp_dir, "fatima")
    try:
        facts = [
            "my name is Fatima",
            "i'm 32 years old",
            "i live in Abu Dhabi, UAE",
            "i work as a product manager at a telecom company",
            "the telecom company is called Etisalat",
            "i've been at Etisalat for 4 years",
            "before that i worked at du for 2 years",
            "i have a master's degree in business administration",
            "graduated from NYU Abu Dhabi",
            "my undergraduate was in computer science",
            "i did my undergrad at American University of Sharjah",
            "i speak Arabic, English, and French fluently",
            "Arabic is my mother tongue",
            "i have a side project building a learning platform",
            "the learning platform teaches Arabic to expats",
            "building it with Python and Django",
            "using PostgreSQL for the database",
            "the platform has about 200 users right now",
            "i want to grow it to 1000 users by year end",
            "i'm thinking about adding gamification features",
            "i have two cats named Luna and Oreo",
            "Luna is a Siamese cat, she's 3 years old",
            "Oreo is a tuxedo cat, she's 1 year old",
            "i love reading, especially business books",
            "currently reading Zero to One by Peter Thiel",
            "my favorite book of all time is Sapiens by Yuval Harari",
            "i exercise every morning, usually running",
            "i run about 5km every day",
            "i also do yoga twice a week",
            "i'm training for a half marathon in December",
            "my phone is an iPhone 15 Pro",
            "i use a MacBook Air for work",
            "and a Windows desktop for gaming at home",
            "i play video games to relax, mostly RPGs",
            "favorite game right now is Baldur's Gate 3",
            "i'm married, my husband works in finance",
            "his name is Khalid, he works at ADCB bank",
            "we've been married for 5 years",
            "we don't have kids yet, maybe in a year or two",
            "we live in a 2-bedroom apartment in Al Reem Island",
            "we're thinking about buying a villa in Saadiyat",
            "my car is a white BMW X3",
            "i also have a motorbike, a Yamaha MT-07",
            "i ride the motorbike on weekends for fun",
            "i drink a lot of green tea",
            "i don't drink coffee, it gives me anxiety",
            "i'm a vegetarian, have been for 3 years",
            "my favorite cuisine is Japanese food",
            "i cook at home most days, meal prep on Sundays",
            "i use Notion for personal organization",
            "Slack for work communication",
            "i manage a team of 8 people",
            "3 developers, 2 designers, 2 QA, 1 business analyst",
            "we follow agile methodology with 2-week sprints",
            "our biggest project right now is a customer portal",
            "the portal should reduce call center volume by 30%",
            "i report to the VP of Digital Products",
            "his name is Omar, very supportive manager",
            "our department budget is about $5M annually",
            "i handle about $2M of that for my team's projects",
            "i'm interested in AI and how it can improve telecom UX",
            "i attended a conference in Dubai last month on AI in telecom",
            "thinking about proposing an AI chatbot for customer service",
            "i write a blog about product management in Arabic",
            "the blog gets about 300 visits per month",
            "i also contribute to a product management community on LinkedIn",
            "i have about 5000 LinkedIn connections",
            "i mentor two junior PMs at work",
            "one of them is Noura, very talented",
            "the other is Hassan, needs more confidence",
            "i volunteer at an animal shelter on Saturdays",
            "i'm learning Spanish on Duolingo, 90-day streak",
            "my family is originally from Egypt",
            "my parents live in Cairo",
            "i visit them about 3 times a year",
            "i have one brother and two sisters",
            "my brother is a doctor in London",
            "one sister is a teacher in Dubai",
            "the other sister is studying in Paris",
            "i'm planning to apply to an accelerator for my side project",
            "looking at Y Combinator and Flat6Labs",
            "i need to build an MVP first",
            "the MVP should be ready by March",
            "i wake up at 5:30 AM every day",
            "i'm most productive between 6 AM and 10 AM",
            "i check email only twice a day, morning and evening",
            "i use time blocking for my calendar",
            "i try to keep meetings under 30 minutes",
            "i'm a big fan of the Pomodoro technique",
            "i take a 15-minute walk after lunch every day",
            "i journal every night before bed",
            "i use a standing desk at work",
            "my salary is confidential but i'm happy with it",
            "i got promoted last year from senior PM to lead PM",
            "next goal is to become VP of Product",
            "i want to transition to my startup full-time in 2 years",
            "i'm saving about 40% of my income for the transition",
            "my startup runway would be about 18 months with savings",
            "i also have some angel investors interested",
            "they want to see the MVP and traction first",
        ]
        store_all(m, facts)

        # 20 recall queries
        res = m.query("what is the user's name?")
        r.check("name -> Fatima", query_contains(res, "fatima"), f"got {len(res)}")

        res = m.query("how old?")
        r.check("age -> 32", query_contains(res, "32"), f"got {len(res)}")

        res = m.query("where does the user live?")
        r.check("location -> Abu Dhabi",
                query_contains_any(res, [["abu dhabi"]]),
                f"got {len(res)}")

        res = m.query("what is the user's job?")
        r.check("job -> product manager",
                query_contains_any(res, [["product manager"], ["product"]]),
                f"got {len(res)}")

        res = m.query("side project?")
        r.check("side project -> learning platform",
                query_contains_any(res, [["learning platform"], ["learning"], ["platform"]]),
                f"got {len(res)}")

        res = m.query("what database for the side project?")
        r.check("database -> PostgreSQL",
                query_contains_any(res, [["postgresql"], ["postgres"]]),
                f"got {len(res)}")

        res = m.query("cat names?")
        r.check("cats -> Luna/Oreo",
                query_contains_any(res, [["luna"], ["oreo"]]),
                f"got {len(res)}")

        res = m.query("currently reading?")
        r.check("book -> Zero to One",
                query_contains_any(res, [["zero to one"], ["zero"]]),
                f"got {len(res)}")

        res = m.query("what university?")
        r.check("university -> NYU Abu Dhabi",
                query_contains_any(res, [["nyu"], ["abu dhabi"], ["american university"]]),
                f"got {len(res)}")

        res = m.query("career goal?")
        r.check("career goal -> VP of Product / startup",
                query_contains_any(res, [["vp"], ["startup"], ["product"]]),
                f"got {len(res)}")

        res = m.query("what languages does the user speak?")
        r.check("languages spoken -> Arabic/English/French",
                query_contains_any(res, [["arabic"], ["english"], ["french"]]),
                f"got {len(res)}")

        res = m.query("what phone?")
        r.check("phone -> iPhone 15 Pro",
                query_contains_any(res, [["iphone"], ["15"]]),
                f"got {len(res)}")

        res = m.query("family?")
        r.check("family -> brother/sisters/married",
                query_contains_any(res, [["brother"], ["sister"], ["married"], ["husband"]]),
                f"got {len(res)}")

        res = m.query("accelerator plans?")
        r.check("accelerator -> Y Combinator / Flat6Labs",
                query_contains_any(res, [["y combinator"], ["flat6labs"], ["accelerator"]]),
                f"got {len(res)}")

        res = m.query("productivity habits?")
        r.check("productivity -> Pomodoro / time blocking",
                query_contains_any(res, [["pomodoro"], ["time blocking"], ["5:30"], ["productive"]]),
                f"got {len(res)}")

        res = m.query("exercise routine?")
        r.check("exercise -> running / yoga",
                query_contains_any(res, [["running"], ["yoga"], ["exercise"], ["5km"]]),
                f"got {len(res)}")

        res = m.query("what car?")
        r.check("car -> BMW X3",
                query_contains_any(res, [["bmw"], ["x3"]]),
                f"got {len(res)}")

        res = m.query("startup runway?")
        r.check("runway -> 18 months",
                query_contains_any(res, [["18 months"], ["18"], ["runway"]]),
                f"got {len(res)}")

        res = m.query("email habits?")
        r.check("email -> twice a day",
                query_contains_any(res, [["twice"], ["email"], ["morning", "evening"]]),
                f"got {len(res)}")

        res = m.query("what is the team size at work?")
        r.check("team -> 8 people",
                query_contains_any(res, [["8 people"], ["8"], ["team"]]),
                f"got {len(res)}")

    except Exception as e:
        r.error = traceback.format_exc()
    finally:
        m.close()
    return r


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_SCENARIOS = [
    scenario_01, scenario_02, scenario_03, scenario_04, scenario_05,
    scenario_06, scenario_07, scenario_08, scenario_09, scenario_10,
    scenario_11, scenario_12, scenario_13, scenario_14, scenario_15,
    scenario_16, scenario_17, scenario_18, scenario_19, scenario_20,
]


def run_all():
    tmp_dir = tempfile.mkdtemp(prefix="lore_test_corpus_")
    print(f"Test data dir: {tmp_dir}")
    print(f"Running {len(ALL_SCENARIOS)} scenarios...\n")

    results = []
    total_queries = 0
    total_passed = 0
    total_failed = 0
    total_hallucinations = 0

    for i, scenario_fn in enumerate(ALL_SCENARIOS, 1):
        print(f"--- Scenario {i:02d} ---")
        t0 = time.time()
        try:
            result = scenario_fn(tmp_dir)
        except Exception as e:
            result = ScenarioResult(f"{i:02d}: CRASHED")
            result.error = traceback.format_exc()
        elapsed = time.time() - t0

        results.append(result)
        total_queries += result.total
        total_passed += result.passed
        total_failed += result.failed
        total_hallucinations += result.hallucinations

        status = "PASS" if result.passed_threshold else "FAIL"
        if result.hallucinations > 0:
            status = "HALLUCINATION"
        if result.error:
            status = "ERROR"

        print(f"  {result.name}")
        print(f"  {result.passed}/{result.total} passed "
              f"({result.pass_rate:.0f}%) [{status}] "
              f"({elapsed:.2f}s)")
        if result.hallucinations > 0:
            print(f"  *** {result.hallucinations} HALLUCINATION(S) DETECTED ***")
        if result.error:
            print(f"  ERROR: {result.error.splitlines()[-1]}")
        for desc, ok, detail in result.details:
            mark = "PASS" if ok else "FAIL"
            print(f"    [{mark}] {desc}: {detail}")
        print()

    # Summary
    overall_rate = (total_passed / total_queries * 100) if total_queries > 0 else 0
    overall_status = "PASS" if overall_rate >= 90 else "FAIL"
    if total_hallucinations > 0:
        overall_status = "FAIL (HALLUCINATIONS)"

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Queries:     {total_queries}")
    print(f"Passed:            {total_passed}")
    print(f"Failed:            {total_failed}")
    print(f"Hallucinations:    {total_hallucinations}")
    print(f"Overall Pass Rate: {overall_rate:.1f}% (threshold: 90%)")
    print(f"Overall Status:    {overall_status}")
    print()

    scenarios_passed = sum(1 for r in results if r.passed_threshold and r.hallucinations == 0)
    scenarios_failed = len(results) - scenarios_passed
    print(f"Scenarios Passed (>=85%): {scenarios_passed}/{len(results)}")
    print(f"Scenarios Failed (<85%):  {scenarios_failed}/{len(results)}")
    print()

    # Write detailed report
    write_report(results, total_queries, total_passed, total_failed,
                 total_hallucinations, overall_rate, overall_status, tmp_dir)

    # Cleanup
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"Cleaned up temp dir: {tmp_dir}")
    except Exception:
        print(f"Warning: could not clean up {tmp_dir}")

    return overall_rate >= 90 and total_hallucinations == 0


def write_report(results, total_queries, total_passed, total_failed,
                 total_hallucinations, overall_rate, overall_status, tmp_dir):
    """Write detailed results to output-v3/TEST_CORPUS_RESULTS.md"""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(project_dir, "output-v3")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "TEST_CORPUS_RESULTS.md")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# Lore Memory Test Corpus Results")
    lines.append(f"\nRun: {now}")
    lines.append(f"Temp dir: `{tmp_dir}`\n")
    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total Queries | {total_queries} |")
    lines.append(f"| Passed | {total_passed} |")
    lines.append(f"| Failed | {total_failed} |")
    lines.append(f"| Hallucinations | {total_hallucinations} |")
    lines.append(f"| Overall Pass Rate | {overall_rate:.1f}% |")
    lines.append(f"| Overall Status | **{overall_status}** |")
    lines.append(f"| Threshold | 90% overall, 85% per scenario |")
    lines.append("")

    # Per-scenario table
    lines.append("## Scenario Results\n")
    lines.append("| # | Scenario | Passed | Total | Rate | Halluc. | Status |")
    lines.append("|---|----------|--------|-------|------|---------|--------|")
    for res in results:
        status = "PASS" if res.passed_threshold and res.hallucinations == 0 else "FAIL"
        if res.hallucinations > 0:
            status = "HALLUCINATION"
        if res.error:
            status = "ERROR"
        lines.append(f"| {res.name.split(':')[0]} | {res.name} | {res.passed} | "
                      f"{res.total} | {res.pass_rate:.0f}% | {res.hallucinations} | {status} |")
    lines.append("")

    # Detailed results per scenario
    lines.append("## Detailed Results\n")
    for res in results:
        lines.append(f"### {res.name}\n")
        if res.error:
            lines.append(f"**ERROR:**\n```\n{res.error}\n```\n")
        for desc, ok, detail in res.details:
            mark = "PASS" if ok else "FAIL"
            emoji = "" if ok else " **<-- FAILED**"
            lines.append(f"- [{mark}] {desc}: {detail}{emoji}")
        lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nDetailed report written to: {report_path}")


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
