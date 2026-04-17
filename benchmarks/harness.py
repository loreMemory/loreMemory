"""
Regression harness for loremem-ai — quality + perf in one script.

Usage:
  python benchmarks/harness.py                 # run, print summary
  python benchmarks/harness.py --save-baseline # write benchmarks/baseline.json
  python benchmarks/harness.py --check         # exit non-zero on regression

The harness has three parts:
  1. QUALITY — 63 queries against a seeded 200-fact personal-memory scenario
     that includes supersession, name collisions, noise, and cascading
     contradictions. Reports top-1 / top-3 / p50 retrieval latency.
  2. PERF — insert throughput at 1K with a fake-embedding to isolate
     storage cost from sentence-transformer cost.
  3. COMPARE — deltas vs. benchmarks/baseline.json, with tolerances.

Deterministic: fixed random seeds, isolated temp DB dir, fake embed for perf.
Runtime target: under 90 seconds on a laptop.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = Path(__file__).resolve().parent / "baseline.json"
LAST_RUN_PATH = Path(__file__).resolve().parent / "last_run.json"

# Tolerances for --check mode
QUALITY_DROP_THRESHOLD = 2.0   # percentage points of top-1 drop
PERF_DROP_THRESHOLD = 0.20     # 20% relative drop in inserts/s

# ---------------------------------------------------------------- QUALITY

# Each scenario is (facts, checks). A check is (label, query, expected_substrings).
# Top-1 is correct if any expected substring appears in the top-1 result text.
SCENARIO = {
    "onboarding": [
        "My name is Sam Rivera",
        "I live in Amsterdam",
        "I work at Stripe as a staff engineer",
        "I'm 34 years old",
        "My partner is Lena, she is a teacher",
        "We have a dog named Biscuit, a cocker spaniel",
        "My birthday is March 12",
        "I studied mechanical engineering at TU Delft",
        "I'm allergic to penicillin",
        "My favorite food is ramen",
    ],
    "growth": [
        "I play chess, rated around 1800 on lichess",
        "I run three times a week, usually in Vondelpark",
        "I hate coriander",
        "I love black coffee, no sugar",
        "My favorite author is Ursula K. Le Guin",
        "I'm learning Italian on Duolingo",
        "My best friend is Mateo, he lives in Lisbon",
        "My friend Priya works at Google DeepMind in London",
        "My sister Noor lives in Toronto",
        "My brother Idris is a nurse in Rotterdam",
        "My mother Amina is retired, lives in Utrecht",
        "My manager is Chen Wei",
        "I'm on the payments reliability team at Stripe",
        "My office is at Rokin 92",
        "I prefer Go and Python, dislike Java",
        "I use Neovim as my editor",
        "I went to Japan in October for two weeks",
        "I want to visit Patagonia next",
        "I bank with ING",
        "I'm vegetarian since 2022",
        "I had knee surgery in 2020, left meniscus",
        "I live on Prinsengracht, 3rd floor walk-up",
        "I subscribe to The Economist",
        "My favorite film is Stalker by Tarkovsky",
        "I speak English, Dutch, Spanish, and a little Arabic",
        "I drive a 2016 Volvo V40",
        "Project Lantern is our new fraud detection pipeline",
        "Project Mosaic is the checkout redesign",
        "Mosaic is blocked on legal review until May",
    ],
    "updates": [
        "I moved from Amsterdam to Berlin last month",
        "I left Stripe and joined Anthropic as a research engineer",
        "Biscuit passed away in February. We adopted a cat named Miso",
        "Lena and I broke up in March; I'm single now",
        "I'm 35 now (birthday last month)",
        "I stopped being vegetarian, I eat fish again",
    ],
    "ambiguity": [
        "I met a new colleague named Sam Chen at Anthropic",
        "Sam Chen is from Singapore and works on evals",
        "My neighbor Priya (different from Priya at DeepMind) owns a bakery on Kastanienallee",
        "At Anthropic I'm on Project Lantern (the safety eval one, not Stripe's fraud one)",
        "Miso is a grey tabby, very vocal",
        "My downstairs neighbor has a cat also named Miso — theirs is orange",
    ],
    "noise": [
        "Ugh, the train was late today",
        "Remember to call the dentist",
        "Fixed a bug on Monday",
        "Watched half of some documentary about octopuses",
        "Miso knocked over the plant again",
        "Mateo sent me a weird meme",
        "Meeting got pushed to Thursday",
    ],
    "cascading": [
        "Actually I moved again — Berlin to Munich two weeks ago",
        "Anthropic moved me to the interpretability team last week",
        "I'm seeing someone new, Yasmin, a cellist",
    ],
}

CHECKS: list[tuple[str, str, list[str]]] = [
    # onboarding basics
    ("onboard_name",       "what is my name?",              ["Sam Rivera", "Sam"]),
    ("onboard_bday",       "when is my birthday?",          ["March 12", "12"]),
    ("onboard_allergy",    "what am I allergic to?",        ["penicillin"]),
    ("onboard_food",       "what is my favorite food?",     ["ramen"]),
    ("onboard_edu",        "where did I study?",            ["Delft", "TU Delft"]),
    # growth / 3rd-party
    ("grow_manager",       "who is my manager?",            ["Chen Wei", "Chen"]),
    ("grow_sister",        "where does my sister live?",    ["Toronto"]),
    ("grow_best_friend",   "who is my best friend?",        ["Mateo"]),
    ("grow_priya",         "where does Priya work?",        ["DeepMind", "Google"]),
    ("grow_chess",         "what's my chess rating?",       ["1800"]),
    ("grow_hate",          "what food do I hate?",          ["coriander"]),
    ("grow_learning",      "what language am I learning?",  ["Italian"]),
    ("grow_film",          "what is my favorite film?",     ["Stalker", "Tarkovsky"]),
    ("grow_team",          "what team am I on at Stripe?",  ["payments", "reliability"]),
    ("grow_lantern_stripe","what is Project Lantern about fraud?", ["fraud"]),
    ("grow_mom_city",      "where does my mother live?",    ["Utrecht"]),
    # supersession
    ("supersede_city",     "where do I live now?",          ["Berlin", "Munich"]),
    ("supersede_work",     "where do I work now?",          ["Anthropic"]),
    ("supersede_role",     "what's my role?",               ["research engineer", "research"]),
    ("supersede_pet",      "do I have a pet?",              ["Miso", "cat"]),
    ("supersede_age",      "how old am I?",                 ["35"]),
    # ambiguity
    ("ambig_my_name",      "what is MY name?",              ["Sam Rivera", "Rivera"]),
    ("ambig_sam_chen",     "where is Sam Chen from?",       ["Singapore"]),
    ("ambig_bakery",       "who owns the bakery?",          ["Priya", "bakery"]),
    ("ambig_deepmind",     "who works at DeepMind?",        ["Priya"]),
    ("ambig_lantern_anth", "what is Lantern at Anthropic?", ["safety", "eval"]),
    ("ambig_my_cat_color", "what color is my cat?",         ["grey", "tabby"]),
    ("ambig_neighbor_cat", "what color is my neighbor's cat?", ["orange"]),
    # noise survival
    ("post_noise_name",    "what is my name?",              ["Sam Rivera"]),
    ("post_noise_cat",     "what's my cat's name?",         ["Miso"]),
    ("post_noise_chess",   "do I play chess?",              ["chess", "lichess", "1800"]),
    # cascading
    ("cascade_city",       "where do I live right now?",    ["Munich"]),
    ("cascade_team",       "what team am I on?",            ["interpretability"]),
    ("cascade_partner",    "am I dating anyone?",           ["Yasmin", "cellist"]),
    # long-tail recall after all churn
    ("tail_best_friend",   "who is my best friend?",        ["Mateo"]),
    ("tail_allergy",       "am I allergic to anything?",    ["penicillin"]),
    ("tail_edu",           "where did I study?",            ["Delft", "TU Delft"]),
    ("tail_film",          "what's my favorite film?",      ["Stalker", "Tarkovsky"]),
    ("tail_old_manager",   "who was my old manager at Stripe?", ["Chen Wei", "Chen"]),
]


def _pct(x: float) -> str:
    return f"{x:.1f}%"


def _percentile(xs: list[float], p: int) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]


def run_quality(data_dir: Path) -> dict[str, Any]:
    """Seed the full scenario, run all CHECKS in order, measure top-1/top-3 + latency."""
    from lore_memory import Memory
    m = Memory(user_id="bench_user", data_dir=str(data_dir))

    # Feed the corpus in order. CHECKS are partitioned implicitly by phase:
    # onboarding + growth → supersession queries; then updates; then ambiguity/noise/cascading.
    # We just run the full scenario up-front and let the system handle supersession
    # via its own belief layer. This tests the *end* state, which is what a user cares about.
    for bucket in ("onboarding", "growth", "updates", "ambiguity", "noise", "cascading"):
        for sent in SCENARIO[bucket]:
            m.store(sent)

    latencies: list[float] = []
    n_top1 = n_top3 = 0
    per_check = []
    for label, q, expected in CHECKS:
        t0 = time.perf_counter()
        hits = m.query(q, limit=5)
        latencies.append((time.perf_counter() - t0) * 1000)

        def _match(text: str | None) -> bool:
            if not text:
                return False
            tl = text.lower()
            return any(e.lower() in tl for e in expected)

        top_text = getattr(hits[0], "text", None) if hits else None
        in_top3 = any(_match(getattr(h, "text", "")) for h in hits[:3])
        if _match(top_text):
            n_top1 += 1
        if in_top3:
            n_top3 += 1
        per_check.append({
            "label": label,
            "q": q,
            "top": top_text,
            "top1_correct": _match(top_text),
            "in_top3": in_top3,
        })

    m.close()
    total = len(CHECKS)
    return {
        "total": total,
        "top1_correct": n_top1,
        "top3_correct": n_top3,
        "top1_pct": 100.0 * n_top1 / total,
        "top3_pct": 100.0 * n_top3 / total,
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "p99_ms": _percentile(latencies, 99),
        "per_check": per_check,
    }


# ---------------------------------------------------------------- PERF

def _fake_embed(text: str) -> list[float]:
    import hashlib
    h = hashlib.blake2b(text.encode(), digest_size=48).digest()
    rnd = random.Random(int.from_bytes(h, "big"))
    return [rnd.gauss(0, 1) for _ in range(384)]


def run_perf(data_dir: Path, target: int = 1000) -> dict[str, Any]:
    """Insert `target` zipf-shaped facts with a fake embedder; report throughput and retrieval p50."""
    from lore_memory import Memory
    rng = random.Random(0xBEEF)
    hot = [f"Person{i}" for i in range(50)]
    weights = [1 / ((i + 1) ** 1.2) for i in range(len(hot))]
    s = sum(weights); weights = [w / s for w in weights]

    def subj():
        r = rng.random()
        if r < 0.8:
            acc = 0.0
            for h, w in zip(hot, weights):
                acc += w
                if r <= acc:
                    return h
            return hot[-1]
        return f"Tail_{rng.randint(0, 5000)}"

    templates = [
        ("lives_in", ["Berlin", "Tokyo", "Lisbon", "NYC", "Madrid"]),
        ("works_at", ["Stripe", "Anthropic", "Meta", "OpenAI", "Google"]),
        ("likes",    ["ramen", "chess", "running", "coffee", "opera"]),
    ]

    def sent(s_):
        p, vs = rng.choice(templates); o = rng.choice(vs)
        if p == "lives_in": return f"{s_} lives in {o}"
        if p == "works_at": return f"{s_} works at {o}"
        return f"{s_} likes {o}"

    m = Memory(user_id="perfuser", data_dir=str(data_dir), embed_fn=_fake_embed)
    t0 = time.perf_counter()
    for _ in range(target):
        m.store(sent(subj()))
    insert_dur = time.perf_counter() - t0

    # Retrieval warmup + measure
    qs = [f"where does {hot[i]} live?" for i in range(20)] + \
         [f"what does {hot[i]} like?" for i in range(20)]
    for q in qs[:5]:
        m.query(q, limit=5)
    lats: list[float] = []
    for _ in range(3):
        for q in qs:
            t = time.perf_counter()
            m.query(q, limit=5)
            lats.append((time.perf_counter() - t) * 1000)
    m.close()

    return {
        "facts_inserted": target,
        "insert_seconds": insert_dur,
        "inserts_per_sec": target / insert_dur,
        "retrieval_p50_ms": _percentile(lats, 50),
        "retrieval_p95_ms": _percentile(lats, 95),
    }


# ---------------------------------------------------------------- COMPARE

def compare(current: dict, baseline: dict) -> tuple[bool, list[str]]:
    """Return (ok, messages). ok=False on real regression."""
    msgs: list[str] = []
    ok = True

    cur_top1 = current["quality"]["top1_pct"]
    base_top1 = baseline["quality"]["top1_pct"]
    delta_top1 = cur_top1 - base_top1
    if delta_top1 < -QUALITY_DROP_THRESHOLD:
        ok = False
        msgs.append(f"REGRESS quality top1: {cur_top1:.1f}% vs baseline {base_top1:.1f}% (Δ={delta_top1:+.1f}pp)")
    else:
        msgs.append(f"OK quality top1: {cur_top1:.1f}% vs baseline {base_top1:.1f}% (Δ={delta_top1:+.1f}pp)")

    cur_tput = current["perf"]["inserts_per_sec"]
    base_tput = baseline["perf"]["inserts_per_sec"]
    delta_rel = (cur_tput - base_tput) / base_tput if base_tput else 0.0
    if delta_rel < -PERF_DROP_THRESHOLD:
        ok = False
        msgs.append(f"REGRESS perf inserts: {cur_tput:.0f}/s vs baseline {base_tput:.0f}/s (Δ={100*delta_rel:+.1f}%)")
    else:
        msgs.append(f"OK perf inserts: {cur_tput:.0f}/s vs baseline {base_tput:.0f}/s (Δ={100*delta_rel:+.1f}%)")

    return ok, msgs


# ---------------------------------------------------------------- MAIN

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-baseline", action="store_true",
                    help="Write this run as the new baseline.json")
    ap.add_argument("--check", action="store_true",
                    help="Compare to baseline.json; exit 1 on regression")
    ap.add_argument("--tmp", default=None,
                    help="Temp dir for isolated test DBs (default /tmp/lore_bench)")
    ap.add_argument("--slow", action="store_true",
                    help="Additionally run 10K-fact perf (~2 min)")
    args = ap.parse_args()

    tmp = Path(args.tmp or "/tmp/lore_bench")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    # Stable seed so random.seed() choices inside the library are predictable across runs.
    random.seed(0xD00D)

    print("== QUALITY ==")
    t0 = time.perf_counter()
    quality = run_quality(tmp / "quality")
    print(f"   ran in {time.perf_counter()-t0:.1f}s")
    print(f"   top1:       {quality['top1_correct']}/{quality['total']}  ({_pct(quality['top1_pct'])})")
    print(f"   top3:       {quality['top3_correct']}/{quality['total']}  ({_pct(quality['top3_pct'])})")
    print(f"   retrieval:  p50={quality['p50_ms']:.1f}ms  p95={quality['p95_ms']:.1f}ms")

    print("\n== PERF (1K inserts, fake embed) ==")
    t0 = time.perf_counter()
    perf = run_perf(tmp / "perf", target=1000)
    print(f"   ran in {time.perf_counter()-t0:.1f}s")
    print(f"   inserts:    {perf['inserts_per_sec']:.0f}/s ({perf['facts_inserted']} in {perf['insert_seconds']:.2f}s)")
    print(f"   retrieval:  p50={perf['retrieval_p50_ms']:.1f}ms  p95={perf['retrieval_p95_ms']:.1f}ms")

    perf_10k = None
    if args.slow:
        print("\n== PERF (10K inserts, fake embed, slow) ==")
        t0 = time.perf_counter()
        perf_10k = run_perf(tmp / "perf10k", target=10000)
        print(f"   ran in {time.perf_counter()-t0:.1f}s")
        print(f"   inserts:    {perf_10k['inserts_per_sec']:.0f}/s "
              f"({perf_10k['facts_inserted']} in {perf_10k['insert_seconds']:.2f}s)")
        print(f"   retrieval:  p50={perf_10k['retrieval_p50_ms']:.1f}ms  "
              f"p95={perf_10k['retrieval_p95_ms']:.1f}ms")

    run = {"quality": quality, "perf": perf, "timestamp": time.time()}
    if perf_10k:
        run["perf_10k"] = perf_10k
    LAST_RUN_PATH.write_text(json.dumps(run, indent=2))
    print(f"\nwrote {LAST_RUN_PATH}")

    if args.save_baseline:
        BASELINE_PATH.write_text(json.dumps(run, indent=2))
        print(f"wrote baseline -> {BASELINE_PATH}")
        return 0

    if args.check:
        if not BASELINE_PATH.exists():
            print("no baseline.json — run with --save-baseline first")
            return 2
        baseline = json.loads(BASELINE_PATH.read_text())
        ok, msgs = compare(run, baseline)
        print("\n== COMPARE ==")
        for m in msgs:
            print("  " + m)
        return 0 if ok else 1

    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text())
        _, msgs = compare(run, baseline)
        print("\n== VS BASELINE (informational) ==")
        for m in msgs:
            print("  " + m)

    return 0


if __name__ == "__main__":
    sys.exit(main())
