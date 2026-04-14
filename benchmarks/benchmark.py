"""Lore Benchmarks — all contexts + isolation + latency."""

import json, os, sys, time, random, statistics, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lore_memory.engine import Engine, Config

def mk(tmp):
    return Engine(Config(data_dir=tmp, embedding_dims=64))

def bench_personal():
    tmp = tempfile.mkdtemp()
    try:
        e = mk(tmp)
        facts = [("I live in Amsterdam","lives_in","Amsterdam"),("I work at Google","works_at","Google"),
                 ("I like Python","likes","Python"),("My name is Alex","name","Alex"),
                 ("I prefer dark mode","prefers","dark mode")]
        for t,_,_ in facts: e.store_personal("u1", t)
        hits = sum(1 for t,p,v in facts if any(v.lower() in r.memory.object_value.lower() for r in e.recall("u1", t)))
        e.close()
        return {"accuracy": f"{hits}/{len(facts)}", "score": hits/len(facts)}
    finally: shutil.rmtree(tmp, ignore_errors=True)

def bench_chat():
    tmp = tempfile.mkdtemp()
    try:
        e = mk(tmp)
        msgs = [("I prefer React over Vue","prefers"),("We decided to use PostgreSQL","decided"),
                ("Alice lives in Tokyo","lives_in")]
        for t,_ in msgs: e.store_chat("u1", t, session_id="s1")
        hits = sum(1 for t,_ in msgs if e.recall("u1", t))
        e.close()
        return {"recall": f"{hits}/{len(msgs)}", "score": hits/len(msgs)}
    finally: shutil.rmtree(tmp, ignore_errors=True)

def bench_isolation():
    tmp = tempfile.mkdtemp()
    try:
        e = mk(tmp)
        for i in range(10): e.store_personal(f"u{i}", f"My code is CODE{i}X")
        leaks = 0
        for i in range(10):
            for r in e.recall(f"u{i}", "code"):
                if r.memory.user_id != f"u{i}": leaks += 1
        e.close()
        return {"users": 10, "leaks": leaks, "passed": leaks == 0}
    finally: shutil.rmtree(tmp, ignore_errors=True)

def bench_multiuser():
    tmp = tempfile.mkdtemp()
    try:
        e = mk(tmp)
        e.store_company("alice", "acme", "Our mission is to democratize AI")
        e.store_company("bob", "acme", "Our pricing model is freemium")
        e.store_personal("alice", "My salary is 200k")
        a = bool(e.recall("alice", "mission", org_id="acme"))
        b = bool(e.recall("bob", "pricing", org_id="acme"))
        leak = any("200k" in r.memory.object_value for r in e.recall("bob", "salary", org_id="acme"))
        e.close()
        return {"shared_visible": a and b, "salary_leak": leak, "passed": not leak and a and b}
    finally: shutil.rmtree(tmp, ignore_errors=True)

def bench_latency():
    results = {}
    for n in [100, 500, 1000, 2000, 5000]:
        tmp = tempfile.mkdtemp()
        try:
            e = mk(tmp)
            t0 = time.perf_counter()
            for i in range(n):
                e.store_fact("private", "personal", f"e_{i%200}", f"p_{i%20}", f"v_{i}", user_id="bench")
            wms = (time.perf_counter()-t0)*1000
            lats = []
            for _ in range(20):
                q = f"e_{random.randint(0,199)} p_{random.randint(0,19)}"
                t0 = time.perf_counter()
                e.recall("bench", q)
                lats.append((time.perf_counter()-t0)*1000)
            p50 = statistics.median(lats)
            results[f"n={n}"] = {"write_per_fact_ms": round(wms/n,3), "read_p50_ms": round(p50,2)}
            print(f"  n={n}: write={wms/n:.3f}ms/fact  read_p50={p50:.2f}ms")
            e.close()
        finally: shutil.rmtree(tmp, ignore_errors=True)
    return results

def bench_feedback():
    tmp = tempfile.mkdtemp()
    try:
        e = mk(tmp)
        e.store_personal("u1", "I like Python")
        e.store_personal("u1", "I like Java")
        r = e.recall("u1", "programming language I like")
        for x in r:
            if "Python" in x.memory.object_value:
                e.feedback("u1", x.memory.id, helpful=True)
                e.feedback("u1", x.memory.id, helpful=True)
        r2 = e.recall("u1", "programming")
        weights = e.get_weights("u1")
        e.close()
        return {"weights_adapted": weights["updates"] > 0, "weights": weights}
    finally: shutil.rmtree(tmp, ignore_errors=True)

def main():
    print("=" * 60)
    print("  Lore Benchmarks")
    print("=" * 60)
    all_results = {}
    for name, fn in [("Personal", bench_personal), ("Chat", bench_chat),
                     ("Isolation", bench_isolation),
                     ("Multi-User", bench_multiuser), ("Latency", bench_latency),
                     ("Feedback", bench_feedback)]:
        print(f"\n--- {name} ---")
        try:
            r = fn(); all_results[name] = r; print(json.dumps(r, indent=2))
        except Exception as ex:
            print(f"  ERROR: {ex}"); import traceback; traceback.print_exc()
    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nDone.")

if __name__ == "__main__":
    main()
