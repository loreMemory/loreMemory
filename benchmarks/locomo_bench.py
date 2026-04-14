"""
LoCoMo Benchmark for Lore Memory.

Runs the standard LoCoMo evaluation:
1. Feed each conversation turn-by-turn into Lore
2. For each QA pair, query Lore and check if the answer is in the results
3. Score by category (1-4, excluding 5 which has no ground truth)

Usage:
    python benchmarks/locomo_bench.py /path/to/locomo10.json
"""

import json
import sys
import time
import tempfile
import shutil
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lore_memory.engine import Engine, Config


def run_benchmark(data_path: str, max_convs: int = 10):
    results_by_cat: dict[int, list[bool]] = {1: [], 2: [], 3: [], 4: []}
    total_turns = 0
    total_time_store = 0.0
    total_time_query = 0.0

    with open(data_path) as f:
        conversations = json.load(f)

    for conv_idx, conv_data in enumerate(conversations[:max_convs]):
        conv = conv_data["conversation"]
        speaker_a = conv.get("speaker_a", "A")
        speaker_b = conv.get("speaker_b", "B")
        sample_id = conv_data.get("sample_id", f"conv-{conv_idx}")

        # Fresh engine per conversation (no cross-conversation leakage)
        tmp_dir = tempfile.mkdtemp(prefix="lore_locomo_")
        engine = Engine(Config(data_dir=tmp_dir, embedding_dims=384))

        # Phase 1: Ingest all turns
        sessions = []
        for key in sorted(conv.keys()):
            if key.startswith("session_") and not key.endswith("_date_time"):
                sessions.append(conv[key])

        turn_count = 0
        for session in sessions:
            for turn in session:
                speaker = turn.get("speaker", "unknown")
                text = turn.get("text", "")
                if not text.strip():
                    continue

                t0 = time.perf_counter()
                # Store as chat from the speaker's perspective
                user_id = "user_a" if speaker == speaker_a else "user_b"
                engine.store_chat(user_id, text, speaker=speaker,
                                  session_id=sample_id)
                total_time_store += time.perf_counter() - t0
                turn_count += 1

        total_turns += turn_count

        # Phase 2: Answer QA pairs
        for qa in conv_data["qa"]:
            category = qa["category"]
            if category == 5:
                continue  # No ground truth for adversarial questions

            question = qa["question"]
            expected = str(qa["answer"]).lower()

            t0 = time.perf_counter()
            # Query as user_a (the "rememberer")
            recall_results = engine.recall("user_a", question, top_k=10)
            total_time_query += time.perf_counter() - t0

            # Check: is the expected answer anywhere in the top results?
            found = False
            for r in recall_results:
                result_text = (r.memory.source_text + " " + r.memory.object_value).lower()
                # Check if key parts of the expected answer appear in results
                answer_words = [w for w in expected.split() if len(w) > 2]
                if not answer_words:
                    answer_words = expected.split()
                matches = sum(1 for w in answer_words if w in result_text)
                if matches >= max(1, len(answer_words) * 0.5):
                    found = True
                    break

            results_by_cat[category].append(found)

        engine.close()
        shutil.rmtree(tmp_dir)

        # Progress
        cat_scores = {c: (sum(r) / len(r) * 100 if r else 0)
                      for c, r in results_by_cat.items()}
        print(f"  [{sample_id}] {turn_count} turns ingested | "
              f"Running: Cat1={cat_scores[1]:.0f}% Cat2={cat_scores[2]:.0f}% "
              f"Cat3={cat_scores[3]:.0f}% Cat4={cat_scores[4]:.0f}%")

    # Final scores
    print("\n" + "=" * 60)
    print("LOCOMO BENCHMARK RESULTS")
    print("=" * 60)

    cat_names = {
        1: "Single-hop QA      ",
        2: "Temporal QA         ",
        3: "Multi-hop reasoning ",
        4: "Open-ended summary  ",
    }

    total_correct = 0
    total_questions = 0
    for cat in [1, 2, 3, 4]:
        r = results_by_cat[cat]
        correct = sum(r)
        total = len(r)
        pct = correct / total * 100 if total else 0
        total_correct += correct
        total_questions += total
        print(f"  Cat {cat} ({cat_names[cat]}): {correct}/{total} ({pct:.1f}%)")

    overall = total_correct / total_questions * 100 if total_questions else 0
    print(f"\n  OVERALL (Cat 1-4): {total_correct}/{total_questions} ({overall:.1f}%)")
    print(f"\n  Turns ingested: {total_turns}")
    print(f"  Avg store time: {total_time_store / total_turns * 1000:.1f}ms/turn")
    print(f"  Avg query time: {total_time_query / total_questions * 1000:.1f}ms/query")

    return {
        "overall": round(overall, 1),
        "cat1": round(sum(results_by_cat[1]) / len(results_by_cat[1]) * 100, 1) if results_by_cat[1] else 0,
        "cat2": round(sum(results_by_cat[2]) / len(results_by_cat[2]) * 100, 1) if results_by_cat[2] else 0,
        "cat3": round(sum(results_by_cat[3]) / len(results_by_cat[3]) * 100, 1) if results_by_cat[3] else 0,
        "cat4": round(sum(results_by_cat[4]) / len(results_by_cat[4]) * 100, 1) if results_by_cat[4] else 0,
        "total_turns": total_turns,
        "total_questions": total_questions,
    }


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "locomo10.json"
    if not Path(data_path).exists():
        # Try common locations
        for p in [
            Path.home() / "projects/locomo/data/locomo10.json",
            Path("data/locomo10.json"),
        ]:
            if p.exists():
                data_path = str(p)
                break

    print(f"Running LoCoMo benchmark on: {data_path}")
    print(f"Engine: Lore Memory with sentence-transformers")
    print()
    run_benchmark(data_path)
