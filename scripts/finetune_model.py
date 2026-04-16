"""
Fine-tune MiniLM-L6-v2 for personal memory retrieval.

Uses TripletLoss with (query, positive_fact, negative_fact) triplets
to teach the model that a query should be closer to its answer than
to unrelated facts.

Output: saves fine-tuned model to ./models/lore-minilm-v1/
"""

import json
import os
import sys

# Check dependencies
try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import TripletEvaluator
    from torch.utils.data import DataLoader
except ImportError:
    print("Install: pip install sentence-transformers torch")
    sys.exit(1)

# ============================================================
# Config
# ============================================================

BASE_MODEL = "all-MiniLM-L6-v2"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "lore-minilm-v1")
TRAIN_FILE = os.path.join(os.path.dirname(__file__), "train_pairs.json")
VAL_FILE = os.path.join(os.path.dirname(__file__), "val_pairs.json")

EPOCHS = 3
BATCH_SIZE = 16
WARMUP_RATIO = 0.1
LR = 5e-6  # very conservative — preserve base model quality

# ============================================================
# Load data
# ============================================================

print(f"Loading training data...")
with open(TRAIN_FILE) as f:
    train_data = json.load(f)
with open(VAL_FILE) as f:
    val_data = json.load(f)

print(f"  Train: {len(train_data)} triplets")
print(f"  Val: {len(val_data)} triplets")

# Convert to InputExample format — (query, positive) pairs for MNRL
# MNRL uses in-batch negatives, much more effective than explicit negatives
train_examples = [
    InputExample(texts=[d["query"], d["positive"]])
    for d in train_data
]

# Validation evaluator
val_anchors = [d["query"] for d in val_data]
val_positives = [d["positive"] for d in val_data]
val_negatives = [d["negative"] for d in val_data]

evaluator = TripletEvaluator(
    anchors=val_anchors,
    positives=val_positives,
    negatives=val_negatives,
    name="lore-val",
    show_progress_bar=False,
)

# ============================================================
# Load base model
# ============================================================

print(f"\nLoading base model: {BASE_MODEL}")
model = SentenceTransformer(BASE_MODEL)

# Evaluate base model before fine-tuning
print("Evaluating base model...")
base_result = evaluator(model)
base_score = base_result if isinstance(base_result, float) else base_result.get("lore-val_cosine_accuracy", list(base_result.values())[0] if base_result else 0.0)
print(f"  Base model triplet accuracy: {base_score:.4f}")

# ============================================================
# Fine-tune
# ============================================================

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model=model)
warmup_steps = int(len(train_dataloader) * EPOCHS * WARMUP_RATIO)

print(f"\nFine-tuning for {EPOCHS} epochs...")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LR}")
print(f"  Warmup steps: {warmup_steps}")
print(f"  Total steps: {len(train_dataloader) * EPOCHS}")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": LR},
    output_path=OUTPUT_DIR,
    show_progress_bar=True,
    evaluation_steps=len(train_dataloader),  # Evaluate every epoch
)

# ============================================================
# Evaluate fine-tuned model
# ============================================================

print(f"\nLoading fine-tuned model from {OUTPUT_DIR}")
finetuned = SentenceTransformer(OUTPUT_DIR)

print("Evaluating fine-tuned model...")
ft_result = evaluator(finetuned)
ft_score = ft_result if isinstance(ft_result, float) else ft_result.get("lore-val_cosine_accuracy", list(ft_result.values())[0] if ft_result else 0.0)
print(f"  Fine-tuned triplet accuracy: {ft_score:.4f}")
print(f"  Improvement: {(ft_score - base_score) * 100:+.1f}%")

# ============================================================
# Quick sanity check
# ============================================================

print("\nSanity check — query vs positive vs negative distances:")
test_cases = [
    ("Where do I work?", "I work at Google as a software engineer", "I graduated from MIT"),
    ("What's my pet's name?", "I have a dog named Luna", "My name is Alex"),
    ("Where did I live before?", "I used to live in Berlin", "I live in Lisbon"),
    ("Who is my fiancee?", "My girlfriend is Sarah, works at Google", "I like coffee"),
    ("What doesn't the person like?", "I don't like Java", "I love Python"),
]

for q, pos, neg in test_cases:
    embs = finetuned.encode([q, pos, neg])
    from sentence_transformers.util import cos_sim
    pos_sim = cos_sim(embs[0:1], embs[1:2]).item()
    neg_sim = cos_sim(embs[0:1], embs[2:3]).item()
    correct = "✓" if pos_sim > neg_sim else "✗"
    print(f"  {correct} Q: {q[:40]}")
    print(f"    +: {pos_sim:.4f}  -: {neg_sim:.4f}  gap: {pos_sim - neg_sim:+.4f}")

print(f"\nModel saved to: {OUTPUT_DIR}")
print("To use in lore-memory, set LORE_MODEL_PATH environment variable:")
print(f"  export LORE_MODEL_PATH={os.path.abspath(OUTPUT_DIR)}")
