from datasets import load_dataset
import json
import random

# load dataset
ds = load_dataset("truthful_qa", "generation")

samples = []

validation = ds["validation"]

# 50 questions × 2 answers = 100 samples
for i in range(50):

    row = validation[i]

    question = row["question"]

    # GOOD answer
    samples.append({
        "id": len(samples) + 1,
        "domain": "TruthfulQA",
        "question": question,
        "answer": row["best_answer"],
        "expected_verdict": "Excellent"
    })

    # BAD answer
    samples.append({
        "id": len(samples) + 1,
        "domain": "TruthfulQA",
        "question": question,
        "answer": random.choice(row["incorrect_answers"]),
        "expected_verdict": "Poor"
    })

# save benchmark file
with open("benchmark_truthfulqa.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print(f"Created benchmark_truthfulqa.json with {len(samples)} samples")