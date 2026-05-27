import json
import time
from groq import Groq

client = Groq()

def evaluate_single(question, answer, domain):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert AI evaluator in {domain}.
Respond ONLY with valid JSON. No markdown, no backticks, nothing else.
Score 0-10 on: accuracy, clarity, completeness, reasoning.
Also give: overall (accuracy 40% + others 20% each), verdict using:
- overall >= 9.0 → Excellent
- overall >= 7.0 → Good
- overall >= 5.0 → Fair
- overall < 5.0 → Poor
summary, suggestions (list of 2-3).
Exact shape:
{{"accuracy":7.5,"clarity":8.0,"completeness":6.0,"reasoning":7.0,"overall":7.2,"verdict":"Good","summary":"...","suggestions":["...","..."]}}"""
            },
            {
                "role": "user",
                "content": f"Domain: {domain}\nQuestion: {question}\nAnswer to evaluate:\n{answer}"
            }
        ],
        temperature=0.3,
    )
    raw = response.choices[0].message.content
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)

def run_benchmark():
    with open("benchmark.json") as f:
        samples = json.load(f)

    results = []
    correct = 0
    total = len(samples)

    print(f"\nRunning benchmark on {total} samples...\n")
    print(f"{'ID':<5} {'Domain':<15} {'Expected':<12} {'Got':<12} {'Match'}")
    print("-" * 55)

    for sample in samples:
        try:
            result = evaluate_single(
                sample["question"],
                sample["answer"],
                sample["domain"]
            )
            predicted = result.get("verdict", "")
            expected  = sample["expected_verdict"]
            match     = predicted == expected

            if match:
                correct += 1

            results.append({
                "id": sample["id"],
                "domain": sample["domain"],
                "expected": expected,
                "predicted": predicted,
                "match": match,
                "overall_score": result.get("overall")
            })

            print(f"{sample['id']:<5} {sample['domain']:<15} {expected:<12} {predicted:<12} {'✓' if match else '✗'}")
            time.sleep(0.5)  # avoid rate limit

        except Exception as e:
            print(f"  ERROR on sample {sample['id']}: {e}")

    # ── SUMMARY ──────────────────────────────────────────
    agreement_rate = (correct / total) * 100

    print("\n" + "=" * 55)
    print(f"  BENCHMARK RESULTS")
    print("=" * 55)
    print(f"  Total samples     : {total}")
    print(f"  Correct           : {correct}")
    print(f"  Agreement Rate    : {agreement_rate:.1f}%")
    print("=" * 55)

    # per-domain breakdown
    domains = {}
    for r in results:
        d = r["domain"]
        if d not in domains:
            domains[d] = {"correct": 0, "total": 0}
        domains[d]["total"] += 1
        if r["match"]:
            domains[d]["correct"] += 1

    print("\n  Per-domain breakdown:")
    for domain, counts in domains.items():
        rate = (counts["correct"] / counts["total"]) * 100
        print(f"  {domain:<15} {counts['correct']}/{counts['total']}  ({rate:.0f}%)")

    # save results
    with open("benchmark_results.json", "w") as f:
        json.dump({
            "total": total,
            "correct": correct,
            "agreement_rate": round(agreement_rate, 1),
            "per_domain": domains,
            "details": results
        }, f, indent=2)

    print(f"\n  Full results saved to benchmark_results.json")
    return agreement_rate

if __name__ == "__main__":
    run_benchmark()