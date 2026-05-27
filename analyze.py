import json

with open("benchmark_results.json") as f:
    data = json.load(f)

print(f"\n{'Expected':<12} {'Predicted':<12} {'Count'}")
print("-" * 36)

confusion = {}
for r in data["details"]:
    key = (r["expected"], r["predicted"])
    confusion[key] = confusion.get(key, 0) + 1

for (exp, pred), count in sorted(confusion.items()):
    marker = "✓" if exp == pred else "✗"
    print(f"{exp:<12} {pred:<12} {count}  {marker}")