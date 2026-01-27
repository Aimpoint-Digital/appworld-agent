import json
from collections import Counter, defaultdict
import numpy as np

# Load results
with open("failure_analysis_test_normal_present.json") as f:
    data = json.load(f)

failures = data["failures"]

# -------------------------
# 1) Primary category counts
# -------------------------
primary_counts = Counter(
    f["classification"]["primary_category"]
    for f in failures
    if f.get("classification")
)

print("\nPrimary failure categories:")
for k, v in primary_counts.most_common():
    print(f"{k:40s} {v}")

# -------------------------
# 2) Secondary category counts
# -------------------------
secondary_counts = Counter(
    c
    for f in failures
    if f.get("classification")
    for c in f["classification"].get("secondary_categories", [])
)

print("\nSecondary failure categories:")
for k, v in secondary_counts.most_common():
    print(f"{k:40s} {v}")

# -------------------------
# 3) Confidence-weighted counts
# -------------------------
weighted = defaultdict(float)

for f in failures:
    cls = f.get("classification")
    if not cls:
        continue
    weighted[cls["primary_category"]] += float(cls.get("confidence", 1.0))

print("\nConfidence-weighted failures:")
for k, v in sorted(weighted.items(), key=lambda x: -x[1]):
    print(f"{k:40s} {v:.2f}")

# -------------------------
# 4) Failure category by difficulty
# -------------------------
by_difficulty = defaultdict(list)

for f in failures:
    cls = f.get("classification")
    if not cls:
        continue
    diff = f["evaluation"].get("difficulty", "unknown")
    by_difficulty[diff].append(cls["primary_category"])

print("\nFailure categories by difficulty:")
for diff in sorted(by_difficulty.keys()):
    print(f"\nDifficulty {diff}:")
    for k, v in Counter(by_difficulty[diff]).most_common():
        print(f"  {k:35s} {v}")

# -------------------------
# 5) Evidence density (debuggability)
# -------------------------
evidence_lengths = [
    len(f["classification"].get("evidence", []))
    for f in failures
    if f.get("classification")
]

print("\nEvidence stats:")
print(f"  mean:   {np.mean(evidence_lengths):.2f}")
print(f"  median: {np.median(evidence_lengths)}")
print(f"  min/max:{min(evidence_lengths)} / {max(evidence_lengths)}")

# -------------------------
# 6) Executive summary line
# -------------------------
top3 = primary_counts.most_common(3)
summary = ", ".join(f"{k} ({v})" for k, v in top3)

print(
    f"\nOut of {data['num_failed_tasks']} failed tasks, "
    f"the most common failure modes were: {summary}."
)
