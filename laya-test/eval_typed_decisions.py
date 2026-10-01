"""
Evaluate laya-typed-decisions on the held-out test split of LocalLLaMA/typed-decisions.

Usage:
    python eval_typed_decisions.py                    # all 4 workflows, 400 test cases
    python eval_typed_decisions.py customer_service   # one workflow (100 cases)
    python eval_typed_decisions.py all 20             # quick smoke test on 20 cases

Workflows: all, customer_service, invoice_processing, security_incidents, agent_trace_observability
"""
import csv
import json
import sys
import time
from collections import defaultdict

from datasets import load_dataset
import laya

WORKFLOW = sys.argv[1] if len(sys.argv) > 1 else "all"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None

# Use the TEST split: the model was fine-tuned on the train split.
ds = load_dataset("LocalLLaMA/typed-decisions", WORKFLOW, split="test")
if LIMIT:
    ds = ds.select(range(min(LIMIT, len(ds))))
print(f"Loaded {len(ds)} test cases ({WORKFLOW})")

# If you downloaded the checkpoint to a local folder, use:
# agent = laya.load("./laya-models/typed-decisions", device="mps")
agent = laya.load("convaiinnovations/laya", subfolder="typed-decisions", device="mps")


def predicted_label(qtype, ans):
    """Turn a Laya answer into a label comparable with the dataset's gold label."""
    if qtype == "choice":
        return str(ans["choice"])
    if qtype == "noul":
        return "true" if ans["noul"] >= 0.5 else "false"
    if qtype == "score":
        # Laya returns the expected level; round it to the nearest level.
        return str(int(round(ans["score"])))
    raise ValueError(f"Unknown question type: {qtype}")


correct = defaultdict(int)
total = defaultdict(int)
rows_out = []
t0 = time.time()

for i, row in enumerate(ds):
    state = json.loads(row["state"])
    questions = json.loads(row["questions"])
    gold = json.loads(row["gold"])

    answers = agent.predict(state, questions)["answers"]

    for q, spec in questions.items():
        qtype = spec["type"]
        try:
            pred = predicted_label(qtype, answers[q])
        except KeyError:
            print(f"Unexpected answer format for '{q}':", answers[q])
            raise
        gold_label = str(gold[q]["label"])
        hit = pred == gold_label

        for key in [("ALL", "ALL"), (row["workflow"], "ALL"), ("ALL", qtype)]:
            total[key] += 1
            correct[key] += hit

        rows_out.append({"id": row["id"], "workflow": row["workflow"], "question": q,
                         "type": qtype, "gold": gold_label, "pred": pred, "correct": hit})

    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(ds)} cases, running accuracy "
              f"{correct[('ALL', 'ALL')] / total[('ALL', 'ALL')]:.3f}")

elapsed = time.time() - t0

print("\n=== Accuracy ===")
for key in sorted(total):
    label = "overall" if key == ("ALL", "ALL") else (f"workflow: {key[0]}" if key[1] == "ALL" else f"type: {key[1]}")
    print(f"{label:40s} {correct[key] / total[key]:.3f}  ({correct[key]}/{total[key]})")
print(f"\n{len(ds)} cases in {elapsed:.1f}s ({1000 * elapsed / len(ds):.0f} ms/case)")

with open("typed_decisions_predictions.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
    writer.writeheader()
    writer.writerows(rows_out)
print("Per-question predictions saved to typed_decisions_predictions.csv")