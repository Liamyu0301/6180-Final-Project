"""Summarise the pilot: accuracy per condition, error types, memory sizes, cost. Writes
pilot/results/summary.json, pilot_accuracy.png and pilot_errors.png.

  python pilot/analyze.py
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent / "results"
ORDER = ["full", "rag", "oneshot_L0", "oneshot_L1", "oneshot_L2", "incr_L0", "incr_L1", "incr_L2"]
LABEL = {"full": "Full history", "rag": "Retrieval (BM25)",
         "oneshot_L0": "One-shot · prose", "oneshot_L1": "One-shot · JSON", "oneshot_L2": "One-shot · anchors",
         "incr_L0": "Incremental · prose", "incr_L1": "Incremental · JSON", "incr_L2": "Incremental · anchors"}


def rows(name):
    p = RES / name
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


answers = {(a["cond"], a["qid"]): a for a in rows("answers.jsonl")}
judg = defaultdict(list)
for j in rows("judgments.jsonl"):
    judg[j["cond"]].append(j)
attr = defaultdict(Counter)
for x in rows("attribution.jsonl"):
    attr[x["cond"]][x["label"]] += 1

mem_tokens = {}
for m in rows("memories.jsonl"):
    key = ("oneshot_" if m["mode"] == "oneshot" else "incr_") + m["fmt"]
    if m["mode"] == "oneshot" or m["round"] >= mem_tokens.get(key, (0, 0))[1]:
        mem_tokens[key] = (m["tokens"], m.get("round") or 0)

summary = {"meta": json.load(open(RES / "meta.json")), "conditions": {}, "cost_usd": {}}
for c in ORDER:
    js = judg.get(c, [])
    if not js:
        continue
    k, n = sum(j["correct"] for j in js), len(js)
    lo, hi = wilson(k, n)
    wrong = [j for j in js if not j["correct"]]
    row = {"n": n, "correct": k, "accuracy": round(100 * k / n, 1), "ci95": [round(100 * lo, 1), round(100 * hi, 1)],
           "answered_unknown": sum(answers[(c, j["qid"])]["answer"].strip().upper().startswith("UNKNOWN") for j in js)}
    if c in mem_tokens:
        row["memory_tokens"] = mem_tokens[c][0]
        row["errors"] = dict(attr[c])
    if c == "rag":
        row["errors"] = {"EVIDENCE_NOT_RETRIEVED": sum(not answers[("rag", j["qid"])].get("evidence_retrieved") for j in wrong),
                         "EVIDENCE_RETRIEVED": sum(bool(answers[("rag", j["qid"])].get("evidence_retrieved")) for j in wrong)}
    summary["conditions"][c] = row

cost = defaultdict(float)
calls = rows("calls.jsonl")
for x in calls:
    cost[x["step"] + ":" + x["model"]] += x["cost_usd"]
summary["cost_usd"] = {k: round(v, 3) for k, v in sorted(cost.items())}
summary["cost_usd"]["total"] = round(sum(cost.values()), 3)
summary["calls"] = len(calls)
json.dump(summary, open(RES / "summary.json", "w"), indent=1)

print("history %(history_tokens)d tokens, budget %(budget_tokens)d" % summary["meta"])
for c, r in summary["conditions"].items():
    print("%-22s %2d/%d = %5.1f%%  CI %s  unknown=%d  mem=%s  errors=%s" % (
        LABEL[c], r["correct"], r["n"], r["accuracy"], r["ci95"], r["answered_unknown"],
        r.get("memory_tokens", "-"), r.get("errors", "-")))
print("cost:", summary["cost_usd"], "calls:", summary["calls"])

# ---- figure 1: accuracy with 95% intervals
conds = [c for c in ORDER if c in summary["conditions"]]
acc = [summary["conditions"][c]["accuracy"] for c in conds]
err = [[a - summary["conditions"][c]["ci95"][0] for a, c in zip(acc, conds)],
       [summary["conditions"][c]["ci95"][1] - a for a, c in zip(acc, conds)]]
colour = ["#1C2331" if c in ("full", "rag") else "#1F6F8B" if c.startswith("oneshot") else "#C8551B" for c in conds]
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.barh([LABEL[c] for c in conds][::-1], acc[::-1], xerr=[err[0][::-1], err[1][::-1]], color=colour[::-1], capsize=3)
ax.set_xlim(0, 100)
ax.set_xlabel("Accuracy (%%), n = %d questions, 95%% Wilson interval" % summary["conditions"][conds[0]]["n"])
ax.set_title("Pilot on LoCoMo %s at %dx compression" % (summary["meta"]["sample"], summary["meta"]["ratio"]))
fig.tight_layout()
fig.savefig(RES / "pilot_accuracy.png", dpi=160)

# ---- figure 2: why the wrong answers were wrong (memory conditions only)
mem_conds = [c for c in conds if "errors" in summary["conditions"][c] and c != "rag"]
if mem_conds:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    left = [0] * len(mem_conds)
    for label, col in (("MISSING", "#C8551B"), ("PRESENT", "#1F6F8B"), ("CONFLICT", "#1C2331")):
        vals = [summary["conditions"][c]["errors"].get(label, 0) for c in mem_conds]
        ax.barh([LABEL[c] for c in mem_conds][::-1], vals[::-1], left=left[::-1], color=col,
                label={"MISSING": "Lost in compression", "PRESENT": "In memory, not read", "CONFLICT": "Memory written wrong"}[label])
        left = [l + v for l, v in zip(left, vals)]
    ax.set_xlabel("Number of wrong answers (out of %d questions)" % summary["conditions"][mem_conds[0]]["n"])
    ax.set_title("Why the wrong answers were wrong")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "pilot_errors.png", dpi=160)
