"""Summarise the LongMemEval extension. Writes lme/results/summary.json and lme/results/accuracy.png.

  python lme/analyze.py

Every question has its own history and its own memories, so questions are independent here: intervals
are 95% Wilson intervals over questions, and comparisons are paired counts on the same questions with an
exact two-sided sign test. With 20 questions per type, only large gaps can be told apart.
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent / "results"
FORMATS = ["L0", "L1", "L2"]
RATIO = 20
NAME = {"L0": "prose", "L1": "JSON", "L2": "anchors + summary"}
COLOUR = {"L0": "#C8551B", "L1": "#1A7BB0", "L2": "#7A8B2E"}        # same validated palette as the main figures
INK, MUTED, GRID = "#1C2331", "#6B7280", "#E5E2DA"
RAG = "rag_r%d" % RATIO
ONE = {f: "oneshot_%s_r%d" % (f, RATIO) for f in FORMATS}
INC = {f: "incr4_%s_r%d" % (f, RATIO) for f in FORMATS}
CONDS = ["oracle", "full", RAG] + [ONE[f] for f in FORMATS] + [INC[f] for f in FORMATS]
LABEL = {"oracle": "Evidence sessions only", "full": "Full history", RAG: "Retrieval (BM25), same budget"}
for f in FORMATS:
    LABEL[ONE[f]] = "Compressed once · " + NAME[f]
    LABEL[INC[f]] = "Rewritten every 4 sessions · " + NAME[f]
TYPES = {"knowledge-update": "Knowledge update", "single-session-preference": "Preference"}


def rows(name):
    p = RES / name
    return [json.loads(l) for l in open(p, encoding="utf-8").read().split("\n") if l] if p.exists() else []


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def sign_test(a, b):
    """Exact two-sided p-value for a and b discordant pairs under 'either is equally likely'."""
    n, k = a + b, min(a, b)
    if n == 0:
        return 1.0
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), 4)


meta = json.load(open(RES / "meta.json"))
answers = {(a["conv"], a["cond"]): a for a in rows("answers.jsonl")}
J = {(j["conv"], j["cond"]): j["correct"] for j in rows("judgments.jsonl")}
ids = [q for q in meta if all((q, c) in J for c in CONDS)]
if not ids:
    raise SystemExit("no question has every condition judged yet")
groups = {"all": ids}
groups.update({t: [q for q in ids if meta[q]["type"] == t] for t in TYPES})

summary = {"questions": len(ids), "not_finished": [q for q in meta if q not in ids],
           "per_type": {t: len(v) for t, v in groups.items()}, "conditions": {}, "comparisons": []}
final = {(m["conv"], "%s_%s_r%d" % (m["mode"], m["fmt"], m["ratio"])): m for m in rows("memories.jsonl")
         if m["mode"] == "oneshot" or m["round"] == meta[m["conv"]]["sessions"]}
attr = defaultdict(Counter)
for x in rows("attribution.jsonl"):
    if x["conv"] in ids:
        attr[x["cond"]][x["label"]] += 1
        attr[(x["cond"], x["category"])][x["label"]] += 1

for cond in CONDS:
    row = {}
    for g, qs in groups.items():
        k = sum(J[(q, cond)] for q in qs)
        row[g] = {"n": len(qs), "correct": k, "accuracy": round(100 * k / len(qs), 1), "ci95": wilson(k, len(qs))}
    row["answered_unknown"] = sum(answers[(q, cond)]["answer"].strip().upper().startswith("UNKNOWN") for q in ids)
    sizes = [100 * final[(q, cond)]["tokens"] / final[(q, cond)]["budget"] for q in ids if (q, cond) in final]
    if sizes:
        row["final_memory_pct_of_budget"] = [round(min(sizes)), round(sum(sizes) / len(sizes)), round(max(sizes))]
        row["errors"] = dict(attr[cond])
        row["errors_by_type"] = {t: dict(attr[(cond, t)]) for t in TYPES}
    if cond.startswith("rag"):
        pct = [100 * answers[(q, cond)]["context_tokens"] / meta[q]["budgets"][str(RATIO)] for q in ids]
        row["context_pct_of_budget"] = [round(min(pct)), round(sum(pct) / len(pct)), round(max(pct))]
        row["evidence_retrieved"] = sum(bool(answers[(q, cond)]["evidence_retrieved"]) for q in ids)
    summary["conditions"][cond] = row


def compare(a, b, pairs=None, name=None):
    """Paired on the same questions. `pairs` lists (a, b) condition pairs to pool, e.g. the three formats."""
    pairs = pairs or [(a, b)]
    for g, qs in groups.items():
        only_a = sum(J[(q, x)] and not J[(q, y)] for q in qs for x, y in pairs)
        only_b = sum(J[(q, y)] and not J[(q, x)] for q in qs for x, y in pairs)
        summary["comparisons"].append({"a": name or a, "b": "" if name else b, "group": g, "right_only_with_a": only_a,
                                       "right_only_with_b": only_b, "sign_test_p": sign_test(only_a, only_b)})


compare("oracle", "full")
compare("full", RAG)
compare(RAG, ONE["L0"])
for f in FORMATS:
    compare(ONE[f], INC[f])
compare(None, None, [(ONE[f], INC[f]) for f in FORMATS],
        "compressed once vs rewritten every 4 sessions, three formats pooled (not independent)")
for mode in (ONE, INC):
    for f, g in (("L0", "L1"), ("L0", "L2"), ("L1", "L2")):
        compare(mode[f], mode[g])

mem = [m for m in rows("memories.jsonl") if m["conv"] in ids]
summary["length_control"] = {"rounds": len(mem), "in_range": sum(m["in_range"] for m in mem),
                             "over_budget": sum(m["over_budget"] for m in mem),
                             "under_budget": sum(m["under_budget"] for m in mem)}
h = [meta[q]["history_tokens"] for q in ids]
summary["history_tokens"] = [min(h), round(sum(h) / len(h)), max(h)]
cost = defaultdict(float)
for x in rows("calls.jsonl"):
    cost["%s (%s)" % (x["step"], "batch" if x["batch"] else "regular")] += x["cost_usd"]
summary["cost_usd"] = dict({k: round(v, 2) for k, v in sorted(cost.items())}, total=round(sum(cost.values()), 2))
json.dump(summary, open(RES / "summary.json", "w"), indent=1)

C = summary["conditions"]
print("%d questions %s; history tokens %s" % (len(ids), summary["per_type"], summary["history_tokens"]))
for cond in CONDS:
    r = C[cond]
    print("%-18s all %2d/%d = %5.1f%% %-13s | KU %2d/%d | pref %2d/%d | unknown=%-2d mem%%=%-15s errors=%s" % (
        cond, r["all"]["correct"], r["all"]["n"], r["all"]["accuracy"], r["all"]["ci95"],
        r["knowledge-update"]["correct"], r["knowledge-update"]["n"], r["single-session-preference"]["correct"],
        r["single-session-preference"]["n"], r["answered_unknown"], r.get("final_memory_pct_of_budget", "-"),
        r.get("errors", "-")))
for c in summary["comparisons"]:
    if c["group"] == "all":
        print("  %-40s vs %-16s only-a %2d only-b %2d p=%s" % (c["a"][:40], c["b"], c["right_only_with_a"],
                                                                c["right_only_with_b"], c["sign_test_p"]))
print("length control:", summary["length_control"], "| retrieval context % of budget:", C[RAG].get("context_pct_of_budget"))
print("cost:", summary["cost_usd"])

# ---------------------------------------------------------------- figure: accuracy by condition, one panel per type
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True)
order = CONDS[::-1]
for ax, (t, title) in zip(axes, TYPES.items()):
    ax.set_facecolor("white")
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    for y, cond in enumerate(order):
        r = C[cond][t]
        col = COLOUR.get(cond.split("_")[1], INK) if "_L" in cond else (INK if cond != RAG else MUTED)
        ax.errorbar(r["accuracy"], y, xerr=[[r["accuracy"] - r["ci95"][0]], [r["ci95"][1] - r["accuracy"]]], color=col,
                    marker="s" if cond.startswith("incr4") else "o", markersize=8, markeredgecolor="white",
                    markeredgewidth=1.5, capsize=3, elinewidth=1.5, linewidth=0)
        ax.text(106, y, "%d/%d" % (r["correct"], r["n"]), va="center", fontsize=9, color=MUTED)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABEL[c] for c in order], color=INK)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Questions answered correctly (%)", color=INK)
    ax.set_title("%s (%d questions)" % (title, C["full"][t]["n"]), color=INK, fontsize=12, loc="left")
fig.suptitle("LongMemEval subset at %dx compression" % RATIO, color=INK, fontsize=13, x=0.01, ha="left")
fig.text(0.01, 0.01, "Bars: 95% Wilson interval over questions. Circles: memory written once or baseline; squares: "
         "memory rewritten every 4 sessions.", color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(RES / "accuracy.png", dpi=160)
