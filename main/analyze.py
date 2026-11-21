"""Summarise the main run. Writes main/results/summary.json and three figures:
accuracy.png (accuracy by ratio and format), drift.png (accuracy by number of rewrites at 10x),
errors.png (why wrong answers were wrong).

  python main/analyze.py

Only conversations with every condition answered and judged are included. All questions of one
conversation share one memory per condition, so intervals come from a bootstrap that resamples
conversations (10,000 draws, fixed seed), not single questions.
"""
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent / "results"
RATIOS, FORMATS = [4, 10, 20], ["L0", "L1", "L2"]
NAME = {"L0": "Prose", "L1": "JSON", "L2": "Anchors + summary"}
COLOUR = {"L0": "#C8551B", "L1": "#1A7BB0", "L2": "#7A8B2E"}        # checked with the dataviz validator
ERR = [("MISSING", "Lost in compression", "#7E57C2"), ("PRESENT", "In memory, not read", "#00968F"),
       ("CONFLICT", "Memory contradicts the answer", "#B5830F")]
INK, MUTED, GRID = "#1C2331", "#6B7280", "#E5E2DA"
CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


def rows(name):
    p = RES / name
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def conditions():
    out = ["full"] + ["rag_r%d" % r for r in RATIOS]
    for r in RATIOS:
        out += ["oneshot_%s_r%d" % (f, r) for f in FORMATS] + ["incr_%s_r%d" % (f, r) for f in FORMATS]
    return out + ["incr4_%s_r10" % f for f in FORMATS]


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


meta = json.load(open(RES / "meta.json"))
answers = {(a["conv"], a["cond"], a["qid"]): a for a in rows("answers.jsonl")}
J = defaultdict(dict)                       # J[(conv, cond)][qid] = correct
for j in rows("judgments.jsonl"):
    J[(j["conv"], j["cond"])][j["qid"]] = j["correct"]
CONDS = conditions()
convs = [c for c in meta if all(len(J[(c, k)]) == meta[c]["questions"] for k in CONDS)]
skipped = [c for c in meta if c not in convs]
if not convs:
    raise SystemExit("no conversation has every condition judged yet")
rng = random.Random(0)
draws = [[rng.choice(convs) for _ in convs] for _ in range(10000)]


K = {key: sum(v.values()) for key, v in J.items()}     # correct answers per (conversation, condition)
N = {key: len(v) for key, v in J.items()}


def acc(cond, cs):
    return sum(K[(c, cond)] for c in cs) / sum(N[(c, cond)] for c in cs)


def interval(fn):
    """95% interval of fn(list of conversations) under the conversation bootstrap."""
    vals = sorted(fn(d) for d in draws)
    return vals[249], vals[9749]


summary = {"conversations": convs, "not_finished": skipped, "questions": sum(meta[c]["questions"] for c in convs),
           "interval": "bootstrap over conversations" if len(convs) > 1 else "Wilson over questions (one conversation)",
           "conditions": {}, "comparisons": [], "by_category": {}}

final = {}                                  # final memory of each (conv, cond)
for m in rows("memories.jsonl"):
    key = (m["conv"], "%s_%s_r%d" % (m["mode"], m["fmt"], m["ratio"]))
    if m["mode"] == "oneshot" or m["round"] == meta[m["conv"]]["sessions"]:
        final[key] = m
attr = defaultdict(Counter)
for x in rows("attribution.jsonl"):
    if x["conv"] in convs:
        attr[x["cond"]][x["label"]] += 1

for cond in CONDS:
    n = sum(len(J[(c, cond)]) for c in convs)
    k = sum(sum(J[(c, cond)].values()) for c in convs)
    lo, hi = interval(lambda cs: acc(cond, cs)) if len(convs) > 1 else wilson(k, n)
    row = {"n": n, "correct": k, "accuracy": round(100 * k / n, 1), "ci95": [round(100 * lo, 1), round(100 * hi, 1)],
           "per_conversation": {c: round(100 * acc(cond, [c]), 1) for c in convs},
           "answered_unknown": sum(answers[(c, cond, q)]["answer"].strip().upper().startswith("UNKNOWN")
                                   for c in convs for q in J[(c, cond)])}
    sizes = [100 * final[(c, cond)]["tokens"] / final[(c, cond)]["budget"] for c in convs if (c, cond) in final]
    if sizes:
        row["final_memory_pct_of_budget"] = [round(min(sizes)), round(sum(sizes) / len(sizes)), round(max(sizes))]
        row["errors"] = dict(attr[cond])
    if cond.startswith("rag"):
        wrong = [answers[(c, cond, q)] for c in convs for q, ok in J[(c, cond)].items() if not ok]
        row["errors"] = {"EVIDENCE_RETRIEVED": sum(bool(a.get("evidence_retrieved")) for a in wrong),
                         "EVIDENCE_NOT_RETRIEVED": sum(not a.get("evidence_retrieved") for a in wrong)}
    summary["conditions"][cond] = row
    cats = defaultdict(lambda: [0, 0])
    for c in convs:
        for q, ok in J[(c, cond)].items():
            cats[answers[(c, cond, q)]["category"]][0] += ok
            cats[answers[(c, cond, q)]["category"]][1] += 1
    summary["by_category"][cond] = {CATEGORY[g]: [k_, n_, round(100 * k_ / n_, 1)] for g, (k_, n_) in sorted(cats.items())}


def compare(a, b):
    """Accuracy of a minus accuracy of b on the same questions."""
    only_a = sum(J[(c, a)][q] and not J[(c, b)][q] for c in convs for q in J[(c, a)])
    only_b = sum(J[(c, b)][q] and not J[(c, a)][q] for c in convs for q in J[(c, a)])
    row = {"a": a, "b": b, "diff_points": round(100 * (acc(a, convs) - acc(b, convs)), 1),
           "right_only_with_a": only_a, "right_only_with_b": only_b,
           "conversations_where_a_higher": sum(acc(a, [c]) > acc(b, [c]) for c in convs)}
    if len(convs) > 1:
        lo, hi = interval(lambda cs: acc(a, cs) - acc(b, cs))
        row["ci95"] = [round(100 * lo, 1), round(100 * hi, 1)]
    summary["comparisons"].append(row)


compare("full", "rag_r10")
for r in RATIOS:
    for f in FORMATS:
        compare("oneshot_%s_r%d" % (f, r), "incr_%s_r%d" % (f, r))          # the cost of rewriting
    for mode in ("oneshot", "incr"):
        for f, g in (("L0", "L1"), ("L0", "L2"), ("L1", "L2")):
            compare("%s_%s_r%d" % (mode, f, r), "%s_%s_r%d" % (mode, g, r))  # format against format
    compare("rag_r%d" % r, "oneshot_L0_r%d" % r)
for f in FORMATS:
    compare("oneshot_%s_r10" % f, "incr4_%s_r10" % f)
    compare("incr4_%s_r10" % f, "incr_%s_r10" % f)

mem = [m for m in rows("memories.jsonl") if m["conv"] in convs]
summary["length_control"] = {"rounds": len(mem), "in_range": sum(m["in_range"] for m in mem),
                             "over_budget": sum(m["over_budget"] for m in mem),
                             "under_budget": sum(m["under_budget"] for m in mem)}
cost = defaultdict(float)
for x in rows("calls.jsonl"):
    cost["%s (%s)" % (x["step"], "batch" if x["batch"] else "regular")] += x["cost_usd"]
summary["cost_usd"] = dict({k: round(v, 2) for k, v in sorted(cost.items())}, total=round(sum(cost.values()), 2))
json.dump(summary, open(RES / "summary.json", "w"), indent=1)

C = summary["conditions"]
print("%d conversation(s), %d questions; intervals: %s" % (len(convs), summary["questions"], summary["interval"]))
if skipped:
    print("not finished, left out:", skipped)
for cond in CONDS:
    r = C[cond]
    print("%-18s %4d/%d = %5.1f%%  CI %-14s unknown=%-4d mem%%=%-16s errors=%s" % (
        cond, r["correct"], r["n"], r["accuracy"], r["ci95"], r["answered_unknown"],
        r.get("final_memory_pct_of_budget", "-"), r.get("errors", "-")))
print("length control:", summary["length_control"])
print("cost:", summary["cost_usd"])


# ---------------------------------------------------------------- figures
def style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)


def series(ax, xs, conds, colour, label, dashed=False, marker="o"):
    ys = [C[c]["accuracy"] for c in conds]
    err = [[y - C[c]["ci95"][0] for y, c in zip(ys, conds)], [C[c]["ci95"][1] - y for y, c in zip(ys, conds)]]
    ax.errorbar(xs, ys, yerr=err, color=colour, linewidth=2, linestyle="--" if dashed else "-", marker=marker,
                markersize=7, markeredgecolor="white", markeredgewidth=1.5, capsize=3, elinewidth=1, label=label)


note = "%d conversation%s, %d questions. Bars: 95%% interval (%s)." % (
    len(convs), "s" if len(convs) > 1 else "", summary["questions"], summary["interval"])

# figure 1: accuracy by compression ratio, one panel per mode
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, mode, title in zip(axes, ("oneshot", "incr"), ("Compressed once", "Rewritten after every session")):
    style(ax)
    xs = list(range(len(RATIOS)))
    ax.axhline(C["full"]["accuracy"], color=INK, linewidth=2, label="Full history (no compression)")
    series(ax, xs, ["rag_r%d" % r for r in RATIOS], MUTED, "Retrieval (BM25), same budget", marker="^")
    for i, f in enumerate(FORMATS):
        series(ax, [x + (i - 1) * 0.06 for x in xs], ["%s_%s_r%d" % (mode, f, r) for r in RATIOS], COLOUR[f], NAME[f],
               marker=["o", "s", "D"][i])
    ax.set_xticks(xs)
    ax.set_xticklabels(["%dx" % r for r in RATIOS])
    ax.set_xlabel("Compression ratio (history tokens / memory tokens)", color=INK)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.set_ylim(0, 100)
axes[0].set_ylabel("Questions answered correctly (%)", color=INK)
fig.legend(*axes[1].get_legend_handles_labels(), frameon=False, fontsize=9.5, loc="lower center", ncol=5,
           labelcolor=INK, bbox_to_anchor=(0.5, 0.045))
fig.suptitle("Accuracy by memory format and compression ratio (LoCoMo)", color=INK, fontsize=13, x=0.01, ha="left")
fig.text(0.01, 0.01, note, color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.1, 1, 0.96))
fig.savefig(RES / "accuracy.png", dpi=160)

# figure 2: accuracy by number of rewrites at 10x
fig, ax = plt.subplots(figsize=(7.5, 4.8))
style(ax)
for i, f in enumerate(FORMATS):
    series(ax, [x + (i - 1) * 0.05 for x in range(3)], ["%s_%s_r10" % (m, f) for m in ("oneshot", "incr4", "incr")],
           COLOUR[f], NAME[f], marker=["o", "s", "D"][i])
ax.set_xticks(range(3))
ax.set_xticklabels(["Once\n(1 rewrite)", "Every 4 sessions\n(5 to 8 rewrites)", "Every session\n(19 to 32 rewrites)"])
ax.set_ylim(0, 100)
ax.set_ylabel("Questions answered correctly (%)", color=INK)
ax.set_title("More rewrites, same final size: accuracy at 10x compression", color=INK, fontsize=12, loc="left")
ax.legend(frameon=False, fontsize=9, labelcolor=INK)
fig.text(0.01, 0.01, note, color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(RES / "drift.png", dpi=160)

# figure 3: error types as a share of all questions, one panel per ratio
if any(C[c].get("errors") for c in CONDS if not c.startswith(("full", "rag"))):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, r in zip(axes, RATIOS):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=1)
        conds = ["%s_%s_r%d" % (m, f, r) for m in ("oneshot", "incr") for f in FORMATS]
        labels = ["%s · %s" % ("Once" if c.startswith("oneshot") else "Every session", NAME[c.split("_")[1]]) for c in conds]
        left = [0.0] * len(conds)
        for key, label, col in ERR:
            vals = [100 * C[c]["errors"].get(key, 0) / C[c]["n"] for c in conds]
            ax.barh(labels[::-1], vals[::-1], left=left[::-1], color=col, height=0.62, label=label,
                    edgecolor="white", linewidth=[1 if v else 0 for v in vals[::-1]])
            left = [l + v for l, v in zip(left, vals)]
        ax.set_xlim(0, 100)
        ax.set_title("%dx" % r, color=INK, fontsize=12, loc="left")
        ax.set_xlabel("Wrong answers (% of all questions)", color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9.5, loc="lower center", ncol=3,
               labelcolor=INK, bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("Why the wrong answers were wrong", color=INK, fontsize=13, x=0.01, ha="left")
    fig.text(0.01, 0.01, "%d questions per bar." % summary["questions"], color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(RES / "errors.png", dpi=160)
