"""Gate for the length control: replay the compression rounds that missed the allowed range in the
first conv-30 attempt (main/results_gate1/memories.jsonl) with the current compress_once, starting from
the same previous memory, and report how many now land in range. Nothing is added to memories.jsonl.

  python main/gate_length_control.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R  # noqa: E402

old = [json.loads(l) for l in open(R.ROOT / "main" / "results_gate1" / "memories.jsonl", encoding="utf-8")]
by = {(m["ratio"], m["mode"], m["fmt"], m["round"]): m for m in old}
missed = [m for m in old if not m["in_range"]]


def replay(m):
    sessions, _ = R.load(m["conv"])
    tag = {"step": "gate", "conv": m["conv"], "ratio": m["ratio"], "mode": m["mode"], "fmt": m["fmt"], "round": m["round"]}
    if m["mode"] == "oneshot":
        body = "CONVERSATION:\n" + R.P.render_history(sessions) + "\n\nWrite the memory now."
        r = R.compress_once(m["fmt"], m["budget"], body, m["source_tokens"], 0, tag)
    else:                                   # every missed incremental round here folds in one session
        prev = by[(m["ratio"], m["mode"], m["fmt"], m["round"] - 1)]
        body = R.incr_body(prev["text"], [sessions[m["round"] - 1]])
        r = R.compress_once(m["fmt"], m["budget"], body, m["source_tokens"], prev["tokens"], tag, prev["tok_per_word"])
    print("r%-2d %-7s %s round %-4s before %5d  now %5d  allowed %d-%d  attempts %d  sizes tried %s  %s" % (
        m["ratio"], m["mode"], m["fmt"], m["round"], m["tokens"], r["tokens"], r["lo"], round(m["budget"] * 1.05),
        r["attempts"], r["tries"], "ok" if r["in_range"] else "STILL OUT"), flush=True)
    return r


with ThreadPoolExecutor(16) as ex:
    results = list(ex.map(replay, missed))
print("%d of %d rounds now in range; mean attempts %.1f; spent on main/ so far $%.2f" % (
    sum(r["in_range"] for r in results), len(results), sum(r["attempts"] for r in results) / len(results), R._spent))
