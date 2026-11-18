"""Gate for the length control: replay the incremental rounds that missed the 85-105% band in the
aborted second run, a few trials each, and report how many land in range and in how many attempts.

  python pilot/gate_length_control.py [trials]

Inputs come from pilot/results_v2_aborted/memories.jsonl (the memory before each missed round).
Calls are logged to pilot/results/calls.jsonl with step "gate".
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pilot

SRC = Path(__file__).resolve().parent / "results_v2_aborted" / "memories.jsonl"
MISSED = [("L0", 9), ("L1", 3), ("L2", 4), ("L2", 8), ("L2", 17), ("L0", 6)]
trials = int(sys.argv[1]) if len(sys.argv) > 1 else 2

sessions, _ = pilot.load()
budget = json.load(open(pilot.RES / "meta.json"))["budget_tokens"]
chain = {(m["fmt"], m["round"]): m for m in map(json.loads, open(SRC)) if m["mode"] == "incr"}


def one(job):
    fmt, rnd, trial = job
    prev = chain[(fmt, rnd - 1)]
    body = ("CURRENT MEMORY:\n%s\n\nNEW SESSION:\n%s\n\nWrite the full updated memory now. Keep what still "
            "matters from the current memory and add what matters from the new session."
            % (prev["text"], pilot.render_session(sessions[rnd - 1])))
    ratio = prev["tokens"] / max(1, len(prev["text"].split()))
    m = pilot.compress_once(fmt, budget, body, {"step": "gate", "fmt": fmt, "round": rnd, "trial": trial}, ratio)
    return fmt, rnd, trial, m["tokens"], m["attempts"], m["in_range"]


with ThreadPoolExecutor(6) as ex:
    out = list(ex.map(one, [(f, r, t) for f, r in MISSED for t in range(trials)]))
for row in sorted(out):
    print("%s round %2d trial %d: %4d tokens (%3d%% of budget), attempts %d, in range %s" % (
        row[0], row[1], row[2], row[3], round(100 * row[3] / budget), row[4], row[5]))
ok = sum(r[5] for r in out)
print("in range: %d of %d; within 3 attempts: %d; mean attempts %.1f" % (
    ok, len(out), sum(r[5] and r[4] <= 3 for r in out), sum(r[4] for r in out) / len(out)))
