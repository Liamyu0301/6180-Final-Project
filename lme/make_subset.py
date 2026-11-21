"""Draw the LongMemEval subset used in this project: 20 knowledge-update and 20 single-session-preference
questions (abstention questions left out), chosen with a fixed seed. Writes data/longmemeval_subset40.json.

Source file (277 MB, MIT licence, not in this repo): longmemeval_s_cleaned.json from
https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned  -> put it in data/longmemeval/

  python3 -I lme/make_subset.py        (standard library only)

Sessions are sorted by date, because 211 of the 500 questions list them out of order and incremental
compression must see them in the order they happened.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "longmemeval" / "longmemeval_s_cleaned.json"
OUT = ROOT / "data" / "longmemeval_subset40.json"
TYPES = ["knowledge-update", "single-session-preference"]
PER_TYPE, SEED = 20, 6180

data = json.load(open(SRC, encoding="utf-8"))
rng = random.Random(SEED)
subset = []
for typ in TYPES:
    pool = sorted((q for q in data if q["question_type"] == typ and not q["question_id"].endswith("_abs")),
                  key=lambda q: q["question_id"])
    for q in sorted(rng.sample(pool, PER_TYPE), key=lambda q: q["question_id"]):
        order = sorted(range(len(q["haystack_sessions"])), key=lambda i: q["haystack_dates"][i])   # stable
        sessions = []
        for n, i in enumerate(order, 1):
            sessions.append({"idx": n, "date": q["haystack_dates"][i], "session_id": q["haystack_session_ids"][i],
                             "evidence": q["haystack_session_ids"][i] in q["answer_session_ids"],
                             "turns": [{"speaker": "User" if t["role"] == "user" else "Assistant", "text": t["content"],
                                        "dia_id": "S%d:%d" % (n, k), "has_answer": bool(t.get("has_answer"))}
                                       for k, t in enumerate(q["haystack_sessions"][i], 1)]})
        subset.append({"question_id": q["question_id"], "question_type": typ, "question": q["question"],
                       "question_date": q["question_date"], "answer": str(q["answer"]),
                       "evidence_sessions": [s["idx"] for s in sessions if s["evidence"]], "sessions": sessions})
json.dump(subset, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
for typ in TYPES:
    qs = [q for q in subset if q["question_type"] == typ]
    print(typ, len(qs), "questions; sessions per question", min(len(q["sessions"]) for q in qs), "to",
          max(len(q["sessions"]) for q in qs), "; ids", [q["question_id"] for q in qs])
print("wrote", OUT.name, "%.1f MB" % (OUT.stat().st_size / 1e6))
