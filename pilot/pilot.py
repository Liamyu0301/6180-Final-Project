"""Pilot: how much does a chat agent forget when its history is compressed 10x?

One LoCoMo conversation (conv-30, 81 answerable questions), 8 conditions:
  full          the whole conversation (upper bound, no compression)
  rag           BM25 retrieval of raw turns at the same token budget
  oneshot_L0/L1/L2   compress the whole history once  (prose / JSON / anchors+summary)
  incr_L0/L1/L2      rewrite the memory after every session (19 rounds)

Steps (each one resumes where it stopped and appends to pilot/results/*.jsonl):
  python pilot/pilot.py compress [oneshot|incr]
  python pilot/pilot.py answer   [condition ...] [--limit N]
  python pilot/pilot.py judge
  python pilot/pilot.py attribute
"""
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "pilot" / "results"
RES.mkdir(parents=True, exist_ok=True)
for _line in (ROOT / ".env").read_text().splitlines():
    if "=" in _line:
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

client = anthropic.Anthropic(max_retries=8)

WORKER = "claude-haiku-5-5"   # compressor and reader
JUDGE = "claude-sonnet-5-5"   # grader and error attribution
SAMPLE = "conv-30"
RATIO = 10
FORMATS = ["L0", "L1", "L2"]
CONDITIONS = ["full", "rag"] + ["oneshot_" + f for f in FORMATS] + ["incr_" + f for f in FORMATS]
WORKERS = 8
SPEND_CAP_USD = 8.0
TOKENS_PER_CHAR = 0.37        # measured on conv-30: 19,274 tokens / 52,146 characters
# USD per million tokens: (input, output, cache read). Haiku's cache-read price is an assumption (10% of input).
PRICE = {WORKER: (0.10, 0.50, 0.01), JUDGE: (2.00, 10.00, 0.10)}

_lock = threading.Lock()
_spent = 0.0


# ---------------------------------------------------------------- data
def load():
    data = json.load(open(ROOT / "data" / "locomo10.json", encoding="utf-8"))
    d = next(x for x in data if x["sample_id"] == SAMPLE)
    conv = d["conversation"]
    n = max(int(k.split("_")[1]) for k in conv if re.fullmatch(r"session_\d+", k))
    sessions = [{"idx": i, "date": conv["session_%d_date_time" % i], "turns": conv["session_%d" % i]}
                for i in range(1, n + 1) if ("session_%d" % i) in conv]
    qa = [dict(q, qid=i) for i, q in enumerate(d["qa"]) if q.get("category") != 5]
    return sessions, qa


def render_turn(t):
    s = "%s: %s" % (t["speaker"], t.get("text", ""))
    if t.get("blip_caption"):
        s += " [shares a photo: %s]" % t["blip_caption"]
    return s


def render_session(s):
    return "[Session %d, %s]\n" % (s["idx"], s["date"]) + "\n".join(render_turn(t) for t in s["turns"])


def render_history(sessions):
    return "\n\n".join(render_session(s) for s in sessions)


# ---------------------------------------------------------------- API helpers
def append(name, rec):
    with _lock:
        with open(RES / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(name):
    p = RES / name
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def call(model, system, content, max_tokens, tag):
    """One model call. Logs usage to calls.jsonl and returns (text, stop_reason)."""
    global _spent
    if _spent > SPEND_CAP_USD:
        raise RuntimeError("spend cap reached: $%.2f" % _spent)
    t0 = time.time()
    # `content` is one user turn, or a full list of turns when a draft is sent back for revision.
    is_turns = isinstance(content, list) and content and "role" in content[0]
    resp = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                  messages=content if is_turns else [{"role": "user", "content": content}])
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    u = resp.usage
    cr = getattr(u, "cache_read_input_tokens", 0) or 0
    cw = getattr(u, "cache_creation_input_tokens", 0) or 0
    pin, pout, pcr = PRICE[model]
    cost = (u.input_tokens * pin + cw * pin * 1.25 + cr * pcr + u.output_tokens * pout) / 1e6
    with _lock:
        _spent += cost
    append("calls.jsonl", dict(tag, model=resp.model, stop_reason=resp.stop_reason,
                               input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                               cache_read=cr, cache_write=cw, cost_usd=round(cost, 6),
                               seconds=round(time.time() - t0, 2)))
    return text, resp.stop_reason


def count_tokens(text):
    return client.messages.count_tokens(model=WORKER, messages=[{"role": "user", "content": text}]).input_tokens


# ---------------------------------------------------------------- compression
COMPRESS_SYSTEM = (
    "You maintain the long-term memory of a chat assistant. The memory is the only thing a future "
    "assistant will see about past conversations. It will be used to answer questions about the two "
    "speakers: facts, dates, plans, preferences, and how things changed over time. You do not know the "
    "questions in advance. Session headers give the date of each session; words like 'yesterday' are "
    "relative to that date.")

FORMAT_RULES = {
    "L0": "FORMAT: free-form prose paragraphs. No lists, no JSON.",
    "L1": ("FORMAT: one JSON object with exactly these keys: \"facts\", \"preferences\", \"state_updates\", "
           "\"timeline\". Each value is a list of short strings. Each state_updates entry has the form "
           "\"subject: old value -> new value (date)\". Each timeline entry has the form \"date: event\". "
           "Output only the JSON object."),
    "L2": ("FORMAT: two parts. Part 1, under the heading ANCHORS: lines copied word for word from the "
           "conversation for exact numbers, dates, names and the latest state of things. Each line has the "
           "form [session date] Speaker: \"exact quote\". Part 2, under the heading SUMMARY: a short "
           "summary in your own words of everything else."),
}


def compress_once(fmt, budget, body, tag, tok_per_word=1.35):
    """Ask for a memory within the token budget.

    A memory is accepted only when its measured token count is between 85% and 105% of the budget:
    too long is unfair to the other formats, and too short throws information away for good in
    incremental mode. The lower bound is dropped while the source itself is still shorter than
    about 1.3 budgets (early sessions).

    The model cannot count tokens, so a draft that misses is sent back with its measured size and a
    request to cut or extend it by a percentage (up to 6 attempts). Revising the same draft
    converges; asking for a fresh memory with a different word count does not. `tok_per_word` is the
    ratio seen in the last accepted memory of this format, used to turn the budget into a word count.
    """
    hi = budget * 1.05
    lo = budget * 0.85 if len(body) * TOKENS_PER_CHAR >= budget * 1.3 else 0
    target = budget * 0.95
    first = {"role": "user", "content": "%s\n\nLENGTH: write close to %d words. Do not go over, and do not stop far "
             "short of it.\n\n%s\n\nOutput only the memory." % (FORMAT_RULES[fmt], int(target / tok_per_word), body)}
    turns, tries = [first], []
    for attempt in range(1, 7):
        text, stop = call(WORKER, COMPRESS_SYSTEM, turns, 12000, dict(tag, attempt=attempt))
        if stop != "end_turn" or not text:
            continue                      # cut off or empty: a failed attempt, ask again
        n = count_tokens(text)
        tries.append((text, n))
        if lo <= n <= hi:
            break
        change = round(abs(target - n) / n * 100)
        if n > hi:
            ask = ("That memory is %d tokens and the limit is %d. Cut it by about %d%%: drop the least important "
                   "details and keep the format." % (n, round(hi), change))
        else:
            ask = ("That memory is %d tokens but about %d are available. Make it about %d%% longer by adding more "
                   "of the important details from the source above (exact facts, dates, names, changes). Keep the "
                   "format." % (n, round(target), change))
        turns = [first, {"role": "assistant", "content": text},
                 {"role": "user", "content": ask + " Output only the full revised memory."}]
    else:
        if not tries:
            raise RuntimeError("no usable memory after 6 attempts (%s)" % tag)
        # No attempt landed in range: take the one closest to it, preferring not to exceed the budget.
        within = [t for t in tries if t[1] <= hi]
        text, n = max(within, key=lambda t: t[1]) if within else min(tries, key=lambda t: t[1])
    return {"text": text, "tokens": n, "attempts": attempt, "in_range": lo <= n <= hi,
            "over_budget": n > hi, "under_budget": n < lo, "tok_per_word": round(n / max(1, len(text.split())), 3)}


def step_compress(which):
    sessions, _ = load()
    history = render_history(sessions)
    done = {(m["mode"], m["fmt"], m.get("round")) for m in read_jsonl("memories.jsonl")}
    meta_path = RES / "meta.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
    else:
        h = count_tokens(history)
        meta = {"sample": SAMPLE, "sessions": len(sessions), "history_tokens": h, "budget_tokens": round(h / RATIO),
                "history_chars": len(history), "worker": WORKER, "judge": JUDGE, "ratio": RATIO}
        json.dump(meta, open(meta_path, "w"), indent=1)
    budget = meta["budget_tokens"]
    print("history %d tokens, budget %d tokens" % (meta["history_tokens"], budget))

    def oneshot(fmt):
        if ("oneshot", fmt, None) in done:
            return
        m = compress_once(fmt, budget, "CONVERSATION:\n" + history + "\n\nWrite the memory now.",
                          {"step": "compress", "mode": "oneshot", "fmt": fmt})
        append("memories.jsonl", dict(m, mode="oneshot", fmt=fmt, round=None))
        print("oneshot", fmt, m["tokens"], "tokens, attempts", m["attempts"])

    def incremental(fmt):
        rounds = sorted([m for m in read_jsonl("memories.jsonl") if m["mode"] == "incr" and m["fmt"] == fmt],
                        key=lambda m: m["round"])
        memory = rounds[-1]["text"] if rounds else ""
        ratio = rounds[-1].get("tok_per_word", 1.35) if rounds else 1.35
        for s in sessions[len(rounds):]:
            body = ("CURRENT MEMORY:\n%s\n\nNEW SESSION:\n%s\n\nWrite the full updated memory now. Keep what still "
                    "matters from the current memory and add what matters from the new session."
                    % (memory or "(empty)", render_session(s)))
            m = compress_once(fmt, budget, body, {"step": "compress", "mode": "incr", "fmt": fmt, "round": s["idx"]},
                              ratio)
            memory, ratio = m["text"], m["tok_per_word"]
            append("memories.jsonl", dict(m, mode="incr", fmt=fmt, round=s["idx"]))
            print("incr", fmt, "round", s["idx"], m["tokens"], "tokens")

    jobs = []
    if which in ("oneshot", "all"):
        jobs += [(oneshot, f) for f in FORMATS]
    if which in ("incr", "all"):
        jobs += [(incremental, f) for f in FORMATS]
    with ThreadPoolExecutor(6) as ex:
        futures = [(fn.__name__, f, ex.submit(fn, f)) for fn, f in jobs]
    failed = [(name, f, fut.exception()) for name, f, fut in futures if fut.exception()]
    for name, f, err in failed:           # report every failed chain, not just the first
        print("FAILED", name, f, repr(err))
    if failed:
        raise RuntimeError("%d compression job(s) failed" % len(failed))


def final_memories():
    out = {}
    for m in read_jsonl("memories.jsonl"):
        if m["mode"] == "oneshot":
            out["oneshot_" + m["fmt"]] = m
        elif "incr_" + m["fmt"] not in out or m["round"] > out["incr_" + m["fmt"]]["round"]:
            out["incr_" + m["fmt"]] = m
    return out


# ---------------------------------------------------------------- retrieval baseline (BM25 over single turns)
def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def build_rag(sessions, budget, tokens_per_char):
    docs = [(s, t, tok(render_turn(t))) for s in sessions for t in s["turns"]]
    df = Counter(w for _, _, ws in docs for w in set(ws))
    avg = sum(len(ws) for _, _, ws in docs) / len(docs)

    def retrieve(question):
        q = tok(question)
        scored = []
        for i, (_, _, ws) in enumerate(docs):
            tf = Counter(ws)
            score = sum(math.log(1 + (len(docs) - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * 2.5
                        / (tf[w] + 1.5 * (0.25 + 0.75 * len(ws) / avg)) for w in q if w in tf)
            scored.append((score, i))
        picked, used = [], 0
        for _, i in sorted(scored, reverse=True):
            s, t, _ = docs[i]
            line = "[%s] %s" % (s["date"], render_turn(t))
            cost = len(line) * tokens_per_char + 1
            if used + cost > budget:
                break
            picked.append((i, line, t["dia_id"]))
            used += cost
        picked.sort()
        return "\n".join(l for _, l, _ in picked), [d for _, _, d in picked]

    return retrieve


# ---------------------------------------------------------------- answering
READER_SYSTEM = (
    "You answer questions about a long conversation between two people. Use ONLY the context you are given. "
    "Reply with a short answer: a few words, a date, or a short list. No explanation. "
    "If the context does not contain the answer, reply exactly: UNKNOWN")


def step_answer(conds, limit):
    sessions, qa = load()
    meta = json.load(open(RES / "meta.json"))
    history = render_history(sessions)
    mems = final_memories()
    retrieve = build_rag(sessions, meta["budget_tokens"], meta["history_tokens"] / meta["history_chars"])
    done = {(a["cond"], a["qid"]) for a in read_jsonl("answers.jsonl")}
    todo = [(c, q) for c in conds for q in (qa[:limit] if limit else qa) if (c, q["qid"]) not in done]
    print("answering", len(todo), "items")

    def one(item):
        cond, q = item
        extra = {}
        if cond == "full":
            context = history
        elif cond == "rag":
            context, ids = retrieve(q["question"])
            ev = q.get("evidence") or []
            extra = {"retrieved": len(ids), "evidence_retrieved": bool(ev) and all(e in ids for e in ev)}
        else:
            context = mems[cond]["text"]
        content = [{"type": "text", "text": "CONTEXT:\n" + context, "cache_control": {"type": "ephemeral"}},
                   {"type": "text", "text": "QUESTION: " + q["question"]}]
        text, stop = call(WORKER, READER_SYSTEM, content, 2000, {"step": "answer", "cond": cond, "qid": q["qid"]})
        append("answers.jsonl", dict(extra, cond=cond, qid=q["qid"], category=q["category"], question=q["question"],
                                     gold=str(q["answer"]), answer=text, stop_reason=stop))

    # Warm the cache with the first item of each condition, then run the rest in parallel.
    first = {}
    for it in todo:
        first.setdefault(it[0], it)
    for it in first.values():
        one(it)
    rest = [it for it in todo if it not in first.values()]
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, rest))


# ---------------------------------------------------------------- judging
JUDGE_SYSTEM = (
    "You grade answers to questions about a conversation. You get a question, the gold answer and a model's "
    "answer. The model's answer is CORRECT if it states the same fact as the gold answer. Accept paraphrases, "
    "different date formats, and extra harmless detail. If the gold answer lists several items, the model must "
    "give the main ones. An answer of UNKNOWN, a different fact, or a different date is WRONG. "
    "Reply with exactly one word: CORRECT or WRONG.")


def step_judge():
    done = {(j["cond"], j["qid"]) for j in read_jsonl("judgments.jsonl")}
    todo = [a for a in read_jsonl("answers.jsonl") if (a["cond"], a["qid"]) not in done]
    print("judging", len(todo), "answers")

    def one(a):
        prompt = "QUESTION: %s\nGOLD ANSWER: %s\nMODEL ANSWER: %s" % (a["question"], a["gold"], a["answer"])
        text, stop = call(JUDGE, JUDGE_SYSTEM, prompt, 1500, {"step": "judge", "cond": a["cond"], "qid": a["qid"]})
        words = re.findall(r"CORRECT|WRONG", text.upper())
        append("judgments.jsonl", {"cond": a["cond"], "qid": a["qid"], "correct": bool(words) and words[-1] == "CORRECT",
                                   "raw": text, "stop_reason": stop})

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, todo))


# ---------------------------------------------------------------- error attribution
ATTR_SYSTEM = (
    "A model answered a question wrongly using only a MEMORY of a conversation. Decide why, by checking the "
    "memory against the gold answer. Reply with exactly one word:\n"
    "MISSING  - the memory does not contain the information needed for the gold answer.\n"
    "PRESENT  - the memory contains the information (stated or clearly derivable), so the reader missed it.\n"
    "CONFLICT - the memory states something that contradicts the gold answer.")


def step_attribute():
    mems = final_memories()
    answers = {(a["cond"], a["qid"]): a for a in read_jsonl("answers.jsonl")}
    done = {(x["cond"], x["qid"]) for x in read_jsonl("attribution.jsonl")}
    todo = [answers[(j["cond"], j["qid"])] for j in read_jsonl("judgments.jsonl")
            if not j["correct"] and j["cond"] in mems and (j["cond"], j["qid"]) not in done]
    todo.sort(key=lambda a: a["cond"])
    print("attributing", len(todo), "wrong answers")

    def one(a):
        content = [{"type": "text", "text": "MEMORY:\n" + mems[a["cond"]]["text"], "cache_control": {"type": "ephemeral"}},
                   {"type": "text", "text": "QUESTION: %s\nGOLD ANSWER: %s\nMODEL ANSWER: %s" % (
                       a["question"], a["gold"], a["answer"])}]
        text, stop = call(JUDGE, ATTR_SYSTEM, content, 1500, {"step": "attribute", "cond": a["cond"], "qid": a["qid"]})
        words = re.findall(r"MISSING|PRESENT|CONFLICT", text.upper())
        append("attribution.jsonl", {"cond": a["cond"], "qid": a["qid"], "label": words[-1] if words else "UNPARSED",
                                     "raw": text, "stop_reason": stop})

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    args = sys.argv[1:]
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    names = [a for a in args[1:] if not a.startswith("--") and not a.isdigit()]
    step = args[0] if args else ""
    if step == "compress":
        step_compress(names[0] if names else "all")
    elif step == "answer":
        step_answer(names or CONDITIONS, limit)
    elif step == "judge":
        step_judge()
    elif step == "attribute":
        step_attribute()
    else:
        print(__doc__)
    print("spent this run: $%.3f" % _spent)
