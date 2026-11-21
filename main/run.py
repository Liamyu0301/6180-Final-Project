"""Main experiment: 10 LoCoMo conversations x 3 compression ratios (4x, 10x, 20x).

Conditions, per conversation:
  full                  the whole conversation (upper bound, no compression)
  rag_r{R}              BM25 retrieval of raw turns at the token budget of ratio R
  oneshot_{F}_r{R}      compress the whole history once
  incr_{F}_r{R}         rewrite the memory after every session
  incr4_{F}_r10         rewrite the memory after every 4 sessions (10x only)
F is L0 (prose), L1 (JSON) or L2 (anchors + summary). The prompts are imported from pilot/pilot.py.

Steps (each one resumes from main/results/*.jsonl):
  python main/run.py meta
  python main/run.py compress             regular API: drafts are sent back for revision
  python main/run.py answer               Batch API
  python main/run.py judge                Batch API
  python main/run.py attribute            Batch API (add --live for the regular API with a warm cache)
  python main/run.py status
Options: --convs conv-30,conv-26   --live   --limit N (first N questions of each conversation)
"""
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pilot"))
import pilot as P  # noqa: E402  prompts, rendering and BM25 only; its call()/append() write to pilot/results

client = P.client
WORKER, JUDGE, FORMATS, PRICE = P.WORKER, P.JUDGE, P.FORMATS, P.PRICE
RES = ROOT / "main" / "results"
RES.mkdir(parents=True, exist_ok=True)
RATIOS = [4, 10, 20]
THREADS = int(os.environ.get("THREADS", 20))
SPEND_CAP_USD = float(os.environ.get("SPEND_CAP_USD", 150))
DATA = json.load(open(ROOT / "data" / "locomo10.json", encoding="utf-8"))
CONVS = [d["sample_id"] for d in DATA]

_lock = threading.Lock()


# ---------------------------------------------------------------- storage
def append(name, rec):
    line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
    with _lock:
        with open(RES / name, "ab", buffering=0) as f:    # one write call per line, so two processes cannot interleave
            f.write(line)


def read(name):
    p = RES / name
    if not p.exists():
        return []
    with _lock:
        lines = open(p, encoding="utf-8").read().split("\n")
    # a last line without a newline is still being written by another process: leave it for the next read
    return [json.loads(l) for l in lines[:-1] if l]


def load(conv):
    d = next(x for x in DATA if x["sample_id"] == conv)
    c = d["conversation"]
    n = max(int(k.split("_")[1]) for k in c if re.fullmatch(r"session_\d+", k))
    sessions = [{"idx": i, "date": c["session_%d_date_time" % i], "turns": c["session_%d" % i]}
                for i in range(1, n + 1) if ("session_%d" % i) in c]
    qa = [dict(q, qid=i) for i, q in enumerate(d["qa"]) if q.get("category") != 5]
    return sessions, qa


def conditions():
    out = ["full"] + ["rag_r%d" % r for r in RATIOS]
    for r in RATIOS:
        out += ["oneshot_%s_r%d" % (f, r) for f in FORMATS] + ["incr_%s_r%d" % (f, r) for f in FORMATS]
    return out + ["incr4_%s_r10" % f for f in FORMATS]


# ---------------------------------------------------------------- spend tracking and API calls
_spent = sum(c.get("cost_usd", 0) for c in read("calls.jsonl"))   # the cap holds across restarts


def check_cap(extra=0.0):
    if _spent + extra > SPEND_CAP_USD:
        raise RuntimeError("spend cap: $%.2f spent + $%.2f planned > $%.0f" % (_spent, extra, SPEND_CAP_USD))


def log_usage(tag, msg, batch, seconds=None):
    """Record the tokens and cost of one reply. Batch requests are billed at half price."""
    global _spent
    u = msg.usage
    cr = getattr(u, "cache_read_input_tokens", 0) or 0
    cw = getattr(u, "cache_creation_input_tokens", 0) or 0
    model = WORKER if msg.model.startswith(WORKER) else JUDGE
    pin, pout, pcr = PRICE[model]
    cost = (u.input_tokens * pin + cw * pin * 1.25 + cr * pcr + u.output_tokens * pout) / 1e6 * (0.5 if batch else 1)
    with _lock:
        _spent += cost
    append("calls.jsonl", dict(tag, model=msg.model, batch=batch, stop_reason=msg.stop_reason,
                               input_tokens=u.input_tokens, output_tokens=u.output_tokens, cache_read=cr,
                               cache_write=cw, cost_usd=round(cost, 6), seconds=seconds))


def text_of(msg):
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def patient(fn):
    """Run fn(), waiting out rate limits and server errors (the account caps concurrent requests)."""
    for i in range(60):
        try:
            return fn()
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError):
            if i == 59:
                raise
            time.sleep(30)


def count_tokens(text):
    return patient(lambda: P.count_tokens(text))


def call(params, tag, stream=False):
    """One regular (non-batch) request. Returns (text, stop_reason)."""
    check_cap()
    t0 = time.time()

    def go():
        if stream:      # long outputs: streaming avoids HTTP timeouts
            with client.messages.stream(**params) as s:
                return s.get_final_message()
        return client.messages.create(**params)

    msg = patient(go)
    log_usage(tag, msg, False, round(time.time() - t0, 2))
    return text_of(msg), msg.stop_reason


# ---------------------------------------------------------------- meta: sizes and budgets
def step_meta(convs):
    meta = json.load(open(RES / "meta.json")) if (RES / "meta.json").exists() else {}
    for conv in convs:
        if conv in meta:
            continue
        sessions, qa = load(conv)
        history = P.render_history(sessions)
        with ThreadPoolExecutor(8) as ex:
            stoks = list(ex.map(lambda s: P.count_tokens(P.render_session(s)), sessions))
        h = P.count_tokens(history)
        meta[conv] = {"sessions": len(sessions), "questions": len(qa), "history_tokens": h,
                      "history_chars": len(history), "session_tokens": stoks,
                      "budgets": {str(r): round(h / r) for r in RATIOS}}
        json.dump(meta, open(RES / "meta.json", "w"), indent=1)
        print(conv, "sessions", len(sessions), "questions", len(qa), "history", h, "budgets", meta[conv]["budgets"])
    return meta


def get_meta():
    return json.load(open(RES / "meta.json"))


# ---------------------------------------------------------------- compression
TOK_PER_WORD = {"L0": 1.7, "L1": 2.2, "L2": 2.0}     # measured on conv-30 memories; used for the first request only
ATTEMPTS = 8


def compress_once(fmt, budget, body, source_tokens, prev_tokens, tag, tok_per_word=None):
    """Ask for a memory and hold it to the length rule, measured in tokens with count_tokens.

    Rule: a memory is never above 105% of the budget. Once the material to compress (previous memory
    plus new sessions, or the whole history) reaches the budget, the memory must also be at least 85%
    of the budget. While the material is still below the budget nothing needs to be dropped, so the
    memory must not be shorter than the previous memory (or than 85% of the budget, if that is smaller).

    A draft that misses is sent back with its measured size and a request to cut or extend it (up to
    8 attempts). The model over-reacts to such requests on long memories (asked for 20% longer, it
    writes 70% longer), so each request is scaled by how strongly the model reacted to the previous
    one, and it always revises the draft that is closest to the target, not the latest one.
    A reply that is cut off or empty counts as a failed attempt.
    """
    hi = budget * 1.05
    fits = source_tokens < budget
    lo = min(prev_tokens, 0.85 * budget) if fits else 0.85 * budget
    target = budget * 0.95
    words = int(target / (tok_per_word or TOK_PER_WORD[fmt]))
    if fits:
        length = ("LENGTH: at most %d words. The material is shorter than this limit, so keep every detail that "
                  "could matter. Do not pad and do not repeat." % words)
    else:
        length = "LENGTH: write close to %d words. Do not go over, and do not stop far short of it." % words
    first = {"role": "user", "content": "%s\n\n%s\n\n%s\n\nOutput only the memory." % (P.FORMAT_RULES[fmt], length, body)}
    max_tokens = int(hi * 1.5) + 8000          # room for the memory plus thinking
    turns, tries, attempt, gain, asked = [first], [], 0, 1.0, None
    for attempt in range(1, ATTEMPTS + 1):
        text, stop = call(dict(model=WORKER, max_tokens=max_tokens, system=P.COMPRESS_SYSTEM, messages=turns),
                          dict(tag, attempt=attempt), stream=True)
        if stop != "end_turn" or not text:
            continue
        n = count_tokens(text)
        tries.append((text, n))
        if lo <= n <= hi:
            break
        if asked:                              # how strongly did the model react to the last request?
            wanted, got = math.log(asked[1] / asked[0]), math.log(n / asked[0])
            if wanted * got > 0:
                gain = min(4.0, max(0.5, got / wanted))
        aim = min(lo * 1.1, target) if (fits and n < lo) else target
        base, bn = min(tries, key=lambda t: abs(math.log(t[1] / aim)))     # the draft closest to the aim
        goal = bn * (aim / bn) ** (1 / gain)                               # ask for less if it over-reacts
        asked = (bn, goal)
        w = len(base.split())
        gw, pct = round(w * goal / bn), max(2, round(abs(goal - bn) / bn * 100))
        if bn > aim:
            ask = ("That memory is %d tokens (about %d words) and the limit is %d tokens. Cut it to about %d words, "
                   "about %d%% shorter: drop the least important details and keep the format. Do not cut more than "
                   "that." % (bn, w, round(hi), gw, pct))
        elif fits:
            ask = ("That memory is %d tokens (about %d words), but the memory before this update was %d tokens and "
                   "there is room for everything. Put back the details that were dropped, keep the new material, and "
                   "extend it to about %d words, about %d%% longer. Keep the format." % (bn, w, prev_tokens, gw, pct))
        else:
            ask = ("That memory is %d tokens (about %d words) but about %d tokens are available. Extend it to about "
                   "%d words, about %d%% longer, by adding more of the important details from the source above (exact "
                   "facts, dates, names, changes). Keep everything already in it and keep the format. Do not add more "
                   "than that." % (bn, w, round(target), gw, pct))
        turns = [first, {"role": "assistant", "content": base},
                 {"role": "user", "content": ask + " Output only the full revised memory."}]
    else:
        if not tries:
            raise RuntimeError("no usable memory after %d attempts (%s)" % (ATTEMPTS, tag))
        within = [t for t in tries if t[1] <= hi]   # none in range: the largest that fits, else the smallest
        text, n = max(within, key=lambda t: t[1]) if within else min(tries, key=lambda t: t[1])
    return {"text": text, "tokens": n, "attempts": attempt, "in_range": lo <= n <= hi, "over_budget": n > hi,
            "under_budget": n < lo, "tok_per_word": round(n / max(1, len(text.split())), 3),
            "budget": budget, "lo": round(lo), "source_tokens": source_tokens,
            "tries": [t[1] for t in tries]}


def incr_body(memory, group):
    many = len(group) > 1
    return ("CURRENT MEMORY:\n%s\n\nNEW SESSION%s:\n%s\n\nWrite the full updated memory now. Keep what still "
            "matters from the current memory and add what matters from the new session%s."
            % (memory or "(empty)", "S" if many else "", "\n\n".join(P.render_session(s) for s in group),
               "s" if many else ""))


def run_chain(conv, ratio, fmt, mode, meta, done):
    """One compression chain. `done` holds the rounds of this chain already saved."""
    sessions, _ = load(conv)
    budget = meta[conv]["budgets"][str(ratio)]
    stoks = meta[conv]["session_tokens"]
    base = {"conv": conv, "ratio": ratio, "mode": mode, "fmt": fmt}
    tag = dict(base, step="compress")
    if mode == "oneshot":
        if done:
            return
        m = compress_once(fmt, budget, "CONVERSATION:\n" + P.render_history(sessions) + "\n\nWrite the memory now.",
                          meta[conv]["history_tokens"], 0, dict(tag, round=None))
        append("memories.jsonl", dict(m, **base, round=None))
        print(conv, "r%d" % ratio, mode, fmt, m["tokens"], "tokens, attempts", m["attempts"], flush=True)
        return
    size = 1 if mode == "incr" else 4
    groups = [sessions[i:i + size] for i in range(0, len(sessions), size)]
    done = sorted(done, key=lambda m: m["round"])
    memory, mtok, tpw = (done[-1]["text"], done[-1]["tokens"], done[-1]["tok_per_word"]) if done else ("", 0, None)
    for g in groups[len(done):]:
        source = mtok + sum(stoks[s["idx"] - 1] for s in g)
        m = compress_once(fmt, budget, incr_body(memory, g), source, mtok, dict(tag, round=g[-1]["idx"]), tpw)
        memory, mtok, tpw = m["text"], m["tokens"], m["tok_per_word"]
        append("memories.jsonl", dict(m, **base, round=g[-1]["idx"]))
        print(conv, "r%d" % ratio, mode, fmt, "round", g[-1]["idx"], m["tokens"], "of", budget,
              "attempts", m["attempts"], "" if m["in_range"] else "OUT OF RANGE", flush=True)


def chains(convs):
    out = [(c, r, f, mode) for c in convs for r in RATIOS for f in FORMATS for mode in ("oneshot", "incr")]
    return out + [(c, 10, f, "incr4") for c in convs for f in FORMATS]


def step_compress(convs):
    meta = get_meta()
    saved = defaultdict(list)
    for m in read("memories.jsonl"):
        saved[(m["conv"], m["ratio"], m["fmt"], m["mode"])].append(m)
    # the first conversation named goes first (it is the one checked before the rest), then longest chains first
    jobs = sorted(chains(convs), key=lambda j: (j[0] != convs[0], -meta[j[0]]["budgets"][str(j[1])]))
    with ThreadPoolExecutor(THREADS) as ex:
        futures = [(j, ex.submit(run_chain, *j, meta, saved[j])) for j in jobs]
    failed = [(j, fut.exception()) for j, fut in futures if fut.exception()]
    for j, err in failed:
        print("FAILED", j, repr(err))
    if failed:
        raise RuntimeError("%d compression chain(s) failed" % len(failed))


def final_memories(convs):
    """{(conv, condition): memory record}. Raises if a chain has not reached the last session."""
    last = {c: load(c)[0][-1]["idx"] for c in convs}
    out = {}
    for m in read("memories.jsonl"):
        key = (m["conv"], "%s_%s_r%d" % (m["mode"], m["fmt"], m["ratio"]))
        if m["conv"] in last and (m["mode"] == "oneshot" or m["round"] == last[m["conv"]]):
            out[key] = m
    want = {(c, "%s_%s_r%d" % (mode, f, r)) for c, r, f, mode in chains(convs)}
    if want - set(out):
        raise RuntimeError("compression not finished for: %s" % sorted(want - set(out))[:6])
    return out


# ---------------------------------------------------------------- running many requests: batch or live
def cid_of(step, conv, cond, qid):
    cid = "%s__%s__%s__%d" % (step, conv, cond, qid)
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cid), cid      # the Batch API's rule for custom_id
    return cid


def run_batch(step, build, handle):
    """Submit everything `build()` returns as batches, wait, and pass good replies to `handle`.

    Batch ids are written to batches.jsonl before polling, and batches left over from an earlier run
    are collected before anything new is submitted. A reply that errored, expired, was cut off or is
    empty is not saved, so the next pass submits it again (3 passes).
    """
    for _ in range(3):
        items = build()
        collected = {b["id"] for b in read("batches.jsonl") if b.get("collected")}
        pending = [b for b in read("batches.jsonl") if b["step"] == step and "n" in b and b["id"] not in collected]
        if not pending:
            if not items:
                return
            chunk, size, chunks = [], 0, []
            for cid, params, _ in items:
                s = len(json.dumps(params))
                if chunk and (len(chunk) >= 10000 or size + s > 50e6):
                    chunks.append((chunk, size))
                    chunk, size = [], 0
                chunk.append({"custom_id": cid, "params": params})
                size += s
            chunks.append((chunk, size))
            for chunk, size in chunks:
                pin, pout, _ = PRICE[chunk[0]["params"]["model"]]
                check_cap((size * 0.3 * pin + len(chunk) * 300 * pout) / 1e6 * 0.5)   # rough upper estimate
                b = client.messages.batches.create(requests=chunk)
                rec = {"id": b.id, "step": step, "n": len(chunk), "bytes": size, "created": time.strftime("%F %T")}
                append("batches.jsonl", rec)
                pending.append(rec)
                print("submitted", step, "batch", b.id, len(chunk), "requests", flush=True)
        by = {cid: info for cid, _, info in items}
        for b in pending:
            while True:
                st = client.messages.batches.retrieve(b["id"])
                if st.processing_status == "ended":
                    break
                print(time.strftime("%T"), b["id"], "processing", st.request_counts.processing,
                      "succeeded", st.request_counts.succeeded, flush=True)
                time.sleep(60)
            ok = bad = 0
            for r in client.messages.batches.results(b["id"]):
                if r.result.type != "succeeded":
                    bad += 1
                    continue
                msg = r.result.message
                log_usage({"step": step, "cid": r.custom_id, "batch_id": b["id"]}, msg, True)
                text = text_of(msg)
                if msg.stop_reason != "end_turn" or not text or r.custom_id not in by:
                    bad += 1
                elif handle(by[r.custom_id], text, msg.stop_reason):
                    ok += 1
                else:
                    bad += 1
            append("batches.jsonl", {"id": b["id"], "step": step, "collected": True, "ok": ok, "not_saved": bad})
            print("collected", b["id"], "saved", ok, "not saved", bad, flush=True)
    left = len(build())
    if left:
        print("WARNING: %d %s item(s) still not done after 3 passes" % (left, step))


def run_live(step, build, handle):
    """Regular API. The first request of each (conversation, condition) goes first to warm the cache."""
    for _ in range(3):
        items = build()
        if not items:
            return
        firsts, rest = {}, []
        for it in items:
            g = (it[2]["conv"], it[2]["cond"])
            if g in firsts:
                rest.append(it)
            else:
                firsts[g] = it

        def one(it):
            cid, params, info = it
            text, stop = call(params, {"step": step, "cid": cid})
            if stop == "end_turn" and text:
                handle(info, text, stop)

        with ThreadPoolExecutor(THREADS) as ex:
            list(ex.map(one, firsts.values()))
            list(ex.map(one, rest))
    left = len(build())
    if left:
        print("WARNING: %d %s item(s) still not done after 3 passes" % (left, step))


def block(context, question, cache=True):
    first = {"type": "text", "text": context}
    if cache:
        first["cache_control"] = {"type": "ephemeral"}
    return [{"role": "user", "content": [first, {"type": "text", "text": question}]}]


# ---------------------------------------------------------------- answer, judge, attribute
def step_answer(convs, limit, live):
    meta, mems = get_meta(), final_memories(convs)

    def build():
        done = {(a["conv"], a["cond"], a["qid"]) for a in read("answers.jsonl")}
        items = []
        for conv in convs:
            sessions, qa = load(conv)
            history = P.render_history(sessions)
            tpc = meta[conv]["history_tokens"] / meta[conv]["history_chars"]
            rag = {r: P.build_rag(sessions, meta[conv]["budgets"][str(r)], tpc) for r in RATIOS}
            for cond in conditions():
                for q in (qa[:limit] if limit else qa):
                    if (conv, cond, q["qid"]) in done:
                        continue
                    info = {"conv": conv, "cond": cond, "qid": q["qid"], "category": q["category"],
                            "question": q["question"], "gold": str(q["answer"])}
                    if cond == "full":
                        context = history
                    elif cond.startswith("rag"):
                        context, ids = rag[int(cond.split("_r")[1])](q["question"])
                        ev = q.get("evidence") or []
                        info.update(retrieved=len(ids), evidence_retrieved=bool(ev) and all(e in ids for e in ev))
                    else:
                        context = mems[(conv, cond)]["text"]
                    # retrieved context differs for every question, so there is nothing to cache
                    params = dict(model=WORKER, max_tokens=4000, system=P.READER_SYSTEM,
                                  messages=block("CONTEXT:\n" + context, "QUESTION: " + q["question"],
                                                 cache=not cond.startswith("rag")))
                    items.append((cid_of("ans", conv, cond, q["qid"]), params, info))
        print("answer: %d to do" % len(items), flush=True)
        return items

    def handle(info, text, stop):
        append("answers.jsonl", dict(info, answer=text, stop_reason=stop))
        return True

    (run_live if live else run_batch)("answer", build, handle)


def step_judge(convs, live):
    def build():
        done = {(j["conv"], j["cond"], j["qid"]) for j in read("judgments.jsonl")}
        items = []
        for a in read("answers.jsonl"):
            if a["conv"] in convs and (a["conv"], a["cond"], a["qid"]) not in done:
                prompt = "QUESTION: %s\nGOLD ANSWER: %s\nMODEL ANSWER: %s" % (a["question"], a["gold"], a["answer"])
                params = dict(model=JUDGE, max_tokens=3000, system=P.JUDGE_SYSTEM,
                              messages=[{"role": "user", "content": prompt}])
                items.append((cid_of("jdg", a["conv"], a["cond"], a["qid"]), params,
                              {"conv": a["conv"], "cond": a["cond"], "qid": a["qid"]}))
        print("judge: %d to do" % len(items), flush=True)
        return items

    def handle(info, text, stop):
        words = re.findall(r"CORRECT|WRONG", text.upper())
        if not words:
            return False        # no verdict in the reply: ask again instead of scoring it as wrong
        append("judgments.jsonl", dict(info, correct=words[-1] == "CORRECT", raw=text, stop_reason=stop))
        return True

    (run_live if live else run_batch)("judge", build, handle)


def step_attribute(convs, live):
    mems = final_memories(convs)

    def build():
        answers = {(a["conv"], a["cond"], a["qid"]): a for a in read("answers.jsonl")}
        done = {(x["conv"], x["cond"], x["qid"]) for x in read("attribution.jsonl")}
        items = []
        for j in read("judgments.jsonl"):
            key = (j["conv"], j["cond"], j["qid"])
            if j["correct"] or j["conv"] not in convs or (j["conv"], j["cond"]) not in mems or key in done:
                continue
            a = answers[key]
            params = dict(model=JUDGE, max_tokens=4000, system=P.ATTR_SYSTEM,
                          messages=block("MEMORY:\n" + mems[(j["conv"], j["cond"])]["text"],
                                         "QUESTION: %s\nGOLD ANSWER: %s\nMODEL ANSWER: %s"
                                         % (a["question"], a["gold"], a["answer"])))
            items.append((cid_of("att", *key), params, {"conv": j["conv"], "cond": j["cond"], "qid": j["qid"]}))
        items.sort(key=lambda it: (it[2]["conv"], it[2]["cond"]))
        print("attribute: %d to do" % len(items), flush=True)
        return items

    def handle(info, text, stop):
        words = re.findall(r"MISSING|PRESENT|CONFLICT", text.upper())
        if not words:
            return False
        append("attribution.jsonl", dict(info, label=words[-1], raw=text, stop_reason=stop))
        return True

    (run_live if live else run_batch)("attribute", build, handle)


# ---------------------------------------------------------------- status
def step_status():
    mem = read("memories.jsonl")
    print("memories: %d rounds saved" % len(mem))
    for r in RATIOS:
        for mode in ("oneshot", "incr", "incr4"):
            ms = [m for m in mem if m["ratio"] == r and m["mode"] == mode]
            if ms:
                tight = [m for m in ms if m["source_tokens"] >= m["budget"]]
                print("  r%-2d %-7s rounds %4d  in range %4d  over budget %d  under %d  | once over budget: %d rounds, "
                      "size %.0f%%-%.0f%% of budget  | mean attempts %.1f"
                      % (r, mode, len(ms), sum(m["in_range"] for m in ms), sum(m["over_budget"] for m in ms),
                         sum(m["under_budget"] for m in ms), len(tight),
                         100 * min([m["tokens"] / m["budget"] for m in tight] or [0]),
                         100 * max([m["tokens"] / m["budget"] for m in tight] or [0]),
                         sum(m["attempts"] for m in ms) / len(ms)))
    for name in ("answers", "judgments", "attribution"):
        rows = read(name + ".jsonl")
        print("%s: %d rows, %d conversation(s)" % (name, len(rows), len({x["conv"] for x in rows})))
    agg = defaultdict(lambda: [0, 0, 0, 0, 0, 0.0])
    for c in read("calls.jsonl"):
        a = agg[(c["step"], "batch" if c["batch"] else "live")]
        for i, k in enumerate(("input_tokens", "cache_read", "cache_write", "output_tokens")):
            a[i + 1] += c[k]
        a[0] += 1
        a[5] += c["cost_usd"]
    for k, a in sorted(agg.items()):
        print("  %-10s %-5s calls %6d  cost $%7.3f  ($%.5f per call)  cached share of input %.0f%%"
              % (k[0], k[1], a[0], a[5], a[5] / a[0], 100 * a[2] / max(1, a[1] + a[2] + a[3])))
    stops = Counter((c["step"], c["stop_reason"]) for c in read("calls.jsonl") if c["stop_reason"] != "end_turn")
    print("replies that did not end normally:", dict(stops) or "none")
    print("total spent: $%.2f of cap $%.0f" % (sum(a[5] for a in agg.values()), SPEND_CAP_USD))


if __name__ == "__main__":
    args = sys.argv[1:]
    step = args[0] if args else ""
    convs = args[args.index("--convs") + 1].split(",") if "--convs" in args else CONVS
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    live = "--live" in args
    if step == "meta":
        step_meta(convs)
    elif step == "compress":
        step_compress(convs)
    elif step == "answer":
        step_answer(convs, limit, live)
    elif step == "judge":
        step_judge(convs, live)
    elif step == "attribute":
        step_attribute(convs, live)
    elif step == "status":
        step_status()
    else:
        print(__doc__)
    print("spent so far (all runs of main/): $%.2f" % _spent)
