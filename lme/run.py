"""LongMemEval extension: 40 questions (20 knowledge-update, 20 single-session-preference), 20x compression.

LoCoMo has no knowledge-update or preference questions, so this adds them at one ratio. In LongMemEval
every question has its own history (41 to 55 sessions), so every question gets its own memories.

Conditions, per question:
  oracle             only the session(s) that hold the answer (a check of the reader and the judge)
  full               the whole history (upper bound, no compression)
  rag_r20            BM25 retrieval of raw turns, trimmed to at most 105% of the token budget
  oneshot_{F}_r20    compress the whole history once
  incr4_{F}_r20      rewrite the memory after every 4 sessions
F is L0 (prose), L1 (JSON) or L2 (anchors + summary).

Why 20x and not 10x: a 10x pilot on three questions (lme/results_pilot10x/) cost $2.70 per question for
compression alone, and its 17,000-token memories missed the length range in 8 of 103 rounds. At 20x the
memories are about 8,400 tokens, the size range where the length control was verified in the main run.

The compressor prompts, the length control and the batch code are the ones of the main experiment,
imported from main/run.py. The reader and attribution prompts are adapted to a user-and-assistant chat,
and the grading prompts are the benchmark's own (src/evaluation/evaluate_qa.py in the LongMemEval repo),
run with claude-sonnet-5-5 instead of the benchmark's GPT-4o.

Steps (each resumes from lme/results/*.jsonl):
  python lme/run.py meta
  python lme/run.py oracle                  regular API: answer and judge the oracle condition
  python lme/run.py compress                regular API
  python lme/run.py fixfinal                redo the last round where the final memory is out of range
  python lme/run.py answer                  Batch API
  python lme/run.py judge                   Batch API
  python lme/run.py attribute               Batch API
  python lme/run.py status
Options: --ids id1,id2 (question ids)   --live
"""
import json
import os
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "main"))
os.environ.setdefault("SPEND_CAP_USD", "80")
import run as R  # noqa: E402  the main experiment's helpers; nothing in main/ is edited or written

P = R.P
WORKER, JUDGE, FORMATS, PRICE = R.WORKER, R.JUDGE, R.FORMATS, R.PRICE
RES = ROOT / "lme" / "results"
RES.mkdir(parents=True, exist_ok=True)
RATIO = 20
SUBSET = {q["question_id"]: q for q in json.load(open(ROOT / "data" / "longmemeval_subset40.json", encoding="utf-8"))}
IDS = list(SUBSET)
RAG = "rag_r%d" % RATIO
CONDITIONS = (["full", RAG] + ["oneshot_%s_r%d" % (f, RATIO) for f in FORMATS]
              + ["incr4_%s_r%d" % (f, RATIO) for f in FORMATS])


# ---------------------------------------------------------------- point the shared helpers at lme/results
def log_usage(tag, msg, batch, seconds=None):
    """Like the main experiment's, plus Haiku's second rate card: a prompt above 100K tokens is billed at
    5x ($0.50 / $2.50 per million). Batch requests are half price."""
    u = msg.usage
    cr = getattr(u, "cache_read_input_tokens", 0) or 0
    cw = getattr(u, "cache_creation_input_tokens", 0) or 0
    model = WORKER if msg.model.startswith(WORKER) else JUDGE
    pin, pout, pcr = PRICE[model]
    long_prompt = model == WORKER and u.input_tokens + cr + cw > 100_000
    cost = ((u.input_tokens * pin + cw * pin * 1.25 + cr * pcr + u.output_tokens * pout) / 1e6
            * (5 if long_prompt else 1) * (0.5 if batch else 1))
    with R._lock:
        R._spent += cost
    R.append("calls.jsonl", dict(tag, model=msg.model, batch=batch, stop_reason=msg.stop_reason,
                                 input_tokens=u.input_tokens, output_tokens=u.output_tokens, cache_read=cr,
                                 cache_write=cw, long_prompt=long_prompt, cost_usd=round(cost, 6), seconds=seconds))


_main_call = R.call


def call(params, tag, stream=False):
    """One-shot compression resends the whole history (over 100K tokens) on every length-control attempt,
    so that block is marked cacheable. The text sent to the model is unchanged."""
    if tag.get("step") == "compress":
        # Room to finish: with the main experiment's limit (1.5 budgets + 8,000) one chain hit the output cap
        # on all 8 attempts. A draft that is too long is then measured and sent back to be cut.
        params = dict(params, max_tokens=max(params["max_tokens"], 48000))
    if tag.get("step") == "compress" and tag.get("mode") == "oneshot":
        first = params["messages"][0]
        params = dict(params, messages=[{"role": "user", "content": [
            {"type": "text", "text": first["content"], "cache_control": {"type": "ephemeral"}}]}] + params["messages"][1:])
    return _main_call(params, tag, stream)


def load(qid):
    q = SUBSET[qid]
    return q["sessions"], [{"qid": 0, "question": q["question"], "answer": q["answer"], "category": q["question_type"]}]


def chains(ids):
    return [(i, RATIO, f, mode) for i in ids for f in FORMATS for mode in ("oneshot", "incr4")]


# The model's first draft is usually longer than the word count it is asked for, and by a similar factor
# from round to round. Ask for fewer words by that factor: a prior per format (median of first draft over
# target in the 10x pilot), then updated from each round's own first draft. This only changes the word
# count in the first request of a round; the length rule and the revise loop are the main experiment's.
OVERSHOOT = {("incr4", "L0"): 1.28, ("incr4", "L1"): 1.24, ("incr4", "L2"): 1.38,
             ("oneshot", "L0"): 1.20, ("oneshot", "L1"): 1.00, ("oneshot", "L2"): 1.50}
_over = {}
_main_compress = R.compress_once


def compress_once(fmt, budget, body, source_tokens, prev_tokens, tag, tok_per_word=None):
    key = (tag["conv"], tag["mode"], fmt)
    factor = _over.get(key, OVERSHOOT[(tag["mode"], fmt)])
    m = _main_compress(fmt, budget, body, source_tokens, prev_tokens, tag, (tok_per_word or R.TOK_PER_WORD[fmt]) * factor)
    if source_tokens >= budget and m["tries"]:
        _over[key] = min(1.8, max(0.8, factor * (m["tries"][0] / (budget * 0.95)) ** 0.5))
    return dict(m, first_request_factor=round(factor, 3))


R.RES, R.log_usage, R.call, R.load, R.chains, R.compress_once = RES, log_usage, call, load, chains, compress_once
R._spent = sum(c.get("cost_usd", 0) for c in R.read("calls.jsonl"))     # this extension's spending only
assert R.RES == RES and "main" not in RES.parts


# ---------------------------------------------------------------- meta
def step_meta(ids):
    path = RES / "meta.json"
    meta = json.load(open(path)) if path.exists() else {}
    for qid in ids:
        if qid in meta:
            continue
        sessions = SUBSET[qid]["sessions"]
        history = P.render_history(sessions)
        with ThreadPoolExecutor(8) as ex:
            stoks = list(ex.map(lambda s: R.count_tokens(P.render_session(s)), sessions))
        h = R.count_tokens(history)
        meta[qid] = {"type": SUBSET[qid]["question_type"], "sessions": len(sessions), "questions": 1,
                     "history_tokens": h, "history_chars": len(history), "session_tokens": stoks,
                     "evidence_sessions": SUBSET[qid]["evidence_sessions"], "budgets": {str(RATIO): round(h / RATIO)}}
        json.dump(meta, open(path, "w"), indent=1)
        print(qid, meta[qid]["type"], "sessions", len(sessions), "history", h, "budget", meta[qid]["budgets"][str(RATIO)],
              flush=True)


# ---------------------------------------------------------------- prompts
READER_SYSTEM = (
    "You are a chat assistant with a long history of conversations with one user. The CONTEXT is everything "
    "you have from those past conversations. Answer the user's question using the context. When the user asks "
    "for a suggestion, advice or an explanation, always give one, and build it on the specific things the "
    "context says about this user (what they own, did, like and plan), naming those details. When the user asks "
    "for a fact from the past conversations, give the latest value the context supports. Be brief: at most four "
    "sentences, no preamble. Only if the question asks for a fact about the past conversations and the context "
    "has nothing about it, reply exactly: UNKNOWN")
# First version (kept in lme/results_check1/): it told the reader to answer UNKNOWN whenever the context lacked
# "the answer" and to use one to three sentences. With only the evidence session as context it answered UNKNOWN
# to 3 of 20 preference questions, so it was revised once, before any compression was run.

# The benchmark's grading prompts, word for word (evaluate_qa.py, get_anscheck_prompt).
JUDGE_PROMPT = {
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
        "response contains the correct answer. Otherwise, answer no. If the response contains some previous "
        "information along with an updated answer, the response should be considered as correct as long as the "
        "updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a response from a model. "
        "Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does "
        "not need to reflect all the points in the rubric. The response is correct as long as it recalls and "
        "utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."),
}

ATTR_SYSTEM = (
    "A model answered a user's question wrongly using only a MEMORY of the user's past conversations. Decide why, "
    "by checking the memory against the reference. The reference is a correct answer, or, for a question that "
    "asks for a suggestion, a rubric that says what a good answer must use. Reply with exactly one word:\n"
    "MISSING  - the memory does not contain the information the reference needs.\n"
    "PRESENT  - the memory contains that information (stated or clearly derivable), so the reader missed it.\n"
    "CONFLICT - the memory states something that contradicts the reference, for example it only has an "
    "outdated value.")


def reader_params(context, q):
    return dict(model=WORKER, max_tokens=4000, system=READER_SYSTEM,
                messages=R.block("CONTEXT:\n" + context,
                                 "TODAY'S DATE: %s\nQUESTION: %s" % (q["question_date"], q["question"]), cache=False))


def judge_params(q, answer):
    # The benchmark's prompt is the user message. The system line only fixes the output format: without it
    # the judge sometimes writes its reasoning, and a reply that starts with "no" can end with "yes".
    return dict(model=JUDGE, max_tokens=3000, system="Reply with exactly one word: yes or no.",
                messages=[{"role": "user", "content": JUDGE_PROMPT[q["question_type"]]
                           .format(q["question"], q["answer"], answer)}])


def save_answer(info, text, stop):
    R.append("answers.jsonl", dict(info, answer=text, stop_reason=stop))
    return True


def save_judgment(info, text, stop):
    words = re.findall(r"[a-z]+", text.lower())
    if not words or words[-1] not in ("yes", "no"):
        return False            # the verdict is the last word (usually the only one); otherwise ask again
    R.append("judgments.jsonl", dict(info, correct=words[-1] == "yes", raw=text, stop_reason=stop))
    return True


def info_of(qid, cond):
    q = SUBSET[qid]
    return {"conv": qid, "cond": cond, "qid": 0, "category": q["question_type"], "question": q["question"],
            "gold": q["answer"]}


# ---------------------------------------------------------------- retrieval, trimmed to a measured size
_rag_lock = threading.Lock()


def retrieve(qid, meta):
    """BM25 over single turns. The index is filled by an estimate of tokens per character, so the result
    is measured and the fill is lowered until it is at most 105% of the budget (the memories' limit)."""
    q, budget = SUBSET[qid], meta[qid]["budgets"][str(RATIO)]
    tpc = meta[qid]["history_tokens"] / meta[qid]["history_chars"]
    for scale in (1.0, 0.96, 0.92, 0.88, 0.84, 0.8, 0.7):
        context, ids = P.build_rag(q["sessions"], budget * scale, tpc)(q["question"])
        n = R.count_tokens(context)
        if n <= budget * 1.05:
            break
    ev_turns = {t["dia_id"] for s in q["sessions"] for t in s["turns"] if t["has_answer"]}
    return context, {"retrieved": len(ids), "context_tokens": n, "evidence_retrieved": bool(ev_turns) and ev_turns <= set(ids)}


# ---------------------------------------------------------------- steps
def judge_items(ids, conds=None):
    done = {(j["conv"], j["cond"]) for j in R.read("judgments.jsonl")}
    items = []
    for a in R.read("answers.jsonl"):
        if a["conv"] in ids and (a["conv"], a["cond"]) not in done and (conds is None or a["cond"] in conds):
            items.append((R.cid_of("jdg", a["conv"], a["cond"], 0), judge_params(SUBSET[a["conv"]], a["answer"]),
                          {"conv": a["conv"], "cond": a["cond"], "qid": 0, "category": a["category"]}))
    print("judge: %d to do" % len(items), flush=True)
    return items


def step_oracle(ids):
    """Reader and judge on the evidence sessions only. If this is low, the prompts are the problem."""
    def build():
        done = {(a["conv"], a["cond"]) for a in R.read("answers.jsonl")}
        items = []
        for qid in ids:
            if (qid, "oracle") not in done:
                q = SUBSET[qid]
                context = P.render_history([s for s in q["sessions"] if s["evidence"]])
                items.append((R.cid_of("ans", qid, "oracle", 0), reader_params(context, q), info_of(qid, "oracle")))
        print("oracle answers: %d to do" % len(items), flush=True)
        return items

    R.run_live("answer", build, save_answer)
    R.run_live("judge", lambda: judge_items(ids, {"oracle"}), save_judgment)


def step_answer(ids, live):
    meta, mems = R.get_meta(), R.final_memories(ids)

    def build():
        done = {(a["conv"], a["cond"]) for a in R.read("answers.jsonl")}
        items = []
        for qid in ids:
            q = SUBSET[qid]
            for cond in CONDITIONS:
                if (qid, cond) in done:
                    continue
                info = info_of(qid, cond)
                if cond == "full":
                    context = P.render_history(q["sessions"])
                elif cond == RAG:
                    context, extra = retrieve(qid, meta)
                    info.update(extra)
                else:
                    context = mems[(qid, cond)]["text"]
                items.append((R.cid_of("ans", qid, cond, 0), reader_params(context, q), info))
        print("answer: %d to do" % len(items), flush=True)
        return items

    (R.run_live if live else R.run_batch)("answer", build, save_answer)


def step_judge(ids, live):
    (R.run_live if live else R.run_batch)("judge", lambda: judge_items(ids, set(CONDITIONS)), save_judgment)


def step_attribute(ids, live):
    mems = R.final_memories(ids)

    def build():
        answers = {(a["conv"], a["cond"]): a for a in R.read("answers.jsonl")}
        done = {(x["conv"], x["cond"]) for x in R.read("attribution.jsonl")}
        items = []
        for j in R.read("judgments.jsonl"):
            key = (j["conv"], j["cond"])
            if j["correct"] or j["conv"] not in ids or key not in mems or key in done:
                continue
            a, q = answers[key], SUBSET[j["conv"]]
            ref = "RUBRIC FOR A GOOD ANSWER" if q["question_type"] == "single-session-preference" else "CORRECT ANSWER"
            params = dict(model=JUDGE, max_tokens=4000, system=ATTR_SYSTEM,
                          messages=R.block("MEMORY:\n" + mems[key]["text"], "QUESTION: %s\n%s: %s\nMODEL ANSWER: %s"
                                           % (q["question"], ref, q["answer"], a["answer"]), cache=False))
            items.append((R.cid_of("att", j["conv"], j["cond"], 0), params,
                          {"conv": j["conv"], "cond": j["cond"], "qid": 0, "category": j["category"]}))
        print("attribute: %d to do" % len(items), flush=True)
        return items

    def handle(info, text, stop):
        words = re.findall(r"MISSING|PRESENT|CONFLICT", text.upper())
        if not words:
            return False
        R.append("attribution.jsonl", dict(info, label=words[-1], raw=text, stop_reason=stop))
        return True

    (R.run_live if live else R.run_batch)("attribute", build, handle)


def step_fixfinal(ids):
    """Redo the last round of any chain whose final memory is outside the length range (at most twice).
    The replaced records go to memories_replaced.jsonl."""
    meta = R.get_meta()
    for _ in range(2):
        mem = R.read("memories.jsonl")
        used = {(a["conv"], a["cond"]) for a in R.read("answers.jsonl")}     # never replace a memory already answered from
        bad = [m for m in mem if m["conv"] in ids and not m["in_range"]
               and (m["mode"] == "oneshot" or m["round"] == meta[m["conv"]]["sessions"])
               and (m["conv"], "%s_%s_r%d" % (m["mode"], m["fmt"], m["ratio"])) not in used]
        print("final memories out of range and not yet answered from: %d" % len(bad), flush=True)
        if not bad:
            return
        drop = {(m["conv"], m["mode"], m["fmt"], m["round"]) for m in bad}
        tmp = RES / "memories.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for m in mem:
                if (m["conv"], m["mode"], m["fmt"], m["round"]) not in drop:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
        for m in bad:
            R.append("memories_replaced.jsonl", m)
        tmp.replace(RES / "memories.jsonl")
        R.step_compress(ids)


def step_status():
    mem = R.read("memories.jsonl")
    for mode in ("oneshot", "incr4"):
        ms = [m for m in mem if m["mode"] == mode]
        if ms:
            print("%-7s rounds %4d  in range %4d  over %d  under %d  mean attempts %.1f" % (
                mode, len(ms), sum(m["in_range"] for m in ms), sum(m["over_budget"] for m in ms),
                sum(m["under_budget"] for m in ms), sum(m["attempts"] for m in ms) / len(ms)))
    for name in ("answers", "judgments", "attribution"):
        print("%s: %d rows" % (name, len(R.read(name + ".jsonl"))))
    agg = defaultdict(lambda: [0, 0, 0.0])
    for c in R.read("calls.jsonl"):
        a = agg[(c["step"], "batch" if c["batch"] else "live")]
        a[0] += 1
        a[1] += c["long_prompt"]
        a[2] += c["cost_usd"]
    for k, a in sorted(agg.items()):
        print("  %-10s %-5s calls %5d  above 100K tokens %4d  cost $%.2f" % (k[0], k[1], a[0], a[1], a[2]))
    print("total spent on lme/: $%.2f of cap $%.0f" % (sum(a[2] for a in agg.values()), R.SPEND_CAP_USD))


if __name__ == "__main__":
    args = sys.argv[1:]
    step = args[0] if args else ""
    ids = args[args.index("--ids") + 1].split(",") if "--ids" in args else IDS
    live = "--live" in args
    if step == "meta":
        step_meta(ids)
    elif step == "oracle":
        step_oracle(ids)
    elif step == "compress":
        R.step_compress(ids)
    elif step == "fixfinal":
        step_fixfinal(ids)
    elif step == "answer":
        step_answer(ids, live)
    elif step == "judge":
        step_judge(ids, live)
    elif step == "attribute":
        step_attribute(ids, live)
    elif step == "status":
        step_status()
    else:
        print(__doc__)
    print("spent so far on lme/: $%.2f" % R._spent)
