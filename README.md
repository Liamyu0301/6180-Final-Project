# Does memory format matter under repeated compression?

Course project (NEU 6180): how much a chat agent forgets when its conversation history is compressed
into a short memory, depending on the memory format, the compression ratio, and how many times the
memory is rewritten.

- **Main experiment** (`main/`): all 10 LoCoMo conversations, 1,540 questions, 25 conditions. Run on 2026-11-19.
- **LongMemEval extension** (`lme/`): 40 questions of two types that LoCoMo does not have (knowledge update,
  preference), at 20x. Run on 2026-11-20.
- **Pilot** (`pilot/`): one conversation at one ratio, run on 2026-11-17 for the progress review.

## Install

Python 3.10+ and two packages:

```bash
pip install anthropic matplotlib
```

Put an Anthropic API key in a file named `.env` in this folder (it is git-ignored):

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Data

`data/locomo10.json` is the public LoCoMo release (10 conversations), downloaded unchanged from
<https://github.com/snap-research/locomo/blob/main/data/locomo10.json>
(Maharana et al., 2024, arXiv 2402.17753). LoCoMo is released under CC BY-NC 4.0; the file is
redistributed here unchanged, for non-commercial coursework. We use the 1,540 questions that are left
after dropping the adversarial category 5. The conversations have 19 to 32 sessions and 19,274 to
35,956 tokens each (310K tokens in total, counted with the API's token counter).

`data/longmemeval_subset40.json` holds the 40 LongMemEval questions used in the extension, each with its
own history, cut from `longmemeval_s_cleaned.json` (Wu et al., 2024, arXiv 2410.10813; MIT licence;
<https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned>). The source file is 277 MB and is not in
this repo; `lme/make_subset.py` rebuilds the subset from it (fixed seed, sessions sorted by date).

## Reproduce the main result

One command runs every step and rebuilds the summary and the three figures:

```bash
python main/run.py meta && python main/run.py compress && python main/run.py answer && python main/run.py judge && python main/run.py attribute && python main/analyze.py
```

Each step skips work already saved in `main/results/`, so with the files in this repo the command only
rebuilds `summary.json` and the figures. To start from zero, delete the `.jsonl` files in `main/results/`.
A fresh run makes about 105,000 model calls, cost us $63, and took about three
hours: compression goes through the regular API (about two hours at 28 parallel requests; set
`THREADS` lower if your account allows fewer concurrent requests), and answering, judging and error
attribution go through the Batch API (under an hour for us; a batch can take up to 24 hours). A fresh
run will not give exactly the same numbers, because the
models are not deterministic. `SPEND_CAP_USD` (default 150) stops the run if spending passes it.

## Main result

Accuracy on 1,540 questions, from `main/results/summary.json`. Brackets are 95% intervals from a
bootstrap over the 10 conversations.

| Condition | 4x | 10x | 20x |
|---|---|---|---|
| Full history (no compression) | 82.1 [80.1–83.8] | same | same |
| Retrieval (BM25), same token budget | 67.5 [65.4–70.1] | 61.8 [59.7–64.3] | 55.6 [53.7–58.0] |
| Compressed once · prose | 66.3 [62.0–70.3] | 47.0 [43.1–51.6] | 27.6 [26.1–29.3] |
| Compressed once · JSON | 55.0 [51.8–57.8] | 35.9 [33.1–38.9] | 22.9 [21.2–24.7] |
| Compressed once · anchors + summary | 54.0 [51.5–56.7] | 34.2 [32.1–36.5] | 20.8 [18.6–23.5] |
| Rewritten every 4 sessions · prose | – | 37.3 [32.3–42.2] | – |
| Rewritten every 4 sessions · JSON | – | 29.8 [27.1–32.8] | – |
| Rewritten every 4 sessions · anchors + summary | – | 25.5 [22.2–29.3] | – |
| Rewritten every session · prose | 48.4 [45.6–51.9] | 25.6 [22.4–29.1] | 16.7 [13.9–19.7] |
| Rewritten every session · JSON | 47.7 [44.8–50.5] | 26.9 [25.1–28.8] | 15.5 [12.7–18.0] |
| Rewritten every session · anchors + summary | 35.1 [31.7–38.6] | 22.0 [19.0–25.1] | 15.0 [13.5–16.4] |

Figures: `main/results/accuracy.png`, `main/results/drift.png`, `main/results/errors.png`.

What the numbers show (differences are in percentage points, paired on the same questions, with the
bootstrap interval; all are listed under `comparisons` in `summary.json`):

1. **Rewriting costs accuracy in every format at every ratio.** Compressed once minus rewritten every
   session, at the same final size: prose +17.9 / +21.4 / +10.9 at 4x / 10x / 20x, JSON +7.3 / +9.0 / +7.5,
   anchors +19.0 / +12.2 / +5.8. Every interval excludes zero, and the one-shot memory wins in 9 or 10 of
   the 10 conversations in all nine cases.
   The incremental memories end slightly larger than the one-shot ones (on average 101% vs 93% of the
   budget at 4x, 97% vs 95% at 10x and 20x), so memory size does not explain the gap.
2. **More rewrites, lower accuracy, most clearly for prose.** At 10x, prose goes 47.0 (once) → 37.3
   (every 4 sessions) → 25.6 (every session), and both steps are outside their intervals. JSON goes
   35.9 → 29.8 → 26.9 and anchors 34.2 → 25.5 → 22.0: the first step is clear, the second is not
   (+2.9 [−0.2, 6.2] and +3.4 [−1.1, 8.6]).
3. **Format matters when the memory is written once, and much less after repeated rewriting.** Written
   once, prose beats JSON by 11.3 [6.0, 15.7] at 4x and 11.1 [6.9, 15.7] at 10x, and JSON and anchors are
   close to each other. Rewritten every session, prose and JSON are tied (+0.7, −1.2, +1.2 at 4x, 10x,
   20x). Anchors is behind both at 4x (−13.4 against prose) and behind JSON at 10x (−4.9 [−7.9, −1.6]);
   at 20x the three are within their intervals.
4. **Retrieval of raw turns is a strong baseline.** At 4x it is level with the best memory (67.5 vs
   66.3); at 10x and 20x it wins by 14.8 and 28.1 points. Its context is filled by an estimate of
   tokens per character and came out at about 106% of the budget (median), while the memories average
   about 95%, so retrieval had roughly a tenth more text. That is enough to matter for the 4x
   comparison and far too little to explain the 10x and 20x gaps. The exception is multi-hop questions
   at 10x, where the one-shot prose memory scores 51.1 against 35.8 for retrieval (`by_category` in
   `summary.json`).
5. **Most errors are information lost in compression.** Of 21,418 wrong answers from memories, 17,050
   (80%) are missing from the memory, 3,918 (18%) are in the memory but were not read out, and 450 (2%)
   are cases where the memory contradicts the gold answer. "In the memory but not read" is a larger share
   at 4x (38% of errors for one-shot memories) than at 20x (14%).

## Where each number comes from (main experiment)

| Number or figure | Produced by | Stored in |
|---|---|---|
| History size, session sizes, token budgets | `main/run.py meta` | `main/results/meta.json` |
| Every memory, every rewrite round, its size and the sizes tried | `main/run.py compress` | `main/results/memories.jsonl` |
| Model answers | `main/run.py answer` | `main/results/answers.jsonl` |
| Correct / wrong per answer | `main/run.py judge` | `main/results/judgments.jsonl` |
| Error type per wrong answer | `main/run.py attribute` | `main/results/attribution.jsonl` |
| Token usage and cost of every model call | all steps | `main/results/calls.jsonl` |
| Batch ids and how many replies each returned | batch steps | `main/results/batches.jsonl` |
| Accuracy table, intervals, paired comparisons, accuracy by question type, error counts, cost | `main/analyze.py` | `main/results/summary.json` |
| The three figures | `main/analyze.py` | `main/results/accuracy.png`, `drift.png`, `errors.png` |

Rows are keyed by conversation, condition and question id (`conv`, `cond`, `qid`). Condition names:
`full`, `rag_r{ratio}`, `oneshot_{format}_r{ratio}`, `incr_{format}_r{ratio}` (rewritten every session),
`incr4_{format}_r10` (rewritten every 4 sessions); formats are `L0` prose, `L1` JSON, `L2` anchors + summary.

## Setup of the main experiment

- Compressor and reader: `claude-haiku-5-5`. Judge and error attribution: `claude-sonnet-5-5`. Default
  thinking settings. The prompts are the pilot's, imported from `pilot/pilot.py` (`COMPRESS_SYSTEM`,
  `FORMAT_RULES`, `READER_SYSTEM`, `JUDGE_SYSTEM`, `ATTR_SYSTEM`).
- The memory budget of a conversation is its history's token count divided by 4, 10 or 20.
- Incremental compression: after each session (or each 4 sessions) the model gets the current memory
  and the new session(s) and writes a new memory. The budget is the same in every round, so the first
  rounds are below it: the raw history first exceeds the 4x budget at session 5 to 9, the 10x budget at
  session 2 to 4, and the 20x budget at session 1 or 2.
- Length rule (`compress_once` in `main/run.py`), measured with the API's token counter: a memory is
  never above 105% of the budget; once the material to compress (previous memory plus new sessions, or
  the whole history) reaches the budget, the memory must be at least 85% of the budget; before that,
  it must not be shorter than the previous memory. A draft that misses is sent back with its size and a
  request to cut or extend it, up to 8 attempts. Each request is scaled by how strongly the model
  reacted to the previous one, because on long memories it over-corrects.
- Length control result: 2,740 of 2,754 rounds in range. The 14 misses are between 82% and 111% of the
  budget. Of the 210 final memories, 209 are in range; one (conv-42, prose, rewritten every session, 4x)
  is at 110%.
- Retrieval baseline: BM25 over single turns, using the question as the query, filled up to the same
  token budget by an estimate of tokens per character. Measured from the requests' input tokens in
  `calls.jsonl`, the retrieved context is about 106% of the budget at the median, and 55% to 72% of the
  contexts are above the 105% limit that the memories are held to.
- Answering, judging and error attribution ran through the Batch API (24 batches, 98,418 requests,
  none failed or cut off). A reply that does not end normally is never scored; it is submitted again.
- Intervals: all questions of one conversation share one memory per condition, so the intervals
  resample conversations (10,000 draws), not single questions.

Cost of the main run, estimated from the token usage in `calls.jsonl` at list prices (the Haiku
cache-read price is assumed to be a tenth of its input price): compression $24.35, answering $3.37,
judging $15.08, error attribution $20.39, gate replay $0.14; total $63.34. The account's billing page
has the exact figure.

## LongMemEval extension (2026-11-20)

LoCoMo has no knowledge-update or preference questions, so 40 LongMemEval questions were added: 20
knowledge-update and 20 single-session-preference, drawn with a fixed seed. Each question has its own
history of 41 to 55 sessions and 165K to 171K tokens, so each question gets its own memories. One ratio
(20x, a budget of about 8,400 tokens) and one rewriting schedule (every 4 sessions, 11 to 14 rewrites).

```bash
python lme/run.py meta && python lme/run.py oracle && python lme/run.py compress && python lme/run.py fixfinal && python lme/run.py answer && python lme/run.py judge && python lme/run.py attribute && python lme/analyze.py
```

As in the main experiment, every step skips what is saved in `lme/results/`, so with the files in this
repo the command only rebuilds `summary.json` and the figure (tested: no file changes, no model calls).

Correct answers out of 20 per type, from `lme/results/summary.json`; brackets are 95% Wilson intervals
over the 40 questions. Figure: `lme/results/accuracy.png`.

| Condition | All 40 | Knowledge update | Preference |
|---|---|---|---|
| Evidence sessions only (check of reader and judge) | 80.0% [65.2–89.5] | 19 | 13 |
| Full history | 82.5% [68.0–91.3] | 20 | 13 |
| Retrieval (BM25), same budget | 85.0% [70.9–92.9] | 19 | 15 |
| Compressed once · prose | 80.0% [65.2–89.5] | 19 | 13 |
| Compressed once · JSON | 80.0% [65.2–89.5] | 18 | 14 |
| Compressed once · anchors + summary | 82.5% [68.0–91.3] | 18 | 15 |
| Rewritten every 4 sessions · prose | 65.0% [49.5–77.9] | 14 | 12 |
| Rewritten every 4 sessions · JSON | 72.5% [57.2–83.9] | 16 | 13 |
| Rewritten every 4 sessions · anchors + summary | 57.5% [42.2–71.5] | 15 | 8 |

What it shows, and how far it goes:

1. **Rewriting costs accuracy here too.** Compressed once against rewritten every 4 sessions, paired on
   the same questions (right only with the first / only with the second): prose 9 / 3, JSON 5 / 2,
   anchors 12 / 2. The direction is the same in all three formats; only anchors is clear on its own
   (exact sign test p = 0.013; prose 0.15, JSON 0.45). Pooled over the three formats it is 26 / 7
   (p = 0.001), but the three formats share the same questions, so that pooled test overstates the evidence.
   On the knowledge-update questions alone, where the ceiling is not in the way, it is 5 / 0, 2 / 0 and
   4 / 1 (pooled 11 / 1).
2. **A memory written once at 20x does as well as the full history on these questions** (32 or 33 of 40
   against 33). On LoCoMo at 20x the same memories reached 21% to 28% against 82%. This contrast is not
   controlled: the two benchmarks differ in history structure, question types, reader prompt and
   grading prompt. One likely reason, untested here, is that a LongMemEval history is mostly sessions
   unrelated to the question and the answer sits in one or two sessions, while LoCoMo asks about many
   details of one long relationship.
3. **The formats cannot be told apart with 40 questions.** Written once they are within one question of
   each other.
4. **The preference questions have a low ceiling.** With only the evidence session as context the
   reader gets 13 of 20, and every condition but one sits between 12 and 15. The knowledge-update
   questions carry the rewriting effect: 18 or 19 of 20 written once, 14 to 16 rewritten.
5. **Errors.** Of 23 wrong answers from memories written once, 16 are labelled "in the memory but not
   read", 6 missing, 1 conflict; of 42 from rewritten memories, 23 present and 19 missing. For the
   preference questions "present" mostly means the ceiling of point 4 (14 of the 16 such errors written
   once are preference questions), not a reader that overlooked something. Reading all 20
   knowledge-update errors by hand: the failure this type was chosen to find, an answer that gives the
   outdated value or swaps old and new, happens in 2 or 3 of them. More often one of the two values is
   gone, or the memory keeps both without a clear order (5 of the 20 are one question about which gadget
   was bought first), or the reader sees both values and declines to pick one.

Setup and deviations from the main experiment (all in `lme/run.py`; nothing in `main/` was changed):

- Compressor prompts, length rule, revise loop and batch code are imported from `main/run.py`.
- Reader prompt: written for a user-and-assistant chat and given the question's date. It was revised
  once, before any compression was run: the first version answered UNKNOWN to 3 of 20 preference
  questions with the evidence in view (`lme/results_check1/`).
- Grading: the benchmark's own prompts for these two question types, with `claude-sonnet-5-5` instead
  of the benchmark's GPT-4o and a system line asking for one word. 44 of 360 replies still contained
  reasoning; the verdict is taken from the last word. The judge is strict on the preference rubrics.
- Retrieval is trimmed to a measured size of at most 105% of the budget (median 98%; one context is at 62%).
- The word count in the first request of each round is scaled down by how much the model overshot in
  earlier rounds (`OVERSHOOT` in `lme/run.py`). The output cap was raised after one chain hit it 8 times.
- Sessions are sorted by date. In 4 knowledge-update questions a few sessions are stamped some hours
  after the question; they are kept, as in the benchmark.
- Length control: 1,534 of 1,608 rounds in range (68 over, 6 under). After the last round of
  out-of-range chains was redone up to twice (`fixfinal`; replaced records in `memories_replaced.jsonl`),
  232 of 240 final memories are in range. The 8 misses are all prose rewritten every 4 sessions, at 107%
  to 125% of the budget, which favours that condition.
- A 10x pilot on three questions was abandoned (`lme/results_pilot10x/`): compression cost $2.70 per
  question and the 17,000-token memories missed the length range in 8 of 103 rounds.
- Cost, estimated from token usage with Haiku's second rate card for prompts above 100K tokens:
  $68.93 (compression $66.02, answering $1.92, judging $0.34, error attribution $0.65), plus $7.81 for
  the abandoned 10x pilot.
- Limits: 20 questions per type, one ratio, one schedule, one sample of each memory; no human check of
  the judge or of the error labels.

## Runs and folders

| Folder | What it is |
|---|---|
| `main/results/` | The main experiment. Authoritative. |
| `main/results_gate1/` | First attempt on conv-30 only. Its length control missed the range in 16 rounds, mostly at 4x (the model over-corrected back and forth). Kept as the input of `main/gate_length_control.py`, which replays those 16 rounds with the fixed control (16 of 16 in range). Not results; cost $1.19. |
| `lme/results/` | The LongMemEval extension. |
| `lme/results_check1/` | Answers and verdicts from the first reader prompt and the first judge parser, on the evidence-only check. Superseded. |
| `lme/results_pilot10x/` | The abandoned 10x pilot on three LongMemEval questions. Not results. |
| `pilot/results/` | Pilot, second run (2026-11-17). |
| `pilot/results_v1/`, `pilot/results_v2_aborted/` | Pilot, first run and an aborted attempt. Superseded. |

`main/judge_check_conv30.csv` is a sheet of 50 randomly chosen conv-30 answers with the judge's verdict
left out, for labelling by hand.

## Known limits

- One model family. The main result is on one benchmark; the LongMemEval extension adds 40 questions of
  two other types at one ratio, which is enough to see the direction of the rewriting effect and little else.
- The LLM judge has not been checked against human labels yet (the sheet above is for that). Where the
  reader gave the same answer text in the pilot and in the main run (79 answers), the judge gave the
  same verdict every time. It can be lenient: one answer that led with a different fact and mentioned
  the gold one second was graded correct. Answers are about the same length in every condition (median
  5 or 6 words), so this should not favour one condition. The error-type labels have no human check at all.
- Three things differ from what the pilot suggested: retrieval beats every memory at 10x and 20x; the
  anchors format does not collapse under rewriting at 10x the way it did on conv-30 in the pilot; and
  prose and JSON are tied after rewriting. The Nov 18 slides show the pilot's numbers.
- Each memory was generated once, so there are 10 memories per cell (one per conversation). Intervals
  from 10 conversations are rough.
- The compressor does not know the questions; the retrieval baseline uses the question as its query.
- In incremental mode the memory is rewritten even while everything still fits in the budget. This is
  part of the condition being tested, and it matters most at 4x.
- The conv-30 numbers here differ from the pilot's: the length rule changed, and memories are sampled
  again. Full history on conv-30 is 84.0% here and 79.0% in the pilot; the 4 questions that changed all
  had different answer text from the reader.
- Open-domain questions often need knowledge from outside the conversation: full history gets only 39.6% of them.
- Some "contradicts the gold answer" labels reflect questionable gold answers rather than a wrong memory.
- `calls.jsonl` also holds calls from compression rounds that were interrupted when two jobs together
  passed the account's limit on concurrent requests, and the gate replay, so the cost total is slightly
  above one clean run.

## Pilot (2026-11-17)

One conversation (`conv-30`, 81 questions) at 10x, used for the Nov 18 progress review. Reproduce with:

```bash
python pilot/pilot.py compress && python pilot/pilot.py answer && python pilot/pilot.py judge && python pilot/pilot.py attribute && python pilot/analyze.py
```

Second-run accuracy, from `pilot/results/summary.json` (95% Wilson intervals over questions): full
history 79.0% (68.9–86.5); retrieval 66.7% (55.9–76.0); one-shot prose / JSON / anchors 56.8 / 45.7 / 45.7;
incremental (19 rewrites) 37.0 / 28.4 / 13.6. Paired counts, right only with the first / only with the
second: one-shot vs incremental 68 / 12 pooled over formats; one-shot prose vs JSON 16 / 7. Of 302 wrong
answers from memories: 243 missing, 51 present but not read, 8 conflict. First run, for comparison:
one-shot 55.6 / 34.6 / 40.7, incremental 42.0 / 42.0 / 33.3.

Pilot files: `pilot/results/meta.json` (sizes), `memories.jsonl`, `answers.jsonl`, `judgments.jsonl`,
`attribution.jsonl`, `calls.jsonl`, `summary.json`, `pilot_accuracy.png`, `pilot_errors.png`. The pilot
used an earlier length rule (85–105% of the budget, with the lower bound skipped while the source looked
short), which let two incremental memories end slightly under the range. Total API spend on the pilot
was about $3.20.

Project decisions are logged in `DECISIONS.md`. The speaker script for the Nov 18 progress review is in `presentation_script.md`.
