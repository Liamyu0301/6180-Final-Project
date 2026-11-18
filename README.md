# Does memory format matter under repeated compression?

Course project (NEU 6180): how much a chat agent forgets when its conversation history is compressed
into a short memory, depending on the memory format and on how many times the memory is rewritten.

This repo currently holds the **pilot** (one LoCoMo conversation). The main experiment is not run yet.

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
redistributed here unchanged, for non-commercial coursework. The pilot uses conversation `conv-30`
(19 sessions, 81 questions after dropping the adversarial category 5).

## Reproduce the pilot result

One command runs every step and rebuilds the summary and both figures:

```bash
python pilot/pilot.py compress && python pilot/pilot.py answer && python pilot/pilot.py judge && python pilot/pilot.py attribute && python pilot/analyze.py
```

Each step skips work already saved in `pilot/results/`, so delete that folder's `.jsonl` files to start from zero.
A fresh run calls the API (the Nov 17 run made 1,797 calls and cost about $1.57) and will not give byte-identical memories, because the models are not deterministic.

## Pilot result (run on 2026-11-17)

This first run is archived in `pilot/results_v1/`. A rerun with a corrected length control (see
"Known limits" below) writes to `pilot/results/`.

Accuracy on 81 questions, 10x compression, from `pilot/results_v1/summary.json`:

| Condition | Accuracy | 95% interval |
|---|---|---|
| Full history | 79.0% | 68.9 – 86.5 |
| Retrieval (BM25), same budget | 66.7% | 55.9 – 76.0 |
| One-shot, prose | 55.6% | 44.7 – 65.9 |
| One-shot, JSON | 34.6% | 25.1 – 45.4 |
| One-shot, anchors + summary | 40.7% | 30.7 – 51.6 |
| Incremental, prose | 42.0% | 31.8 – 52.8 |
| Incremental, JSON | 42.0% | 31.8 – 52.8 |
| Incremental, anchors + summary | 33.3% | 24.0 – 44.1 |

Of 285 wrong answers from compressed memories: 223 missing from the memory, 56 present but not read, 6 where the memory conflicts with the gold answer.

## Where each number comes from

| Number or figure | Produced by | Stored in |
|---|---|---|
| History size and token budget | `pilot.py compress` (`count_tokens`) | `pilot/results/meta.json` |
| Compressed memories, their token counts, every incremental round | `pilot.py compress` | `pilot/results/memories.jsonl` |
| Model answers | `pilot.py answer` | `pilot/results/answers.jsonl` |
| Correct / wrong per answer | `pilot.py judge` | `pilot/results/judgments.jsonl` |
| Error type per wrong answer | `pilot.py attribute` | `pilot/results/attribution.jsonl` |
| Token usage and cost of every API call | all steps | `pilot/results/calls.jsonl` |
| Accuracy table, confidence intervals, cost totals | `analyze.py` | `pilot/results/summary.json` |
| Accuracy figure, error figure | `analyze.py` | `pilot/results/pilot_accuracy.png`, `pilot_errors.png` |

## Pilot setup

- Compressor and reader: `claude-haiku-5-5`. Judge and error attribution: `claude-sonnet-5-5`.
- Compression ratio 10x: the memory budget is the history's token count divided by 10.
- Conditions: full history; BM25 retrieval of raw turns at the same budget; one-shot compression in
  three formats (L0 prose, L1 JSON, L2 verbatim anchors plus summary); incremental compression in the
  same three formats, where the memory is rewritten after each of the 19 sessions.
- Prompts are in `pilot/pilot.py` (`COMPRESS_SYSTEM`, `FORMAT_RULES`, `READER_SYSTEM`, `JUDGE_SYSTEM`, `ATTR_SYSTEM`).

## Known limits of the pilot

One conversation, one run, one model family. The LLM judge has not been checked against human labels.
With 81 questions, differences under roughly 15 points are within noise.

**The incremental numbers are confounded by the length control.** `compress_once` keeps the shortest of up
to five attempts and accepts anything not more than 5% over budget, so a round that comes out far too short
is accepted (for example anchors round 9 was 918 tokens against a 1,927 budget) and that loss carries into
every later round. Some intermediate rounds were also carried forward over budget (12 of 60 memories were
more than 5% over). The drop from one-shot to incremental prose therefore cannot yet be attributed to
rewriting alone. Fix before the main run: accept a memory only when it is between about 85% and 105% of the budget.

Final memory sizes also differ: one-shot memories used 83–88% of the budget, incremental ones 96–104%,
and retrieval was filled to the full budget.

Some "conflict" labels reflect questionable gold answers in the benchmark rather than a wrong memory.

Project decisions are logged in `DECISIONS.md`.
