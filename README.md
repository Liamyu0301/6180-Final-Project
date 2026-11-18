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
A fresh run makes about 1,500 API calls, costs roughly $1.50, and will not give the same numbers,
because the models are not deterministic (see "Two runs" below).

## Pilot result (second run, 2026-11-17)

Accuracy on 81 questions at 10x compression, from `pilot/results/summary.json`.
The chat is 19,274 tokens; the memory budget is 1,927 tokens.

| Condition | Accuracy | 95% interval | Final memory (tokens) |
|---|---|---|---|
| Full history | 79.0% | 68.9 – 86.5 | – |
| Retrieval (BM25), same budget | 66.7% | 55.9 – 76.0 | – |
| One-shot, prose | 56.8% | 45.9 – 67.0 | 1,956 |
| One-shot, JSON | 45.7% | 35.3 – 56.5 | 1,997 |
| One-shot, anchors + summary | 45.7% | 35.3 – 56.5 | 1,994 |
| Incremental (19 rewrites), prose | 37.0% | 27.3 – 47.9 | 2,009 |
| Incremental (19 rewrites), JSON | 28.4% | 19.7 – 39.0 | 2,006 |
| Incremental (19 rewrites), anchors + summary | 13.6% | 7.8 – 22.7 | 1,579 |

Paired on the same 81 questions (right only with the first / right only with the second):

- One-shot vs incremental: prose 22 / 6, JSON 19 / 5, anchors 27 / 1. Pooled over formats: 68 / 12.
- One-shot prose vs one-shot JSON: 16 / 7. One-shot prose vs one-shot anchors: 13 / 4.
- Retrieval vs one-shot prose: 19 / 11.

Of 302 wrong answers from compressed memories: 243 missing from the memory, 51 present but not read,
8 where the memory conflicts with the gold answer.

## Where each number comes from

| Number or figure | Produced by | Stored in |
|---|---|---|
| History size and token budget | `pilot.py compress` (`count_tokens`) | `pilot/results/meta.json` |
| Compressed memories, their token counts, every incremental round | `pilot.py compress` | `pilot/results/memories.jsonl` |
| Model answers | `pilot.py answer` | `pilot/results/answers.jsonl` |
| Correct / wrong per answer | `pilot.py judge` | `pilot/results/judgments.jsonl` |
| Error type per wrong answer | `pilot.py attribute` | `pilot/results/attribution.jsonl` |
| Token usage and cost of every API call | all steps | `pilot/results/calls.jsonl` |
| Accuracy table, confidence intervals, error counts, cost totals | `analyze.py` | `pilot/results/summary.json` |
| Accuracy figure, error figure | `analyze.py` | `pilot/results/pilot_accuracy.png`, `pilot_errors.png` |

The paired counts above are computed from `judgments.jsonl` (same `qid` under two conditions).

## Pilot setup

- Compressor and reader: `claude-haiku-5-5`. Judge and error attribution: `claude-sonnet-5-5`. Default thinking settings.
- Compression ratio 10x: the memory budget is the history's token count divided by 10.
- Conditions: full history; BM25 retrieval of raw turns at the same budget; one-shot compression in
  three formats (L0 prose, L1 JSON, L2 verbatim anchors plus summary); incremental compression in the
  same three formats, where the memory is rewritten after each of the 19 sessions.
- Length control: a memory is accepted only between 85% and 105% of the budget (measured with
  `count_tokens`). A draft that misses is sent back with its measured size and a request to cut or
  extend it by a percentage, up to 6 attempts (`compress_once` in `pilot/pilot.py`).
- Prompts are in `pilot/pilot.py` (`COMPRESS_SYSTEM`, `FORMAT_RULES`, `READER_SYSTEM`, `JUDGE_SYSTEM`, `ATTR_SYSTEM`).

## Two runs, and what changed between them

| Folder | What it is |
|---|---|
| `pilot/results_v1/` | First run. Its length control accepted memories far under budget (one incremental round was 918 tokens against a 1,927 budget), and one-shot JSON and anchors memories used only 83–88% of the budget. Superseded. |
| `pilot/results_v2_aborted/` | Memories from a second attempt whose length control still missed the band and then crashed. Kept only as input for `pilot/gate_length_control.py`, which replays the rounds that missed. Not results. |
| `pilot/results/` | Second run, with the length control described above. The full-history and retrieval rows are carried over from the first run (they do not depend on compression). |

Accuracy in the first run, for comparison: one-shot prose / JSON / anchors 55.6 / 34.6 / 40.7,
incremental 42.0 / 42.0 / 33.3. The gaps between formats and the incremental numbers differ a lot from
the second run, so a single run is not a stable estimate.

## Known limits of the pilot

- One conversation and one model family. The LLM judge has not been checked against human labels.
- With 81 questions, differences under roughly 15 points are within noise, and the two runs above show
  that run-to-run variation is large as well.
- The first and second runs differ in the length control, so they are not two samples of the same setup.
- In the second run the one-shot memories were produced by an earlier retry loop (re-asking with a
  rescaled word count) and the incremental ones by the revise-the-draft loop. Both use the same 85–105% band.
- Two incremental anchors memories are slightly under the band (round 6 at 83%, the final one at 82%).
  The final one was accepted because the lower bound is skipped when the source looks short, and that
  check uses a characters-to-tokens estimate that is too low for quote-heavy text.
- Final memory sizes still differ by format (1,579 to 2,009 tokens), and retrieval is filled to the full budget.
- Some "conflict" labels reflect questionable gold answers in the benchmark rather than a wrong memory.
- `calls.jsonl` in `pilot/results/` also contains the aborted attempt and the gate replay, so its cost total
  ($1.79) is higher than one clean run. Total API spend across all runs so far is about $3.20.

Project decisions are logged in `DECISIONS.md`. The speaker script for the Nov 18 progress review is in `presentation_script.md`.
