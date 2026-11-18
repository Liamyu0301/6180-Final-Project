# Speaker script: progress review, Nov 18

Deck: https://claude.ai/artifact/FrpaBsQ4zuYeMbpeK57Y8V

Simple spoken English, about 600 words in total (about 8.5 minutes at 72 words per minute).

## Slide 1: Does memory format matter under repeated compression? (40 words)

Hi everyone. Our project is about memory in chat agents. When a chat gets long, the agent keeps a short summary and rewrites it again and again. We ask what gets lost, and if the format of the memory matters.

## Slide 2: Agents forget quietly (63 words)

Here is the problem. A long chat does not fit in the model’s window. So the agent keeps a short memory. But a summary can drop small things, like a date or a number. And it is rewritten after every session, so a lost detail never comes back. The agent does not know it forgot. It still answers, and the answer is wrong.

## Slide 3: What we ask (58 words)

So this is our question. Every memory gets the same size, so the cost is the same. Does the format matter? And what happens when the memory is rewritten many times? We change three things: how small the memory is, which format it uses, and how many times it is rewritten. The third one is what we added.

## Slide 4: Three ways to write the same memory (67 words)

Here are the three formats. All three lines come from our own pilot, and they are about the same fact: Jon lost his job. Prose is free text. JSON has fixed keys, like facts and timeline. The third one keeps exact quotes with the date, plus a short summary. Look at the third one. It says “yesterday”, so the reader has to work out the real date.

## Slide 5: How we test it (71 words)

This is the pipeline. We take a long dialogue from LoCoMo. A model compresses it into a memory. A reader answers the questions using only that memory. A stronger model grades each answer. We do not train anything. We compress in two ways: all at once, or step by step after every session. We have two baselines: the full chat, and simple retrieval. And for each wrong answer, we ask why.

## Slide 6: What is known, and what we add (69 words)

We checked the closest papers. Wang and others showed that a summary can be updated step by step. ACE showed that rewriting again and again wears away detail, but not on dialogue. WhenLoss splits errors into two kinds. Another paper found that compression can hurt update and time questions. So rewriting a summary is not new. We add the drift for each format, and a third kind of error.

## Slide 7: The pilot runs end to end (111 words)

This is where we are. Yesterday we ran the whole pipeline on one LoCoMo conversation, with 81 questions. The memory is ten times smaller than the chat. With the full chat, the reader gets 79 percent. Every compressed memory is much lower. Compressed once, prose is the best. After 19 rewrites, prose is lower, 42 percent. But we found a problem in our length control, so that number is not final. Simple retrieval scores highest here. And most wrong answers are wrong because the fact is not in the memory any more. Please be careful with these numbers: one conversation, one run, and we have not checked the judge by hand.

## Slide 8: Four weeks to the poster (64 words)

Here is the plan. This week we check the pilot: we read fifty judge decisions by hand and fix the length control. Next week we run all ten LoCoMo conversations. In early December we draw the drift curves and label the errors. We also add a small LongMemEval set, because LoCoMo has no update or preference questions. The last week is for the poster.

## Slide 9: What could go wrong (76 words)

Last slide: the risks. The judge is a model, so it can be wrong. We will check fifty answers by hand. One conversation is a small sample, so we will run all ten. LoCoMo has no update or preference questions, so we plan to add a LongMemEval set. For cost, we use a cheap model and the batch API. And if the gap goes away in the full run, that is still a result. Thank you.
