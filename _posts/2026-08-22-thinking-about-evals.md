---
layout: page
title: "Your eval is a measuring instrument"
description: "Notes on building evals worth trusting: traces over questions, rubric before instrument, quality before cost, and failure names specific enough to test."
date: 2026-08-22
image: /assets/blog/thinking-about-evals/fig7.png
---

Someone changes a prompt. The score goes from 71 to 74. Is the product better?

Nobody knows. So three people try it on the examples they happen to remember, agree it feels better, and ship.

That meeting is the failure. Not the model. The eval was supposed to settle the question, didn't, and everything downstream got decided by vibes with a number stapled on for cover.

An eval is a measuring instrument. Uncalibrated, it produces decoration.

## Start where you already know the answer

Not your most important workflow. The one you know cold — where you can look at an output and know in two seconds that it's wrong.

That instinct is your ground truth. It's sitting in your head where nobody else can use it. Getting it out is the whole job.

![A three-panel comic. Panel one: a person beside a prompt-to-tool-call-to-answer chain, saying they can tell in two seconds when the answer is wrong. Panel two: the same person beside a clipboard headed RUBRIC with three ticked criteria - cites the source, says when it can't, asks before guessing. Panel three: the chain feeding a box labelled 'graded against the rubric', with a starburst score of 4 out of 5, captioned as the part of your judgment that now survives outside your head.](/assets/blog/thinking-about-evals/fig1.png)

*Figure 1 — Every eval starts as a judgment you already make in your head.*

Pick a domain you don't know and you're building the instrument and learning the subject at the same time. When they disagree, you can't tell which one is broken.

## Study the traces, not the questions

Users don't arrive one turn at a time. They ask something vague, get half an answer, refine it, come back.

So write down what good looks like at each step — and separately, what good looks like for the whole run.

![Two panels side by side. The left panel, headed 'What most eval sets hold', shows a single question and single answer with a large orange X drawn through them. The right panel, headed 'What your users actually do', shows a four-turn document-QA conversation, with a dashed line from each assistant turn to a step-level check: found the right section, carried context across the turn, said so when the answer wasn't there. A brace under the whole exchange adds one end-to-end grade - did the user walk away with a correct, sourced answer.](/assets/blog/thinking-about-evals/fig2.png)

*Figure 2 — Users don't arrive one turn at a time, so don't grade one turn at a time.*

You need both. Step checks tell you where a run broke, which is what you need to fix it. The outcome tells you whether it mattered, which is what you need to prioritise. Grade only the steps and you optimise for a tidy transcript. Grade only the outcome and you know you're broken without knowing where.

## Then break them on purpose

Traces collected from a working system are traces where things went right. That's not where products fail.

![A wide top panel shows a clean four-step trace: clear question, tool returns clean rows, grounded answer, user is happy, with a comic starburst reading POW and the line 'and now wreck it on purpose'. Below it, three panels each show a way the trace degrades - the tool returns an empty result set, a retrieved chunk stops mid-sentence, and the user's request is ambiguous. Each panel names the failure response and the passing response beneath it.](/assets/blog/thinking-about-evals/fig3.png)

*Figure 3 — The happy path is the easy part. Write the traces where things arrive broken.*

Empty the tool result but keep the success status. Truncate the retrieved chunk mid-sentence. Strip the disambiguating noun out of the request.

For each one, write the failure you expect and the behaviour that counts as passing. "Said it found nothing" is a pass. "Produced something fluent" is not — and if your rubric can't tell them apart, it will score the second one just as highly.

## Write the rubric before you pick the instrument

The first question people ask is usually "should we use an LLM judge?" That's the second question.

First: what does good look like, written down tightly enough that two people grade the same output the same way.

![A diagram headed 'First: what does good look like?'. At the top is a box labelled THE RUBRIC listing four ticked criteria about grounding, refusal, asking before assuming, and never inventing a citation. Three arrows fan out below it to three instruments - a human reads it, an LLM judges it, a script checks it - each annotated with what it is good for and tagged with a cost of three dollar signs, two dollar signs, and one cent respectively. A strip along the bottom warns that picking the instrument first quietly rewrites the rubric into whatever that instrument finds easy to score.](/assets/blog/thinking-about-evals/fig4.png)

*Figure 4 — Write the rubric first. The instrument you pick will otherwise rewrite it for you.*

Order matters, because instruments are opinionated about what they can see. Pick the judge first and the rubric drifts toward what a judge scores consistently. Pick string matching first and it drifts toward exact-answer questions. Either way the criteria you actually cared about fall off the list, and nothing in the dashboard records that they're gone.

More of it is script-checkable than you'd expect. Did the citation resolve. Was the tool called with the arguments the case specified. Those are assertions, not judgments — free, exact, and they don't drift.

## Quality first. Cost next.

Treat evals like frontier models: establish the quality frontier, then work your way down the cost curve.

Version one should be the most expensive thing you can stand to run. Pay the annotators. Use the big judge model. Read fifty transcripts yourself.

![A line chart with cost per full eval run on the horizontal axis and how much you trust the signal on the vertical axis. A green curve rises steeply and then flattens into a long plateau. A stick figure plants a flag at the far right of the plateau, labelled step one - humans, the expensive judge, whatever it takes before you believe the number. An orange arrow runs leftwards along the plateau to a marked knee, labelled step two - smaller judge, sampling, a plain script wherever a script can decide it. A shaded zone at the far left is labelled cheap and useless, and a note points into it saying most teams start there and never learn the eval was the broken part.](/assets/blog/thinking-about-evals/fig5.png)

*Figure 5 — Establish the frontier first. Walking down the cost curve is the second problem.*

Cost reduction needs a reference. Swap in a smaller judge and the only way to know what you lost is to compare against a measurement you already trusted.

Start cheap and nothing ever tells you the instrument is broken. The dashboard reads 71, then 74, and nobody has grounds to doubt either number.

Then walk left. Distil the judge and measure agreement against the expensive one. Sample on commits, run the full set on releases. Convert every judged criterion that a script could decide.

## Name the failure

This is the first thing to build once you have v1. Pull the last 500 or 1,000 interactions, find the failures, cluster them, name the clusters.

Get specific. "Bad answer" isn't a cluster name, it's a shrug.

![Two panels. The left panel reads '1,000 traces reviewed, 63 of them failed' above a grid of exactly sixty-three grey dots crossed out with a large orange X, labelled 'one bin, labelled bad answer' and 'one pile, nothing to fix on Monday'. The right panel splits the same sixty-three dots into five named bins holding eighteen, fourteen, twelve, eleven and eight failures: wrong document retrieved, right document wrong section, context there but ignored and invented instead, should have punted but made something up, and ambiguous ask guessed instead of asking. Arrows run from each bin to a strip reading 'five different bugs, five different fixes'.](/assets/blog/thinking-about-evals/fig6.png)

*Figure 6 — Sixty-three failures. One pile teaches you nothing; five named clusters are five work items.*

Wrong document retrieved. Right document, wrong section. The context was there and it answered from memory anyway. Should have punted, made something up. Question was ambiguous and it guessed instead of asking.

Five bugs, three owners, five different fixes. Average them into one quality score and you'll spend a quarter improving retrieval while the damage is coming from ungrounded confidence.

A named failure also converts straight into a test. Once "right document, wrong section" exists as a category, you can write ten cases where the answer lives in §4.2 of a document whose §1 looks similar, and assert on the retrieved section ID rather than on the text of the answer. Cheap, deterministic, fails for exactly one reason. You can't write it until you've named it.

## Four kinds, four jobs

![A comic mountain scene illustrating four eval types. A stick-figure climber planting a flag near the summit is labelled hill-climb evals, hard cases at the edge of what the product can do today. A rope running down to an anchor and a safety net across the slope is labelled regression evals, everything that worked yesterday re-checked today. A smoke detector at the mountain's base is labelled smoke tests, not hard but unsurvivable if wrong. A rocket lifting off to the right is labelled launch evals, real-ish traffic with the least control and the most realism.](/assets/blog/thinking-about-evals/fig7.png)

*Figure 7 — Four kinds of eval, doing four different jobs. You need all four.*

**Hill-climb** — the frontier of what the product can do. Short shelf life by design: the week you pass them all, they stop telling you anything. Refresh constantly.

**Regression** — did we break today's product while climbing? Boring, almost always passes, should block. The urge to prune them is the urge to cut the net down because nothing has fallen into it.

**Smoke** — not hard, just can't be wrong. Product identity. The obvious refusal. Does it come up at all. Every single run.

**Launch** — closest to real traffic, least control, most realistic. The last thing that can still surprise you, because everything upstream is made of situations you already thought of.

## Keep it running, and keep it current

Two things kill a suite.

Friction. If a full run needs an afternoon of someone's attention, it runs when someone has an afternoon, which is never the moment it would have mattered. One command, fixed seed, fixed set, triggered by merges rather than by good intentions.

Drift. The cases you wrote in March measure March. The suite keeps returning confident numbers the whole time, which is exactly what makes it dangerous.

![Two panels. The left panel, headed 'Cheap to run, or it won't get run', shows a person saying they'll run the eval when they have an afternoon, then below a dividing line the automated version: every merge triggers the suite, producing 142 passes and 3 failures, with a note that the same seed and same set mean a change in the score is a change in the product. The right panel shows two bell curves on a shared baseline - a solid blue one labelled 'your eval set, written in March' and a dashed orange one shifted to the right labelled 'what users are actually asking now' - with the gap between them annotated 'nobody was asking this in March'. Beneath runs a three-step refresh loop: sample last week's real traffic, cluster the new failures, add cases and retire dead ones, repeating monthly.](/assets/blog/thinking-about-evals/fig8.png)

*Figure 8 — An eval that needs a volunteer runs roughly never — and the one you wrote in March is already stale.*

Sample last week's traffic, cluster the new failures, add cases, retire the dead ones. Monthly.

Watch the mix while you do it. If 90% of your set came out of the failure taxonomy, your headline number describes edge cases rather than users.

## What I don't have a good answer for

**The judge ceiling.** Strong judges agree with human preference around 80% of the time — roughly the rate at which humans agree with each other. Your judge is about as good as a second annotator, and that floor isn't moving.

**Self-preference.** Models identify their own outputs at better than chance, and the strength of that recognition tracks how much they favour those outputs. Randomising presentation order doesn't touch it.

**Multi-turn credit assignment.** A small unstated assumption at turn one that becomes a confidently wrong answer at turn four passes every step check you thought to write.

**Labels.** Every technique here consumes human judgment somewhere. Distilling the judge moves that cost around rather than removing it.

**Goodhart.** The moment the score is a target, it starts capturing effort that would otherwise have gone into the product. Keep a holdout nobody optimises against, and be suspicious of any number that only moves up.

## Further reading

- Zheng et al. (2023), **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.** [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) — the 80% agreement figure, plus position, verbosity and self-enhancement bias.
- Panickssery, Bowman & Feng (2024), **LLM Evaluators Recognize and Favor Their Own Generations.** [arXiv:2404.13076](https://arxiv.org/abs/2404.13076) — why grading a model with a close relative of itself is a conflict of interest.
- Ribeiro et al. (2020), **Beyond Accuracy: Behavioral Testing of NLP Models with CheckList.** [ACL 2020](https://aclanthology.org/2020.acl-main.442/) — named failure categories as tests. Their user study found practitioners wrote twice as many tests and caught nearly three times as many bugs.
- Zhou et al. (2023), **Instruction-Following Evaluation (IFEval).** [arXiv:2311.07911](https://arxiv.org/abs/2311.07911) — 25 verifiable instruction types over ~500 prompts; the reference point for how much a script can decide.
- Recht et al. (2019), **Do ImageNet Classifiers Generalize to ImageNet?** [arXiv:1902.10811](https://arxiv.org/abs/1902.10811) — usually cited as proof of test-set overfitting; the authors actually concluded the 11–14 point drop came from harder images, not adaptivity.
