---
layout: page
title: "Who Plays the User? The State of Full-Duplex Evaluation"
description: "A dozen benchmarks now measure full-duplex speech models and they disagree about almost everything. The reason is one shared unsolved problem: who plays the user."
date: 2026-08-15
image: /assets/blog/duplex-eval/fig2.png
---

*Last time, a friend asked me why voice AI still feels like a walkie-talkie. He came back with the harder question: okay — so how would you know if one got good?*

---

**The short version, if you're skimming**

- A dozen benchmarks now measure full-duplex models, and they disagree about almost everything.
- They disagree because of **one** thing: who plays the user. Recorded clip, TTS script, live LLM, real human. Everything else follows.
- The number that reframed the field for me: voice agents keep only **30–45%** of their text-mode ability on identical tasks [11].
- The best model follows explicit turn-taking instructions **64.4%** of the time [15]. None clears **50%** at revising its plan when you interrupt [10].
- What nobody measures yet: **grounding, proactivity, adversarial safety, other languages, error bars.**

The rest of this is me walking you through how the field got here, one question at a time.

---

## Part 1: The question that starts everything

You're on a support call. Halfway through, you realize you gave the wrong order number, so you cut in — "sorry, actually it's the one from Tuesday" — while they're still mid-sentence. They stop, absorb it, carry on from the corrected premise.

Now score that.

Not the final answer. The whole thing: whether stopping was right, how fast, whether they lost the thread, whether resuming forced them to throw away reasoning they'd already said out loud.

Here's the trouble. There's no single moment to grade, and no transcript that captures it — because the interesting part is what **both streams** were doing at the same time.

So before we can talk about benchmarks, we need names for the things we're trying to grade.

---

## Part 2: Four words you need

| Term | What it is | What should happen |
|---|---|---|
| **Pause** | You stop, but haven't finished your turn | System stays quiet |
| **Backchannel** | "Mm-hm," "right" — attention without claiming the floor | System keeps talking |
| **Barge-in** | You genuinely take the floor mid-response | System stops |
| **Floor-holding** | System talks through overlap it judged non-interrupting | Depends on the product |

Read those four rows again and notice something: **all four are the same physical event.** Sound arriving on the input channel while the model is speaking. The difference between them is entirely in what it *meant*.

That's why the obvious engineering answer doesn't work. A voice activity detector hears "mm-hm," "wait, no," a cough, and someone else's voice in the room as the same thing — energy. Telling them apart needs semantics, and semantics arrive too late if you wait for a complete utterance. So VAD systems pick a threshold, and **every threshold is simultaneously too eager for backchannels and too slow for real interruptions.**

It's also why the obvious *evaluation* answer doesn't work. Every one of those four events is defined by **timing, not content**. WER, MOS, an LLM judging a transcript — all of them score a finished turn. The transcript of a botched barge-in and a clean one can be the *same string*.

Which leaves us stuck. If you can't score the artifact, you have to score the interaction. And an interaction needs two participants.

---

## Part 3: The one idea worth taking away

> **The hard problem in full-duplex evaluation isn't scoring the model. It's building the other half of the conversation.**

This is the thing I'd stop and repeat if we were actually talking. Once you see it, every benchmark below stops being a random acronym and becomes a position on a single question: *what's on the other end of the line?*

| Who plays the user | You get | You give up |
|---|---|---|
| Pre-recorded clip | Exact reproducibility | Any reaction — the file delivers the correction at 12.4s whether the model's ready or not |
| LLM script + TTS | Multiple rounds, cheap scale | Real disfluency; turn boundaries you had to invent |
| Live LLM examiner | Corrections, entity tracking, adaptivity | Determinism — you can't tell a model regression from an examiner mood swing |
| Real human recordings | Authentic disfluency and acoustics | Adaptivity, again |
| Live humans | Everything real | Ground truth entirely |

Nobody gets all three of reproducible, realistic, and verifiable. Every benchmark in this post is a different answer to which two you keep.

![Five stacked bands forming a ladder, numbered one at the bottom to five at the top: pre-recorded audio file, scripted TTS pipeline, live LLM examiner, real human recordings, and live humans in an arena. Each band is tagged with the benchmarks that sit on it, from FDB v1 and v1.5 at the bottom to Voice Showdown at the top. An arrow down the left is labelled reproducibility; an arrow up the right is labelled realism.](/assets/blog/duplex-eval/fig1.png)

*Figure 1 — The ladder of user simulators.*

So let's climb it. Five groups of researchers, five different bets about which two to keep.

---

## Part 4: Five bets

| | The bet | What they found |
|---|---|---|
| **FDB v1 / v1.5** [3,4] | Behaviors are stimulus–response, so a fixed clip is enough | Two *opposite* strategies exist: **responsive** vs. **floor-holding** |
| **FD-Bench / MTR** [7,8] | Multi-round is the gap; slice the stream into turns | Performance degrades round over round |
| **FDB-v2 / EVA / τ-voice** [5,12,11] | The user has to be able to react | Voice keeps **30–45%** of text capability |
| **FDB-v3 / HumDial** [6,16] | Synthetic speech is the confound | Fastest system has the *worst* turn-take rate |
| **Voice Showdown** [17] | Only humans can judge naturalness | The multilingual gap is the #1 differentiator |

Five rows, five stories. Here they are in order.

### The bottom rung: a clip that can't react

FDB v1 and v1.5 do the cheapest sensible thing — build an audio clip that *should* provoke a backchannel, play it, watch what happens.

It worked, and it turned up the finding I keep coming back to. Across five agents, v1.5 found two coherent and *opposite* strategies. Some models are **responsive**: react fast to anything you say. Others are **floor-holding**: filter overlap to protect their own flow.

Neither is wrong. They're different products. But a single leaderboard number ranks one above the other and quietly hides that they're optimizing different things — which is the first sign that "who plays the user" isn't the only thing benchmarks are smuggling in.

**The catch:** the clip can't respond. If the model says something that would change what a real person says next, the file plays on regardless. Which is fine for one probe — and falls apart the moment you want a conversation.

### One rung up: inventing turn boundaries

So FD-Bench and MTR-DuplexBench synthesize whole conversations with an LLM, voice them with TTS, and score multiple rounds.

MTR-DuplexBench is the one that names why this is hard: **blurred turn boundaries** (there's no clean point where round three ends and round four begins) and **context inconsistency** (by round four the model is conditioning on its own earlier output, which may have drifted from the script the rest of the conversation assumes).

Its fix is explicit turn segmentation — and to its credit, it's honest that this is an *imposition*. You're drawing boundaries on a medium whose defining property is that boundaries overlap.

**The catch:** the user is still a script. Which raises the obvious next question — what if you let the user think?

### The big shift: the benchmark becomes a service

This is where the field is now, and it's a genuine architectural change. FDB-v2, EVA-Bench and τ-voice put a **live examiner** on a real-time audio channel. The benchmark stops being a dataset you download and becomes something you connect to.

Two design details are worth stealing:

**FDB-v2 runs two pacing setups, Fast and Slow.** The same task delivered by an impatient user and a patient one exposes different failures — the cheapest way I've seen to separate a model genuinely tracking state from one riding the script's rhythm.

**EVA-Bench validates the simulator.** It detects when the *user simulator itself* went off-script and regenerates the conversation before scoring. Sit with that: once your benchmark contains a live LLM, **the benchmark can fail**, and you need a mechanism to notice.

And then τ-voice made the sharpest move in the literature. It took τ²-bench — an established *text* agent benchmark — and ported it to voice with the tasks, tools, policies and evaluator kept byte-identical. Same task, same scoring, both modalities. Then it decoupled the simulator from the clock:

```python
# Audio playback is real-time. Deciding what to say next is not.
session.pause_clock()
reply = frontier_llm.decide(heard)   # can take 4s of thinking
session.resume_clock()               # the model never sees the gap
```

That's what makes the comparison clean, and the comparison is brutal. On 278 grounded customer-service tasks, GPT-5 with reasoning scores **85% in text**. The same tasks over live full-duplex audio: **31–51% clean, 26–38% with noise and accents.**

Before τ-voice you couldn't tell whether a voice agent failed because the task was hard or because the channel was audio. Now you can. It's the channel.

**The catch:** none of this is deterministic. Two runs produce two different conversations, and every number downstream inherits that variance.

![A horizontal bar chart on a zero to one hundred percent axis. A single long bar shows GPT-5 reasoning scoring 85 percent in text. Below it, two much shorter floating range bands show voice agents scoring 31 to 51 percent on clean audio and 26 to 38 percent on realistic audio. A brace on the right spans the gap, annotated 30 to 45 percent of text capability retained.](/assets/blog/duplex-eval/fig2.png)

*Figure 2 — Voice keeps a third of text capability.*

### Back down the ladder, on purpose: real human audio

Here's the move I didn't expect. Having climbed to live simulators, FDB-v3 and HumDial climb back *down* — to fixed, non-reactive audio — because they think the synthetic voice was the thing corrupting the measurements all along.

FDB-v3 builds its whole dataset from real human speech annotated for fillers, pauses, hesitations, false starts and self-corrections. And the results are a good advertisement for reporting honestly, because **nobody wins on more than one axis:**

| System | Pass@1 | Latency | Turn-take rate |
|---|---|---|---|
| GPT-Realtime | **0.600** | — | — |
| Gemini Live 3.1 | — | **4.25s** (fastest) | **78.0%** (lowest) |
| Cascaded (Whisper→GPT-4o→TTS) | — | 10.12s (slowest) | **100%** |

Look at row two, then row three. The fastest system is the one most likely to fail to take its turn at all — **"low latency" and "responds when it should" are different axes**, and a model can buy one with the other. And the cascaded pipeline, the architecture everyone calls obsolete, wins the turn-take column outright.

**The catch:** real audio is *fixed* audio. Authentic disfluency, no adaptivity — you're back on rung one, just with better material.

### The top rung: ask a lot of people

Voice Showdown is the first real preference arena for voice: 60+ languages, 11 frontier models, blind pairwise comparisons inside conversations users were already having. The de-confounding is careful — voices swapped so you can't recognize your usual model, gender-matched, both streaming simultaneously so speed doesn't proxy for quality.

It surfaces things no automated benchmark structurally can. Multilingual ability is the single biggest differentiator between models. GPT Realtime answers **in English to non-English prompts about 20% of the time** — including Hindi, Spanish and Turkish. And within one model, the best voice wins **30 percentage points** more often than the worst, which means voice selection reorders rankings.

**And here's the punchline: Voice Showdown is turn-based.** Scale says full-duplex preference evaluation is coming next. As of today, the paradigm that anchors all of text evaluation has **no full-duplex equivalent in production.**

### The one that doesn't fit the ladder

Before we leave the benchmarks — one paper changes the *scoring function* rather than the user, so it sits off to the side.

**Talking Turns** [13] trains a supervised model to predict turn-taking events from human–human conversation, then uses it as a judge: does this system behave the way the predictor expects a human would? Its user-study findings are still the most quotable in the field — systems **don't know when to speak up, interrupt too aggressively, and rarely backchannel.**

**The catch:** a judge trained on one corpus inherits that corpus's norms. Switchboard telephone calls are not customer-service calls are not tutoring sessions.

So: you've picked a rung, you've picked a judge. Now you need numbers — and this is where it gets slippery.

---

## Part 5: The metrics, and where they lie to you

| Metric | Measures | The catch |
|---|---|---|
| **TOR** (Takeover Rate) | How often the model grabs the floor in a window | Value depends entirely on the window, and papers define it differently |
| **Latency** | See below | At least five different quantities share this name |
| **JSD vs. human timing** | Aggregate rhythm | Says nothing about any individual moment |
| **Backchannel frequency** | How often, not how well | Constant random backchanneling scores well |
| **LLM-as-judge** | Post-interruption quality | Softest number in every paper that uses it |
| **pass@k vs. pass^k** | Peak vs. *reliable* capability | EVA-Bench median gap: **0.44** |

If you only take one row from that table, take the last one. pass@k is best-of-k; pass^k demands *all* k succeed. A median gap of 0.44 means peak and dependable capability are wildly different quantities — and a system that works on the third try isn't one you put on a support line.

The row that's caused me the most confusion, though, is latency. Five distinct things travel under that name:

1. End of user speech → model starts (**response latency**)
2. Interruption → model *stops* (**stop latency**)
3. Interruption → model's *new* response
4. **Time to first audio byte** — excludes decision-making entirely
5. **End-to-end task time** — what FDB-v3's 4.25s and 10.12s measure

These differ by an order of magnitude. A vendor quoting 200ms on #4 and a benchmark reporting 4.25s on #5 aren't contradicting each other. They're measuring different things, and any cross-paper latency comparison that doesn't pin down which is worthless.

![A two-track timeline, user above and model below. The user speaks, stops, and later interrupts; the model responds, visibly stops mid-response, stays silent, then begins a new response. Five measurement spans are drawn beneath with different start and end points: response latency, time to first audio which is a tiny sliver, stop latency, latency after interruption, and end-to-end task time spanning the whole timeline. A note explains that a vendor quoting 200 milliseconds means the fourth span while a benchmark reporting 4.25 seconds means the fifth.](/assets/blog/duplex-eval/fig3.png)

*Figure 3 — Five things called latency.*

That's the state of what *is* measured. The more interesting list is what isn't.

---

## Part 6: What nobody is measuring

Step out of audio for a second. Text, vision and omni-modal evaluation have spent years building axes that full-duplex simply doesn't have yet.

| Axis | Mature elsewhere | Status in full-duplex |
|---|---|---|
| **Grounding / hallucination** | A decade of attribution work; WorldSense [20] — 3,172 QA over 1,662 synced videos | One sub-component of EVA-Bench's EVA-A. Otherwise nothing |
| **Proactivity** | OmniMMI [19] — proactive alerting, model decides *when* to speak | FLEXI's emergency interruption [9] is the only case. Everything else is reactive |
| **Instruction following** | Deterministic verifiable checkers | Instruct-FD [15] at **64.4%** best adherence; Game-Time [14] on tempo. Both LLM-judged, not verified |
| **Adversarial safety** | AJailBench [21] — 1,495 prompts, 10 categories; JALMBench [22] | Both **half-duplex**. Nobody tests attacks arriving *mid-generation* |
| **Multilingual** | Standard | Mostly English-only — yet Scale finds English fallback **~20%** of the time on well-supported languages [17] |
| **Long-horizon state** | Long-context benchmarks | EchoChain [10]: single interruption, **no model >50%** pass |
| **Error bars** | Increasingly standard | Duplex is *more* stochastic and reports confidence intervals *less* |
| **Cost** | Standard | Nobody reports it |

Three of those rows deserve more than a cell.

**Grounding is the biggest hole.** There's no duplex version of "did this answer follow from the document I gave you" — which is exactly the setting for tutoring, support, and every retrieval-grounded voice product being built right now. And the genuinely duplex-native version of the question isn't even being asked: *does grounding degrade when you interrupt the model mid-citation?*

**EchoChain has the most elegant control design.** It names three failure modes — contextual inertia, interruption amnesia, objective displacement — then runs the *same* tasks half-duplex. Total failures drop **40.2%**. That single comparison localizes the difficulty to state revision under interruption, rather than to the tasks being hard. More benchmarks should do this.

**Multilingual is the one that should embarrass us.** It's the failure mode with the largest measured real-world impact, and the thinnest automated coverage of anything on the list.

![A triangle with vertices labelled reproducible, realistic, and verifiable. Each edge is labelled with the benchmark family that achieves the two properties it connects: static scripted probes, live LLM examiners, and human preference arenas. Text in the centre reads that no published benchmark occupies the middle.](/assets/blog/duplex-eval/fig4.png)

*Figure 4 — Reproducible, realistic, verifiable: pick two.*

---

## Where that leaves us

If my friend only remembers one thing, I'd want it to be this.

The field has agreed on *what* to measure — pause handling, backchanneling, barge-in, floor-holding — and roughly on the vocabulary for measuring it. It has not agreed on **who plays the user**, and that one unresolved choice explains every disagreement downstream. Static probes are perfectly reproducible and can't react. Script-and-TTS pipelines buy multiple rounds by inventing turn boundaries. Live examiners buy adaptivity by giving up determinism — and hand us the field's most useful number along the way. Real-human audio buys authenticity by giving adaptivity back. Human arenas buy everything except ground truth.

Nobody has all three. And the gaps that remain — grounding, proactivity, adversarial safety, other languages, honest error bars — are most interesting in their duplex-native forms. Not *"does it hallucinate,"* but *"does it start hallucinating after you cut it off."*

That's the benchmark I want to see next.

---

## References

**Foundations** — [1] Nguyen, T. A., et al. (2023). **Generative Spoken Dialogue Language Modeling (dGSLM).** *TACL* 11:250–266. arXiv:2203.16502. · [2] Chen, Y., & Yu, H. (2025). **From Turn-Taking to Synchronous Dialogue: A Survey of Full-Duplex Spoken Language Models.** arXiv:2509.14515.

**Full-Duplex-Bench line** — [3] Lin, G.-T., Lian, J., Li, T., Wang, Q., Anumanchipalli, G., Liu, A. H., & Lee, H.-y. (2025). **Full-Duplex-Bench.** arXiv:2503.04721. · [4] Lin, G.-T., Kuan, S.-Y. S., Wang, Q., Lian, J., Li, T., Watanabe, S., & Lee, H.-y. (2025). **Full-Duplex-Bench v1.5: Evaluating Overlap Handling.** ICASSP 2026. arXiv:2507.23159. · [5] Lin, G.-T., Kuan, S.-Y. S., Shi, J., Chang, K.-W., Arora, S., Watanabe, S., & Lee, H.-y. (2026). **Full-Duplex-Bench-v2.** ACL 2026. arXiv:2510.07838. · [6] Lin, G.-T., Chen, C., Chen, Z., & Lee, H.-y. (2026). **Full-Duplex-Bench-v3.** arXiv:2604.04847. · Code: https://github.com/DanielLin94144/Full-Duplex-Bench

**Multi-round and scenario** — [7] Peng, Y., Chao, Y.-W., Ng, D., Ma, Y., Ni, C., Ma, B., & Chng, E. S. (2025). **FD-Bench.** arXiv:2507.19040. · [8] Zhang, H., Cui, W., Xu, H., Li, X.-H., Zhu, L., Bai, H., Ma, S., & King, I. (2026). **MTR-DuplexBench.** *Findings of ACL 2026*, 5334–5351. arXiv:2511.10262. · [9] Ge, Y., Chen, S., Xiao, J., Liu, X., Xiao, T., et al. (2025). **FLEXI.** arXiv:2509.22243. · [10] Modi, S. N., Mahajan, G., Wetter, M., & Welles, R. (2026). **EchoChain.** arXiv:2604.16456.

**Task-grounded** — [11] Ray, S., Dhandhania, K., Barres, V., & Narasimhan, K. (2026). **τ-Voice.** arXiv:2603.13686. · Code: https://github.com/sierra-research/tau2-bench · [12] Bogavelli, T., Gauthier Melançon, G., Stankiewicz, K., Bamgbose, O., Riols, F., et al. (2026). **EVA-Bench.** ServiceNow. arXiv:2605.13841.

**Timing, instructions, judges** — [13] Arora, S., Lu, Z., Chiu, C.-C., Pang, R., & Watanabe, S. (2025). **Talking Turns.** ICLR 2025. arXiv:2503.01174. · [14] Chang, K.-W., Hu, E.-P., Kuan, C.-Y., Ren, W., Chen, W.-C., Lin, G.-T., Tsao, Y., Sun, S.-H., Lee, H.-y., & Glass, J. (2026). **Game-Time.** ICASSP 2026, 16302–16306. arXiv:2509.26388. · [15] Tang, Y., Ma, W., Zhao, X., et al. (2026). **Instruct-FD.** Boson AI. arXiv:2607.20460.

**Human data and preference** — [16] Wang et al. (2026). **Full-Duplex Interaction in Spoken Dialogue Systems: ICASSP 2026 HumDial Challenge.** arXiv:2604.21406. · Data: https://github.com/ASLP-lab/HumDial-FDBench *[full author list unverified — complete before publishing]* · [17] Gu, J., Gosai, A., & Siegel, M. (2026). **Voice Showdown: The First Arena for Voice AI.** Scale AI. https://scale.com/blog/voice-showdown · [18] Jiang, F., Lin, Z., Bu, F., Du, Y., Wang, B., & Li, H. (2025). **S2S-Arena.** arXiv:2503.05085.

**Outside audio** — [19] Wang, Y., Wang, Y., Chen, B., Wu, T., Zhao, D., & Zheng, Z. (2025). **OmniMMI.** CVPR 2025. arXiv:2503.22952. · [20] Hong, J., Yan, S., Cai, J., Jiang, X., Hu, Y., & Xie, W. (2025). **WorldSense.** arXiv:2502.04326. · [21] Song, Z., Jiang, Q., Cui, M., Li, M., Gao, L., et al. (2026). **AJailBench.** ACL 2026, 27294–27308. arXiv:2505.15406. · [22] Peng, Z., Liu, Y., Sun, Z., Li, M., Luo, Z., et al. (2026). **JALMBench.** ICLR 2026. arXiv:2505.17568.

---

*Code snippet above is illustrative pseudocode, not a runnable implementation.*
