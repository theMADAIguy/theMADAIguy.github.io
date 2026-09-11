---
layout: post
title: "TurnBench: Grading Turn-Taking Like a Linguist, Not a Timer"
description: "A short, figure-led breakdown of TurnBench: a multi-domain, conversation-analysis-grounded benchmark for end-of-turn and interruption detection in spoken dialogue."
date: 2026-09-11
image: /assets/blog/turnbench-turn-taking/fig3.png
---

*Ask most voice agents to hold a real conversation and the cracks show fast: they cut you off during a backchannel, or they wait a beat too long after you've clearly finished.*

---

**The short version, if you're skimming**

- Every listener sound (backchannel, interruption, or a clean turn handoff) looks alike in the waveform. Most benchmarks score them separately anyway, with their own private definitions.
- TurnBench scores all of it against **one** shared, conversation-analysis-grounded annotation: 30 hours, 154 dialogues, triple-annotated at Fleiss κ = 0.78, across **six** distinct conversation styles.
- Humans start speaking a median **151ms before** the current turn ends. The best system (VAP) trails by **368ms** on end-of-turn and **994ms** on interruptions.
- One pattern holds for all 14 systems tested: interruption false positives are worse in casual, backchannel-dense conversation than in argumentative conversation.

---

## The core idea

At the exact moment a listener starts talking, that sound can mean three different things: a backchannel ("mm-hmm," keep going), an interruption (taking the floor), or the listener simply starting their own turn because the speaker finished. All three look alike in the waveform, so scoring them in isolation hides exactly the confusion a real system has to resolve. TurnBench builds one annotation, grounded in conversation-analysis theory going back to Sacks, Schegloff, and Jefferson's 1974 turn-taking paper, and scores every system against it.

![Six stat cards summarizing TurnBench's scale: 30 hours of hand-labeled dialogue audio, 154 dyadic conversations, 106 voice actors in 53 pairs, 6 conversation types, 14 turn-taking systems benchmarked, and a Fleiss kappa of 0.78 for the triple annotation.](/assets/blog/turnbench-turn-taking/fig1.png)

*Figure 1 — TurnBench at a glance.*

## Six conversations, six personalities

The corpus isn't one register. Recording sessions were assigned one of six interaction styles, casual chat, task-oriented exchange, instructional teaching, collaborative problem-solving, argumentative debate, and one-sided narrative, and the styles behave nothing alike.

![Bar chart of average turn length in seconds for six conversation types (Casual 8.1, Task-Oriented 8.9, Collaborative 7.9, Argumentative 10.0, Narrative 10.2, Instructional 10.5), with orange dots overlaid showing interruptions per minute for each type (2.05, 1.75, 2.64, 2.48, 1.50, 1.52 respectively).](/assets/blog/turnbench-turn-taking/fig2.png)

*Figure 2 — Turn length and interruption rate vary a lot by conversation type.*

Collaborative sessions have the shortest turns and the most overlap; instructional ones have the longest turns and the least floor competition. A system tuned on one type can look great and still fail on another, which is why the benchmark reports results per type instead of pooling everything into one number.

## How it's scored

Two tracks, one shared protocol:

- **End-of-turn (EOT)**: does the floor actually pass to the other speaker, or is this just a mid-sentence pause?
- **Interruption (INT)**: does the listener's speech take the floor away mid-turn, or is it a backchannel that leaves the speaker still holding it?

**The metrics**

- **Recall** = TP / (TP + FN). Of all the real events in the gold annotation, how many did the system catch?
- **False positive rate (FPR)** = FP / (FP + TN). Of all the negative spans (mid-turn pauses for EOT; backchannels and noise for INT), how many did the system wrongly fire on?
- **Latency** = signed time gap between a system's prediction and the gold event (negative means the system committed early), reported at the 10th, 50th, and 90th percentiles.

**The matching rule**

- For each gold positive at time *t*, the scorer looks for a submitted event in the window [*t* − 0.25s, *t* + 3.0s]. The earliest unclaimed prediction in that window counts as a true positive.
- Inside a negative span, firing counts as at most one false positive; staying silent counts as one true negative.
- Predictions inside excluded intervals (ambiguous cases like non-floor-taking interruption attempts) are ignored entirely, neither rewarded nor penalized.
- The public leaderboard ranks systems by test-set recall, but only among those that stay under a 0.15 FPR ceiling. Go over that, and you rank below every system that qualified, no matter how high your recall is.

## The headline result: humans lead, models lag

Averaged across the corpus, human listeners begin speaking a median 151ms *before* the current turn ends when the handoff is smooth. No evaluated system gets close to that without also firing constantly on backchannels.

![Horizontal bar chart comparing timing relative to the true turn boundary: human listeners begin speaking 151 milliseconds before the boundary, VAP commits an end-of-turn 368 milliseconds after, VAP commits an interruption 994 milliseconds after, and Gemini 3.1 Live commits an end-of-turn 1234 milliseconds after.](/assets/blog/turnbench-turn-taking/fig3.png)

*Figure 3 — The best system is still roughly half a second behind a human listener, and a full second behind on interruptions.*

## Nobody wins on both axes

Across 14 systems, from a bare energy-threshold VAD to Gemini 3.1 Live and Moshi running zero-shot, the same tradeoff shows up everywhere: fire early and you catch more real turn-ends, but you also fire on every backchannel and pause.

![Scatter plot of end-of-turn recall versus false positive rate for seven representative systems. RMS VAD and OpenAI Server VAD sit at high recall (0.72-0.96) but very high false positive rates (0.53-0.63). OpenAI Semantic VAD, Gemini 3.1 Live, and Moshi sit at low false positive rates (0.02-0.04) but low recall (0.23-0.66). VAP and Kyutai SVAD land closest to the upper left, with recall around 0.77-0.85 and false positive rates around 0.06, just inside the 0.1 dev budget line.](/assets/blog/turnbench-turn-taking/fig4.png)

*Figure 4 — VAP, which predicts continuous floor-holding probability rather than reacting to silence, comes closest to the upper-left corner.*

## How the baselines perform

- **RMS VAD** (energy threshold, no linguistic information): the floor of the benchmark. Anticipatory latency (−117ms) but a 0.51–0.70 EOT FPR and 0.38–0.55 INT FPR, it fires on nearly every silence and every onset.
- **OpenAI Realtime, Server VAD**: commits on silence duration alone. Saturates EOT recall (0.94–0.96) at almost the same FPR as RMS VAD (~0.53), acoustic-only endpointing is structurally over-eager.
- **OpenAI Realtime, Semantic VAD**: the same API's linguistically-aware mode. Recall drops hard (0.30 EOT, 0.48 INT) but so does FPR (0.02 EOT, 0.27 INT), it waits for linguistic completion and pays for it in the highest EOT latency of any non-full-duplex system (793ms).
- **Kyutai SVAD** (streaming ASR + semantic EOT head): the best balance among the linguistically-informed systems, 0.77 EOT recall / 0.06 FPR and 0.90 INT recall / 0.08 FPR.
- **SmartTurn v3**: solid EOT (0.75 recall / 0.05 FPR) but its interruption recall collapses to 0.11, committing fast (159ms) costs it almost every real interruption.
- **VAP** (voice activity projection): the strongest in-budget system on both tracks, 0.845 EOT recall / 0.055 FPR at 368ms, 0.945 INT recall / 0.107 FPR at 994ms.
- **Mimi-EP** (codec-token endpointer): a similar profile to VAP but a step slower and noisier, 0.78 EOT / 0.078 FPR, 0.90 INT / 0.106 FPR.
- **WavLM-Large, causal vs. anchor**: causal (left-context-only) recall is poor (0.40 EOT); giving the same model a bidirectional 4s window (anchor) lifts it to 0.80 EOT / 0.87 INT, but at the highest latency of any in-budget system (1076–1412ms).
- **Gemini 3.1 Live**: the most conservative system tested, lowest FPR (0.022) but only 0.657 EOT recall and the highest latency overall (1234ms).
- **Moshi**: stays inside the FPR budget (0.044) but recall falls to 0.233 and keeps falling as sessions get longer, it progressively goes silent.

One pattern holds across every single system: interruption false positives are higher in casual conversation than in argumentative conversation, because casual talk is dense with backchannels that sound like interruptions at onset.

## What's still hard

- **Interruptions are ambiguous at the moment they start.** A floor-taking interruption and a backchannel look identical for the first ~100-200ms; systems that commit fast pay for it in false positives.
- **No system handles all six conversation types equally well.** The per-type breakdown, not just the pooled average, is where real gaps show up.
- **Full-duplex models (Gemini 3.1 Live, Moshi) weren't built with explicit turn-taking labels**, and it shows: both sit far from the human anticipation number, and Moshi's recall degrades further as sessions get longer.

---

## References

**The paper**

**[1]** Jiang, F., Sanabria, R., Deshmukh, S., Veluri, B., Vuch Williams, S. M., Suen, E. K., Lee, G., Choi, K. Y., Umeki, T., Kubo, R., Udupa, S., Huang, C., Kuan, S.-Y. S., Tao, Z., Krishna, S., Eskimez, S. E., Tsao, Y., Lee, H., & Watanabe, S. (2026). [TurnBench: A Multi-Domain Benchmark for Turn-Taking Dynamics in Spoken Dialogue](https://arxiv.org/abs/2608.25218). arXiv:2608.25218.

**Foundations**

**[2]** Sacks, H., Schegloff, E. A., & Jefferson, G. (1974). A Simplest Systematics for the Organization of Turn-Taking for Conversation. *Language*, 50(4), 696-735.

**Systems TurnBench evaluates**

**[3]** Ekstedt, E., & Skantze, G. (2022). [Voice Activity Projection: Self-Supervised Learning of Turn-Taking Events](https://aclanthology.org/2022.interspeech-1). *Proceedings of Interspeech 2022*, 5190-5194. — the strongest in-budget system in TurnBench's own results.

**[4]** Arora, S., Lu, Z., Chiu, C.-C., Pang, R., & Watanabe, S. (2025). [Talking Turns: Benchmarking Audio Foundation Models on Turn-Taking Dynamics](https://arxiv.org/abs/2503.01174). ICLR 2025. arXiv:2503.01174.

**[5]** Défossez, A., Mazaré, L., Orsini, M., Royer, A., Pérez, P., Jégou, H., Grave, E., & Zeghidour, N. (2024). [Moshi: A Speech-Text Foundation Model for Real-Time Dialogue](https://arxiv.org/abs/2410.00037). arXiv:2410.00037.

---

*Figures 1-4 are original diagrams built from the numbers reported in the paper, not reproductions of the paper's own figures.*
