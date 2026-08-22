---
layout: page
title: "Where Does a System Prompt Go? Taking PersonaPlex Apart"
description: "NVIDIA's PersonaPlex adds role conditioning and zero-shot voice cloning to a full-duplex speech model without changing the architecture. A two-person walkthrough of the Hybrid System Prompt, the synthetic data pipeline, the six-hour fine-tune, and Service-Duplex-Bench."
date: 2026-08-22
image: /assets/blog/personaplex/fig1.png
---

*Ana builds speech language models. Ravi ships voice agents on cascaded stacks. Last time they [took Moshi apart](/blog/2026/08/20/moshi-architecture/). This time Ravi has a question about the thing Moshi can't do.*

---

**Ravi:** I tried the obvious thing after our last conversation. Took Moshi, wrote "you are a support agent for National Health Coverage," and went looking for somewhere to put it. There's no system field. There's no prompt at all.

**Ana:** There isn't. Moshi has one voice and one personality, baked in at training time, and that's most of what keeps it out of production. PersonaPlex is NVIDIA's answer, and the answer is stranger than adding a field.

**Ravi:** Stranger how?

**Ana:** They don't add an input. They fabricate a stretch of past. Remember the shape of Moshi's input — three parallel streams, one frame every 80 milliseconds, forever. User audio, agent text, agent audio. There is no "before the conversation" in that picture. So PersonaPlex writes the persona *into those same three streams* and prepends it to the real conversation. As far as the model is concerned, the system prompt already happened.

**Ravi:** So it's a prefix.

**Ana:** It's a prefix. That's the whole idea, and everything odd about the design falls out of it.

---

## The two segments

**Ravi:** What's actually in the prefix?

**Ana:** Two segments, temporally concatenated, which they call the *Hybrid System Prompt*. The **text prompt segment** does role conditioning: they force the scenario text tokens onto the agent's *text* channel — "You are an agent named Brody Murphy working for National Health Coverage…" — and hold the agent's audio channel silent. The **voice prompt segment** does voice cloning: a short speech sample goes on the agent's *audio* channel while the text channel gets padding.

**Ravi:** Wait. The voice sample goes on the *agent* audio channel? Not the user's?

**Ana:** That's the trick. If you put the sample on the user channel, you've shown the model a person it should reply to. Put it on the agent channel and you've shown the model *itself* speaking in that voice. Autoregression does the rest — every subsequent agent frame continues the voice it just apparently produced. Zero-shot cloning with no speaker encoder, no adapter, no architectural change at all.

![A three-row channel diagram over a shared time axis. The user audio row holds a 440 Hz sine wave through both prompt segments, then switches to live user microphone. The agent text row holds padding tokens during the voice prompt segment, then the system role description, then dialogue tokens. The agent audio row holds a speaker sample during the voice prompt, silence during the text prompt, then generated audio. A dashed delimiter separates the Hybrid System Prompt, which is masked out of the training loss, from generation.](/assets/blog/personaplex/fig1.png)

*Figure 1 — The Hybrid System Prompt is not a new input — it is a prefix written into the three streams the model already has.*

**Ravi:** And the user channel during all this?

**Ana:** A 440 Hz sine wave. Not silence — a constant tone, for the duration of the prefix.

**Ravi:** Why not silence? Silence seems like the honest answer for "nobody is talking yet."

**Ana:** Because here silence isn't the absence of input, it's a token — and it's the most heavily trained token in the user channel. The model has spent its whole life learning that user silence means *your turn, keep going.* A silent prefix would be thousands of frames of the strongest cue it has for "start speaking now." A tone is unambiguous: nothing in training sounds like that, so no conversational policy fires. The paper says the sine wave is there for stable conditioning without spelling out why; that reading is mine, but the failure mode it avoids is real. Custom text and audio delimiters mark where the prefix ends.

---

## Order, prefilling, and the loss mask

**Ravi:** Does the order of the two segments matter?

**Ana:** They report no difference either way. But they put the voice segment *first*, and that's a latency decision, not a quality one. If you're not cloning on this call — default voice — the voice segment is fixed, so it can be prefilled once and reused. Only the text segment varies per conversation. Put the variable part last and you pay for it once.

**Ravi:** Nice. And during training, what stops the model from just learning to generate persona descriptions?

**Ana:** They mask loss backpropagation over the system prompt entirely. The prefix is context, never a prediction target. Beyond that they inherit Moshi's rebalancing: loss on non-semantic audio tokens is downweighted by 0.02, and on padded text tokens by 0.3. Both numbers exist because the frame is mostly padding and mostly acoustic detail, and without the reweighting the model optimizes the boring parts.

---

## Where 2,250 hours of conversation came from

**Ravi:** So they need paired data — a persona description, plus a two-speaker conversation that obeys it. Nobody has that lying around.

**Ana:** Nobody does, so they generated all of it. Transcripts come from Qwen-3-32B and GPT-OSS-120B, built hierarchically: sample a service domain (restaurant, bank), then a scenario inside it (refund, information request, general enquiry), write a high-level description, expand it into a two-speaker transcript, and generate the matching role context. A smaller second pile covers two-turn question answering under one fixed teacher persona.

**Ravi:** And the audio?

**Ana:** This is the part I'd steal. Two different TTS paths, chosen for different reasons. For the service dialogs they used Dia, which takes *two* speaker samples and generates the continuation for both speakers jointly — so overlaps, interruption timing, and room tone come out of the model instead of being assembled. For the QA dialogs they used Chatterbox per turn, which is single-speaker, so the turns have to be stitched. And when you're stitching you control the gap: positive silence padding gives you natural turn-taking, and *negative* padding — overlapping the turns — manufactures barge-in.

**Ravi:** You can synthesize interruptions by subtracting silence.

**Ana:** For the training distribution, yes. Voice prompts come from 26,296 single-speaker samples across VoxCeleb, Libriheavy, LibriTTS, CommonAccent and Fisher, with 2,630 held out for speaker-similarity measurement. The final corpus is 1,840 hours of customer service across 105,410 dialogs, plus 410 hours of QA across 39,322.

---

## Six hours on eight A100s

**Ravi:** How long does the training run take?

**Ana:** Six hours on 8×A100.

**Ravi:** That can't be right.

**Ana:** It's a fine-tune. Initialize from Moshi's weights, then run 24,576 steps at batch size 32, Adam with cosine annealing, learning rate 4e-6 for the depth transformer and 2e-6 for the temporal transformer. Sequence length caps at 2,048 tokens, which they note is 163.84 seconds — divide it out and you recover the 12.5 Hz frame rate.

**Ravi:** Two different learning rates.

**Ana:** The depth transformer runs within a frame, over the codebook hierarchy; the temporal transformer runs across frames. They get twice the rate on the depth side, which is where the voice identity lives. The headline is that adding persona and voice control to a duplex model costs an afternoon of compute, because the architecture didn't change. That's the argument the paper is really making.

---

## Testing whether a persona sticks

**Ravi:** How do you even measure "stayed in character"?

**Ana:** Full-Duplex-Bench measures turn-taking, and only turn-taking, with one assistant role. So they extended it: **Service-Duplex-Bench**, 50 service scenarios with 7 single-turn probes each — 350 new questions on top of the benchmark's 400. Each probe attacks a different failure mode. Q0 asks the agent to name its own employer, testing proper-noun recall. Q1 and Q2 ask about details that only exist in the role context. Q3 requests something the context forbids. Q4 is a rude customer. Q5 is in-domain but unspecified, Q6 is off-topic entirely. Training scenarios are disjoint from evaluation ones.

![A role context card listing an agent name, employer, customer SSN, three insurance plans and an enrollment time, connected by a bus bar to seven probe rows labelled Q0 through Q6: proper noun, context detail, context detail, unfulfillable request, rudeness, unspecified, and unrelated. A footer notes seven probes per scenario across fifty scenarios giving 350 questions, added to Full-Duplex-Bench's existing 400.](/assets/blog/personaplex/fig2.png)

*Figure 2 — Service-Duplex-Bench: one role context, seven ways to break character.*

**Ravi:** Q1 is the interesting one.

**Ana:** It is. In their published example, the role context lists the customer's SSN as 076-65-0542 and Q1 asks the agent to confirm 076-75-0542. The digits don't match. Whether that's a deliberate trap or a typo in the table isn't stated — but a probe that asks a model to confirm something slightly wrong is exactly the right shape of test, because agreeing is the fluent, agreeable, incorrect thing to do.

---

## What the numbers say, including the parts that don't flatter

**Ana:** Speaker similarity is the cleanest win: 0.57 by WavLM-TDNN cosine similarity, against 0.10 for Moshi, 0.07 for Qwen-2.5-Omni, 0.05 for Freeze-Omni, and 0.00 for Gemini — those systems have fixed voices, so there's nothing to be similar to. Naturalness, rated by 202 human evaluators on Service-Duplex-Bench, comes out at 3.59 against Gemini's 3.22 and Moshi's 2.83.

**Ravi:** And role adherence?

**Ana:** Gemini wins. 4.73 mean to PersonaPlex's 4.48 on the GPT-4o-judged Service-Duplex-Bench. PersonaPlex beats every open system by a distance — Freeze-Omni 4.02, Qwen-2.5-Omni 2.76, Moshi 1.75 — and loses to the commercial one. The paper says so plainly, which I appreciate.

**Ravi:** Anything else it loses?

**Ana:** Pause handling, badly. Take-over rate during a mid-sentence pause is 0.584 on synthetic pauses and 0.662 on real Candor conversations, where Gemini sits at 0.255 and 0.310. Lower is better — PersonaPlex barges in on a thinking pause roughly twice as often as Gemini does. It's also worth reading the response-latency numbers next to the quality ones: Moshi answers interruptions in 0.257 seconds and scores 0.765 for what it says; PersonaPlex takes 0.400 and scores 4.210. Latency alone would rank them backwards.

**Ravi:** How much of this is just data volume?

**Ana:** Their ablation splits it cleanly, and it's my favourite table in the paper. Cut the corpus to 25% and speaker similarity barely moves — 0.54 against 0.57. Role adherence drops from 4.48 to 4.20 and keeps climbing with every addition. Voice is a cheap thing to teach; staying in character is not.

![Two line charts against fraction of the synthetic training corpus, at 0, 25, 50 and 100 percent. Speaker similarity rises from 0.10 to 0.54 by 25 percent and then flattens at 0.56 and 0.57. Role adherence on Service-Duplex-Bench rises from 1.75 to 4.20 at 25 percent, then 4.24 and 4.48, still climbing at full data.](/assets/blog/personaplex/fig3.png)

*Figure 3 — Voice is cheap to teach. Staying in character is not.*

---

## What's still hard

**Ravi:** Where does this leave someone who wants to deploy it?

**Ana:** Three open problems. First, everything the model knows about being a service agent it learned from LLM-written transcripts spoken by TTS, and synthetic conversation has a smoothness real conversation doesn't. The released checkpoint quietly concedes this: 1,217 hours of real Fisher telephone conversations, annotated at three levels of prompt detail from "You enjoy having a good conversation" up to a paragraph of biography. It also drops the mixed TTS setup for Chatterbox throughout, which lifts speaker similarity to 0.65 and pulls synthetic-pause take-over from 0.584 to 0.358.

**Ravi:** So the released model isn't the paper's model.

**Ana:** It isn't, and its naturalness study used a different annotator pool, so those scores don't compare across tables. Read Table 6 and Table 7, not Table 1, if you're deciding whether to deploy. Second problem: the probes are single-turn. A persona that holds for one question is a much weaker claim than one that holds for twenty, and the failure mode people actually hit — drifting out of character over a long call — isn't measured here at all. Third: role adherence still trails a closed commercial system, and nothing in the paper suggests where the remaining gap lives.

---

## The one-paragraph version

Moshi has no place to put a system prompt, because a duplex speech model has no notion of anything happening before the conversation. PersonaPlex adds one without touching the architecture: it writes a persona description onto the agent's text channel and a voice sample onto the agent's audio channel, holds the user channel at a 440 Hz tone so the model doesn't mistake the prefix for its cue to speak, and prepends the whole thing as fabricated history. Training is a six-hour fine-tune from Moshi weights on 2,250 hours of LLM-scripted, TTS-voiced dialog, with the prefix masked out of the loss. It buys real zero-shot voice cloning (0.57 speaker similarity, 0.65 in the released checkpoint) and the best role adherence of any open duplex model, while still losing to Gemini Live on staying in character and losing badly on when to stay quiet.

---

*Code blocks and figures in this post are illustrative rather than implementation-accurate. All numbers are from the paper and the released model card; where the two disagree, both are given.*

## References

1. Roy, R., Raiman, J., Lee, S., Ene, T.-D., Kirby, R., Kim, S., Kim, J., & Catanzaro, B. (2026). **PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models.** arXiv:2602.06053. https://arxiv.org/abs/2602.06053 — preprint under review at ICASSP 2026.
2. NVIDIA. **personaplex-7b-v1 model card.** https://huggingface.co/nvidia/personaplex-7b-v1 — the released checkpoint, which differs from the paper's experimental setup.
3. NVIDIA. **PersonaPlex code.** https://github.com/NVIDIA/personaplex
4. Défossez, A., Mazaré, L., Orsini, M., Royer, A., Pérez, P., Jégou, H., Grave, E., & Zeghidour, N. (2024). **Moshi: a speech-text foundation model for real-time dialogue.** arXiv:2410.00037. https://arxiv.org/abs/2410.00037 — the architecture and initial weights.
5. Lin, G.-T., Lian, J., Li, T., Wang, Q., Anumanchipalli, G., Liu, A. H., & Lee, H.-y. (2025). **Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities.** arXiv:2503.04721. https://arxiv.org/abs/2503.04721
6. Nari Labs. **Dia.** https://github.com/nari-labs/dia — two-speaker joint generation for the service dialogs.
7. Resemble AI. **Chatterbox.** https://github.com/resemble-ai/chatterbox — per-turn generation for the QA dialogs and the released checkpoint.
8. Qwen Team. (2025). **Qwen3 Technical Report.** — transcript generation.
9. OpenAI. (2025). **gpt-oss-120b & gpt-oss-20b model card.** — transcript generation and prompt annotation.
10. Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., Li, J., Kanda, N., Yoshioka, T., Xiao, X., et al. (2022). **WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing.** *IEEE JSTSP*, 16(6), 1505–1518. — the speaker verification model behind every SSIM number above.
11. Cieri, C., Miller, D., & Walker, K. (2004). **The Fisher Corpus: a Resource for the Next Generations of Speech-to-Text.** *LREC 2004.* — the real conversations added in the released checkpoint.
