---
layout: page
title: "The User Is Not a Turn: Omni-Flow and MiniCPM-o 4.5"
description: "MiniCPM-o 4.5 treats user speech as part of the observed environment rather than a conversational turn. A two-person walkthrough of Omni-Flow's chunked serialization, why one-second chunks beat 100 ms, TAIL's time-aligned interleaving, and the four-stage training pipeline."
date: 2026-08-22
image: /assets/blog/minicpm-o-omniflow/fig1.png
---

*Ana builds speech language models. Ravi ships voice agents on cascaded stacks. They have already [taken Moshi apart](/blog/2026/08/20/moshi-architecture/) and worked out [where a system prompt goes](/blog/2026/08/22/personaplex/). Ravi thought that was the end of it.*

---

**Ravi:** I thought we'd finished this arc. Moshi deletes the turn, PersonaPlex writes a persona into the frames. What's left?

**Ana:** Deleting the conversational partner. MiniCPM-o 4.5 is a 9B omni-modal model from the MiniCPM-o team at OpenBMB, and its framework — Omni-Flow — makes one move that neither of the others makes. Moshi still models a conversation: two audio streams, you and it. Omni-Flow says the user is not a participant. The user is weather.

**Ravi:** Weather.

**Ana:** There are three streams in their formulation. **env-visual** is the live camera. **env-audio** is the acoustic scene, and the paper is explicit that user speech is just something that happens to arrive on that channel. **out-stream** is the model's own text and speech. Your request has no privileged status; it's one more thing occurring in an environment the model is watching anyway. And once you write it down that way, the model can't wait for a trigger, because there is no trigger. It has to decide, continuously, whether this is a moment to speak.

**Ravi:** That's where the proactive behaviour comes from.

**Ana:** It's not a feature they added. It's what's left when you delete the trigger. Same framework produces "answer the question," "warn them the pan is smoking," and "say nothing."

---

## Chunks, not frames

**Ravi:** So how does it actually get into a causal LM?

**Ana:** Time-division multiplexing, which is the analogy the paper reaches for and it's the right one. Cut the interaction into windows of duration *t*. For the *k*th window, encode whatever arrived on env-visual into a visual token sequence, whatever arrived on env-audio into an audio token sequence, and the model's own output into an output sequence. Group them, `g_k = [v_k; a_k; o_k]`, concatenate the groups, feed the whole thing to a standard causal transformer. Within a chunk the perceptual tokens come first, so every output token is conditioned on the freshest observation available.

**Ravi:** And when it shouldn't say anything?

**Ana:** `o_k` contains exactly one token: `[listen]`. That's the whole silence mechanism, and it's why they can claim reduced reliance on an external VAD — the decision to stay quiet is a token prediction inside the same sequence, not a signal from a module outside the model.

![Top: three parallel stream rows across four one-second chunks. env-visual carries 64 tokens per chunk and shows a pan on the hob, then smoke rising. env-audio carries 10 tokens per chunk and shows room tone and sizzling, with no user speech. out-stream shows a listen token for the first two chunks, then speak tokens as the model comments on the smoke unprompted. Bottom: the same content flattened into one causal sequence, each group written as a boundary token followed by v, a and o, with a brace labelling it g equals v semicolon a semicolon o.](/assets/blog/minicpm-o-omniflow/fig1.png)

*Figure 1 — One chunk, three streams, one sequence. Silence is a token, not a signal from outside the model.*

**Ravi:** What's the token budget per chunk?

**Ana:** Concrete numbers, because they're the reason everything else is shaped the way it is. Audio: a Whisper Medium encoder produces 50 feature tokens a second, then a two-layer MLP projector does 5× temporal compression, so the backbone sees 10 audio tokens per second. Vision: in full-duplex streaming mode the max resolution drops to 448×448, and with 14×14 patches that's exactly 1,024 patches, which a resampler compresses to 64 tokens — a 16× ratio against the 4× that's more common. So at one frame per second, a one-second chunk carries roughly 64 visual and 10 audio tokens.

---

## Why one second and not eighty milliseconds

**Ravi:** Hold on. Moshi's frame is 80 milliseconds. Omni-Flow's chunk is a *second*? That's twelve times slower.

**Ana:** And they tried to make it faster, and it fell apart. This is the ablation I'd send anyone who thinks finer granularity is free.

![Two grouped bar charts on MMLU, spoken QA and IFEval. Left: shrinking the chunk from 1.0 to 0.2 to 0.1 seconds drops MMLU from 0.65 to 0.45 to 0.32, spoken QA from 0.36 to 0.09 to 0.13, and IFEval from 0.29 to 0.10 to 0.10. Right: at one second, explicit boundary with Listen-Speak control scores 0.65, 0.36 and 0.29, explicit with Listen-Text scores 0.56, 0.35 and 0.24, and implicit with Listen-Text scores 0.45, 0.28 and 0.22.](/assets/blog/minicpm-o-omniflow/fig2.png)

*Figure 2 — Two ablations that argue against the obvious design choices.*

**Ana:** Holding everything else fixed and shrinking the chunk from 1.0 s to 0.2 s: MMLU falls from 0.65 to 0.45, spoken QA from 0.36 to 0.09, IFEval from 0.29 to 0.10. At 0.1 s, MMLU is 0.32 — the model has lost half its knowledge scores to a scheduling parameter.

**Ravi:** Why? Moshi doesn't have this problem.

**Ana:** Moshi's backbone emits audio tokens, so every 80 ms frame is a full unit of work — seventeen tokens in the grid. MiniCPM-o's backbone only emits *text*, at roughly 3–4 tokens per second, which is human speaking speed. Divide that out: a 0.1-second chunk holds about a third of a text token. The chunk boundary starts falling inside words. That arithmetic is my reading rather than theirs — they say only that short chunks leave insufficient information for stable decisions — but the numbers line up with it.

**Ravi:** Two other columns in that table.

**Ana:** Both worth having. **Explicit boundaries beat implicit ones** — marking where observation ends and generation begins is apparently a nontrivial thing to ask the model to infer, and telling it outright is worth a few points across the board. And **Listen-Speak beats Listen-Text**: predicting a binary speak/don't-speak control token *before* generating content works better than folding `[listen]` into the ordinary text vocabulary. MMLU 0.65 versus 0.56. Deciding *whether* to speak and deciding *what* to say want to be separate predictions.

---

## The backbone doesn't sing

**Ravi:** You said the backbone only emits text. Where does the audio come from?

**Ana:** A lightweight Llama-style speech token decoder, about 0.3B. For each text token, they sum the backbone's hidden state — reshaped through an MLP — with the speech decoder's own embedding, and the decoder generates S3 speech tokens from that. Prosody has already been decided upstream by the 8B model, so the small decoder can spend its capacity on acoustics. A streaming flow-matching decoder turns the tokens into waveform, conditioned on reference audio in the system prompt.

**Ravi:** Which is the PersonaPlex trick again.

**Ana:** Same idea, different plumbing — voice cloning by conditioning rather than by architecture. But the reason for the split is the interesting part. Backbones that generate speech tokens directly run at about 25 tokens a second, which is expensive, and the paper cites work suggesting it degrades core language ability.

**Ravi:** The knowledge tax.

**Ana:** Exactly the thing we looked at with Moshi, where the same 7B backbone dropped from 54.3 to 49.7 on MMLU and 56.4 to 22.8 on TriviaQA once it had to speak. So does the split pay off? Partly. Against its own Qwen3-8B backbone, MiniCPM-o 4.5 comes out slightly *ahead* on average, 82.1 to 81.6, and well ahead on BBH, 81.1 to 69.4. But MMLU drops from 81.7 to 77.0 and MATH-500 from 84.0 to 77.0. Their post-training data mixture is doing work in both directions; it isn't a clean "no tax" result, and the average alone would hide that.

---

## Making the speech keep up with the world

**Ravi:** Here's what I don't understand. If the model generates a sentence's worth of text in one chunk, that sentence takes several seconds to say out loud. By the time you hear the end of it, the scene has moved on.

**Ana:** That's the problem TAIL exists for, and you've stated it better than I would. Time-Aligned Interleaving. Existing streaming TTS does one of two things: generate a long text lead and synthesize behind it, or interleave text and speech at a fixed ratio. Neither pins the speech to the interaction clock. Text runs ahead, audio lags, and the model ends up narrating a moment that's over.

**Ravi:** So TAIL slows the text down.

**Ana:** Adaptively, and — this is the part I like — with memory. At the *k*th chunk it doesn't just match that chunk's speech duration. It looks at accumulated playback progress across the whole interaction, and adjusts how much text to emit so that after vocalizing, the audio stream arrives at the boundary *kt*. If earlier chunks introduced a small delay, this chunk generates fewer text tokens and lets the speech catch up. It's a controller with an integral term, in effect.

![Three schematic timelines over four one-second chunks, each with a text row and a speech row. Panel a, non-interleaving, puts all text tokens at the start and spreads speech across the whole timeline. Panel b, fixed ratio, repeats two text tokens and five speech tokens in every block. Panel c, TAIL, varies the text count per chunk from seven to four to six to four, with orange look-ahead tokens whose speech is deferred into the following chunk, shown by curved arrows.](/assets/blog/minicpm-o-omniflow/fig3.png)

*Figure 3 — Schematic. TAIL sets each chunk's text budget from how far the audio has already fallen behind.*

**Ravi:** How do you get supervision for that?

**Ana:** From timestamps. They take full-duplex streaming training data, collect the start and end time of every text token, and assign each token — plus its speech tokens — to the chunk its start time falls into. The number of text tokens per chunk then varies across the sequence according to how far behind the audio has drifted, and the model learns that history-dependent pattern rather than a fixed ratio.

**Ravi:** What about pronunciation? You need to know the next word sometimes.

**Ana:** "The apple" versus "the car" — their example. TAIL defers the speech tokens of the last few text tokens in chunk *k* into chunk *k+1*. Bounded look-ahead: enough future context for prosody, not enough for text to run away from playback.

**Ravi:** And the cost?

**Ana:** Real, and they print it. On the Seed-TTS test set, fixed-ratio interleaving gets 2.38 English WER; TAIL gets 3.93. Chinese CER goes 0.86 to 1.04. So temporal alignment costs about 65% relative on English word error. Their framing is that TAIL is for the harder full-duplex setting and the trade is worth it, which I think is right — but if you're deploying turn-based, use the fixed-ratio mode.

---

## Where the data comes from

**Ravi:** Who has time-indexed video-plus-audio-plus-speech data?

**Ana:** Nobody, so it's assembled. Bulk coverage comes from web audio-video, filtered hard: segments dominated by a single speaker get dropped, so do segments where audio and video aren't actually related. Then OCR-based subtitle removal, talking-head detection, and filtering on ASR transcripts — all of it aimed at killing shortcuts, because a model that can read the subtitles isn't learning to listen.

**Ravi:** And the good stuff?

**Ana:** Two pieces. A small manually constructed set of full-duplex task scenarios — continuous scene description, proactive reminding — with annotated instruction-following data. And for speech, my favourite detail in the paper: they generate colloquial dialogue with a text LLM, then have professional voice actors re-record a subset in a studio, delivering conversationally instead of reading the script verbatim, varying emotion and rate and emphasis while holding one vocal identity. That's a deliberate decision to buy the thing synthetic data can't fake.

---

## Four stages

**Ravi:** And training?

**Ana:** Staged, starting from a MiniCPM-V 4.5 pretraining checkpoint and a pretrained Whisper encoder. **Stage one**, speech pretraining: freeze everything pretrained, train only the new modules — audio projector, LLM-to-speech projector, speech decoder — so the vision and language abilities can't be damaged while the speech path finds the hidden space. **Stage two**, joint pretraining: unfreeze everything, train on a balanced mixture. There's a nice piece of engineering here — different modality combinations are assigned to different data-parallel ranks, which fixes the modality ratio at every single step rather than in expectation.

**Ravi:** That's a real problem in practice. Batch composition drifts and your loss curve gets weird.

**Ana:** **Stage three**, supervised fine-tuning in two phases, broad then human-annotated, with resolution and frame rate randomized — 0.2–0.4 megapixels, 1–5 FPS — so one checkpoint can serve different quality-latency budgets. **Stage four**, RL: GRPO for reasoning and instruction following, a general reward model that specifically suppresses unintended English-Chinese code-mixing, and RLAIF-V for hallucination. And they report that hallucination mitigation learned from static image-text data transfers to streaming — which is the sort of claim I'd want to see measured directly, but it's a plausible and useful one.

---

## What's still hard

**Ravi:** Give me the honest version.

**Ana:** The evaluation gap is the big one, and to their credit they say so. The headline capability is full-duplex omni-modal interaction. The full-duplex number they report is on LiveSports-3K-CC — a win rate of 54.4, against 45.6 for StreamingVLM and 41.5 for LiveCC. That's a real result, and it's an **audio-free** benchmark. Vision-only. The paper states plainly that benchmarks for real-time omni-modal full-duplex interaction barely exist.

**Ravi:** So the audio half of the flagship claim is demonstrated by...

**Ana:** A demo website. There's no Full-Duplex-Bench evaluation, no turn-taking or barge-in or backchannel numbers, and no head-to-head against Moshi or PersonaPlex — which is not a criticism of these authors so much as a statement about the field. Nobody has agreed on how to score a model whose main skill is knowing when not to talk.

**Ravi:** Anything else?

**Ana:** Their own limitations list is unusually candid: speech generation in streaming mode is sometimes unstable, with mispronunciation and English-Chinese code-mixing, and proactive behaviour is still simple — reminders and observations, not context-aware planning. Add TAIL's word-error cost and the 1.0-second floor on chunk size, and you have a model whose responsiveness is bounded by a design parameter they couldn't push further.

---

## The one-paragraph version

Omni-Flow reframes real-time interaction as time-division multiplexing over three streams — live video, live audio, and the model's own output — sliced into one-second chunks and serialized into a single causal sequence, where staying quiet is a `[listen]` token rather than a decision made by a VAD outside the model. The consequence is that user speech loses its privileged status as a "turn" and becomes part of the observed environment, which is what makes proactive behaviour fall out of the same machinery as answering a question. MiniCPM-o 4.5 pairs this with a backbone that emits only text at human speaking rate while a 0.3B decoder handles speech tokens, and with TAIL, an adaptive interleaving scheme that throttles text generation so audio playback tracks the interaction clock — at a measurable cost in word error. The ablations are the most useful part of the paper: finer chunks are dramatically worse, explicit boundaries help, and deciding *whether* to speak wants to be a separate prediction from deciding *what* to say.

---

*Code and figures in this post are illustrative rather than implementation-accurate. All numbers are from the paper.*

## References

1. Cui, J., Xu, B., Wang, C., Yu, T., Sun, W., et al. (2026). **MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction.** arXiv:2604.27393. https://arxiv.org/abs/2604.27393 — 36 authors; MiniCPM-o Team, OpenBMB.
2. OpenBMB. **MiniCPM-o 4.5 code and demo.** https://github.com/OpenBMB/MiniCPM-o-Demo
3. Défossez, A., Mazaré, L., Orsini, M., Royer, A., Pérez, P., Jégou, H., Grave, E., & Zeghidour, N. (2024). **Moshi: a speech-text foundation model for real-time dialogue.** arXiv:2410.00037. https://arxiv.org/abs/2410.00037
4. Roy, R., Raiman, J., Lee, S., Ene, T.-D., Kirby, R., Kim, S., Kim, J., & Catanzaro, B. (2026). **PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models.** arXiv:2602.06053. https://arxiv.org/abs/2602.06053
5. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). **Robust Speech Recognition via Large-Scale Weak Supervision (Whisper).** *ICML 2023.* — the audio encoder.
6. Qwen Team. (2025). **Qwen3 Technical Report.** — the 8B backbone.
7. Du, Z., Wang, Y., Chen, Q., Shi, X., et al. (2024). **CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models.** arXiv:2412.10117. https://arxiv.org/abs/2412.10117 — S3 speech tokens and the streaming flow-matching decoder.
8. Guo, Z., Xu, R., Yao, Y., Cui, J., et al. (2024). **LLaVA-UHD: an LMM Perceiving Any Aspect Ratio and High-Resolution Images.** *ECCV 2024.* — the image partitioning strategy.
9. Chen, J., Zeng, Z., Lin, Y., Li, W., Ma, Z., & Shou, M. Z. (2025). **LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale.** *CVPR 2025.* arXiv:2504.16030. https://arxiv.org/abs/2504.16030 — origin of the LiveSports-3K-CC benchmark.
10. Xu, R., Xiao, G., Chen, Y., He, L., Peng, K., Lu, Y., & Han, S. (2025). **StreamingVLM: Real-Time Understanding for Infinite Video Streams.** arXiv:2510.09608. https://arxiv.org/abs/2510.09608
11. Yu, T., Zhang, H., Li, Q., Xu, Q., Yao, Y., et al. (2024). **RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness.** — the hallucination-reduction stage.
12. Lin, G.-T., Lian, J., Li, T., Wang, Q., Anumanchipalli, G., Liu, A. H., & Lee, H.-y. (2025). **Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities.** arXiv:2503.04721. https://arxiv.org/abs/2503.04721 — the evaluation this paper does not report.
