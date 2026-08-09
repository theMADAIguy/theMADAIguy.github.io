---
layout: post
title: "How Full-Duplex Audio Models Actually Work"
description: "Moshi, dGSLM, PersonaPlex, Lychee-FD and NVIDIA VoiceChat 11B, explained from first principles: why voice AI still feels like a walkie-talkie, and what it takes to fix it."
date: 2026-08-09
image: /assets/blog/full-duplex/fig2-latency-budget.png
---

*A friend asked me why voice AI still feels like a walkie-talkie. This is the long answer.*

---

## The walkie-talkie problem

Talk to most voice assistants and you'll notice a rhythm: you speak, you stop, a beat of silence, then it speaks. You wait. It waits. Nobody ever talks at the same time.

Now think about how you actually talk to a friend. You say "mm-hm" while they're still mid-sentence. You cut in when you already know where they're going. They trail off, you finish the thought, they laugh over you. Two people in a conversation are almost never taking clean turns — they're continuously listening and producing at the same time, using overlap itself as a signal.

That's the gap. **Half-duplex** systems alternate. **Full-duplex** systems don't. And the difference isn't a UX polish layer bolted onto a fast pipeline — it's a different thing being modeled entirely.

Let me walk through the terminology, then the architectural split, then five real systems that took genuinely different bets on how to do it.

![Two timeline diagrams. Left: user and agent speech blocks strictly alternating with silent gaps between them. Right: the same tracks overlapping, showing a backchannel on top of the agent's speech and a barge-in cutting the agent off.](/assets/blog/full-duplex/fig1-two-rhythms.png)

*Figure 1 — The two rhythms. Half-duplex alternates; full-duplex overlaps.*

---

## Part 1: The vocabulary

Before the architectures, the words. Most of the confusion in this space comes from people using these loosely.

**Half-duplex / full-duplex.** Borrowed from telecom. Half-duplex means the channel carries traffic one direction at a time (walkie-talkie). Full-duplex means both directions simultaneously (phone call). Applied to dialogue models: can the system be *listening and speaking in the same instant*, or does it swap modes?

**Turn-taking.** The coordination problem of who holds the floor. Humans do this astonishingly fast — median gap between turns in conversation is around 200ms, which is *shorter than the time it takes to plan a word*. Meaning: you started preparing your response while the other person was still talking. Any system that only starts thinking after you stop has already lost.

**Endpointing / EOU (end-of-utterance) detection.** Deciding you've finished. Classic systems do this acoustically — silence for N milliseconds. This is where "I'm thinking... uh... about three things" gets you interrupted mid-sentence, and where "yeah." gets you 700ms of dead air.

**Barge-in.** The user interrupting the agent mid-speech. In a cascaded system this is a special case you have to detect and handle (kill the TTS stream, flush the buffer, re-enter listening). In a full-duplex model it's just... what the data looks like.

**Backchanneling.** The short listener noises — "mm-hm," "right," "yeah" — that you emit *while the other person keeps talking*. Critically, these do not take the floor. A system that can't distinguish "mm-hm" from "actually, wait—" doesn't understand turn-taking, it just understands voice activity.

**VAD (Voice Activity Detection).** A classifier that says "speech / not speech." Cheap, fast, semantically blind. Newer work pushes toward *semantic* endpointing — using an LM's understanding of whether the utterance is syntactically and semantically complete, rather than just measuring silence [10].

**Frame rate.** Full-duplex models discretize time. Everything happens on a clock — typically 12.5 Hz (80ms per frame) or 25 Hz. Every tick, the model consumes one frame of the user's audio and emits one frame of its own. This clock is the single most important design parameter in the whole field, and I'll come back to it.

**Semantic vs. acoustic tokens.** Neural audio codecs compress waveforms into discrete tokens using residual vector quantization (RVQ) — a stack of codebooks where each level encodes the residual left by the previous one. The convention that emerged: level 1 carries *semantic* content (what was said), levels 2–8 carry *acoustic* content (timbre, prosody, who said it, room tone). Mimi, Moshi's codec, explicitly distills its first codebook to match WavLM representations to enforce this split [2].

**TOR (Takeover Rate).** The main turn-taking metric in current benchmarks. Roughly: how often does the model take the floor when it should? Note the arrow direction flips by task — for *pause handling* you want TOR low (don't jump in when the user pauses mid-thought); for *smooth turn-taking* and *user interruption* you want it high [7].

---

## Part 2: What a cascaded voice agent actually is

The standard voice agent, the one behind most production systems, is a pipeline:

```
mic → VAD → ASR → LLM → TTS → speaker
```

Here's the loop in pseudocode, with the part that matters made explicit:

```python
def cascaded_voice_agent(mic, speaker, llm, asr, tts, vad):
    """The walkie-talkie. Note the shape: it's a `while True` around
    a strictly sequential body. Nothing overlaps."""
    history = []

    while True:
        # ---- PHASE 1: LISTEN (and do nothing else) ----
        audio_buffer = []
        while True:
            chunk = mic.read()               # e.g. 20ms
            audio_buffer.append(chunk)
            if vad.is_silence_for(ms=700):   # <-- the fatal hyperparameter
                break

        # ---- PHASE 2: TRANSCRIBE ----
        text = asr.transcribe(audio_buffer)  # +100-300ms

        # ---- PHASE 3: THINK ----
        reply = llm.generate(history + [("user", text)])  # +300-800ms TTFT
        history += [("user", text), ("assistant", reply)]

        # ---- PHASE 4: SPEAK ----
        for audio_chunk in tts.stream(reply):             # +80-200ms to first audio
            speaker.write(audio_chunk)
            # barge-in is a *patch* here, not a capability:
            if vad.detects_speech(mic.peek()):
                speaker.flush()               # kill playback
                tts.cancel()                  # kill synthesis
                break                         # awkwardly re-enter phase 1
```

Two things to notice.

**First, latency is additive and irreducible.** Your floor is `silence_threshold + ASR + LLM_TTFT + TTS_TTFB`. Engineers have gotten heroically good at shaving each term — streaming ASR, speculative endpointing, prefix-cached LLMs, streaming TTS — but that 700ms VAD threshold is doing something you can't optimize away. Lower it and the model interrupts you every time you take a breath. Raise it and the conversation feels dead.

**Second, and more fundamental: the model never hears the user.** The LLM sees `text`. It does not see that the user said it hesitantly, that they trailed off with rising intonation, that they were half-laughing, that they got quieter on the last three words. All of that was destroyed at the ASR boundary. The information required to do good turn-taking is *precisely* the information the pipeline throws away.

And there's a subtler one: in phase 4, the agent is deaf. It's producing audio and has no representation of what's happening on the input channel except a crude VAD flag. It can't backchannel. It can't notice you're confused. It can't trail off gracefully because you started talking.

![Stacked latency bars. The cascaded pipeline totals about 1.5 seconds, dominated by a 700 millisecond voice-activity-detection wait, plus ASR, LLM time-to-first-token and TTS. Below it, a much shorter full-duplex bar totalling 160 milliseconds.](/assets/blog/full-duplex/fig2-latency-budget.png)

*Figure 2 — Where the latency lives. The VAD silence threshold dominates the cascaded budget.*

---

## Part 3: The full-duplex idea in one paragraph

Stop thinking of dialogue as *a sequence of turns*. Think of it as *two audio streams evolving in parallel over a shared clock*.

Now autoregressively model that. Every tick of the clock, the model observes one frame of the user's stream and predicts one frame of its own stream. Silence is not the absence of a token — silence *is* a token. Speaking is a token. Backchanneling is a token. Turn-taking never needs to be detected, decided, or handled, because the model is simply predicting "what does my channel sound like at t+1 given both channels up to t," and in the training data, the answer to that question at the right moment happens to be *start talking*.

That reframe is the whole field. Everything else is engineering.

```python
def full_duplex_agent(mic, speaker, model, frame_ms=80):
    """The core loop. Notice what's NOT here: no VAD, no endpointing,
    no barge-in branch, no phases. Just a clock."""
    state = model.init_streaming_state()

    while True:
        user_frame = mic.read(frame_ms)          # always listening
        user_tokens = model.codec.encode(user_frame)

        # one step: condition on BOTH streams, emit own next frame
        own_tokens, state = model.step(
            user_tokens=user_tokens,
            state=state,
        )

        speaker.write(model.codec.decode(own_tokens))  # always speaking
        # ...where "speaking" often means emitting the token for silence.
```

The agent is *always* consuming and *always* producing. When it has the floor, its output decodes to speech. When it doesn't, its output decodes to silence — but it's still running, still updating state, still 80ms away from being able to say something.

That's why barge-in disappears as a feature. If you start talking while the model is speaking, its next-frame prediction is conditioned on your incoming audio, and a well-trained model's distribution over its own next frame shifts toward silence. It yields because yielding is what the data does.

![A piano-roll grid over 80 millisecond frames with a user row and an agent row. Coloured cells mark speech, grey cells mark silence, and a single orange cell marks a backchannel emitted while the user is still talking.](/assets/blog/full-duplex/fig3-silence-is-a-token.png)

*Figure 3 — Silence is a token. Turn-taking is predicted frame by frame, not detected.*

---

## Part 4: Five architectures, five different bets

### 4.1 dGSLM (2022) — the textless origin

Meta's **dGSLM** [1] was the first model to generate naturalistic two-channel spoken dialogue, and it did it with no text at all. Trained on 2,000 hours of two-channel telephone conversation (the Fisher corpus) with no transcripts and no labels.

The architecture is a **dual-tower transformer with cross-attention**: one tower per speaker channel, each attending to the other's history. Audio is discretized into self-supervised units (HuBERT-style), and a HiFi-GAN vocoder maps units back to waveform.

```python
class DualTower(nn.Module):
    """dGSLM, conceptually. Two towers, one per channel,
    each seeing the other through cross-attention."""

    def forward(self, units_a, units_b):
        h_a = self.embed_a(units_a)
        h_b = self.embed_b(units_b)

        for layer_a, layer_b in zip(self.tower_a, self.tower_b):
            # self-attention: what have I been saying?
            h_a = layer_a.self_attn(h_a, causal=True)
            h_b = layer_b.self_attn(h_b, causal=True)

            # cross-attention: what have they been saying?
            # this is where turn-taking coordination lives
            h_a = layer_a.cross_attn(query=h_a, kv=h_b, causal=True)
            h_b = layer_b.cross_attn(query=h_b, kv=h_a, causal=True)

        return self.head_a(h_a), self.head_b(h_b)
```

**Why it mattered:** it demonstrated that turn-taking dynamics — including laughter and other paralinguistic signals in both channels simultaneously — are *learnable directly from raw conversational audio*, and that doing so produces more naturalistic turn-taking than a text-based cascaded baseline. The dynamics were never explicitly modeled. They fell out of next-token prediction on two channels.

**The catch:** textless means no linguistic grounding. dGSLM produces conversations with the *shape* of dialogue and the *rhythm* of dialogue, but it isn't a useful assistant — there's no world knowledge, no instruction following. It's a proof that the substrate works.

### 4.2 Moshi (2024) — the reference design

Kyutai's **Moshi** [2] is the architecture most people mean when they say "full-duplex model," and almost everything since either extends it or reacts to it.

Four ideas, stacked:

**(1) Mimi, a streaming codec at 12.5 Hz.** Mimi compresses 24kHz audio to a 12.5 Hz token representation at 1.1 kbps, fully causally, with an 80ms frame size — while outperforming non-streaming codecs like SpeechTokenizer (50 Hz, 4 kbps). That 12.5 Hz number is the load-bearing one. Text tokens arrive at roughly 3–4 Hz in normal speech; getting audio down to 12.5 Hz puts it within a small constant factor of text, which is what makes a 7B LM able to run this in real time at all.

**(2) Multi-stream modeling.** Moshi models *its own* audio stream and *the user's* audio stream as parallel token streams. Explicit speaker turns are removed from the formulation entirely, which is what lets it represent arbitrary conversational dynamics — overlap, interruption, simultaneous silence.

**(3) Inner Monologue.** Moshi predicts time-aligned *text* tokens as a prefix to its audio tokens at each step. It writes down what it's about to say, then says it. This substantially improves the linguistic quality of generated speech, and as a bonus gives you streaming ASR and streaming TTS from the same model.

**(4) The RQ-Transformer.** This one needs setup, because the *problem* it solves is easy to miss.

**The problem.** A single 80ms frame isn't one token. It's seventeen: 8 RVQ codebooks for Moshi's own audio, 8 for the user's, plus 1 text token for the inner monologue. And they can't be predicted independently — RVQ level 2 encodes the *residual* left over by level 1, so level 2 is meaningless until you've committed to level 1. They have to come out one after another, each conditioned on the last.

So a plain autoregressive transformer has to run seventeen forward passes to produce one frame. And it has 80ms to do it in, forever, or the audio stutters. Seventeen passes of a 7B model in 80ms is not a thing that happens on any GPU you own.

**The observation that saves it.** Those seventeen tokens are not all asking the same kind of question. Look at what each one actually needs to know:

- *"Should I be speaking at all right now, and what word comes next?"* — this needs the whole conversation. Who the user is, what they asked four minutes ago, whether they just started talking over you.
- *"Given that I've committed to the coarse structure of this 80ms of audio, what's the fine acoustic detail in codebook 6?"* — this needs essentially none of that. It needs to know what codebooks 1–5 said. That's it. The answer doesn't change based on what happened four minutes ago.

One question is long-range and expensive. The other is local and cheap. Flattening the sequence forces you to pay the long-range price sixteen extra times for questions that never needed it.

**The fix: split by axis, not by size.** Instead of one model walking down a flattened list, use two models walking in two different directions through the same grid of tokens.

- The **Temporal Transformer** walks *sideways*, across time. One step per 80ms frame. Its job: absorb everything that has happened in the conversation and compress it into a single vector describing "here is the state of this conversation at this instant." It carries the KV cache for the whole dialogue. This is the model that needs to be 7B, because conversational understanding is what's expensive.
- The **Depth Transformer** walks *downward*, through the seventeen tokens inside one frame. Seventeen steps, then it resets and forgets everything. Its job: take that one vector and unroll it into concrete tokens, coarse to fine. Its entire context is one frame, so it's tiny — not as a compromise, but because a seventeen-position sequence genuinely doesn't need more.

The names describe the axis each one travels. "RQ" is for residual quantization — the depth axis exists because RVQ created it.

The payoff, stated as a rate:

```python
# The hard real-time constraint:
#   12.5 frames/sec, 17 tokens per frame  ->  212.5 tokens/sec, forever.

# Flattened, one model does all of it:
#   7B forward passes:  212.5 / sec     <- impossible

# Factorized:
#   Temporal (7B)    :   12.5 / sec     <- once per frame, not per token
#   Depth   (small)  :  212.5 / sec     <- 17x more often, ~1% of the cost
```

Same 212.5 tokens per second either way. The difference is *which* model is on the hook for them.

Here's one tick:

```python
def rq_transformer_step(temporal, depth, prev_frame, state):
    """Produce one 80ms frame of Moshi."""

    # --- AXIS 1: sideways through time. Runs once. ---
    # Everything the conversation has ever contained, squeezed into h_t.
    h_t, state = temporal.step(embed(prev_frame), state)

    # --- AXIS 2: downward through the frame. Runs 17 times, cheaply. ---
    # `depth` only ever sees h_t plus what it has already emitted below.
    # It has no idea a conversation is happening.
    ctx = h_t

    text_tok = depth.predict_text(ctx)          # what am I saying?
    ctx = ctx + embed_text(text_tok)

    tokens = []
    for k in range(8):                          # RVQ levels, coarse -> fine
        tok_k = depth.predict_audio(ctx, level=k)   # how does it sound?
        ctx = ctx + embed_audio(tok_k, level=k)     # level k+1 is k's residual
        tokens.append(tok_k)

    return text_tok, tokens, state
    # `ctx` is now discarded. Next frame starts fresh from a new h_t.
```

Two details in that inner loop are worth pausing on. The `ctx = ctx + embed(...)` accumulation is what makes the residual structure work — each level literally sees the sum of everything committed above it, which is the same thing the RVQ encoder was doing when it built the codes. And the ordering is deliberate: text first, then semantic codebook, then acoustic codebooks. The model settles *what* it's saying before it settles *how it sounds*. That's Inner Monologue and RVQ hierarchy falling out of the same top-to-bottom sweep.

The general lesson generalizes past Moshi: when your tokens have two independent structural axes, don't flatten them into one. Factorize, and let each axis get a model sized to the questions it actually asks.

**Latency.** Moshi reports a theoretical latency of 160ms — 80ms for the Mimi frame plus 80ms of acoustic delay — and around 200ms in practice on an L4 GPU. Compare that to the cascaded budget above. And note it's in the same range as the ~200ms human turn gap, which is not a coincidence; it's the target.

![A grid of 17 stacked token streams across 8 time frames: one text row, eight Moshi audio rows, eight user audio rows. A vertical arrow inside a single highlighted column marks the Depth Transformer; a wide horizontal arrow beneath marks the Temporal Transformer. The Moshi acoustic rows are offset one column right, showing the acoustic delay.](/assets/blog/full-duplex/fig4-moshi-tensor.png)

*Figure 4 — Moshi's token tensor. The Temporal Transformer walks sideways through time; the Depth Transformer walks down through one frame.*

### 4.3 PersonaPlex (2026) — making it *someone*

NVIDIA's **PersonaPlex** [4] takes the Moshi architecture and weights and attacks a specific limitation: existing duplex models are locked to a fixed role and a fixed voice. Great for a demo, useless if you want a full-duplex model to actually be a customer service agent, a language tutor, or an intake nurse.

The mechanism is a **hybrid system prompt** — role conditioning via *text* prompt, voice cloning via a *speech* sample:

```python
# The persona is set once, before the clock starts running.
prompt = HybridSystemPrompt(
    role_text=(
        "You are Maya, a support agent for a regional airline. "
        "You are warm but efficient. You must verify the booking "
        "reference before discussing any itinerary details."
    ),
    voice_reference=load_wav("target_speaker_10s.wav"),  # cloning
)

state = model.init_streaming_state(prompt=prompt)
# ...then the identical 80ms frame loop from Part 3.
```

The hard part isn't the interface, it's the data. There is no natural corpus of two-channel conversations paired with role descriptions and voice samples. PersonaPlex is trained on a large-scale *synthetic* dataset: open LLMs generate the paired prompts and user–agent conversations, open TTS systems render them to speech. They also extended Full-Duplex-Bench beyond a single assistant role to multi-role customer service scenarios, because the existing benchmark literally could not measure what they built.

Reported result: strong role adherence, speaker similarity, latency, and naturalness relative to prior duplex models and hybrid LLM-based speech systems.

**Why this is a bigger deal than it sounds:** persona control is what moves full-duplex from "research artifact" to "deployable." It also demonstrates something structurally important — that you can inherit a duplex backbone and add controllability on top, rather than retraining the conversational dynamics from scratch. That pattern (freeze/inherit the duplex prior, add a capability) recurs in a lot of recent work.

### 4.4 Lychee-FD (2026) — diagnosing the tax

Here's a problem the field kept observing and not explaining: **take a strong half-duplex speech LM, convert it to native full-duplex, and it gets noticeably dumber.** Its knowledge degrades. Its reasoning degrades. You gain fluid interaction and lose intelligence.

**Lychee-FD** [5] does the diagnostic work. Through analysis of optimization dynamics, they identify the cause as **gradient conflict between acoustic rendering and semantic modeling when both are forced to share a deep parameter space**. The two objectives want different things from the same weights, and in the deeper layers they actively fight. There's a second failure mode too: high-frequency audio tokens numerically swamp the sparse text supervision — semantic dilution.

The fix follows directly from the diagnosis, which is what makes this paper satisfying:

```python
class HierarchicalSeparation(nn.Module):
    """Lychee-FD, conceptually: share early, split late."""

    def __init__(self, n_shared, n_split):
        # Early layers: shared. Low-level features are genuinely common.
        self.shared = nn.ModuleList([Block() for _ in range(n_shared)])
        # Deep layers: separated. This is where the gradients conflict.
        self.semantic_branch = nn.ModuleList([Block() for _ in range(n_split)])
        self.acoustic_branch = nn.ModuleList([Block() for _ in range(n_split)])

    def forward(self, x):
        for blk in self.shared:
            x = blk(x)

        h_sem = x
        for blk in self.semantic_branch:      # reasoning, knowledge, monologue
            h_sem = blk(h_sem)

        h_ac = x
        for blk in self.acoustic_branch:      # timbre, prosody, rendering
            h_ac = blk(h_ac)

        # The semantic alignment channel: keep the internal monologue
        # coherent so acoustic rendering can't wash out meaning.
        h_ac = self.semantic_alignment(h_ac, h_sem)
        return h_sem, h_ac
```

Two components: **hierarchical parameter separation** decoupling the conflicting modalities in deep layers, and a **semantic alignment channel** that preserves a coherent internal monologue so semantic modeling stays robust during training.

The reported numbers: roughly +7.4% on Spoken QA and +28.5% on FullDuplexBench 1.5 over the prior state of the art, without sacrificing inference efficiency. The result I find most interesting is that Lychee-FD doesn't just recover the performance of its half-duplex backbone — it *exceeds* it, which suggests the duplex tax was never inherent, just an artifact of naive parameter sharing.

![Two panels. Left: a single transformer stack with acoustic and semantic gradients pushing against the same deep block, with a declining performance trend. Right: the deep layers fork into separate acoustic and semantic branches linked by a semantic alignment channel, with an improving trend.](/assets/blog/full-duplex/fig5-gradient-conflict.png)

*Figure 5 — The gradient conflict. Sharing deep layers forces acoustic and semantic objectives to fight.*

### 4.5 NVIDIA NemotronLabs VoiceChat 11B (2026) — full-duplex meets agents

Released August 2026, **VoiceChat 11B** [6] takes a visibly different architectural route from the Moshi lineage, and adds a capability nobody else had shipped openly.

The architecture is a hybrid Mamba/Transformer stack assembled from existing components: a **Fast Conformer speech encoder** in front, an **NVIDIA Nemotron Nano v2** LLM backbone in the middle, and an **NVIDIA TTS decoder and codec** behind. Crucially it's one unified model with a single computation graph — not three services with API handoffs — which is what preserves the latency win. Input audio at 16kHz, output at 22.05kHz. Roughly 550k hours of training audio, real and synthetic. Interestingly, PersonaPlex's training datasets are listed among its ingredients.

The distinguishing feature is **tool calling during live conversation** — the first open full-duplex model to do it. There's a separate output channel that emits tool-call scripts alongside the speech generation.

The design problem this solves is genuinely tricky. In a text agent, calling a tool means a pause; nobody notices. In a *voice* agent, calling a tool means dead air, and dead air in conversation reads as the system being broken. VoiceChat's answer: each tool can define an **"on-hold" message** that the agent speaks the moment the LLM generates the tool-triggering text, so the conversation keeps flowing while the tool executes.

```python
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string",
                         "description": "The city name as the user spoke it"}
            },
            "required": ["city"],
        },
        # The conversational cover. Spoken immediately, so the
        # 900ms of API latency doesn't sound like a crash.
        "on_hold_message": "Let me check that for you.",
    },
]

# The model emits, on its dedicated tool channel:
#   <TOOLCALL>[{"name": "get_weather", "arguments": {"city": "Sunnyvale"}}]</TOOLCALL>
# The client executes and returns:
#   <TOOL_RESPONSE>[{"temp_f": 72, "conditions": "clear"}]</TOOL_RESPONSE>
# ...while the audio clock never stopped ticking.
```

Reported numbers from the model card: ~448ms latency on smooth turn-taking (TOR 0.82), 480ms on user interruption (TOR 1.0), #2 among open full-duplex models on VoiceBench, and 56.1% average on the BFCL-v3 tool-calling subset of AU Harness. That last number is worth reading honestly — tool *selection* is 82.5% on Full-Duplex-Bench-v3 but *argument accuracy* is 44.2% and Pass@1 is 33%. Picking the right tool from spoken input is largely solved; extracting correct arguments from disfluent speech is not.

Note also the deployment reality: it wants 80GB of GPU memory. Full-duplex is not cheap.

---

## Part 5: Side by side

<div class="table-scroll" markdown="1">

| | **dGSLM** (2022) | **Moshi** (2024) | **PersonaPlex** (2026) | **Lychee-FD** (2026) | **VoiceChat 11B** (2026) |
|---|---|---|---|---|---|
| **Core bet** | Dynamics are learnable from raw audio | Text + audio in one autoregressive stack | Duplex needs controllability | Duplex tax is a gradient problem | Duplex needs to be *agentic* |
| **Architecture** | Dual-tower + cross-attention | RQ-Transformer (time axis 7B + depth axis) | Moshi arch & weights | Hierarchical parameter separation | Conformer → Nemotron Nano v2 → TTS decoder |
| **Text?** | None (textless) | Inner Monologue | Inner Monologue + role prompt | Semantic alignment channel | Text is the LLM interface |
| **Audio repr.** | Self-supervised units + HiFi-GAN | Mimi, 12.5 Hz, 1.1 kbps, 8 RVQ levels | Mimi | Hierarchical acoustic–semantic | NVIDIA TTS codec |
| **Headline** | First textless spoken dialogue generation | 160ms theoretical / 200ms practical | Role + voice control via hybrid prompt | +7.4% Spoken QA, +28.5% FDBench 1.5 | ~450ms turn-taking; first open FD tool calling |
| **Limitation** | No linguistic grounding | Fixed role & voice | Inherits Moshi's ceiling | Recent; ecosystem still forming | 80GB VRAM; 44% arg. accuracy |

</div>

---

## Part 6: What's still hard

**Data.** This is the real bottleneck. Full-duplex training wants *dual-channel* conversational audio — separate tracks per speaker, so overlap is preserved rather than mixed. Fisher (2,000 hours) is still doing enormous heavy lifting two decades on. Most large speech corpora are single-speaker read speech, which contains exactly zero examples of the phenomenon you're trying to learn. Hence the industry-wide pivot to synthetic conversation generation — and hence the open question of whether models trained on TTS-rendered dialogue learn real human turn-taking timing or a synthetic caricature of it.

**Evaluation.** You can't measure conversational dynamics with WER or an LLM judge on transcripts. The Full-Duplex-Bench family [7, 8, 9] is the current answer: v1 covers pause handling, backchanneling, smooth turn-taking, and user interruption with automatic metrics; v2 adds multi-turn evaluation with an automated examiner; v3 targets tool use under real-world disfluency. VoiceBench [11] covers the intelligence axis. You need both, and models routinely trade one against the other.

**The intelligence/interaction trade-off.** Lychee-FD diagnosed it but didn't dissolve it. Every full-duplex system is still negotiating between "responds beautifully" and "knows things."

**Semantic endpointing.** Even native duplex models inherit some acoustic bias in when they take the floor. Work like Phoenix-VAD [10] pushes toward endpoint detection based on *semantic completeness* — using an LM's judgment of whether the utterance is finished — with different timeout thresholds for complete vs. incomplete input.

**Context length.** At 12.5 Hz with 17 streams, a 30-minute conversation is 22,500 timesteps. The RQ-Transformer factorization keeps this tractable, but long-horizon duplex memory is genuinely unsolved.

**Everything downstream.** Every capability the text-LLM world takes for granted has to be re-invented under a real-time clock. Tool calling took until 2026 to appear openly. Retrieval, long-document grounding, multi-agent coordination, structured output — all of these have to work *without stopping the audio*, and that constraint is much sharper than it first appears.

![A scatter plot with interaction quality on the horizontal axis and speech intelligence on the vertical axis. Cascaded pipelines sit high-left, dGSLM low-right, and Moshi, PersonaPlex, Lychee-FD and VoiceChat 11B progress toward the top-right frontier, with human conversation marked beyond it.](/assets/blog/full-duplex/fig6-evaluation-surface.png)

*Figure 6 — The evaluation surface. Interaction quality and speech intelligence still trade against each other.*

---


## References

1. Nguyen, T. A., Kharitonov, E., Copet, J., Adi, Y., Hsu, W.-N., Elkahky, A., Tomasello, P., Algayres, R., Sagot, B., Mohamed, A., & Dupoux, E. (2023). **Generative Spoken Dialogue Language Modeling.** *Transactions of the ACL*, 11:250–266. arXiv:2203.16502. https://aclanthology.org/2023.tacl-1.15/ · Samples: https://speechbot.github.io/dgslm/

2. Défossez, A., Mazaré, L., Orsini, M., Royer, A., Pérez, P., Jégou, H., Grave, E., & Zeghidour, N. (2024). **Moshi: a speech-text foundation model for real-time dialogue.** arXiv:2410.00037. https://arxiv.org/abs/2410.00037 · Code: https://github.com/kyutai-labs/moshi

3. Veluri, B., Peloquin, B. N., Yu, B., Gong, H., & Gollakota, S. (2024). **Beyond Turn-Based Interfaces: Synchronous LLMs as Full-Duplex Dialogue Agents.** *EMNLP 2024*, pp. 21390–21402. arXiv:2409.15594. https://aclanthology.org/2024.emnlp-main.1192/

4. Roy, R., Raiman, J., Lee, S.-g., Ene, T.-D., Kirby, R., Kim, S., Kim, J., & Catanzaro, B. (2026). **PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models.** arXiv:2602.06053. https://arxiv.org/abs/2602.06053 · Code: https://github.com/NVIDIA/personaplex

5. **Hierarchical Acoustic-Semantic Modeling: Modality Separation and Semantic Coherence for Full-Duplex SLMs (Lychee-FD).** (2026). arXiv:2607.06540. https://arxiv.org/abs/2607.06540 · Code: https://github.com/HITsz-TMG/Lychee-FD

6. NVIDIA (2026). **NVIDIA NemotronLabs VoiceChat 11B.** Model card, released August 3, 2026. https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B · Code: https://github.com/NVIDIA-NeMo/Speech/tree/nemotron-labs-voicechat

7. Lin, G.-T., et al. (2025). **Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities.** arXiv:2503.04721. https://arxiv.org/abs/2503.04721

8. Lin, G.-T., Kuan, S.-Y. S., Shi, J., Chang, K.-W., Arora, S., Watanabe, S., & Lee, H.-y. (2025). **Full-Duplex-Bench v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner.** arXiv:2510.07838. https://arxiv.org/abs/2510.07838

9. **Full-Duplex-Bench-v3: Benchmarking Tool Use for Full-Duplex Voice Agents Under Real-World Disfluency.** (2026). arXiv:2604.04847. https://arxiv.org/abs/2604.04847

10. **Phoenix-VAD: Streaming Semantic Endpoint Detection for Full-Duplex Speech Interaction.** (2025). arXiv:2509.20410. https://arxiv.org/abs/2509.20410

11. Chen, Y., et al. (2024). **VoiceBench: Benchmarking LLM-Based Voice Assistants.** arXiv:2410.17196. https://arxiv.org/abs/2410.17196

12. Hu, K., Hosseini-Asl, E., et al. (2025). **SALM-Duplex: Efficient and Direct Duplex Modeling for Speech-to-Speech Language Model.** arXiv:2505.15670. https://arxiv.org/abs/2505.15670

---

*Code snippets in this post are illustrative pseudocode written to expose architectural intent — they are not runnable implementations. For working code, see the linked repositories.*
