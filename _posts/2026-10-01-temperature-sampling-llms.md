---
layout: page
title: "The Paradox of Perfection: Why LLMs Need Randomness to Sound Human"
date: 2026-10-01
---

# The Paradox of Perfection: Why LLMs Need Randomness to Sound Human

> **TL;DR**
>
> - A language model outputs a probability for every token. *Decoding* is the rule that turns those probabilities into one token.
> - "Always pick the most likely token" (greedy, beam search) gives bland, looping text on open-ended tasks.
> - **Temperature** reshapes the distribution. A **random number generator (RNG)** then draws from it. Top-k, top-p and min-p cut off the unreliable tail.
> - Every number in the figures below is computed from one fixed set of logits, so you can reproduce them.

If you have prepared for an ML interview, you have been asked some version of this: *"We spent millions finding the best weights. Why not just take the best token every time?"*

The intuitive answer is the trap. This post is the mechanism behind the right answer.

---



## 1. The trap: maximizing probability

Early generation systems searched for the highest-probability sequence. Greedy decoding takes the top token at each step. Beam search keeps the top-B partial sequences and returns the best-scoring one.

On open-ended generation this fails in two ways:

- **Repetition loops.** The most likely continuation of a phrase is often a repeat of it. Once a loop starts, its own repetition makes the loop more likely.
- **Blandness.** The safest token is the generic one. Metaphor, rhythm and specific detail all live below the top-1 slot.

Holtzman et al. (2019) ran this experiment and found that maximization-based decoding produces "text that is bland and strangely repetitive", even from a model that scores well on likelihood **[1]**.

## 2. Humans do not write at maximum probability

The same paper compared the per-token probability a model assigns to human text against what it assigns to its own beam-search output **[1]**.

- Machine text under beam search sits at high probability at nearly every step.
- Human text does not. Its probability moves around, with many tokens the model considered unlikely.

Take the prompt *"He took a deep breath and dove into the ___"*. A model puts most of its mass on "water". A human writer sometimes picks "unknown" or "chaos". Those low-probability choices are a large part of why the sentence reads as written by a person.

So the goal of decoding is not "find the best sequence". It is "produce sequences that look like samples from the human distribution".

---



## 3. Temperature: rescale the logits before softmax

The model outputs a logit $z_i$ for each token. Temperature $T$ divides the logits before the softmax **[2]**:

$$p_i(T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- **T < 1** sharpens. The gap between logits grows, so the top token takes more mass.
- **T = 1** is the model's native distribution.
- **T > 1** flattens. Gaps shrink and the tail gains mass.
- **T → 0** converges to greedy decoding (argmax). **T → ∞** converges to uniform over the vocabulary.

Temperature does not add randomness. It only changes the *shape* of the distribution. The randomness comes in the next step.

![Figure 1](/assets/blog/temperature-sampling-llms/fig1.png)
*Figure 1: Eight candidate tokens from one fixed set of logits, softmaxed at four temperatures. Blue is the top token, orange are the "creative" choices (unknown, chaos).*

Reading the figure:




| T   | water | pool  | unknown | chaos | entropy   |
| --- | ----- | ----- | ------- | ----- | --------- |
| 0.2 | 99.6% | 0.4%  | 0.0%    | 0.0%  | 0.04 bits |
| 0.7 | 73.9% | 15.4% | 6.5%    | 2.4%  | 1.25 bits |
| 1.0 | 58.5% | 19.5% | 10.7%   | 5.3%  | 1.83 bits |
| 1.5 | 42.7% | 20.5% | 13.8%   | 8.6%  | 2.36 bits |




- At **T = 0.2** the two creative tokens together have under 0.1% of the mass. They will essentially never appear.
- At **T = 1.0** they have 16%. About one draw in six is a twist.
- At **T = 1.5** they have 22%, but "water" is down to 43% and the other, lower-ranked tokens are also gaining mass. That is where text starts to drift.

![Figure 2](/assets/blog/temperature-sampling-llms/fig2.png)
*Figure 2: Top-1 probability (blue, left axis) and entropy in bits (orange, right axis) as temperature sweeps from 0.05 to 2.5, for the same logits.*

The curves are smooth, with no threshold. Temperature is a dial between "always the same token" and "uniform noise".

---



## 4. The RNG: where the dice roll happens

Once you have $p(T)$, the sampler draws one token. The standard method is inverse-CDF sampling:

```python
def sample(logits, T, rng):
    z = logits / T
    p = softmax(z)              # temperature already applied
    u = rng.random()            # uniform in [0, 1)
    return searchsorted(cumsum(p), u)   # first index where cumsum > u
```

Lay the tokens end to end on the interval [0, 1], each taking a segment as wide as its probability. The RNG throws a dart at that interval. Whichever segment it lands in is the token.

![Figure 3](/assets/blog/temperature-sampling-llms/fig3.png)
*Figure 3: The same RNG draw, u = 0.83, at two temperatures. At T = 0.5 the dart lands inside "water" (87% of the line). At T = 1.5 the segments are more even and the same dart lands on "chaos".*

This is the point most explanations skip. **The RNG value did not change between the two panels. The distribution did.** Temperature decides how much of the interval each token owns, and the RNG decides which spot gets hit.

Two practical consequences:

- With the same prompt, same T and same seed, sampling is reproducible. Production APIs often do not guarantee bit-exact repeats even at low temperature, because floating-point reductions on parallel hardware can differ between runs.
- A single draw that picks a low-probability token changes every later step, because the model now conditions on a different prefix. One roll sends the whole generation down a different branch.

---



## 5. Truncation: stop the tail from ruining the roll

Raising temperature fattens the whole tail, including tokens that are plain wrong. A vocabulary has tens of thousands of entries. Even a tiny per-token probability adds up when there are many of them. Truncation methods remove the tail *before* the RNG draw and renormalize.




| Method                      | Rule                                                                            | Weakness                                                           |
| --------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Top-k** **[3]**           | Keep the k highest-probability tokens                                           | Fixed k is too many when the model is sure, too few when it is not |
| **Top-p (nucleus)** **[1]** | Keep the smallest set whose cumulative probability reaches p                    | Can still admit poor tokens at high temperature                    |
| **Min-p** **[4]**           | Keep tokens with probability at least `min_p` times the top token's probability | Newer, less universally implemented                                |




![Figure 4](/assets/blog/temperature-sampling-llms/fig4.png)
*Figure 4: Top-p = 0.9 on a toy 50-token long-tail distribution (probability proportional to 1/rank^1.15). The cumulative curve crosses 0.9 at 27 tokens. The remaining 23 get zero probability after truncation.*

The key property of top-p and min-p is that the kept set is **adaptive**. When the model is confident, the nucleus is a handful of tokens. When it is uncertain, the nucleus is wide. Top-k cannot do that.

---



Follow-ups:

- *"What does T = 0 do?"* It is argmax, so greedy decoding. In code it is usually special-cased, since dividing by zero is undefined.
- *"Does temperature change the model's ranking of tokens?"* No. Dividing logits by a positive constant preserves their order. Only the probabilities between them change.
- *"Temperature vs top-p?"* Temperature reshapes the whole distribution. Top-p cuts it. They interact: high T with a tight top-p still works because truncation removes the tail temperature just inflated.
- *"When do you want low temperature?"* Tasks with one right answer, such as extraction, code and math, where a single unlucky draw breaks the output.

---



## References

**[1]** Holtzman, Buys, Du, Forbes, Choi. [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751). ICLR 2020. Introduces nucleus (top-p) sampling.

**[2]** Hinton, Vinyals, Dean. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). 2015. The softmax temperature used here.

**[3]** Fan, Lewis, Dauphin. [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833). ACL 2018. Applies top-k sampling to story generation.

**[4]** Nguyen, Baker, Neo, Roush, Kirsch, Shwartz-Ziv. [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs](https://arxiv.org/abs/2407.01082). ICLR 2025.