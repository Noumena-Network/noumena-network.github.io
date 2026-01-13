---
title: "Why Training MoEs is So Hard"
subtitle: "The three challenges that make frontier MoE training different from everything else"
date: 2026-01-01
author: "xjdr"
number: "0001"
---

Recently, I found myself wanting a small, research-focused training repo that I could do quick experiments on. These experiments range from trying out new attention architectures (MLA, SWA, NSA, KDA—all pluggable) to multi-precision training to most recently multi-optimizer setups with "new" optimizers.

I tried the three major contenders (NeMo, Megatron, and Torchtitan) but for many and various reasons they didn't fit the bill. They were all pretty painful to set up, use, and get running stably. I once again missed my tooling from Google, and rewriting my production training stack (which is tailor-made for large infrastructure monitoring and stability) also felt like a poor use of time.

This got me thinking: **why was training frontier-quality "smallish" MoEs (say under 20B params total) so difficult?** Why didn't the repo I wanted already exist?

After thinking about it for a while, most of the challenges came down to three different things:

1. FLOPs / FLOP efficiency
2. Load balancing / router stability
3. Data quality and quantity

---

## FLOPs

Training dense models is pretty straightforward these days. The training dynamics are mostly coupled, and if you have enough params in the architecture, the model will pretty much learn despite your many mistakes (this has bitten me more than once).

[DeepSeek-style ultra-sparse MoEs](https://arxiv.org/abs/2412.19437v2) are different because your training dynamics are **decoupled**. Only a portion of your MLPs are active for a given token, and as training goes on, the active experts change and evolve over time.

This is what makes multi-epoch training and data rephrasing so effective for MoEs (especially larger ones). You get large inference efficiency wins and small training efficiency wins, but at the cost of:

- Decoupled training dynamics (makes it hard to predictably and stably train)
- You have to dump a lot more FLOPs in to make sure you learn a somewhat optimal routing policy and that the experts involved in the various policies are adequately trained

This is where the FLOPs / FLOP efficiency challenge arises.

### The stranded FLOPs problem

By nature, ultra-sparse MoEs take up a tremendous amount of HBM to load up the experts, which means you have a lot of GPUs required—and thus a lot of idle GPUs in your system.

FSDP (and the various other sharding topologies) are mostly relics of dense training, and do not do a good job of adequately leveraging all those stranded FLOPs. This leads to **single-digit MFUs** for most people's ultra-sparse MoE training.

While there are a handful of ways of addressing that (much more on this in the future), I focused on two specific things:

1. **New sharding topology**: A novel expert-parallel dispatch system that keeps GPUs busy
2. **Mixed precision training**: You have all this stranded HBM, so reduce expert precision and cut that by 1/2 or 1/4 (FP8 and NVFP4 respectively)

---

## Load Balancing / Router Stability

I'll leave the new sharding topology for its own dedicated write-up. But mixed precision training is a no-brainer—in theory.

In practice, mixed precision training usually takes *more* HBM because you have to keep your master weights and grads in higher precision, then quantize down to the lower precision representation and cache them for the next forward pass. So it helps inference efficiency (which, as more and more FLOPs go to RL and inference, is a real win) but at the cost of even more HBM and more FLOPs during training.

Reducing the mixed precision overhead is something that should be a specific area of focus. However, anything you touch that reduces precision and accuracy of the weights ultimately leads to instability in the rest of the training dynamics.

**For MoE, the first place this usually shows up is router stability.**

### The DeepSeek approach (and its constraints)

The DeepSeek-V3 tech report describes a very elegant aux-loss-free training setup where there are very few knobs and the dynamics are very clear. These are clearly tools designed for the experienced user—getting the dynamics correct with only a few knobs is incredibly difficult.

Crucially, **DeepSeek relies heavily on massive batch sizes to stabilize their routers**—a luxury we don't have when doing research on limited hardware. So we have to work extra hard to make our small runs stable, efficient, and informative.

### The router learning problem

As I began experimenting with replicating their setup, specifically for mixed-precision experts, it became very clear that **the grads were far too small for FP8 or NVFP4**, causing the routers to not learn and the experts to starve.

I tried everything under the sun to make this work—first with reduced-precision backwards passes, and eventually even with FP32 master weights and grads—but the router collapse persisted.

A well-timed paper was the Character AI blog post describing their various INT8 stability interventions. I tried them all, but they ended up making the system much less stable. So I went through them one at a time.

### The breakthrough

The first one was **muP embedding scaling of 10.66** and the **logits scaling of 0.125**. There were a bunch of very obvious wins here other than router stability, but one clear thing these scales did was take the very very small FP8 and NVFP4 expert grads and scale them to the point where **the router was finally learning!**

However, these wins also caused the BF16 grad norm to absolutely explode, and the "normal best practice" grad clipping and grad scaling was preventing any learning from happening.

So, the solution? **Remove all clipping and YOLO!**

Finally, we had stable mixed-precision routers training.

### Additional interventions

The other intervention that ended up being incredibly useful was a **single bungee virtual scalar at the output of the experts** (pre-output norm) initialized at 2.0 to match the BF16 grad scale and thus training dynamics such that NVFP4 and FP8 more or less presented the same loss curves as the baseline BF16 training runs.

**Result**: FP8-BF16 gap reduced from ~0.8 to <0.1 at 3k steps.

The headlines:

- Add muP scaling
- Remove all other clipping and live on the wild side
- Bungee virtual scalar pre-output norm
- Keep aux-loss-free and tokens-choice routing *(cause we know the difference between right and wrong in this house)*

---

## Data

Now that we had reasonably stable training dynamics, it became clear that if I ever wanted to share this repo with anyone, we would need some better data to really get the most out of this training lib.

Another well-timed release was [OLMo-3](https://allenai.org/blog/olmo3) with its open-source data mixture recipe (shout out AI2!). However, when I tried using the OLMo-3 mixture directly from HuggingFace, I was getting pretty terrible results compared to my typical FineWeb-Edu baseline.

So, the data spelunking began... and the datasets were pretty dirty.

### Building a proper data pipeline

I did what any crazy person would do and set out to build a frontier-inspired data pipeline. The pipeline has a few key components:

**Heuristic pre-filters**: Language ID, length filters, MinHash dedup, n-gram repetition, perplexity outliers, toxicity—the standard stuff to remove obvious garbage before spending GPU cycles.

**SeqIO-style dynamic mixtures**: Deterministic, resumable sampling that maintains your target ratios (40% CC, 20% code, etc.) regardless of total token budget—critical for proxy runs where you're not training on 6.7T tokens.

**Model-based quality scoring**: This is where it gets interesting. Following the [Seed-Coder](https://arxiv.org/abs/2509.25149v1) pattern, I used large oracle models to generate training labels, then distilled into a fast classifier.

### The quality model architecture

I took a frozen GPT-style 20B backbone and attached two small heads:

- **Probe head** at layer 18: Mean-pooled hidden states → Linear(2880→5). Ultra cheap, catches obvious garbage early.
- **Judge head** at layer 24: Full sequence attention → small transformer encoder → Linear(512→5). More expensive but catches nuanced quality issues.

The early-exit design is key—if Probe scores below threshold, we skip Judge entirely. At scale this saves ~15% compute while maintaining quality.

**Results**: My keep rate for the OLMo-3 dataset was about 30% for CC and internet sources, and 50% for code, math, and science. That's a lot of filtering, but the proxy model evals showed clear improvements over the unfiltered baseline.

---

## Where we are now

We finally had a reasonably functional system that approximated all the things I loved about my large-scale training infra and the great tools I had used before—but was purpose-built for training small MoEs for research and small model production runs.

The result? We can now do meaningful MoE research on limited hardware:

- A 7B2A proxy on a single B200 GPU
- A 16B4A on a single 8×B200 node
- Both hitting 30-40k tokens/sec/GPU

More importantly, **the scaling is predictable**: our 1→8 GPU runs show consistent behavior, which gives us confidence that research done on small proxies will transfer to larger runs.

---

## What's next

Over the next few posts, I'll be covering:

1. **MoE Miniseries v1**: How we built the scientific training loop—speedruns for engineering validation, miniseries for scaling science
2. **Routing Geometry**: Why MoE routing doesn't just shatter, and how to test whether your router is healthy
3. **NVFP4 Dynamics**: How we turned "quantization noise" into a measurable, controllable feature

---

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437v2)
- [Inside Kaiju: Building Conversational Models at Scale](https://blog.character.ai/inside-kaiju-building-conversational-models-at-scale/)
- [Moonlight: A Compute-Efficient MoE Training Framework](https://arxiv.org/abs/2506.03524)
- [Seed-Coder Technical Report](https://arxiv.org/abs/2509.25149v1)
- [OLMo-3: The Best Fully Open Model of its Class](https://allenai.org/blog/olmo3)
