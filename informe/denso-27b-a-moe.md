# Dense-to-MoE Conversion at 27B Scale — Deep Research Report

**Date read:** 7 Nov 2026 (all web sources accessed 6–7 Nov 2026; model releases cited are as of that date)
**Scope note:** This is a follow-up to a previous report on 0.8–0.9B models (Komatsuzaki et al. arXiv:2212.05055; NVIDIA Nemotron upcycling arXiv:2410.07524v2; MoEfication; CMoE; ToMoE; LLaMA-MoE; BTX; Liew et al. arXiv:2502.03009v2). The question here is the same conversion, but starting from a ~27B dense LLM.

## Summary — answer first

**No published paper, repo, blog or released model converts a 20–35B dense model to MoE. That is the single most important finding, and it is stated explicitly.** The largest published dense→MoE upcycles are Nemotron-4 **15B** (NVIDIA, 1T tokens) and Skywork-MoE (13B dense → 146B total / 22B active). The scaling-law paper that would predict 27B behaviour (Liew et al.) only fitted models **up to 1B dense / 8x1B MoE and a few hundred B tokens**. Everything at 27B in this report is therefore **extrapolation or my arithmetic from verified parent architectures** — flagged as such throughout — not a measured 27B result.

What the evidence does support:

1. **Best parents are Gemma-2 27B (primary, cleanest case) and Gemma-3 27B, with Qwen2.5-32B / Qwen3-32B as the stronger but larger near-equivalents.** Gemma-2 27B: 46 layers, hidden 4,608, FFN intermediate 36,864 (GeGLU), 13T pretraining tokens, pretrained MMLU 75.2 / GSM8K 74.0 / HumanEval 51.8. Gemma-3 27B is stronger (MMLU 78.6, GSM8K 82.6) but deeper/narrower-FFN (62 layers, hidden 5,376, intermediate 21,504, 14T tokens) and multimodal. Qwen3-32B base is strongest of all (MMLU 83.6, GSM8K 93.4, 36T tokens).
2. **Copy-upcycling a 27B parent explodes total parameters 8×, to ~191B (Gemma-2), ~383 GB in bf16** — for either 8 experts top-2 or 64 fine-grained experts top-8 (Nemotron E8G8T8). Only split-FFN (MoEfication-style) keeps total parameters at ~27B. This memory blow-up, not FLOPs, is the binding constraint at 27B.
3. **Liew's threshold at N₁=27B is D\* ≈ 0.5–0.6B tokens** (0.62B with natural log, 0.48B with log₁₀) — i.e. on the paper's own (far-extrapolated) law, upcycling beats MoE-from-scratch only for upcycling budgets below ~0.6B tokens. **Do not misread this as "don't train past 0.6B tokens":** D\* is defined against MoE-from-scratch, not against continued dense training. Nemotron trained 1T tokens (≫ D\* at 15B, where D\*≈0.8B) and still beat continued dense training by +0.9 pp (iso-FLOP) to +2.3 pp (top-2) MMLU. The honest reading is that at 27B the *efficiency case vs. training a fresh MoE* is weak, while the case *vs. continued dense training on the same tokens* is the one Nemotron actually demonstrated.
4. **Expected gain at 27B, extrapolated from Nemotron 15B: +0.5 to +1.5 pp MMLU for an iso-FLOP upcycle, +1 to +2.5 pp for a 2×-FLOP top-2 upcycle, on ~100B–1T upcycling tokens** — with wide uncertainty and a real chance of ~0 if the upcycling data merely repeats the parent's pretraining distribution. Most of Nemotron's gain (+6.0 pp, 59.3→65.3) came from the extra 1T tokens themselves, not from MoE-ness.
5. **Yes, it requires continued pretraining — zero-training conversion collapses.** Random-split LLaMA-MoE-v2 training-free: MMLU 30.33, perplexity >10k. The best training-free method (CMoE analytical restructuring, Llama-2 7B, 25% sparsity) reaches MMLU 42.50 — still below its own fine-tuned 44.02 and below the dense parent. Upcycling proper needs LR reset to near the pretraining peak, an aux load-balancing loss (1e-3–1e-2), and virtual-group init + weight scaling for fine-grained experts.
6. **Cheapest viable path: split-FFN (ToMoE/CMoE-style) first as a near-free pilot, then — only if the pilot justifies it — a Nemotron-style E8G8T8 iso-FLOP upcycle on 100B tokens.** Do not start with 8×-copy top-2 at 27B: it costs ~2× the FLOPs and 8× the memory for, at 15B, only +1.4 pp over the iso-FLOP variant.

**Confidence:** High on parent architectures and on the *absence* of a 27B conversion (multiple targeted searches returned nothing above 15B dense→MoE). High on Nemotron/Liew/Komatsuzaki numbers (primary papers read in full). Medium on the 27B gain estimate (one anchor point, 15B). Low on any precise 27B benchmark — none exists.

---

## Findings

### 1) 27B dense parent models — architecture & baselines

There is no single canonical 27B parent; Gemma-2/3 27B are the only models actually named 27B. Qwen's nearest dense models are 32B. Llama has no 27B (8B → 70B); Mistral's nearest dense models are 24B (Mistral Small 3) and 123B (Mistral Large 2).

#### Table 1 — Parent models (pretrained/base variants unless marked IT)

| Model | Params (total / non-emb) | Layers | Hidden | FFN intermediate | Heads Q/KV | Vocab / tying | Pretrain tokens | Context |
|---|---|---|---|---|---|---|---|---|
| **Gemma-2 27B** (primary) | 27.2B / 26.05B | 46 | 4,608 | 36,864 (GeGLU) | 32 / 16, head 128 | 256,000, tied | **13T** | 8K |
| **Gemma-3 27B** | 27.0B / 25.6B | 62 | 5,376 | 21,504 (GeGLU) | 32 / 16, head 128 | 262,144, tied* | **14T** | 128K |
| Qwen2.5-32B | 32.5B / 31.0B | 64 | 5,120 | 27,648 (SwiGLU) | 40 / 8, head 128 | 151,936, untied | **18T** | 128K |
| Qwen3-32B | 32.8B / 31.2B | 64 | 5,120 | 25,600 (SwiGLU) | 64 / 8, head 128 | 151,936, untied | **36T** | 128K |
| Llama-3 70B (reference) | 70.6B | 80 | 8,192 | 28,672 | 64 / 8 | 128,256 | ~15T | 128K |
| Mistral Large 2 (reference) | 123B | —† | —† | —† | —† | 32,768 | —† | 128K |

\* Gemma-3 27B also carries a ~0.4B SigLIP vision encoder (frozen during language training); language-model totals above exclude it. † Mistral Large 2 detailed architecture/tokens were not verified from a primary source in this run — do not use the blanks as data.

**Sources & verification notes (index + primary papers):**
- Gemma-2 27B: 46 layers, hidden 4,608, GeGLU FFN, 13T tokens, GQA 32/16 — Gemma 2 technical report arXiv:2408.00118, via https://arxiv.org/html/2408.00118v1 and https://gemma.ai/tutorials/gemma-explained/gemma-2/ (gate/up/down each 4,608→36,864). **Discrepancy resolved:** some secondary pages quote "GeGLU feed-forward of 73,728" — that is gate+up combined (2×36,864); the per-matrix intermediate size in the HF config is 36,864. Non-embedding 26,047,480,320 + embedding 1,180,237,824 = 27.23B total, per the report's parameter table.
- Gemma-3 27B: 62 layers, hidden 5,376, intermediate 21,504, 32Q/16KV heads — HF config reproduced at https://gemma.ai/tutorials/gemma-explained/gemma-3/ (gate/up/down 5,376→21,504) and https://github.com/iopsystems/llm-calc/blob/HEAD/docs/superpowers/specs/2026-05-12-hybrid-attention-design.md; 14T tokens and 5:1 local:global attention (1,024-token sliding window) from the Gemma 3 report arXiv:2503.19786, https://export.arxiv.org/pdf/2503.19786v1.pdf. Non-embedding 25.6B matches the report's Table 1.
- Qwen2.5-32B: 64 layers, 40Q/8KV heads, 18T tokens — Qwen2.5 report, https://arxiv.org/html/2412.15115v2; hidden 5,120 / intermediate 27,648 from config tables summarised at https://huggingface.co/Qwen/Qwen2.5-32B-Instruct (32.5B total, 31.0B non-embedding, 64 layers).
- Qwen3-32B: 64 layers, hidden 5,120, intermediate 25,600, 64Q/8KV, head 128 — https://github.com/zsxh1990/llmservingsim/blob/HEAD/docs/docs/reference/model-config.md and https://arxiv.org/pdf/2510.07486; 36T tokens, Qwen3 report arXiv:2505.09388, https://ar5iv.labs.arxiv.org/html/2505.09388.
- New exact-27B dense parent (2026, secondary only): Qwen3.8-27B (Aug 2026, dense 27.8B, hidden 5,120, 64 layers — 48 linear-attention + 16 full-attention, hybrid) — https://rohitghumare.com/guides/qwen3-8-27b/. Benchmarks not verified; treat as a candidate parent to re-check, not as a baseline.

#### Table 2 — Pretrained baselines (vendor-reported; do not mix shots/protocols across rows)

| Model (base/PT) | MMLU (5-shot) | GSM8K | HumanEval (0-shot) | ARC-C | HellaSwag (10-shot) |
|---|---|---|---|---|---|
| Gemma-2 27B PT | **75.2** | **74.0** (5-shot, maj@1) | **51.8** | **71.4** (25-shot) | **86.4** |
| Gemma-3 27B PT | **78.6** | **82.6** (8-shot)† | **48.8** | n/v | n/v |
| Qwen2.5-32B Base | **83.32** | **92.87** (4-shot CoT) | 66.25 EvalPlus‡ | n/v | n/v |
| Qwen3-32B Base | **83.61** | **93.40** (4-shot CoT) | 72.05 EvalPlus‡ | n/v | n/v |
| Qwen2.5-72B Base | 86.06 | 91.50 | 65.93 EvalPlus‡ | n/v | n/v |
| Llama-3 70B PT | 79.2 | 76.9 | — | 68.8 | 88.0 |

† Gemma-3 GSM8K discrepancy: the Gemma 3 model card reports 82.6 (8-shot), while the Qwen3 report's Table 4 gives 81.20 for Gemma-3-27B (different shot/CoT protocol). Both are cited; use 81–83 as a range, not a point. ‡ EvalPlus is the average of HumanEval/MBPP/HumanEval+/MBPP+, not HumanEval alone. n/v = not verified in this run (Qwen reports do not publish ARC/HellaSwag for these bases in the tables consulted; Gemma-3 PT ARC/HellaSwag likewise not found — do not fill from IT numbers).

**Sources:** Gemma-2 Table 12/13, arXiv:2408.00118 (https://arxiv.org/html/2408.00118v1); Gemma-3 model card (https://huggingface.co/lex-au/Google.Gemma-3-27b-pt-GGUF and https://hub.docker.com/r/ai/gemma3 — MMLU 78.6, GSM8K 82.6, HumanEval 48.8, 14T tokens); Qwen3 report Tables 4–5, arXiv:2505.09388 (https://arxiv.org/pdf/arXiv:2505.09388 — Qwen2.5-32B 83.32/92.87/66.25, Qwen2.5-72B 86.06/91.50/65.93, Gemma-3-27B 78.69/81.20/55.78, Qwen3-32B 83.61/93.40/72.05, Qwen3-30B-A3B 81.38/91.81/71.45).

**Which parent to pick:** Gemma-2 27B is the cleanest upcycling parent — pure text, trained from scratch (not distilled; the 2B/9B were distilled from it), 8K context, GeGLU FFN, and the weakest baseline, so headroom is largest. Gemma-3 27B is stronger but adds a vision tower and 62-layer depth (more layers = more routers, more expert-parallel communication). Qwen3-32B is the strongest parent but is already trained on 36T tokens — the most "ossified" by Liew's interaction term — and is 32B, not 27B. If the goal is *demonstrating* conversion, use Gemma-2 27B. If the goal is *best final model*, use Qwen3-32B and accept a smaller expected delta.

### 2) Published conversions at 20–35B scale — none at 27B

**Explicit finding: no paper, repo, blog or released model was found that converts a 20–35B dense model to MoE.** Targeted searches for "upcycl* Gemma 27B / Qwen 32B / Llama 70B", "dense converted to MoE 32B", and the general literature all return the same ceiling: 15B. The closest scales, in descending order of relevance:

| Work | Dense parent | Result | Tokens | What it proves at 27B |
|---|---|---|---|---|
| **NVIDIA Nemotron upcycling** (He et al., arXiv:2410.07524v2) | Nemotron-4 **15B** (8T pretrain) | E8G8T8: MMLU 66.2; E8G1T2: **67.6**; dense-continued: 65.3; base 59.3 | **1T** | Largest *dense→MoE* upcycle published; the only >10B anchor. §3 below. |
| **Skywork-MoE** (arXiv:2406.06563) | Skywork-**13B** (3.2T + 2T extra) | 146B total / 22B active, 16 experts; MMLU 77.4, GSM8K 76.1, HumanEval 43.9 (vs 13B dense 62.7/60.2/18.9) | Condensed SkyPile subset, token count not stated as a single number | Largest total-parameter upcycle; shows copy-upcycling works at 13B with large gains — but parent was much weaker/less trained than Gemma/Qwen. |
| **Komatsuzaki et al.** (arXiv:2212.05055) | T5 Base/Large/XL (~0.25–0.9B) | +1.5–2 pts SuperGLUE at 46–55% extra training; effective at +10–60% of sunk cost | ~50% of pretraining compute | Foundational method; scale is 30× smaller than 27B. |
| **Liew et al.** (arXiv:2502.03009v2) | Dense **15M–1B** | Scaling laws; D\* formula (§3) | Up to few hundred B | Only quantitative predictor for 27B; fitted range ends at 1B. |
| **LLaMA-MoE v1** (Zhu et al., 2024) | Llama-2 **7B** | Split FFN → 3.5B-active MoE; 200B tokens continual pretraining | 200B | Split recipe at 7B. |
| **LLaMA-MoE v2** (Qu et al., 2024) | Llama-3 **8B** | Split FFN + attention MoE; two-stage post-training | **7B** | Cheapest split recipe; §4. |
| **BTX** (Sukhbaatar et al., arXiv:2403.07816) | Llama-2 **7B** seed | 4 domain experts (math/code/wiki + seed); MoE fine-tune after branch training | Branch + short MoE fine-tune (~2% of pretraining compute for router stage, per secondary summary) | Alternative: domain-specialised experts, not copies of one FFN. |
| **Recycling / Orthogonal Growth** (Wang et al., arXiv:2510.08008) | **17B MoE → 70B MoE** (not dense→MoE) | 17B→35B (depth) →70B (width, 96→192 experts); +10.66% avg accuracy vs from-scratch at same extra FLOPs; 1T tokens | 600B + 300B + 100B | Closest *scale* proof that checkpoint-recycling works at 70B/1T — but the parent was already MoE, so it does not validate dense→MoE at 27B. |
| **MoE-from-scratch references** | — | Mixtral 8x22B (141B/39B act); Qwen3-30B-A3B (30.5B/3.3B act); DeepSeek-V3 (671B/37B act); **Gemma-4 26B-A4B (25.2B/3.8B act, 128 experts top-8 + 1 shared, 30 layers)** | — | Gemma-4 26B-A4B is the native-MoE model *at* 26B scale: LMArena 1441 vs 1452 for the dense Gemma-4 31B — i.e. Google's own answer at this scale was to train a fine-grained MoE from scratch, not to upcycle Gemma-3 27B. Treat as the strongest existence proof that a 26B-scale fine-grained MoE is viable, and as a benchmark for what an upcycle would have to beat. |

**Sources:** Nemotron https://arxiv.org/html/2410.07524v1; Skywork-MoE https://arxiv.org/html/2406.06563v1/; Komatsuzaki https://arxiv.org/abs/2212.05055; Liew https://arxiv.org/html/2502.03009v2; LLaMA-MoE https://github.com/pjlab-sys4nlp/llama-moe (via https://pub-fad10fba280c45dba0823fd585acc08c.r2.dev/pdf/2406.16554.pdf?v=1789095365); BTX https://ar5iv.labs.arxiv.org/html/2403.07816; Recycling https://arxiv.org/pdf/2510.08008; Gemma-4 26B-A4B https://llmcheck.net/blog/qwen-36-vs-gemma-4-deep-technical-comparison/ and https://dev.to/pulkitgovrani/gemma-4-26b-a4b-what-mixture-of-experts-actually-means-for-your-inference-budget-13hj (secondary; no full Gemma-4 technical report was available at time of writing — flag the LMArena numbers as secondary).

### 3) Concrete statistics — total vs active, tokens, FLOPs, GPU-hours, memory

All 27B numbers in this section are **CALCULATED** (my arithmetic from the verified configs in §1, using the standard 6·N·D training-FLOP rule and H100 dense bf16 peak 989 TFLOPS at 40% MFU — Llama-3 reported 38–43% MFU, https://www.aicassindra.com/blogs/gpu/flops_budget_training.html). They are not measured. The method is stated so the requesting agent can recompute; full working is in `notes/calculations.md`.

#### Table 3 — MoE configurations from a Gemma-2 27B parent (FFN 23.44B, attention 2.60B, tied embedding 1.18B; total 27.2B)

| Configuration | Experts × size | Total params | Active params (non-emb) | Active/total | Weights bf16 | Forward FLOPs/token |
|---|---|---|---|---|---|---|
| Dense (parent) | 1 × full FFN | 27.2B | 26.0B | 100% | 54 GB | ~52 GFLOP |
| **8E top-2, copy** (E8G1T2) | 8 × full | **191.3B** | **49.5B** | 26% | **383 GB** | ~99 GFLOP |
| **8E top-1, copy** (E8G1T1) | 8 × full | **191.3B** | **26.0B** | 14% | **383 GB** | ~52 GFLOP |
| **64E top-8, 1/8 size** (E8G8T8, Nemotron fine-grained) | 64 × 1/8 | **191.3B** | **26.0B** | 14% | **383 GB** | ~52 GFLOP |
| **64E top-8, split** (each 1/64) | 64 × 1/64 | 27.2B | 5.5B | 20% | 54 GB | ~11 GFLOP |
| **Split 8E top-2** (each 1/8) | 8 × 1/8 | 27.2B | 8.5B | 31% | 54 GB | ~17 GFLOP |
| **Split 8E top-4** (each 1/8) | 8 × 1/8 | 27.2B | 14.3B | 53% | 54 GB | ~29 GFLOP |

Notes: (a) "64 fine-grained experts top-8" is ambiguous in the mission — both readings are shown. The Nemotron reading (64 experts of 1/8 size, top-8) is iso-FLOP/iso-active with 8× total params; the MoEfication/split reading (64 shards of 1/64, top-8) keeps total params at 27B and activates only 12.5% of the FFN. Do not quote one without saying which. (b) Router params are negligible (<0.1B) and omitted. (c) Active params here are non-embedding, following Liew's N₂ definition; adding the tied embedding/lm_head changes totals by ~1.2B. (d) Same shapes for other parents, scaled by FFN size: Gemma-3 27B copy-upcycle → ~178B total (FFN 21.5B×8 + 6.0B non-FFN), top-2 active ~47B; Qwen3-32B copy-upcycle → ~209B total, top-2 active ~56B; Qwen2.5-32B copy-upcycle → ~225B total (FFN 27.2B×8), top-2 active ~58B. [CALCULATED]

**Memory (weights only, bf16):** dense 27B = 54 GB (1× H100 80GB); any copy/fine-grained upcycle = 383 GB (5× H100 80GB just for weights, before KV cache/activations); split = 54 GB. At int8 the copy-upcycle is ~191 GB; at int4 ~96 GB. **Training states** (bf16 weights + grads + fp32 master + Adam m/v ≈ 16 bytes/param): dense ~435 GB vs copy-upcycle ~3.1 TB — ZeRO-sharded across ≥40 H100s for states alone, vs ~6 for dense. Expert parallelism (Megatron-Core EP) is mandatory for the copy path; the split path fits the parent's existing parallelism. [CALCULATED]

#### Table 4 — Upcycling tokens, FLOPs, GPU-hours (Gemma-2 27B parent; H100, 40% MFU, $2.50/GPU-h illustrative)

| Run | Active N | Tokens D₂ | Train FLOPs (6·N·D) | H100-hours | H100-days | Cost @ $2.50 | % of Gemma-2 pretrain FLOPs* |
|---|---|---|---|---|---|---|---|
| D\* threshold (Liew) | 26B | **0.62B** | 9.6e19 | ~67 | ~3 | ~$170 | 0.005% |
| Pilot: split 25%, 10B tok | 8.5B | 10B | 5.1e20 | ~360 | ~15 | ~$900 | 0.02% |
| Split 25%, 100B tok | 8.5B | 100B | 5.1e21 | ~3,600 | ~150 | ~$9k | 0.24% |
| Iso-FLOP (E8G8T8), 100B tok | 26B | 100B | 1.6e22 | ~11,000 | ~460 | ~$28k | 0.77% |
| Iso-FLOP, **1T tok** (Nemotron scale) | 26B | 1T | 1.6e23 | ~115,000 | ~4,800 | ~$287k | 7.7% |
| Top-2 (E8G1T2), 100B tok | 49.5B | 100B | 3.0e22 | ~21,000 | ~870 | ~$52k | 1.4% |
| Top-2, **1T tok** | 49.5B | 1T | 3.0e23 | ~209,000 | ~8,700 | ~$521k | 14% |
| Dense-continued, 1T tok (control) | 26B | 1T | 1.6e23 | ~115,000 | ~4,800 | ~$287k | 7.7% |

\* Gemma-2 pretraining ≈ 6·27.2e9·13e12 ≈ 2.1e24 FLOPs. For Qwen2.5-32B (≈3.5e24) the same 100B-token iso run is 0.55% of sunk cost; for Qwen3-32B (36T tokens, ≈7.0e24) it is 0.28%. Wall-clock: divide H100-days by GPU count (e.g. the 1T iso run is ~19 days on 256 H100s, ~75 days on 64). At 30% MFU multiply hours by 1.33; at 50% by 0.80. These exclude evals, restarts, ablations and data work — budget 1.5–2× for a real project. [CALCULATED; MFU/price assumptions from https://www.aicassindra.com/blogs/gpu/flops_budget_training.html and https://www.aiagentichandbook.com/terms/pretraining/]

#### The Liew D\* calculation at 27B — worked, with caveats

Formula (Liew et al. Eq. 2, https://arxiv.org/html/2502.03009v2): D\* ≃ 4·(N₁/10⁹)^(−0.7+0.04·log(N₁/10⁹)) B tokens, where N₁ is non-embedding dense size and D\* is the token budget at which MoE-from-scratch catches up with upcycling (Eq. 13: L^MoE(D\*) = L^Upc.(D₁=D\*, D₂=D\*)).

- N₁=1B → D\* = 4.0B (paper's own check: "D\* equal to 4B tokens for an 8x1B model").
- N₁=15B → D\* ≈ 0.81B. Nemotron nevertheless trained 1T tokens — 1,200× D\* — and still beat dense-continued training. This is not a contradiction: D\* compares against *from-scratch MoE*, a comparison Nemotron never ran.
- **N₁=27B → D\* ≈ 0.62B tokens (natural log) or 0.48B (log₁₀).** For Gemma-2 non-embedding N₁=26.05B, D\*≈0.63B. Range to quote: **~0.5–0.6B tokens**.
- Liew also finds compute-optimal upcycling tokens scale as D₂ ∝ N₁^1.8 (§5.2) — nearly quadratically in model size — and that "upcycling is inefficient relative to from-scratch trainings when considering compute optimality" if no pretrained model exists. Both statements are fitted at ≤1B; applying them at 27B is a 27× extrapolation in N₁ and ~35× in D₁ (13T vs the 368B-token SlimPajama-DC pool Liew trained on). **Flag any 27B D\* claim as [EXTRAPOLATED — 27× beyond fitted range].**

#### How many tokens are actually needed? Three different answers — do not conflate them

1. **To beat MoE-from-scratch (Liew):** ≲0.6B tokens. Practically irrelevant at 27B: nobody will train a fresh 191B-total MoE for 0.6B tokens, and the law is unvalidated here.
2. **To beat continued dense training (Nemotron, measured at 15B):** 1T tokens gave +0.9 pp (iso-FLOP) / +2.3 pp (top-2) MMLU over dense-continued on the same 1T tokens; the gain *grew* with token count ("longer token horizons favor MoE models"). Nemotron's ablations used 0.1T (10% of pretraining) and the final run 1T; at 0.1T the granular variant already beat the coarse one. A reasonable 27B plan is therefore **100B tokens minimum to see anything, 1T to replicate Nemotron** — i.e. 0.8–7.7% of Gemma-2's pretraining FLOPs for the iso-FLOP variant.
3. **To beat the parent at all (Komatsuzaki, measured at ≤0.9B):** ~50% of pretraining compute (≈6.5T tokens for Gemma-2) — **do not apply this at 27B**. It would cost ~1e24 FLOPs and contradicts Nemotron, where 1T tokens (12.5% of the 15B parent's 8T pretraining tokens) already produced a clear win over dense-continued. The Komatsuzaki % rule reflects small, undertrained parents; Gemma/Qwen parents are 24–70× Chinchilla-optimal in tokens/param (13–36T tokens for 27–32B params vs ~20 tokens/param optimal), so percentages of sunk cost are meaningless — absolute tokens and data freshness matter.

### 4) Does it require fine-tuning / continued pretraining? Yes — and how

**Short answer: for copy-upcycling, yes — full continued pretraining of all weights, with LR reset to near the pretraining peak. For split/MoEfication, the spectrum runs from zero training (broken) through router-only training (cheap, lossy) to full continual pretraining (200B tokens at 7B). Nothing in the literature suggests a 27B copy-upcycle works without training.**

#### Table 5 — Training requirements by method

| Method | What trains | What is frozen | Tokens (measured) | LR | Router init | Aux loss |
|---|---|---|---|---|---|---|
| **Nemotron copy-upcycle** (15B) | **Everything** (all experts, attention, embeddings, router) | Nothing | 0.1T ablations; **1T final** | **Reset: init at pretraining min-LR 4.5e-4, peak 3e-4, cosine to 1/100 of min-LR** (2B ablation: peak 1e-4–2e-4 = pretraining peak, cosine to 2e-5; constant min-LR 2e-5 plateaus and underperforms) | **Random** for coarse; **virtual-group** for fine-grained (see below) | **1e-2** (best range 1e-3–1e-2); no aux → dead experts, early plateau; z-loss (1e-3) no effect |
| **Komatsuzaki sparse upcycling** | Everything | Nothing | ~50% of pretraining compute | Pretraining LR schedule restarted | Random (Expert Choice routing in original) | — |
| **Liew et al.** | Everything | Nothing | Up to few hundred B | **MoE LR (not dense LR, not constant end-of-dense LR)** works best; router init N(0, 2/5·d_model) | Random | 1e-3 (ablated) |
| **LLaMA-MoE v1** (split) | Everything (split experts + gates) | Nothing | **200B** | Continual-pretraining LR | Random split + rescaling | — |
| **LLaMA-MoE v2** (split) | Everything, two-stage (continual pretrain + post-train) | Nothing | **7B** | Post-training LR | Importance-based neuron partition; router from hidden features | — |
| **ToMoE** (split/prune) | **Only router + small projections + hypernetwork (0.27% of params)** | **All original weights frozen** | **0.02B** (10k iters) | Router-only training | Learned via dynamic structural pruning (differentiable masks) | Sparsity regulariser λ=16 |
| **CMoE analytical** (split) | Nothing (training-free) or LoRA on 2k samples | Everything (training-free) | 0 (or 4M tokens = 2k×2,048) | — | **Analytical, from activation statistics** (no learned router) | — |
| **BTX** | Branch models independently, then all (MoE fine-tune) | Nothing in fine-tune stage | Branch training + short MoE fine-tune | Branch LR, then fine-tune LR | Learned in MoE fine-tune | — |

**Sources:** Nemotron §§3.2–3.3, §4, https://arxiv.org/html/2410.07524v1 (LR reset Fig. 5–6: cosine similarity to base drops to 0.6–0.7 with reset LR vs ~1.0 with constant min-LR — reset LR is what lets experts diversify; batch size 4M tokens (1024×4,096) beat 2M and 32M; aux 1e-2, no z-loss; 1T run: init LR = pretraining min-LR 4.5e-4, peak 3e-4, cosine to 1/100 min-LR; 0.3 yottaFLOPs for the 1T programme); Liew App. A.6, https://arxiv.org/html/2502.03009v2 (MoE LR best; router N(0, 2/5d)); LLaMA-MoE-v1/v2 token counts via CMoE Table 6 (200B/weeks; 7B/days) at https://arxiv.org/pdf/2502.04416 and https://pub-fad10fba280c45dba0823fd585acc08c.r2.dev/pdf/2406.16554.pdf?v=1789095365; ToMoE https://arxiv.org/html/2501.15316v2 (via https://github.com/gaosh/ToMoE and summary at https://lemmy.world/post/52213044/25965870 — 0.02B tokens, 0.27% added params, frozen weights, 10k iterations on 1–4 A100s for 7B); CMoE https://arxiv.org/html/2502.04416v2.

#### Router initialisation — the detail that makes or breaks fine-grained upcycling

- **Coarse (copy) upcycling:** duplicate the FFN into each expert; router randomly initialised. With topK-then-softmax the model is *functionally identical* to dense at step 0 (probabilities sum to 1); with softmax-then-topK it is not, but Nemotron found softmax-then-topK trains better anyway, provided weight scaling compensates (below). Mixtral uses topK-then-softmax; Nemotron recommends softmax-then-topK for upcycling.
- **Fine-grained (E8G8T8) upcycling — virtual-group init + weight scaling (Nemotron §2.3):** shard the dense MLP into G shards along the intermediate dimension, replicate each shard E times (8 shards × 8 replicas = 64 experts). Initialise router weights so each "virtual group" (replicas of one shard) shares identical router weights — guaranteeing exactly one replica of every shard is in the top-K at step 0, so the MoE reproduces the dense output. Then scale each expert's W₁ and W₂ by ∛(E·G²/T) (Eq. 2; for E8G8T8 the factor is 4 — empirically best among 2×/2.5×/4×). Weight scaling alone gives ~1.5% better loss for both coarse and fine-grained upcycles. Naive sharding without this "loss was very high and the network did not converge to the original loss."
- **SwiGLU/GeGLU splitting hazard (applies to all 27B parents — all are gated FFNs):** gate, up and down projections must be split on the *same* intermediate index: expert e owns rows [gate_e, up_e] and columns [down_e] for one index set. Splitting gate and up independently (or permuting one without the other) destroys the learned gate×up product. LLaMA-MoE's random (IndependentRandom) split respects this alignment and was best of its four methods; clustering-based splits (MoEfication, CMoE activation clustering) also operate per-neuron across all three matrices. Any 27B implementation must assert this alignment in code — it is the most likely silent bug.
- **What is frozen:** in copy-upcycling, nothing. In ToMoE/CMoE-analytical, everything except (ToMoE) the router/projections/hypernetwork. A middle option not in the literature at 27B — freeze attention + embeddings and train only experts + router — would cut optimizer states by ~10% (attention is only ~10% of non-embedding params in Gemma-2) and is not worth the risk; Nemotron's cosine-similarity result argues the whole network needs to move to escape the dense minimum.

#### Zero training — measured reference points (all at 7B, Llama-2, 25% sparsity unless noted)

| Method / regime | MMLU (5-shot) | WikiText-2 PPL | C4 PPL | Verdict |
|---|---|---|---|---|
| Dense Llama-2 7B (parent) | ~45–46 (context; CMoE dense PPL 5.27 Wiki / 7.27 C4) | 5.27 | 7.27 | — |
| LLaMA-MoE-v2, **training-free** | **30.33** | **>10,000** | **>7,000** | Collapsed. Random-split experts + untrained router destroy the model. |
| LLaMA-MoE-v2, fine-tuned (7B tok) | 38.02 | 8.68 | 19.76 | Recovers only partially. |
| CMoE (analytical), **training-free** | **42.50** | **7.32** | 11.98 | Best training-free result; still below dense parent and below its own fine-tuned version. Shared (always-on) experts for frequent neurons + routed experts for sparse ones + analytical router. |
| CMoE, fine-tuned (2k samples, 4M tok) | **44.02** | 5.92 | 11.21 | Near-dense; most gain is from the restructuring, not the fine-tune. |
| ToMoE, no weight fine-tune (router trained, 0.02B tok) | — (PPL 6.41 at 50% on Llama-2 7B vs dense 5.12) | — | — | Usable but lossy; best of the "no weight update" family. |

**Sources:** CMoE Tables 2–3, https://arxiv.org/pdf/2502.04416 (LLaMA-MoE-v2 training-free 30.33 / >10k / >7k; CMoE training-free 42.50 / 7.32 / 11.98; CMoE fine-tuned 44.02; LLaMA-MoE-v1 200B / v2 7B token budgets, Table 6). **Extrapolation warning:** these are 7B numbers. At 27B a training-free split would plausibly degrade *less* in relative terms (more redundancy) or *more* (router errors compound over 46–64 layers) — no data exists; do not quote 42.50 as a 27B expectation. The qualitative conclusion transfers: **zero-training copy-upcycling is not viable; zero-training split is a latency play that costs several MMLU points.**

### 5) Comparison table — 27B dense vs 27B upcycled MoE vs MoE-from-scratch

**Read this table as: measured rows are base-model vendor numbers; the "27B upcycled MoE" row is EXTRAPOLATED (no such model exists).** Benchmark protocols differ across vendors (shots, CoT, EvalPlus vs HumanEval) — the table preserves each source's protocol in the notes rather than forcing comparability.

#### Table 6 — Base models (pretrained, not instruction-tuned)

| Model | Arch | Total / Active | MMLU | GSM8K | HumanEval / EvalPlus | ARC-C | HellaSwag |
|---|---|---|---|---|---|---|---|
| Gemma-2 27B PT (parent) | Dense | 27B / 27B | 75.2 | 74.0 | 51.8 HE | 71.4 | 86.4 |
| Gemma-2 27B **continued, 1T tok** [EXTRAPOLATED from Nemotron ratio] | Dense | 27B / 27B | ~76–78? | ~76–80? | ~53–56? | ~72–74? | ~87? |
| Gemma-2 27B **upcycled E8G8T8, 1T tok** [EXTRAPOLATED] | MoE 64E top-8 | 191B / 26B | **~77–79?** (+0.9pp over continued, Nemotron delta) | ~77–81? | ~54–57? | ~73–75? | ~87–88? |
| Gemma-2 27B **upcycled E8G1T2, 1T tok** [EXTRAPOLATED] | MoE 8E top-2 | 191B / 50B | **~78–80?** (+2.3pp over continued) | ~78–82? | ~55–59? | ~74–76? | ~87–88? |
| Gemma-3 27B PT | Dense | 27B / 27B | 78.6 | 82.6† | 48.8 HE | n/v | n/v |
| Qwen2.5-32B Base | Dense | 32B / 32B | 83.32 | 92.87 | 66.25 EvalPlus | n/v | n/v |
| Qwen3-32B Base | Dense | 32B / 32B | 83.61 | 93.40 | 72.05 EvalPlus | n/v | n/v |
| Mixtral 8x22B Base (MoE-from-scratch) | MoE 8E top-2 | 141B / 39B | 77.6 | 80.3 | 53.1 HE | n/v | n/v |
| Qwen3-30B-A3B Base (MoE-from-scratch) | MoE 128E top-8 | 30.5B / 3.3B | 81.38 | 91.81 | 71.45 EvalPlus | n/v | n/v |
| DeepSeek-V3 Base (MoE-from-scratch) | MoE 256E top-8+1 shared | 671B / 37B | 87.1 | 89.3 | 65.2 HE | 95.3 | 88.9 |
| Gemma-4 26B-A4B **IT** (MoE-from-scratch; NOT base) | MoE 128E top-8+1 shared | 25.2B / 3.8B | 82.6 MMLU-Pro‡ | 88.4‡ | ~87‡ | n/v | n/v |

†/‡ See §1 notes: Gemma-3 GSM8K 81.20 in Qwen3's table; Gemma-4 numbers are instruction-tuned and MMLU-Pro, from secondary summaries — included only to show the native-MoE-at-26B ceiling, not as a like-for-like row. **Sources:** as §1–2, plus Mixtral 8x22B / DeepSeek-V3 base numbers from the DeepSeek-V2 analysis table (https://github.com/viktor-shcherb/msc_thesis/blob/HEAD/references/2024-05-deepseek-v2-moe/analysis.md — Mixtral 8x22B 77.6/80.3/53.1) and DeepSeek-V3 README/tech report (https://github.com/Shyboy0499/DeepSeek-V3/blob/HEAD/README.md — 671B/37B, MMLU 87.1, GSM8K 89.3, HumanEval 65.2, ARC-C 95.3, HellaSwag 88.9, 14.8T pretraining tokens); Qwen3-30B-A3B from Qwen3 Table 5 (https://arxiv.org/pdf/arXiv:2505.09388).

**How to read the extrapolated rows:** the only *measured* MoE-vs-dense-continued deltas are Nemotron-15B's: val loss −4.1% (E8G8T8) / −5.2% (E8G1T2) and MMLU +0.9 / +2.3 pp on 1T tokens (Table 1, https://arxiv.org/html/2410.07524v1). I applied those pp deltas to a guessed continued-dense Gemma-2 baseline (+1–3 pp over the 75.2 parent, by analogy with Nemotron's +6.0 pp continued gain — itself likely optimistic for Gemma-2, which at 13T tokens is far more saturated than Nemotron-4 15B was at 8T). The "?" marks are deliberate: **the honest 27B forecast is a range, not a point: iso-FLOP upcycle ≈ parent +1 to +4 pp MMLU after 1T tokens (of which most is the tokens, ~+1 pp is the MoE); top-2 ≈ parent +2 to +5 pp.** GSM8K/HumanEval/ARC/HellaSwag extrapolated cells are proportional guesses with no anchor — treat as illustrative only.

**Inference speed / cost (measured principles + calculated):**
- **Decode is memory-bandwidth bound at batch 1:** per-token cost tracks *active* params, not total. Iso-FLOP upcycle ≈ dense speed (26B active); top-2 ≈ 2× slower per token at batch 1 (~99 vs ~52 GFLOP/token forward, and ~99 GB vs ~54 GB of weights read per token in bf16); split-25% ≈ 3× *faster* than dense per token (~17 GFLOP/token). CMoE measured at 7B: 25% activation → 1.5× end-to-end latency reduction, 75% activation → ~lossless PPL with 5% acceleration (https://arxiv.org/html/2502.04416v2).
- **At large batch, the active-param advantage evaporates:** with 64 concurrent sequences, near-all 128 experts are touched per layer (Qwen3-30B-A3B analysis, https://www.aicassindra.com/blogs/llm/llm_moe_serving.html) — decode bandwidth then tracks *total* params. A 191B-total upcycle therefore needs ~7× the HBM bandwidth provisioning of the dense parent at serving scale, even though batch-1 latency looks iso-FLOP.
- **Memory/residency is set by total params:** 383 GB bf16 for any copy-upcycle (5× H100) vs 54 GB dense; split stays at 54 GB. Quantisation (int8/int4) compresses both proportionally.
- **Throughput reference (measured, 7B):** ToMoE at batch 1536: 2,919 tok/s vs 1,858 dense (+57%) (secondary summary of arXiv:2501.15316). No 27B throughput measurement exists.
- **Cost comparison vs from-scratch MoE:** Qwen3-30B-A3B (from scratch, 36T tokens) reaches MMLU 81.38 with 3.3B active — *above* Gemma-2 27B's 75.2 and above even the optimistic top-2 upcycle forecast (~78–80), while needing only 61 GB bf16 residency. This is the strongest argument against upcycling Gemma-2 specifically: a newer from-scratch fine-grained MoE already beats the forecast upcycle at one-sixth the residency. Upcycling Qwen3-32B (83.61 base) has a higher ceiling but a lower expected delta.

### 6) Practical recommendation for a 27B conversion

#### Cheapest viable path (staged, with kill criteria)

**Stage 0 — Pilot (days, ~$1–5k): split, don't copy.** Take Gemma-2 27B (base, not IT). Apply CMoE analytical restructuring (https://github.com/JarvisPei/CMoE) or ToMoE (https://github.com/gaosh/ToMoE) at 50% activation (top-1 over 2-way split per layer, or 25% if latency is the goal). Cost: minutes (CMoE) to ~10k iterations on a few GPUs (ToMoE, 0.02B tokens). Measure MMLU/GSM8K/HumanEval + WikiText-2 PPL vs dense. **Stop rule:** if training-free MMLU drops >5 pp from 75.2, or PPL more than doubles, the parent's FFN is not cleanly separable — do not proceed to copy-upcycling on hope; the router is telling you expert diversity is low.

**Stage 1 — Cheap continued training (1–2 weeks, ~$10–30k): split + light fine-tune.** If Stage 0 is within ~3 pp, fine-tune the split model (CMoE-style, 2k–100k samples; or LLaMA-MoE-v2-style, up to 7B tokens at 27B ≈ 7B × 6 × 8.5e9 ≈ 3.6e20 FLOPs ≈ 250 GPU-hours for the 25%-active model — trivial). This is the best expected-value play: near-dense quality at 25–50% of dense inference FLOPs, total params unchanged, no memory blow-up. **Expected gain: −2 to +0 pp MMLU vs dense parent, but 2–3× cheaper inference.** If the goal is *quality gain*, this stage will not deliver it — say so explicitly and stop here if inference cost was the goal.

**Stage 2 — Full upcycle, only if quality gain is the goal AND Stage 1 passed (3–6 weeks, ~$30–300k): Nemotron E8G8T8 iso-FLOP.** Copy-upcycle Gemma-2 27B with virtual-group init + ∛(E·G²/T)=4× weight scaling, softmax-then-topK routing, aux loss 1e-2, LR reset (peak ≈ pretraining peak; for Gemma-2 the pretraining peak LR is not public — start at 2e-4, cosine to 2e-5, by analogy with Nemotron-2B's 2e-4 peak; run a 10B-token LR sweep at 1e-4 vs 2e-4 vs 3e-4 before committing), batch 4M tokens, 100B tokens of *fresh or re-blended* data (not a pure repeat of pretraining data — Nemotron used a different continued-training blend). Use Megatron-LM/Megatron-Core `--moe-use-upcycling --moe-upcycling-granularity 8` (https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/moe.html) with expert parallelism EP=8. **Expected gain: +0.5 to +1.5 pp MMLU over a dense-continued control on the same 100B tokens** (Nemotron's +0.9 pp at 1T, scaled down for fewer tokens and a more saturated parent). **Stop rule:** at 25B tokens, compare val loss vs the dense-continued control run in parallel on identical data; if the upcycle is not ahead by ≥0.5% relative val loss, kill it — Liew's interaction term says the gap will not open up later, and the remaining 75B tokens are better spent on the dense control. If it is ahead, continue to 100B, then re-evaluate; extend to 1T only with a fresh budget decision (×10 cost for, at 15B, roughly +1 pp more).

**Do not do:** 8E top-2 copy as a first attempt (2× FLOPs, same 383 GB, +1.4 pp over iso-FLOP at 15B); training-free copy-upcycling (guaranteed collapse per §4); upcycling the IT model (router training will fight the instruction tuning — upcycle the base model, then re-run SFT/RLHF); upcycling Gemma-3 27B first (vision tower + 62 layers add failure modes; do Gemma-2 first and port the recipe).

#### Risks (ranked)

1. **Memory blow-up (copy path):** 191B total / 383 GB bf16 / ~3.1 TB optimizer states. Mitigation: split path, or EP=8 + ZeRO-3 + int8 optimizer states; budget the cluster before the science.
2. **Router collapse / dead experts:** without aux loss 1e-3–1e-2, experts die and loss plateaus early (Nemotron §3.3). Monitor expert utilisation per layer from step 0; also monitor with the DeepSeek-style aux-loss-free bias alternative only if you have that infrastructure — Nemotron's aux-loss recipe is the validated one here.
3. **Expert diversity failure:** all experts start as identical copies; only a reset LR (cosine similarity to base falling to ~0.6–0.7, Nemotron Fig. 6) breaks symmetry. Constant low LR leaves experts near-identical and the MoE ≈ dense. Weight permutation/reinitialisation (Qwen-2's suggestion) did *not* help Nemotron — don't rely on it.
4. **SwiGLU/GeGLU splitting bugs:** gate/up/down index misalignment silently destroys the FFN (see §4). Unit-test: split model with top-K = all experts must reproduce dense logits to numerical precision before any training.
5. **Granularity/MFU penalty:** fine-grained experts mean smaller GEMMs and permute/unpermute overhead; Nemotron notes lower GPU FLOP utilisation for granular vs coarse. Budget MFU at 30–35%, not 40%, for E8G8T8.
6. **Ossification (Liew interaction):** Gemma-2 (13T) and especially Qwen3-32B (36T) are so heavily pretrained that upcycling tokens yield diminishing returns — D₂ ∝ N₁^1.8. The Stage-2 stop rule exists for this.
7. **Evaluation noise:** MMLU differences of 1–2 pp are within run-to-run/harness variance at this scale. Pre-register the eval harness (lm-eval-harness, fixed shots) and the dense-continued control *before* training; do not cherry-pick checkpoints.
8. **Licence/tooling:** Gemma weights are under the Gemma licence (not Apache-2.0); Qwen is Apache-2.0. Megatron-LM upcycling supports GPT-style models; Gemma-2's logit soft-capping, alternating sliding-window attention and pre/post RMSNorm need a Megatron/Bridge port (NVIDIA Megatron Bridge lists Gemma-2 support as deprecated/unmaintained — verify before committing; HF Transformers + a custom upcycling script may be faster for a pilot).

#### Repos / codebases

| Codebase | Use for | Link |
|---|---|---|
| **Megatron-LM / Megatron-Core** | Production upcycling at 27B; `--moe-use-upcycling`, `--moe-upcycling-granularity`, EP/TP/PP, DeepEP | https://github.com/NVIDIA/Megatron-LM; https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/moe.html |
| **CMoE** | Training-free analytical split + 2k-sample fine-tune (pilot) | https://github.com/JarvisPei/CMoE |
| **ToMoE** | Router-only conversion, frozen weights (pilot) | https://github.com/gaosh/ToMoE |
| **LLaMA-MoE / LLaMA-MoE-v2** | Split + continual pretraining recipe (7B reference implementation) | https://github.com/pjlab-sys4nlp/llama-moe; https://github.com/OpenSparseLLMs/LLaMA-MoE-v2 |
| **HF Transformers** | Model definitions (Gemma2/Gemma3/Qwen3 MoE), pilot scripting, eval | https://huggingface.co/docs/transformers/model_doc/gemma2 (Gemma-2 config: hidden 4,608, intermediate 36,864, 46 layers) |
| **Liew scaling-law code** | D\* / joint-law fitting code and loss data | https://github.com/sbintuitions/sparse-upcycling-scaling-laws |
| **Skywork-MoE** | 13B upcycle reference (code/report) | https://github.com/SkyworkAI/Skywork-MoE |
| **BTX** | Domain-expert alternative | arXiv:2403.07816, https://ar5iv.labs.arxiv.org/html/2403.07816 |
| Inference | vLLM / SGLang / TensorRT-LLM (MoE, EP) for serving the result | (standard; verify Gemma-2 MoE support per engine at build time) |

---

## Could not verify / Open questions

1. **No 27B conversion exists** — the central gap. Any 27B benchmark in this report is extrapolated; a real run is the only way to close it.
2. **Gemma-2 pretraining peak LR** (needed for the Stage-2 reset) is not public. Nemotron's 2e-4 (2B) / 3e-4 peak (15B, 1T run) is the proxy used here.
3. **Gemma-3 27B PT ARC-C / HellaSwag and Qwen-32B ARC/HellaSwag** were not found in the tables consulted; the comparison table marks them n/v rather than filling from IT evaluations.
4. **Gemma-3 GSM8K 82.6 vs 81.20** — protocol discrepancy (8-shot vs 4-shot CoT), unresolved; quoted as a range.
5. **Gemma-4 26B-A4B** numbers (LMArena 1441, MMLU-Pro 82.6) are secondary summaries; no full technical report was available. If Google publishes one, it becomes the key native-MoE-at-26B comparator.
6. **Nemotron's "0.3 yottaFLOPs"** for the 1T programme is quoted as stated; its exact scope (one variant vs both) is ambiguous in the paper.
7. **Liew D\* log base** (natural vs base-10) is not stated in the paper; both evaluations are given (0.62B vs 0.48B) — the difference does not change any conclusion.
8. **Mistral Large 2 architecture details and pretraining tokens** not verified from a primary source; excluded from Table 1 detail.
9. **Qwen3.8-27B** (Aug 2026) benchmarks and pretraining tokens not verified; it may supersede Gemma-2 as the best exact-27B parent — re-check before starting.
10. **Inference throughput at 27B** (tok/s, $/Mtok) for any upcycled variant is unmeasured; §5 gives FLOPs/bytes reasoning only.

## Sources

- Gemma 2 technical report, arXiv:2408.00118 — https://arxiv.org/html/2408.00118v1 (architecture Table 1, pretraining Table 12/13, 13T tokens)
- Gemma 2 explained / HF config — https://gemma.ai/tutorials/gemma-explained/gemma-2/ (gate/up/down 4,608→36,864)
- Gemma 3 technical report, arXiv:2503.19786 — https://export.arxiv.org/pdf/2503.19786v1.pdf (14T tokens, 5:1 local/global, 128K context)
- Gemma 3 explained / config — https://gemma.ai/tutorials/gemma-explained/gemma-3/ (62 layers, hidden 5,376, intermediate 21,504)
- Gemma 3 model card (PT benchmarks) — https://huggingface.co/lex-au/Google.Gemma-3-27b-pt-GGUF ; https://hub.docker.com/r/ai/gemma3
- Qwen2.5 technical report, arXiv:2412.15115 — https://arxiv.org/html/2412.15115v2 (18T tokens, 64 layers, 40/8 heads)
- Qwen2.5-32B HF — https://huggingface.co/Qwen/Qwen2.5-32B-Instruct (32.5B total, 31.0B non-emb)
- Qwen3 technical report, arXiv:2505.09388 — https://arxiv.org/pdf/arXiv:2505.09388 ; https://ar5iv.labs.arxiv.org/html/2505.09388 (36T tokens; Tables 4–5 base benchmarks incl. Qwen3-30B-A3B 81.38/91.81/71.45)
- Qwen3-32B config — https://github.com/zsxh1990/llmservingsim/blob/HEAD/docs/docs/reference/model-config.md (hidden 5,120, intermediate 25,600, 64 layers)
- Komatsuzaki et al., Sparse Upcycling, arXiv:2212.05055 — https://arxiv.org/abs/2212.05055
- He et al., Upcycling LLMs into MoE (Nemotron), arXiv:2410.07524v2 — https://arxiv.org/html/2410.07524v1 (§§2–4, Table 1: 65.3 / 66.2 / 67.6 MMLU, 0.3 yottaFLOPs, LR/aux/virtual-group details)
- Liew et al., Scaling Laws for Upcycling MoE, arXiv:2502.03009v2 — https://arxiv.org/html/2502.03009v2 (Eqs. 1–2, 12–14; ≤1B models, few hundred B tokens; code https://github.com/sbintuitions/sparse-upcycling-scaling-laws)
- Skywork-MoE, arXiv:2406.06563 — https://arxiv.org/html/2406.06563v1/ (13B→146B/22B, MMLU 77.4)
- LLaMA-MoE, arXiv:2406.16554 — https://pub-fad10fba280c45dba0823fd585acc08c.r2.dev/pdf/2406.16554.pdf?v=1789095365 (200B tokens, split FFN)
- LLaMA-MoE-v2 — https://github.com/OpenSparseLLMs/LLaMA-MoE-v2 (7B tokens, two-stage)
- CMoE / Analytical FFN-to-MoE Restructuring, arXiv:2502.04416 — https://arxiv.org/pdf/2502.04416 ; https://arxiv.org/html/2502.04416v2 (Tables 2–3, 6: training-free 42.50, LLaMA-MoE-v2 training-free 30.33 / >10k PPL; code https://github.com/JarvisPei/CMoE)
- ToMoE, arXiv:2501.15316 — https://arxiv.org/html/2501.15316v2 ; code https://github.com/gaosh/ToMoE (0.02B tokens, frozen weights)
- BTX, arXiv:2403.07816 — https://ar5iv.labs.arxiv.org/html/2403.07816 (Llama-2 7B seed, domain experts)
- Recycling / Orthogonal Growth, arXiv:2510.08008 — https://arxiv.org/pdf/2510.08008 (17B MoE→70B, 1T tokens, +10.66%)
- Mixtral 8x22B / DeepSeek-V2 table — https://github.com/viktor-shcherb/msc_thesis/blob/HEAD/references/2024-05-deepseek-v2-moe/analysis.md (Mixtral 8x22B base 77.6/80.3/53.1)
- DeepSeek-V3, arXiv:2412.19437 — https://github.com/Shyboy0499/DeepSeek-V3/blob/HEAD/README.md (671B/37B, base MMLU 87.1, GSM8K 89.3, HumanEval 65.2, ARC-C 95.3, HellaSwag 88.9, 14.8T tokens)
- Gemma-4 26B-A4B (secondary) — https://llmcheck.net/blog/qwen-36-vs-gemma-4-deep-technical-comparison/ ; https://dev.to/pulkitgovrani/gemma-4-26b-a4b-what-mixture-of-experts-actually-means-for-your-inference-budget-13hj (25.2B/3.8B, 128 experts top-8 + 1 shared, LMArena 1441 vs 31B dense 1452)
- Qwen3-30B-A3B serving / MoE math — https://www.aicassindra.com/blogs/llm/llm_moe_serving.html (30.5B/3.3B, 48 layers, 128 experts, expert 768; batch-64 expert-coverage argument)
- Megatron-Core MoE / upcycling flags — https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/moe.html (`--moe-use-upcycling`, `--moe-upcycling-granularity`)
- H100 training economics (989 TFLOPS dense bf16, 35–45% MFU, 6ND rule) — https://www.aicassindra.com/blogs/gpu/flops_budget_training.html ; https://www.aiagentichandbook.com/terms/pretraining/
- Qwen3.8-27B (secondary, unverified benchmarks) — https://rohitghumare.com/guides/qwen3-8-27b/
- Calculations (my arithmetic, not a source): `notes/calculations.md` in this research directory — D\* evaluation, Gemma-2/3/Qwen parameter audits, MoE config totals, FLOPs/GPU-hour/memory tables.

## Method note on flags
- **[MEASURED]** = number taken directly from a primary paper/report/model card cited above.
- **[EXTRAPOLATED]** = 27B forecast built by applying a measured delta (Nemotron pp, Liew law) outside its fitted range.
- **[CALCULATED]** = my arithmetic from verified configs (6ND, parameter counts, bf16 bytes); method in `notes/calculations.md`.
- No page encountered during this research contained instructions attempting to redirect the mission; none were followed.
