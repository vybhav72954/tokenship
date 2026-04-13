# TokenShip 

A replication and extension of **TokenSkip: Controllable Chain-of-Thought Compression in LLMs** (EMNLP 2025).

- **Original paper:** https://arxiv.org/abs/2502.12067  
- **Original GitHub:** https://github.com/hemingkx/TokenSkip  
- **Team:** Vybhav Chaturvedi · Anukul Vats · Kingshuk Barua · Lavesh Nihalani  
- **Mentor:** Vaishnovi Arun  

---

## What is TokenSkip?

TokenSkip introduces a controllable Chain-of-Thought (CoT) compression method for LLMs. The core idea is simple: not all tokens in a reasoning chain are equally important. By fine-tuning a model with a compression parameter **γ ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}**, TokenSkip teaches the model to skip low-importance tokens during generation while preserving reasoning accuracy.

At inference time, a single trained adapter handles all compression levels dynamically — no separate model per γ.

---

## Repository Structure

```
TokenShip/
├── notebooks/
│   ├── gsm8k/          # GSM8K pipeline per model size (0.5B → 14B)
│   └── math500/        # MATH-500 pipeline per model size (0.5B → 14B)
├── results/
│   ├── gsm8k/          # Per-model CSVs with accuracy, tokens, latency, ActRatio
│   └── math500/        # Per-model CSVs with full method comparison
├── figures/
│   ├── gsm8k/          # 8 figures per model size
│   └── math500/        # 8 figures per model size incl. subject heatmap
├── .gitignore
└── README.md
```

---

## Models & Benchmarks

| Model | GSM8K | MATH-500 |
|---|---|---|
| Qwen2.5-0.5B-Instruct | ✅ | ✅ |
| Qwen2.5-1.5B-Instruct | ✅ | ✅ |
| Qwen2.5-3B-Instruct | ✅ | ✅ |
| Qwen2.5-7B-Instruct | ✅ | ✅ |
| Qwen2.5-14B-Instruct | ✅ | ✅ |

> Note: The original paper reports MATH-500 results only for LLaMA-3.1-8B-Instruct. Our Qwen2.5 MATH-500 results are a novel contribution — no prior published baseline exists for this configuration.

---

## Methodology

### Overview

Each pipeline notebook is self-contained and runs 7 sequential stages. All experiments run on Kaggle H100 GPUs.

---

### Stage 1 — Load Training Data

For GSM8K, the standard train split is loaded directly. For MATH-500, we load all 7 subjects from `EleutherAI/hendrycks_math` and concatenate them into a unified training set, preserving `Level` (1–5) and `subject` metadata for downstream use.

---

### Stage 2 — Generate Raw CoT Traces

The base model (no adapter) generates full Chain-of-Thought responses for every training problem using greedy decoding. These traces form the raw material for compression.

**Critical implementation detail:** CoT traces must be generated per model — a 7B model's traces cannot be reused for a 1.5B model. Each model size gets its own traces.

**Generation config** (critical for Qwen): Qwen2.5's default generation config (`temperature=0.7, top_p=0.8, top_k=20`) must be explicitly overridden for true greedy decoding:
```python
do_sample=False, temperature=1.0, top_p=1.0, top_k=0
```
Failing to do this causes ~10pp accuracy degradation on GSM8K baseline.

---

### Stage 3 — LLMLingua-2 Compression (Mixed Ratio)

Correct CoT traces (those where the model answered correctly) are compressed using **LLMLingua-2** (`microsoft/llmlingua-2-xlm-roberta-large-meetingbank`). For each training sample, γ is sampled uniformly from `{0.5, 0.6, 0.7, 0.8, 0.9, 1.0}`. Samples with γ=1.0 are passed through unmodified.

This mixed-ratio training is the key mechanism that gives TokenSkip its γ-controllable behaviour at inference — a single adapter learns to handle all compression levels.

**Note on answer filtering:** Only traces where the model's answer is correct are retained for training, matching the paper's Section 3.2.

---

### Stage 4 — LoRA Fine-tuning

The model is fine-tuned using LoRA on the compressed CoT dataset.

**LoRA config:**
```
r=8, alpha=16, dropout=0.05
target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

**Training config:**
```
epochs=3, lr=5e-5, batch=1, grad_accum=8
warmup_ratio=0.1, scheduler=cosine, bf16=True
val_split=10%, cutoff_len=2048
```

**SFT prompt masking:** Loss is computed only on response tokens — prompt tokens are masked with `labels=-100`. This matches LLaMA-Factory's implementation and is essential for the model to learn CoT compression rather than prompt prediction.

**Response format:**
```
{compressed_cot}

The final answer is: $\boxed{answer}$
```

**TokenSkip prompt format** (γ-conditioned):
```
{instruction}
{question}<|eot_id|>{γ}<|eot_id|>
```
At inference, the γ token directly signals the model how aggressively to compress.

---

### Stage 5 — Evaluation

All methods are evaluated on the test set with the same greedy generation config. Results are checkpointed after each method so runs can be resumed safely.

**5a — Prompt-engineering baselines**

| Method | Instruction Suffix |
|---|---|
| Original | — |
| BeConcise | `Be concise.` |
| OnlyNumbers | `Only use numbers or equations.` |
| AbbreWords | `Abbreviate words as much as possible.` |
| LC-Prompt | `Please reduce 50% of the words in your Chain-of-Thought process.` |

**5b — Truncation**

Brute-force compression by capping `max_new_tokens = MAX_NEW_TOKENS × γ`. Serves as the primary baseline to beat.

**5c — LLMLingua compressed evaluation**

Test prompts are compressed at inference time using LLMLingua-2 and passed to the base model. This tests whether prompt-side compression (without fine-tuning) can match TokenSkip.

**5d — LoRA adapter evaluation**

The trained adapter is loaded, merged via `merge_and_unload()`, and evaluated at each γ. Three variants are run:

- **LoRA** — pure γ-conditioned prompt
- **LoRAGuided** — γ-conditioned + BeConcise instruction
- **LoRASoft** — γ-conditioned + LC-Prompt instruction

For MATH-500, `max_new_tokens` is scaled by γ per paper footnote 4 (`max_len × γ`). This does not apply to GSM8K.

**5e — Adaptive Compression (Extension)**

A novel extension that assigns γ dynamically per problem based on MATH difficulty level, using the same trained adapter with no retraining:

| MATH Level | γ |
|---|---|
| Level 1 (easiest) | 0.5 |
| Level 2 | 0.6 |
| Level 3 | 0.7 |
| Level 4 | 0.8 |
| Level 5 (hardest) | 0.9 |

Harder problems are given more token budget; easier problems are compressed more aggressively. Per-level accuracy and token usage breakdowns are logged automatically.

---

### Stage 6 — Figures

8 figures are generated per run:

1. Truncation analysis (accuracy vs token budget, Pareto frontier)
2. Method heatmap (normalized metrics across all methods)
3. Token length distribution (boxplot per method)
4. Accuracy drop vs token savings
5. Truncation vs LoRASoft grouped by ratio
6. LoRA triplet (LoRA / LoRAGuided / LoRASoft by ratio)
7. All methods ranked by accuracy
8. Per-subject accuracy heatmap (MATH-500 only)

---

### Stage 7 — Output ZIP

All outputs are bundled into a single ZIP for download (training checkpoints excluded to save disk).

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | % of problems answered correctly via `\boxed{}` extraction |
| **Avg Tokens** | Average generated token count per problem |
| **ActRatio** | `Avg Tokens / Original Avg Tokens` — actual compression achieved |
| **Token Savings %** | `(1 - ActRatio) × 100` |
| **Latency (s)** | Average inference time per sample |
| **Efficiency Score** | `Accuracy / Avg Tokens × 100` |

---

## Extensions

Beyond replication, TokenShip contributes:

1. **Qwen2.5 MATH-500 evaluation** — no published baseline exists from the original paper for this model family
2. **Adaptive Compression by Difficulty** — dynamic per-problem γ assignment using MATH level metadata; no retraining required
3. **Scaling Law Analysis** — full pipeline across 0.5B → 14B to identify the compression collapse threshold
4. **Syntax-Aware Pruning** *(in progress)* — extending LLMLingua-2 force tokens to protect mathematical operators during compression

---

## Results

See `results/` for per-model CSVs and `figures/` for all plots.

---

## Setup

All notebooks are designed to run on **Kaggle with H100 GPU**. Required packages are installed inline at the top of each notebook.

> **LLaMA models require HuggingFace approval:**  
> https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct  
> Set `HF_TOKEN` as a Kaggle secret before running.

---

## Citation

```bibtex
@inproceedings{tokenskip2025,
  title     = {TokenSkip: Controllable Chain-of-Thought Compression in LLMs},
  author    = {Heming Xia and others},
  booktitle = {EMNLP},
  year      = {2025},
  url       = {https://arxiv.org/abs/2502.12067}
}
```