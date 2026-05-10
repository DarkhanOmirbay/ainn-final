# ML Concepts — Variant 12: LLM Fine-tuning for Business Letters

> **Project:** Fine-tuning RuGPT-3 Large on Russian business letters using LoRA  
> **Group:** AAI-2502M · Darhan Omirbay, Alisher Khairullin

---

## 1. Fine-tuning vs Pre-training

**Pre-training** trains a model from scratch on billions of tokens. It teaches the model language itself — grammar, world knowledge, reasoning. This costs millions of dollars in compute.

**Fine-tuning** takes an already pre-trained model and continues training on a small, task-specific dataset. The model already knows Russian; fine-tuning teaches it *how to write business letters specifically*.

We use **RuGPT-3 Large** (`ai-forever/rugpt3large_based_on_gpt2`) as the base — a 760M parameter GPT-2 architecture pre-trained on Russian text. Our fine-tuning dataset is 5,000 pairs, which is tiny compared to pre-training data.

---

## 2. LoRA — Low-Rank Adaptation

### The core idea

Full fine-tuning updates all 761M parameters. This is slow and memory-intensive. LoRA instead freezes the original weights and injects small trainable matrices alongside the frozen ones.

For a weight matrix **W** of shape `(d, k)`, LoRA adds:

```
W' = W + ΔW = W + B · A
```

Where:
- **A** has shape `(r, k)` — projects down to rank `r`
- **B** has shape `(d, r)` — projects back up
- **r** (rank) is a small number (we use `r = 8`)

During forward pass: `output = x · W^T + x · A^T · B^T · (alpha / r)`

The `alpha / r` scaling controls how strongly the LoRA update affects the output. We use `lora_alpha = 16`, so the scale is `16/8 = 2.0`.

### Why this works

The hypothesis is that the weight changes needed for fine-tuning lie in a low-dimensional subspace. You don't need to update all 761M parameters — a rank-8 update captures the domain shift from "general Russian" to "business letter style."

### Our configuration

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `lora_r` | 8 | Rank of decomposition matrices |
| `lora_alpha` | 16 | Scale factor (effective scale = 16/8 = 2.0) |
| `lora_dropout` | 0.05 | Dropout on LoRA layers for regularization |
| `target_modules` | `c_attn` | Applied to the combined Q/K/V projection in GPT-2 attention |
| Trainable params | 1,179,648 | **0.15%** of 761M total |

### Target module choice

GPT-2 uses `c_attn` (a single `Conv1D` layer) that produces Q, K, V all at once (shape `1536 → 4608`). Targeting this one module with LoRA touches all attention heads simultaneously.

### Result

- Only **1.18M parameters** trained instead of 761M
- Adapter file: **4.5 MB** (vs ~1.5 GB for the full model)
- Training time: **71.7 minutes** on Apple MPS

---

## 3. QLoRA — Quantized LoRA

QLoRA (Dettmers et al., 2023) extends LoRA with quantization to further reduce memory:

1. **4-bit NF4 quantization** — weights stored in 4-bit Normal Float format (designed for normally distributed weights)
2. **Double quantization** — the quantization constants themselves are also quantized
3. **Paged optimizers** — uses CPU RAM when GPU memory is full

This allows fine-tuning 7B models on a single 8GB GPU.

### Why we used LoRA FP16 instead of QLoRA

Our machine runs **Apple MPS** (Metal Performance Shaders). The `bitsandbytes` library that powers QLoRA requires NVIDIA CUDA and does not support MPS. Therefore we used standard LoRA with FP16 precision.

On a CUDA GPU, QLoRA would be preferred for models ≥7B (e.g., Saiga-Mistral-7B). For RuGPT-3 Large (760M), LoRA FP16 is efficient enough.

---

## 4. Dataset

### Format: Alpaca-style

Each example has three fields:

```json
{
  "instruction": "Составьте вежливый профессиональный ответ на входящее деловое письмо.",
  "input": "Уважаемые коллеги,\nПрошу предоставить информацию о ...",
  "output": "Уважаемый(-ая) А.В. Петрова,\n\nБлагодарим за обращение..."
}
```

The model learns to produce `output` given `instruction + input`. This mirrors the InstructGPT / Alpaca training paradigm.

### Splits

| Split | Size | Purpose |
|-------|------|---------|
| Train | 5,000 | Parameter updates |
| Val | 250 | Monitor overfitting during training |
| Test | 250 | Final evaluation (never seen during training) |

### Categories and styles

| Category | Russian name |
|----------|-------------|
| Information request | запрос_информации |
| Complaint | жалоба |
| Commercial proposal | коммерческое_предложение |
| Notification | уведомление |
| Approval | согласование |

Each category has 3 response styles: **вежливое** (polite/formal), **краткое** (concise), **настойчивое** (assertive).

### Why synthetic data is acceptable

The task goal is to teach *style* (formal Russian business letter format), not to learn domain-specific facts. Template-generated data reliably covers the stylistic patterns. A human-curated dataset would be better for factual accuracy but is much harder to obtain at 5,000+ examples.

---

## 5. Tokenization & Prompt Template

### Prompt format

```
### Инструкция:
{instruction}

### Входящее письмо:
{input}

### Ответ:
{output}
```

This is a standard "instruct" prompt format. The model learns to complete text after `### Ответ:`.

### Label masking

During training, we don't want the model to learn the instruction and input — only the output. We set:

```python
labels[:prefix_len] = [-100] * prefix_len
```

`-100` is the PyTorch ignore index for cross-entropy loss. So the loss is only computed on the response tokens, not the prompt tokens. This is critical — without masking, the model wastes capacity memorizing the instructions.

### Max sequence length

We cap at **256 tokens**. Longer sequences increase memory usage quadratically (due to attention). 256 is sufficient for most business letter prompts + replies.

---

## 6. Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_train_epochs` | 3 | Enough passes to converge without overfitting a 5k dataset |
| `learning_rate` | 2×10⁻⁴ | Standard for LoRA fine-tuning; higher than full fine-tuning |
| `per_device_train_batch_size` | 4 | Fits in MPS memory |
| `gradient_accumulation_steps` | 4 | Effective batch = 4 × 4 = **16** |
| `lr_scheduler_type` | cosine | Smoothly decays LR; reduces oscillation at end of training |
| `warmup_ratio` | 0.03 | 3% of steps used for LR warmup; prevents early instability |
| `fp16` | True | Half-precision; halves memory, speeds up compute |
| `max_seq_length` | 256 | Truncates long sequences |
| `seed` | 42 | Reproducibility |

### Gradient accumulation

Instead of updating weights every step with batch=4, we accumulate gradients for 4 steps and then update. This simulates a batch size of 16 without needing 4× more memory.

### Cosine LR schedule

The learning rate starts at 0, warms up to `2e-4`, then follows a cosine curve down toward 0. This gives a smooth convergence curve and avoids sharp drops.

---

## 7. Inference / Generation

### Generation config used in the demo

| Parameter | Value | Effect |
|-----------|-------|--------|
| `max_new_tokens` | 200 | Maximum reply length |
| `do_sample` | True | Sample from probability distribution (not greedy) |
| `temperature` | 0.7 | Sharpens distribution (lower = more deterministic) |
| `top_p` | 0.9 | Nucleus sampling: only consider top 90% probability mass |
| `repetition_penalty` | 1.3 | Penalizes repeating tokens; reduces loops |

### Greedy vs sampling

- **Greedy** (`do_sample=False`): always picks the highest probability token. Deterministic but can produce repetitive, generic text.
- **Sampling** (`do_sample=True`): draws from the probability distribution. More varied, creative output. Temperature < 1 makes it less random.

For the inference benchmark we used greedy (deterministic) to get reproducible latency numbers.

### Stop patterns

The model can keep generating beyond the intended reply. We apply post-processing to cut the output at patterns like `### `, `==`, organizational suffixes, etc. This simulates a proper EOS token that the model hasn't fully learned.

---

## 8. Evaluation Metrics

### Rubric (1–5 scale)

| Score | Style | Relevance | No Hallucinations |
|-------|-------|-----------|-------------------|
| 5 | Perfect formal style | Fully addresses the request | No invented facts |
| 4 | Minor deviations | Addresses with small gaps | Minor imprecisions |
| 3 | Noticeable style issues | Partial response | Questionable facts present |
| 2 | Significant violations | Incomplete/irrelevant | Clear errors |
| 1 | Not business-appropriate | Off-topic | Gross fabrications |

### Automated rubric implementation

**Style** is scored by counting formal business letter markers in the reply:
- Markers: "уважаемый", "с уважением", "благодарим", "просим", etc.
- 4+ hits → 5; 2+ hits → 4; 1 hit → 3; 0 hits → 2

**Relevance** is scored by checking how many significant words from the incoming letter appear in the reply.

**Hallucinations** are flagged by checking for specific risk phrases ("гарантируем возврат", "штраф составит", etc.) and by penalizing over-long replies (>400 chars suggests padding/drift).

### Real results (30 test examples)

| Criterion | Mean score |
|-----------|-----------|
| Style | **4.23 / 5** |
| Relevance | **4.93 / 5** |
| No hallucinations | **3.67 / 5** |
| **Overall** | **4.28 / 5** |

**By category:**

| Category | Avg score |
|----------|-----------|
| согласование | 4.40 |
| запрос_информации | 4.38 |
| уведомление | 4.29 |
| коммерческое_предложение | 4.25 |
| жалоба | 3.89 |

Complaints score lowest — assertive/emotional tone is harder to model correctly with a 760M base.

The hallucination score (3.67) reflects over-generation: the model sometimes produces replies >400 chars that start to drift from the template, which the rubric penalizes.

---

## 9. Inference Speed

Measured with 10 greedy-decoding runs on a fixed benchmark letter (Apple MPS, RuGPT-3 Large + LoRA).

| Metric | Value |
|--------|-------|
| Avg tokens per reply | 200 (hit max_new_tokens) |
| Latency mean | 22.31 sec |
| Latency median | 22.23 sec |
| Latency p95 | 23.51 sec |
| **Tokens/sec (mean)** | **9.0 tok/s** |

**Context:** 9.0 tok/s is the CPU/MPS speed for a 760M model. On an NVIDIA A100, the same model would run at ~200–400 tok/s. A larger model like Saiga-Mistral-7B on A100 with QLoRA runs at ~40–70 tok/s.

The 22-second latency is acceptable for batch offline processing but too slow for real-time interactive use without GPU hardware.

---

## 10. Summary of the Full Pipeline

```
Raw templates
     ↓
generate_dataset(5500)         # synthetic Alpaca pairs
     ↓
train.json / val.json / test.json
     ↓
load RuGPT-3 Large + apply LoRA (r=8)    # 0.15% trainable
     ↓
tokenize + mask labels
     ↓
Trainer(epochs=3, lr=2e-4, fp16)         # 71.7 min
     ↓
lora_adapter/ (4.5 MB)
     ↓
PeftModel.from_pretrained(base, adapter)
     ↓
generate_reply(letter, instruction)       # 9.0 tok/s
     ↓
Expert evaluation: 4.28 / 5 (30 examples)
```
