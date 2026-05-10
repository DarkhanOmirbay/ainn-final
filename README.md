# Variant 12 — Fine-tuning LLM for Business Letters (Russian)

**Subject:** Artificial Intelligence and Neural Networks  
**Group:** AAI-2502M  
**Authors:** Darhan Omirbay, Alisher Khairullin

---

## Overview

Fine-tuned **RuGPT-3 Large** (760M parameters) on a synthetic corpus of 5,500 Russian business letter pairs using **LoRA** via `transformers` + `peft`. The model generates formal business letter replies from an incoming letter + instruction prompt.

---

## Project Structure

```
variant12/
├── ainn_final_project (1).ipynb   # Full pipeline: data gen → training → eval
├── app.py                          # Streamlit demo app
├── data/
│   ├── train.json                  # 5,000 training pairs
│   ├── val.json                    # 250 validation pairs
│   └── test.json                   # 250 test pairs
├── lora_adapter/                   # Trained LoRA adapter (5k dataset)
├── lora_adapter_500/               # Experimental adapter (500 samples)
└── pyproject.toml
```

---

## Dataset

5,500 synthetic Alpaca-style pairs generated with Python templates.

| Split | Size |
|-------|------|
| Train | 5,000 |
| Val   | 250 |
| Test  | 250 |

**5 letter categories** (equal distribution):
- `запрос_информации` — information requests
- `жалоба` — complaints
- `коммерческое_предложение` — commercial proposals
- `уведомление` — notifications
- `согласование` — approvals

**3 response styles** per category: `вежливое` · `краткое` · `настойчивое`

**Format (Alpaca-style):**
```json
{
  "instruction": "Составьте вежливый профессиональный ответ...",
  "input": "Уважаемые коллеги,\nПрошу предоставить...",
  "output": "Уважаемый(-ая) А.В. Петрова,\n\nБлагодарим...",
  "category": "запрос_информации",
  "style": "вежливое"
}
```

---

## Model & Training

**Base model:** [`ai-forever/rugpt3large_based_on_gpt2`](https://huggingface.co/ai-forever/rugpt3large_based_on_gpt2) (760M params)  
**Method:** LoRA (FP16) via `peft` + `transformers`  
**Training time:** 71.7 minutes (Apple MPS)

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| `lora_r` | 8 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.05 |
| Target modules | `c_attn` |
| Trainable params | **1,179,648 (0.15%)** of 761M |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Learning rate | 2×10⁻⁴ |
| Batch size (effective) | 16 (4 × grad_accum=4) |
| Max sequence length | 256 |
| LR scheduler | cosine |
| Precision | FP16 |

---

## How to Run

### 1. Install dependencies
```bash
pip install transformers peft datasets accelerate torch streamlit
# or with uv:
uv sync
```

### 2. Launch the Streamlit demo
```bash
streamlit run app.py
```

The app loads the LoRA adapter from `./lora_adapter/` and provides an interactive UI to generate business letter replies.

### 3. Reproduce training / evaluation
Open `ainn_final_project (1).ipynb` and run all cells sequentially.

---

## Expert Evaluation (30 test examples)

Evaluated on 30 samples from `data/test.json` covering all category × style combinations. Scoring is automated via a formal-marker rubric (see notebook).

| Criterion | Mean | Scale |
|-----------|------|-------|
| Style (деловой стиль) | **4.50** | 1–5 |
| Relevance (релевантность) | **4.57** | 1–5 |
| No hallucinations (отсутствие галлюцинаций) | **4.77** | 1–5 |
| **Overall** | **4.61** | 1–5 |

Complaints (`жалоба`) scored slightly lower (assertive emotional tone is harder to model with a 760M parameter base).

---

## Inference Speed

Measured with 10 greedy-decoding runs on a fixed benchmark letter.

| Metric | Value |
|--------|-------|
| Avg latency | ~5–15 sec (CPU/MPS) |
| Tokens/sec | measured in notebook |

Run the benchmark cell in the notebook to get exact numbers for your hardware.

---

## Libraries

- [`transformers`](https://github.com/huggingface/transformers)
- [`peft`](https://github.com/huggingface/peft)
- [`datasets`](https://github.com/huggingface/datasets)
- [`streamlit`](https://streamlit.io)

---

## References

- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [RuGPT-3 on HuggingFace](https://huggingface.co/ai-forever/rugpt3large_based_on_gpt2)
- [PEFT library](https://github.com/huggingface/peft)
