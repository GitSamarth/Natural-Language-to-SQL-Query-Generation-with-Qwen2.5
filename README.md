<div align="center">

# Natural Language-to-SQL Query Generation with Qwen2.5

**Fine-tuning Qwen2.5 (0.5B & 1.5B) for text-to-SQL generation — a model-scale and data-scale comparison, with a live demo**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-ff4b4b)

</div>

---

Fine-tunes small open LLMs to convert natural language questions + a table schema into SQL queries. Two independent LoRA fine-tunes are run — **Qwen2.5-1.5B-Instruct on 40,000 examples** and **Qwen2.5-0.5B-Instruct on 14,250 examples**, both for 1 epoch — each evaluated against its own zero-shot base model, with a working Streamlit demo for interactive testing.

## Table of Contents

- [Setup](#setup)
- [Evaluation Methodology](#evaluation-methodology)
- [Results](#results)
- [Finding](#finding)
- [Live Demo](#live-demo)
- [Limitations](#limitations)
- [Usage](#usage)
- [Repo Structure](#repo-structure)

## Setup

| | |
|---|---|
| **Method** | LoRA (r=16, alpha=32, dropout=0.05) on `q_proj`, `k_proj`, `v_proj`, `o_proj`, via Hugging Face's `peft` library |
| **Dataset** | [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) — (question, schema, SQL) triples, ~78k total |
| **Hardware** | Google Colab, T4 GPU |
| **Held-out test set** | 3,000 examples, fixed seed, disjoint from both runs' train/validation splits |

| Run | Base Model | Params | Train Examples | Epochs |
|---|---|---|---|---|
| **A** | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | 40,000 | 1 |
| **B** | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | 14,250 | 1 |

## Evaluation Methodology

Two metrics, measured separately because they test different things:

**1. SQL correctness — execution-based column match.**
`sql-create-context` provides schema only, no populated tables, so exact result-set comparison isn't possible. Both gold and predicted SQL are executed against an empty SQLite DB built from the schema; a prediction counts as correct if it executes without error and returns the same output columns as gold.

```python
def execution_match(schema: str, gold_sql: str, pred_sql: str):
    """
    Returns True if pred_sql executes on the empty schema and returns
    the same output columns as gold_sql. Does NOT validate WHERE/JOIN/
    aggregation logic -- see Limitations.
    """
    ...
```

**2. Output format compliance.**
Whether the model returns clean SQL only, vs. wrapping it in markdown fences or prose (the system prompt explicitly asks for "ONLY the SQL query, nothing else").

## Results

| Run | Correctness — Base | Correctness — Fine-tuned | Δ | Format — Base | Format — Fine-tuned |
|---|:---:|:---:|:---:|:---:|:---:|
| A: 1.5B, 40k examples (n=400) | 80.1% | 88.0% | **+7.9** | 82.5% (n=200) | 100% |
| B: 0.5B, 14k examples (n=2000) | 76.3% | 89.4% | **+13.1** | 99.7% | 100% |

## Finding

Two distinct effects emerged, and they don't move together:

1. **Format compliance saturates fast and isn't very sensitive to model or data scale.** Both fine-tuned models hit 100%.
2. **Logical correctness improved more, in absolute terms, for the smaller model.** The 0.5B model gained **+13.1 points** from fine-tuning vs. **+7.9 points** for the 1.5B model — consistent with a pattern where a weaker base model has more headroom for fine-tuning to close, while a stronger base model needs proportionally more data before the same kind of gain shows up.

**Takeaway:** the size of a fine-tuning win depends on how much headroom the base model has left on the task, and how much data you give it — a small model with modest data can show a bigger correctness gain than a larger model with more data, simply because it started further from the ceiling.

## Live Demo

Two Streamlit apps let you interactively test each fine-tuned model — enter a table schema and a question, get generated SQL back, with a live execution check against an in-memory SQLite DB.

```bash
streamlit run app.py          # Qwen2.5-0.5B fine-tuned
streamlit run app_1_5b.py     # Qwen2.5-1.5B fine-tuned
```

Each app shows the model's aggregate evaluation metrics (correctness, format compliance) alongside a live "Try it" panel with preset examples plus a free-text option. See [`RUN_DEMO.md`](./RUN_DEMO.md) for full setup instructions, including running via Google Colab with an `ngrok` tunnel.

## Limitations

- **Correctness metric has a low ceiling.** Column-match on an empty schema doesn't validate WHERE/JOIN/aggregation logic. A stronger eval would use a dataset with populated tables (e.g. Spider) for true execution-accuracy.
- **Not a fully controlled ablation.** Model size, data volume, and eval sample size (n=400 vs n=2000) all differ between Run A and Run B, so the results show a real pattern but don't cleanly isolate model-size effect from data-size effect.
- Both base models already produce mostly-correct SQL zero-shot, which caps how large an improvement any fine-tune can show without a larger/more diverse training set.


## Repo Structure

```
.
├── notebooks/
│   ├── qwen2.5-1.5b-lora-40k.ipynb
│   └── qwen2.5-0.5b-lora-14k.ipynb
├── adapters/
│   ├── qwen2.5-1.5b-sql-lora-adapter-40k-1ep/
│   └── qwen2.5-0.5b-sql-lora-adapter/
├── app.py                # Streamlit demo — 0.5B fine-tuned model
├── app_1_5b.py            # Streamlit demo — 1.5B fine-tuned model
├── RUN_DEMO.md            # Demo setup instructions (Colab + ngrok)
├── README.md

```
