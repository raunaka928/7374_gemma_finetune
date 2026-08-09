# Methodology

**Group 17 — Raunak Amanna, Bryan Alighieri**

## Task

Given a financial question or headline, generate a short response (2-5 sentences) that answers the question using correct financial terminology and domain-appropriate reasoning.
## Model

Gemma-2B-it, fine-tuned with QLoRA. The base model is loaded in 4-bit to save memory, and we add LoRA adapters to the attention layers (q_proj, k_proj, v_proj, o_proj) with rank 16. Only 0.147% of the parameters get trained (3,686,400 of 2,509,858,816).

## Dataset

Financial-Alpaca (~68,900 records). Each has an instruction, optional input, and output. We format them into the Alpaca template and split 80/20 into train/test with a fixed seed so it's reproducible. The test set is never seen during training.

## Preprocessing

Each record is turned into:
```
### Instruction:
{question}

### Input:
{optional}

### Response:
{answer}
```
Then tokenized with max length 512. The model learns to generate the part after `### Response:`.

## Training

| Setting | Value |
|---|---|
| Epochs | 3 |
| Batch size | 4 (effective 16 with grad accumulation) |
| Learning rate | 2e-4 |
| Scheduler | cosine with 3% warmup |
| Optimizer | paged AdamW 32-bit |
| Precision | bfloat16 + 4-bit base |

Learning rate is small on purpose so we don't overwrite what the model already knows.

## Evaluation

We run two experiments with identical prompts and settings — once with the base model (before) and once with the fine-tuned model (after):

1. **Perplexity** on 100 held-out test examples. Target: ≥20% reduction.
2. **BLEU** on 20 generated outputs vs reference answers. Target: +8 points.
3. **LLM-as-a-judge** — an LLM picks the better answer across 20 prompt pairs. Target: ≥65% win rate.

We also break results down by financial topic (monetary policy, earnings, macro, credit, equities) to see where fine-tuning helped most.

## Preliminary Experiment

Before committing to a full training run, we ran the base model on sample prompts to confirm the pipeline works and to establish a baseline. The base model produced fluent but sometimes factually wrong financial answers (for example, describing quantitative tightening as lowering interest rates, which is backwards). This confirms the gap our fine-tuning aims to close and validated that the full pipeline runs end to end on our hardware.
