# Domain-Adaptive Financial Sentiment Generation via QLoRA Fine-Tuning of Gemma-2B

**IE 7374 — Generative AI | Group 17 | Northeastern University | Summer 2026**

**Team:** Raunak Amanna, Bryan Alighieri

## Overview

General language models understand English but not finance. They know the words "yield curve inversion" but not that it signals a likely recession. This project fine-tunes Gemma-2B-it on the Financial-Alpaca dataset using QLoRA so it can generate financially informed responses instead of generic ones, then measures how much it improved.

**Research Questions:**
1. Does QLoRA fine-tuning reduce perplexity on financial text vs the base model?
2. Does the fine-tuned model generate more financially accurate outputs on unseen prompts?
3. How does decoding temperature affect output quality for base vs fine-tuned model?

## Repository Structure

```
├── README.md               project overview and setup
├── requirements.txt        dependencies
├── config.py               all settings and hyperparameters
├── Dockerfile              container setup
├── src/
│   ├── model_runner.py     run inference (main entry point)
│   ├── data_loader.py      load dataset
│   └── train.py            run training
├── utils/helpers.py        shared functions
├── configs/model_config.yaml   config reference
├── data/load_dataset.py    dataset loading + splitting
├── models/model_setup.py   model loading + LoRA setup
├── experiments/
│   ├── baseline_eval.py    evaluate base model
│   ├── finetune.py         QLoRA training
│   └── evaluate.py         evaluate fine-tuned model
├── outputs/                generated samples
└── docs/
    ├── literature_review.md
    └── methodology.md
```

## Setup

Requirements: Python 3.10+, NVIDIA GPU with 12GB+ VRAM, Hugging Face account with Gemma access.

1. Clone the repo:
```bash
git clone https://github.com/raunaka928/7374_gemma_finetune.git
cd 7374_gemma_finetune
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Accept the Gemma license at https://huggingface.co/google/gemma-2b-it, get a token from https://huggingface.co/settings/tokens, and put it in `config.py`:
```python
HF_TOKEN = "your_token_here"
```

## How to Run

Quick test (base model, works right away):
```bash
python src/model_runner.py --num_samples 3
```

Full pipeline:
```bash
python data/load_dataset.py          # prep data
python experiments/baseline_eval.py  # before fine-tuning
python experiments/finetune.py       # train (2-4 hrs on a V100/T4)
python experiments/evaluate.py       # after fine-tuning
python src/model_runner.py --finetuned   # final samples
```

## Model

- Base: Gemma-2B-it (`google/gemma-2b-it`), 2B params, instruction-tuned
- Method: QLoRA (4-bit + LoRA), r=16, alpha=32, targets q/k/v/o projections
- Trainable params: ~20M (~1%)

## Dataset

- Financial-Alpaca (`gbharti/finance-alpaca`), ~68,900 instruction-response pairs
- 80/20 train/test split, seed 42

## Metrics

| Metric | What it measures | Target |
|---|---|---|
| Perplexity | how well the model predicts financial text | ≥20% lower |
| BLEU | output similarity to reference answers | +8 points |
| LLM-as-a-judge | which model's answers are better | ≥65% win rate |


## Preliminary Results

We ran the base Gemma-2B-it model on 10 samples drawn from the Financial-Alpaca test set before any fine-tuning. The model produced fluent, well-structured responses and demonstrated general instruction-following ability — formatting lists, using bold headers, and completing prompts coherently.

However, the base model showed clear limitations on financial topics. On the 401k stock options prompt, the model referenced a non-existent "Form 89-RS" — a hallucinated tax form — illustrating the factual gaps that domain fine-tuning targets. On the investing methods prompt, responses were generic and surface-level, lacking the market reasoning and sentiment analysis that Financial-Alpaca is designed to teach.

Generated samples are saved in `outputs/samples.txt` and `outputs/samples.json`.

## Known Issues / Limitations
- **trl version:** The `trl` library API changed in v0.12+. `requirements.txt` pins `trl==0.9.6` so the training script runs as written.
- **Base model hallucinations:** The base model occasionally fabricates specific financial details (e.g. non-existent regulatory forms). This is the primary motivation for fine-tuning.
- **Mixed dataset content:** Financial-Alpaca contains general instruction-following examples alongside financial ones. The fine-tuned model is evaluated on the same prompts for a fair comparison.
- **Hardware requirement:** Full fine-tuning requires a GPU with 12GB+ VRAM. Inference alone runs on a T4 (15.6GB).
