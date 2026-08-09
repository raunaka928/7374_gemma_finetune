# Domain-Adaptive Financial Text Generation via QLoRA Fine-Tuning of Gemma-2B

**IE 7374 — Generative AI | Group 17 | Northeastern University | Summer 2026**

**Team:** Raunak Amanna, Bryan Alighieri

## Overview

General language models understand English but not finance. They know the words "yield curve inversion" but not that it signals a likely recession — and more consequentially, they may confidently describe one financial instrument as another. This project fine-tunes Gemma-2B-it on Financial-Alpaca using QLoRA, then measures the effect with three complementary metrics.

**Research Questions:**
1. Does QLoRA fine-tuning reduce perplexity on held-out financial text vs the base model?
2. Does the fine-tuned model generate more accurate responses, by BLEU and by pairwise judge preference?
3. Where does domain adaptation succeed, and where does it fail?

## Headline Results

| Metric | Base model | Fine-tuned | Change | Target | Met? |
|---|---|---|---|---|---|
| Perplexity | 33.33 | 6.57 | −80.3% | −30% | Exceeded |
| BLEU | 6.42 | 8.22 | +1.81 | +10 pts | Missed |
| LLM-as-a-judge win rate | — | 25% | — | 65% | Missed |

Training: 3 epochs, 10,335 optimizer steps, 3.4 hours on a single NVIDIA V100-SXM2 32GB. Final mean training loss 1.5999 (from 3.495 at first logged step).

**The three metrics disagree, and that disagreement is the project's main finding.** Perplexity improved dramatically while judge-rated accuracy fell below the base model. Our analysis attributes this to register transfer: the FiQA-derived portion of Financial-Alpaca consists largely of informal forum answers, and the model acquired that discursive style alongside genuine financial knowledge. The style degrades instruction-following in ways perplexity cannot detect, because perplexity is teacher-forced and the model never generates during that measurement.

Full analysis is in the technical report.

## Repository Structure

```
├── README.md                      project overview, results, setup
├── requirements.txt               dependencies
├── config.py                      all settings and hyperparameters
├── Dockerfile                     container setup
├── src/
│   ├── model_runner.py            run inference (main entry point)
│   ├── data_loader.py             load dataset
│   └── train.py                   run training
├── utils/helpers.py               shared functions
├── configs/model_config.yaml      config reference
├── data/load_dataset.py           dataset loading + splitting
├── models/
│   ├── model_setup.py             model loading + LoRA setup
│   └── saved_adapter/             trained LoRA adapter (committed)
├── experiments/
│   ├── baseline_eval.py           evaluate base model
│   ├── finetune.py                QLoRA training
│   ├── eval_finetuned.py          evaluate fine-tuned model
│   └── results/                   all evaluation outputs (committed)
├── outputs/                       generated samples, base and fine-tuned
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

Full pipeline, in order:
```bash
python data/load_dataset.py            # prep data, write eval prompts
python experiments/baseline_eval.py    # before fine-tuning
python experiments/finetune.py         # train (~3.4 hrs on a V100)
python experiments/eval_finetuned.py   # after fine-tuning
python src/model_runner.py --finetuned # generate fine-tuned samples
```

Note that `experiments/eval_finetuned.py` was renamed from `evaluate.py`. The original
name shadowed the HuggingFace `evaluate` library, so the script imported itself instead
of the BLEU metric and failed partway through evaluation.

## Model

- Base: Gemma-2B-it (`google/gemma-2b-it`), 2B params, instruction-tuned
- Method: QLoRA (4-bit NF4 + LoRA), r=16, alpha=32, dropout 0.05, targets q/k/v/o projections
- **Trainable params: 3,686,400 — 0.147% of 2,509,858,816 total**
- Compute dtype: float16. The V100 is Volta architecture and has no hardware bfloat16;
  the HuggingFace Trainer rejects `bf16=True` on pre-Ampere devices.

## Dataset

- Financial-Alpaca (`gbharti/finance-alpaca`), 68,912 instruction-response pairs
- 80/20 train/test split, seed 42 → 55,129 train / 13,783 test
- Instructions average 10.3 words; responses average 78.8 words
- Approximately 30% of records are finance-related; the remainder are general
  instruction data inherited from Stanford Alpaca

## Metrics and Targets

| Metric | What it measures | Target |
|---|---|---|
| Perplexity | how well the model predicts held-out text | ≥30% lower |
| BLEU | n-gram overlap with reference answers | +10 points |
| LLM-as-a-judge | pairwise accuracy preference vs base model | ≥65% win rate |

## Results Files

Everything needed to verify the numbers above is committed:

- `experiments/results/baseline_results.json` — base model perplexity and generations
- `experiments/results/final_summary.json` — all headline metrics
- `experiments/results/comparison.json` — 20 before/after response pairs
- `experiments/results/judge_prompts.txt` — formatted pairs used for judge evaluation
- `experiments/results/eval_prompts.json` — the 20 evaluation prompts and references
- `models/saved_adapter/` — the trained LoRA adapter
- `outputs/samples_base.*` — base model generations
- `outputs/samples.*` — fine-tuned model generations

## Second Experiment: Domain Filtering

After analysing the results above, we found only 7 of the 20 judge prompts were financial
— a consequence of Financial-Alpaca being roughly 30% finance by composition. We
hypothesised that corpus dilution was limiting performance, filtered the corpus to
finance-only records using a keyword classifier (17,193 training examples), and repeated
the full pipeline under identical hyperparameters.

**The hypothesis was not supported.** The judge win rate was unchanged at 25%, and the
relative perplexity reduction was smaller (72.9% vs 80.3%). Note that absolute perplexity
and BLEU are not comparable across the two runs, since each was measured on a different
test distribution.

This is reported as a negative result in the technical report, and shifts the likely
explanation away from corpus composition toward register transfer. The code for this
experiment is not included in the main pipeline; the results reported in this repository
are from the primary run.

## Known Issues / Limitations

- **trl version:** The `trl` library API changed in v0.12+. `requirements.txt` pins
  `trl==0.9.6` so the training script runs as written.
- **tf-keras required:** On systems with Keras 3 installed, `transformers` 4.44 fails to
  import. Installing `tf-keras` resolves it.
- **Fine-tuned model regressions:** The fine-tuned model violates numeric instruction
  constraints (asked for three items, returns ten), exhibits repetition artifacts, and
  fabricates specific figures more readily than the base model. These are documented in
  the technical report and are the reason the judge win rate fell below 50%.
- **Judge protocol:** Response ordering was not randomised — the base model was always
  Response A. Position bias is a documented confound in this methodology.
- **Evaluation set size:** 20 prompts yields wide confidence intervals on the win rate.
- **Hardware:** Fine-tuning requires a GPU with 12GB+ VRAM. Inference alone runs on a T4.
