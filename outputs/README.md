# Outputs

Generated samples from both the base and fine-tuned models, on the same prompts.

| File | Model | Description |
|---|---|---|
| `samples_base.txt` / `.json` | Base Gemma-2B-it | Before fine-tuning |
| `samples.txt` / `.json` | QLoRA fine-tuned | After 3 epochs on Financial-Alpaca |

Prompts are drawn from the held-out test split. Because the same prompts were used for
both models, the two files can be read side by side as a before/after comparison.

## Reproducing

```bash
python src/model_runner.py              # base model    -> samples.txt / samples.json
python src/model_runner.py --finetuned  # fine-tuned    -> samples.txt / samples.json
```

Note that both commands write to the same filenames. The base-model outputs in this
directory were copied to `samples_base.*` before the fine-tuned run overwrote them.

## What these show

The base model produces fluent, well-formatted text but makes domain errors — most
clearly, it describes a 529 education savings plan as an employer-sponsored retirement
account with pre-tax contributions, which describes a 401(k).

The fine-tuned model corrects errors of that kind and engages with financial questions
the base model declines to answer. It also introduces new failure modes: it violates
numeric instruction constraints, occasionally degenerates into repetition, and fabricates
specific figures more readily. Both behaviours are visible in these files and are analysed
in the technical report.
