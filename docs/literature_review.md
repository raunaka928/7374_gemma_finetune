# Literature Review and Benchmarking

**Group 17 — Raunak Amanna, Bryan Alighieri**

## Objective

Our task is domain-adaptive text generation in NLP: taking a general language model and specializing it to generate financially informed commentary. This is a generative task, not classification the model produces multi-sentence responses, not labels.

## Related Work

**FinBERT (Araci, 2019)** showed that fine-tuning BERT on financial text beats the general model on financial sentiment. It proved domain adaptation works, but it's a classifier it outputs a label, not generated text.

**FinGPT (Yang et al., 2023)** extended this to generative models with an open-source framework for fine-tuning LLMs on financial data. Closer to our approach but uses larger models.

**BloombergGPT (Wu et al., 2023)** trained a 50B parameter model from scratch on financial data. State of the art, but needs way more compute than we have.

**LoRA (Hu et al., 2021)** and **QLoRA (Dettmers et al., 2023)** are the parameter-efficient fine-tuning methods we use. QLoRA quantizes the base model to 4-bit and trains small adapter matrices, which is what makes fine-tuning a 2B model possible on a single GPU.

## Model Benchmarking

We compared several models before choosing:

| Model | Params | Instruction-tuned | Modern | Fits on our GPU | Chosen |
|---|---|---|---|---|---|
| GPT-2 | 117M | No | No (2019) | Yes | No |
| DistilBERT | 66M | No | No | Yes | No (classifier) |
| Llama-3-8B | 8B | Yes | Yes | Tight | No |
| Mistral-7B | 7B | Yes | Yes | Tight | No |
| Gemma-2B-it | 2B | Yes | Yes (2024) | Yes | Yes |

We picked Gemma-2B-it because it's modern, instruction-tuned, small enough to fine-tune comfortably with QLoRA, and much better at generating fluent text than older small models like GPT-2. The 7-8B models would work but are tighter on memory and slower to train.

## Dataset Benchmarking

| Dataset | Size | Format | Good for generation | Chosen |
|---|---|---|---|---|
| Financial PhraseBank | ~5,000 | labels only | No | No |
| FiQA | ~6,600 | Q&A | Yes | Partial |
| Financial-Alpaca | ~68,900 | instruction-response | Yes | Yes |

We originally planned to use Financial PhraseBank but switched to Financial-Alpaca. PhraseBank only has sentiment labels, so training on it would teach the model to output "positive/negative/neutral" not full financial commentary. Financial-Alpaca has complete written answers, which is what a generative task needs.

## Evaluation Methods

We use three metrics: perplexity (measures domain fluency), BLEU (measures similarity to reference answers), and LLM-as-a-judge (measures whether the answers are actually accurate). We include LLM-as-a-judge because BLEU misses semantic correctness it rewards word overlap even when the finance is wrong.

## References

- Araci (2019). FinBERT. arXiv:1908.10063
- Dettmers et al. (2023). QLoRA. arXiv:2305.14314
- Hu et al. (2021). LoRA. arXiv:2106.09685
- Gemma Team (2024). Gemma. arXiv:2403.08295
- Wu et al. (2023). BloombergGPT. arXiv:2303.17564
- Yang et al. (2023). FinGPT. arXiv:2306.06031
- Papineni et al. (2002). BLEU. ACL 2002
