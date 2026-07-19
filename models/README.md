# Model — Gemma-2B-it + QLoRA

Base model: `google/gemma-2b-it` (2B params, instruction-tuned, 2024)

We chose it over GPT-2 (too old, generates worse text) and over 7-8B models (too tight on memory). It's modern, follows instructions, and fits comfortably with QLoRA.

Fine-tuning: QLoRA (4-bit base + LoRA adapters)
- rank 16, alpha 32, dropout 0.05
- target layers: q_proj, k_proj, v_proj, o_proj
- ~20M trainable params (~1% of the model)

Tested on Tesla V100 32GB and Tesla T4 (Colab), CUDA 12.x.
