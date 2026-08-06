# evaluates the fine-tuned model and compares it to the baseline

import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_NAME, HF_TOKEN, OUTPUT_DIR, EVAL_PROMPTS_PATH,
    ADAPTER_SAVE_PATH, NUM_PERPLEXITY_SAMPLES,
    TEMPERATURE, MAX_NEW_TOKENS, REPETITION_PENALTY
)
from models.model_setup import load_tokenizer, load_finetuned_model
from data.load_dataset import load_and_prepare
from huggingface_hub import login
import evaluate as hf_evaluate


def compute_perplexity(model, tokenizer, texts, max_length=512):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for text in tqdm(texts, desc="perplexity"):
        enc = tokenizer(text, return_tensors="pt",
                        truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        n = enc["input_ids"].shape[1]
        total_nll += out.loss.item() * n
        total_tokens += n
    return float(np.exp(total_nll / total_tokens))


def generate_response(model, tokenizer, prompt):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=REPETITION_PENALTY,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    print("=== EVALUATE FINE-TUNED MODEL ===")
    login(token=HF_TOKEN)

    # need the baseline to compare against
    baseline_path = f"{OUTPUT_DIR}/baseline_results.json"
    if not os.path.exists(baseline_path):
        print("run baseline_eval.py first")
        return
    with open(baseline_path) as f:
        baseline = json.load(f)

    with open(EVAL_PROMPTS_PATH) as f:
        eval_data = json.load(f)
    eval_prompts = eval_data["prompts"]
    eval_references = eval_data["references"]

    _, test_dataset = load_and_prepare()
    test_texts = [test_dataset[i]["text"]
                  for i in range(min(NUM_PERPLEXITY_SAMPLES, len(test_dataset)))]

    tokenizer = load_tokenizer(MODEL_NAME, HF_TOKEN)
    model = load_finetuned_model(MODEL_NAME, ADAPTER_SAVE_PATH, HF_TOKEN)

    # perplexity comparison
    ft_ppl = compute_perplexity(model, tokenizer, test_texts)
    base_ppl = baseline["perplexity"]
    reduction = (base_ppl - ft_ppl) / base_ppl * 100
    print(f"perplexity: {base_ppl:.2f} -> {ft_ppl:.2f} ({reduction:.1f}% lower)")

    # generate new outputs
    ft_outputs = [generate_response(model, tokenizer, p)
                  for p in tqdm(eval_prompts, desc="generating")]

    # BLEU comparison
    bleu = hf_evaluate.load("bleu")
    refs = [[r] for r in eval_references]
    base_bleu = bleu.compute(predictions=baseline["outputs"], references=refs)["bleu"] * 100
    ft_bleu = bleu.compute(predictions=ft_outputs, references=refs)["bleu"] * 100
    print(f"BLEU: {base_bleu:.2f} -> {ft_bleu:.2f} (+{ft_bleu - base_bleu:.2f})")

    # save before/after comparison for the report
    comparison = []
    for i in range(len(eval_prompts)):
        if "### Instruction:\n" in eval_prompts[i]:
            instruction = eval_prompts[i].split("### Instruction:\n")[1].split("\n\n")[0]
        else:
            instruction = eval_prompts[i][:200]
        comparison.append({
            "instruction": instruction,
            "before": baseline["outputs"][i],
            "after": ft_outputs[i],
        })

    with open(f"{OUTPUT_DIR}/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # judge prompts to paste into an LLM
    with open(f"{OUTPUT_DIR}/judge_prompts.txt", "w") as f:
        for i, c in enumerate(comparison):
            f.write(f"=== {i+1} ===\n")
            f.write(f"Question: {c['instruction']}\n")
            f.write(f"Response A: {c['before']}\n")
            f.write(f"Response B: {c['after']}\n")
            f.write("Which is more accurate? A or B?\n\n")

    summary = {
        "baseline_perplexity": base_ppl,
        "finetuned_perplexity": ft_ppl,
        "perplexity_reduction_pct": reduction,
        "baseline_bleu": base_bleu,
        "finetuned_bleu": ft_bleu,
        "bleu_improvement": ft_bleu - base_bleu,
    }
    with open(f"{OUTPUT_DIR}/final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved comparison.json, judge_prompts.txt, final_summary.json")


if __name__ == "__main__":
    main()
