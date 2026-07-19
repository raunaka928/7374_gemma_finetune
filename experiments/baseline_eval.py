# evaluates the base model before fine-tuning (the "before" numbers)

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
    NUM_PERPLEXITY_SAMPLES, TEMPERATURE, MAX_NEW_TOKENS, REPETITION_PENALTY
)
from models.model_setup import load_tokenizer, load_base_model_for_inference
from data.load_dataset import load_and_prepare
from huggingface_hub import login


def compute_perplexity(model, tokenizer, texts, max_length=512):
    # perplexity = how "surprised" the model is by the text (lower is better)
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
    print("=== BASELINE EVAL (before fine-tuning) ===")
    login(token=HF_TOKEN)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _, test_dataset = load_and_prepare()
    test_texts = [test_dataset[i]["text"]
                  for i in range(min(NUM_PERPLEXITY_SAMPLES, len(test_dataset)))]

    tokenizer = load_tokenizer(MODEL_NAME, HF_TOKEN)
    model = load_base_model_for_inference(MODEL_NAME, HF_TOKEN)

    perplexity = compute_perplexity(model, tokenizer, test_texts)
    print(f"baseline perplexity: {perplexity:.2f}")

    with open(EVAL_PROMPTS_PATH) as f:
        eval_prompts = json.load(f)["prompts"]

    outputs = [generate_response(model, tokenizer, p)
               for p in tqdm(eval_prompts, desc="generating")]

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "perplexity": perplexity,
        "outputs": outputs,
    }
    with open(f"{OUTPUT_DIR}/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved baseline_results.json")


if __name__ == "__main__":
    main()
