# main entry point - runs the model on some prompts and saves the output
# usage: python src/model_runner.py          (base model)
#        python src/model_runner.py --finetuned   (fine-tuned model)

import os
import sys
import json
import argparse
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import (
    MODEL_NAME, HF_TOKEN, ADAPTER_SAVE_PATH, EVAL_PROMPTS_PATH,
    TEMPERATURE, MAX_NEW_TOKENS, REPETITION_PENALTY
)
from utils.helpers import (
    setup_cuda_library_path, generate_response, print_banner,
    ensure_dir, check_gpu
)


def load_sample_prompts(num_samples):
    # Step 1 of the pipeline: load and process the dataset.
    # Evaluation prompts are drawn from the held-out test split of
    # Financial-Alpaca. If they haven't been prepared yet, we load and
    # process the dataset now (download -> format -> 80/20 split -> write
    # eval_prompts.json). The built-in list is a last-resort fallback used
    # only if the dataset can't be reached (e.g. no network).
    if not os.path.exists(EVAL_PROMPTS_PATH):
        try:
            from data.load_dataset import load_and_prepare
            print("No prepared eval prompts found - loading and processing "
                  "the dataset (first run downloads ~200MB)...")
            load_and_prepare()  # writes EVAL_PROMPTS_PATH as a side effect
        except Exception as e:
            print(f"Could not load dataset ({e}); "
                  f"falling back to built-in prompts.")

    if os.path.exists(EVAL_PROMPTS_PATH):
        with open(EVAL_PROMPTS_PATH) as f:
            return json.load(f)["prompts"][:num_samples]

    builtin = [
        "What does an inverted yield curve signal about the economy?",
        "How does the Federal Reserve raising interest rates affect equity markets?",
        "What is quantitative tightening and what are its market implications?",
        "Explain what a company beating earnings estimates by 15% means for its stock.",
        "What does a widening credit spread indicate about market conditions?",
        "How do rising inflation figures typically affect bond prices?",
        "What is the significance of a company announcing a dividend increase?",
        "Explain the market impact of a central bank cutting rates unexpectedly.",
        "What does high market volatility suggest about investor sentiment?",
        "How does quantitative easing influence asset prices?",
    ]
    return [f"### Instruction:\n{p}\n\n### Input:\n\n### Response:\n"
            for p in builtin[:num_samples]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    print_banner("FINANCIAL SENTIMENT MODEL - INFERENCE")
    setup_cuda_library_path()
    check_gpu()

    from huggingface_hub import login
    login(token=HF_TOKEN)

    from models.model_setup import (
        load_tokenizer, load_base_model_for_inference, load_finetuned_model
    )
    tokenizer = load_tokenizer(MODEL_NAME, HF_TOKEN)

    if args.finetuned:
        if not os.path.exists(ADAPTER_SAVE_PATH):
            print("no adapter found - train the model first")
            sys.exit(1)
        model = load_finetuned_model(MODEL_NAME, ADAPTER_SAVE_PATH, HF_TOKEN)
        label = "finetuned"
    else:
        model = load_base_model_for_inference(MODEL_NAME, HF_TOKEN)
        label = "base"

    prompts = load_sample_prompts(args.num_samples)

    results = []
    for i, prompt in enumerate(prompts):
        response = generate_response(
            model, tokenizer, prompt,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            repetition_penalty=REPETITION_PENALTY
        )
        if "### Instruction:\n" in prompt:
            instruction = prompt.split("### Instruction:\n")[1].split("\n\n")[0]
        else:
            instruction = prompt
        results.append({"sample_num": i + 1, "instruction": instruction, "response": response})
        print(f"\n[{i+1}] {instruction}")
        print(response)

    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    ensure_dir(out_dir)

    with open(os.path.join(out_dir, "samples.txt"), "w") as f:
        f.write(f"{label} model - {MODEL_NAME}\n\n")
        for r in results:
            f.write(f"[{r['sample_num']}] {r['instruction']}\n{r['response']}\n\n")

    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump({"model": label, "model_name": MODEL_NAME,
                   "timestamp": datetime.now().isoformat(),
                   "samples": results}, f, indent=2)

    print(f"\nsaved {len(results)} samples to outputs/")


if __name__ == "__main__":
    main()
