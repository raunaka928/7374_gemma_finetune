# loads Financial-Alpaca, formats it, splits into train/test

import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATASET_NAME, TEST_SIZE, RANDOM_SEED,
    NUM_EVAL_PROMPTS, EVAL_PROMPTS_PATH, OUTPUT_DIR
)

from datasets import load_dataset


def format_prompt(example):
    # put each record into the alpaca template
    instruction = example.get("instruction", "")
    extra_input = example.get("input", "")
    output = example.get("output", "")

    text = f"### Instruction:\n{instruction}\n\n"
    text += f"### Input:\n{extra_input}\n\n"
    text += f"### Response:\n{output}"
    return {"text": text}


def load_and_prepare():
    print("loading dataset...")
    raw_dataset = load_dataset(DATASET_NAME)
    print(f"total examples: {len(raw_dataset['train'])}")

    # format everything
    formatted = raw_dataset["train"].map(
        format_prompt,
        remove_columns=raw_dataset["train"].column_names
    )

    # 80/20 split
    split = formatted.train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
    train_dataset = split["train"]
    test_dataset = split["test"]
    print(f"train: {len(train_dataset)}, test: {len(test_dataset)}")

    # grab 20 prompts spread across the test set for evaluation
    eval_indices = [
        i * (len(test_dataset) // NUM_EVAL_PROMPTS)
        for i in range(NUM_EVAL_PROMPTS)
    ]

    eval_prompts = []
    eval_references = []
    for idx in eval_indices:
        full_text = test_dataset[idx]["text"]
        if "### Response:\n" in full_text:
            prompt_part = full_text.split("### Response:\n")[0] + "### Response:\n"
            reference_part = full_text.split("### Response:\n")[1]
        else:
            prompt_part = full_text
            reference_part = ""
        eval_prompts.append(prompt_part)
        eval_references.append(reference_part)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(EVAL_PROMPTS_PATH, "w") as f:
        json.dump({"prompts": eval_prompts, "references": eval_references}, f, indent=2)

    print(f"saved {NUM_EVAL_PROMPTS} eval prompts")
    return train_dataset, test_dataset


def explore_dataset():
    # print some basic stats for the report
    data = load_dataset(DATASET_NAME)["train"]
    print(f"total records: {len(data)}")
    inst_len = [len(ex["instruction"].split()) for ex in data]
    out_len = [len(ex["output"].split()) for ex in data]
    print(f"avg instruction length: {sum(inst_len)/len(inst_len):.1f} words")
    print(f"avg output length: {sum(out_len)/len(out_len):.1f} words")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    load_and_prepare()
    explore_dataset()
