# fine-tunes gemma on the financial dataset using QLoRA

import os
import sys
import torch
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_NAME, HF_TOKEN, CHECKPOINT_DIR, ADAPTER_SAVE_PATH,
    NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM_STEPS, LEARNING_RATE,
    MAX_SEQ_LENGTH, WARMUP_RATIO, LR_SCHEDULER, LOGGING_STEPS,
    SAVE_STRATEGY, SAVE_TOTAL_LIMIT, LORA_R, LORA_ALPHA,
    LORA_DROPOUT, LORA_TARGET_MODULES, RANDOM_SEED
)
from models.model_setup import load_tokenizer, load_model_for_training
from data.load_dataset import load_and_prepare
from transformers import TrainingArguments, set_seed
from trl import SFTTrainer
from huggingface_hub import login


def main():
    print("=== QLoRA FINE-TUNING ===")
    print("takes a few hours, watch the loss go down")
    set_seed(RANDOM_SEED)
    login(token=HF_TOKEN)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(ADAPTER_SAVE_PATH, exist_ok=True)

    train_dataset, _ = load_and_prepare()
    tokenizer = load_tokenizer(MODEL_NAME, HF_TOKEN)

    lora_params = {
        "r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "target_modules": LORA_TARGET_MODULES,
        "lora_dropout": LORA_DROPOUT,
    }
    model = load_model_for_training(MODEL_NAME, HF_TOKEN, lora_params)

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
        optim="paged_adamw_32bit",
        group_by_length=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    start = datetime.now()
    out = trainer.train()
    hours = (datetime.now() - start).total_seconds() / 3600
    print(f"done. final loss: {out.training_loss:.4f}, took {hours:.1f} hrs")

    # save just the adapter (small file)
    trainer.model.save_pretrained(ADAPTER_SAVE_PATH)
    tokenizer.save_pretrained(ADAPTER_SAVE_PATH)
    print(f"adapter saved to {ADAPTER_SAVE_PATH}")


if __name__ == "__main__":
    main()
