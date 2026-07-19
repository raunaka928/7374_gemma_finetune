# functions to load the model (base or fine-tuned) and set up LoRA

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel


def get_bnb_config():
    # 4-bit quantization so the model fits on the GPU
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_name, hf_token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token   # gemma has no pad token by default
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model_for_inference(model_name, hf_token):
    # load the plain model just for generating text
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        token=hf_token,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model


def load_model_for_training(model_name, hf_token, lora_params):
    # load model in 4-bit and add LoRA adapters
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        token=hf_token,
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_params["r"],
        lora_alpha=lora_params["lora_alpha"],
        target_modules=lora_params["target_modules"],
        lora_dropout=lora_params["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_finetuned_model(base_model_name, adapter_path, hf_token):
    # load base model then attach the trained adapter
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        token=hf_token,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model
