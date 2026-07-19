# shared helper functions

import os
import torch


def setup_cuda_library_path():
    # some clusters hide the cuda libs, add them to the path if they exist
    candidate_paths = [
        "/opt/miniconda/envs/genai-gpu/lib/python3.12/site-packages/nvidia/cuda_cupti/lib",
        "/opt/miniconda/envs/genai-gpu/lib/python3.12/site-packages/nvidia/cublas/lib",
        "/usr/local/cuda/lib64",
    ]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = [p for p in candidate_paths if os.path.exists(p)]
    if new_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths) + ":" + existing


def check_gpu():
    available = torch.cuda.is_available()
    print(f"CUDA available: {available}")
    if available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return available


def generate_response(model, tokenizer, prompt, temperature=0.7,
                      max_new_tokens=150, repetition_penalty=1.2):
    # generate text from a prompt
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
        )
    # only keep the newly generated part
    new_tokens = output_tokens[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def print_banner(text):
    print("=" * 60)
    print(text)
    print("=" * 60)
