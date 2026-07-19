# project settings and hyperparameters

HF_TOKEN = "YOUR_HF_TOKEN_HERE"

# model + dataset
MODEL_NAME = "google/gemma-2b-it"
DATASET_NAME = "gbharti/finance-alpaca"

# output paths
OUTPUT_DIR = "./experiments/results"
CHECKPOINT_DIR = "./models/checkpoints"
ADAPTER_SAVE_PATH = "./models/saved_adapter"
EVAL_PROMPTS_PATH = "./experiments/results/eval_prompts.json"

# data split
TEST_SIZE = 0.2
RANDOM_SEED = 42

# LoRA settings
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# training settings
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4          # effective batch size = 16
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
LOGGING_STEPS = 50
SAVE_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 3

# eval settings
NUM_EVAL_PROMPTS = 20
NUM_PERPLEXITY_SAMPLES = 100

# generation settings
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 150
REPETITION_PENALTY = 1.2
