# wrapper to run training from src/

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from experiments.finetune import main as run_training


if __name__ == "__main__":
    run_training()
