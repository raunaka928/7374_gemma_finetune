# wrapper so the dataset can be loaded from src/ too

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from data.load_dataset import load_and_prepare, explore_dataset


def get_datasets():
    return load_and_prepare()


if __name__ == "__main__":
    get_datasets()
    explore_dataset()
