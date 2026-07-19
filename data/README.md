# Dataset — Financial-Alpaca

Source: `gbharti/finance-alpaca` on Hugging Face
Size: ~68,900 instruction-response pairs
License: Apache 2.0

Each record has an instruction (a financial question), an optional input, and an output (a full written answer). We use it because the outputs are complete responses, which is what a generative model needs to learn from unlike a labels-only dataset.

Split 80/20 into train/test with seed 42.

Run `python data/load_dataset.py` to download, format, and split it.
