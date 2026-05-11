# Drone Audio Detection (LGN)

Drone detection pipeline using Logic Gate Networks (LGN).

This repository contains the training, calibration, and benchmarking scripts to build and evaluate C-compiled LGN models for real-time audio classification.

## Setup

Requires Python 3.9+ and `gcc`.

First, clone this repository and its submodules (important for `torchlogix`):
```bash
git clone --recurse-submodules https://github.com/Arturrio112/LGN-drone-detection.git
cd LGN-drone-detection
```

It is highly recommended to create and activate a virtual environment first:
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux:
source .venv/bin/activate
```

Then install dependencies:
```bash
pip install -r req.txt
```
*(Note: datasets is pinned to 2.18.0 to avoid torchcodec issues on Windows)*

## Pre-trained Models

The `models/` directory contains 6 pre-made baseline models for immediate benchmarking:
- **CNN Models (PyTorch):** One standard 2-layer CNN and one 3-layer CNN.
- **LGN Models (C-Compiled):** Three standard 2-layer LGNs (`train.py` neural network, represented by `models/saved_models_lgn_bal`, `models/saved_models_lgn_base`, and `models/saved_models_lgn_base4`) and one 3-layer LGN (`train_3layer.py` neural network, represented by `models/saved_models_lgn_3layer`).

## Usage

### 1. Train
Trains a new LGN model and compiles it to C (`compiled_1d.so`).
*(Note: Training requires the Google Speech Commands dataset to be downloaded locally for background noise balancing. Ensure the path is set correctly in the script's configuration.)*
```bash
python train.py
```
New models are saved in `models/saved_models_train_<timestamp>/`.

### 2. Calibrate
Finds the optimal threshold offset for a compiled LGN model to maximize balanced accuracy on the validation set.
```bash
python calibrate.py
```
This saves an `offset.txt` in the model directory.

### 3. Benchmark
Evaluates all models inside the `models/` folder (both LGN and PyTorch CNNs) on the holdout test set. 
```bash
python benchmark.py
```
Outputs accuracy, latency, RTFx, and memory metrics to `benchmark_final_results.json`.

## Files
- `train.py`: Training script and automatic C compilation.
- `calibrate.py`: Offset optimization.
- `benchmark.py`: Inference testing script.
- `torchlogix/`: LGN operations package.
- `models/`: Contains pre-trained models and outputs from training.
