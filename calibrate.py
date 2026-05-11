import torch
import os
from tqdm import tqdm
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from sklearn.metrics import balanced_accuracy_score

try:
    import torchlogix
except ImportError:
    print("ERROR: 'torchlogix' not found.")
    exit()

CONFIG = {
    "COMMON": {
        "TARGET_SR": 16000,
        "N_FFT": 1024,
        "HOP_LENGTH": 128,
        "N_MELS": 64,
        "NUM_WORKERS": 0,
        "SAVE_DIR": "./models/saved_models_train", # Path to the model folder to calibrate
    },
    "1d": {
        "BATCH_SIZE": 64,
        "NUM_SAMPLES": 4000,   
        "TEST_SAMPLES": 16000, 
    }
}

class DADS_LGN_Dataset(Dataset):
    """Dataset for model calibration using HuggingFace samples."""
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self.common_conf = CONFIG["COMMON"]

    def _process_signal(self, audio_array, original_sr):
        signal = torch.tensor(audio_array, dtype=torch.float32)
        if signal.ndim == 1: signal = signal.unsqueeze(0)
        elif signal.shape[0] > 1: signal = torch.mean(signal, dim=0, keepdim=True)

        if original_sr != self.common_conf["TARGET_SR"]:
            resampler = T.Resample(original_sr, self.common_conf["TARGET_SR"])
            signal = resampler(signal)

        # NOTE: Calibration MUST use the full test window (e.g., 1s / 16000 samples)
        # to ensure the drone signal is captured correctly.
        target_len = CONFIG["1d"]["TEST_SAMPLES"]
        current_len = signal.shape[1]

        if current_len > target_len:
            signal = signal[:, :target_len]
        else:
            padding = target_len - current_len
            signal = torch.nn.functional.pad(signal, (0, padding))
            
        return signal

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio_data = item['audio']
        label = item['label']
        
        signal = self._process_signal(audio_data['array'], audio_data['sampling_rate'])
        max_val = torch.abs(signal).max()
        if max_val > 0: signal = signal / max_val
        
        return signal, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

def get_val_loader():
    """Download and prepare the validation dataset for calibration."""
    print("Loading Validation Data for calibration...")
    full_dataset = load_dataset("geronimobasso/drone-audio-detection-samples", split='train')
    
    train_test_split = full_dataset.train_test_split(test_size=0.15, seed=42)
    train_val_split = train_test_split['train'].train_test_split(test_size=0.176, seed=42)
    val_ds = train_val_split['test']
    
    return DataLoader(DADS_LGN_Dataset(val_ds), batch_size=64, shuffle=False, num_workers=0)

def calibrate_model(folder_path, val_loader):
    """Find the optimal logit threshold (offset) for a compiled C model."""
    lib_path = os.path.join(folder_path, "compiled_1d.so")
    if not os.path.exists(lib_path):
        print(f"Skipping: No compiled model found in {folder_path}")
        return
    
    print(f"\nCalibrating model: {folder_path}")
    
    try:
        compiled_model = torchlogix.CompiledLogicNet.load(
            save_lib_path=lib_path, input_shape=(1, 1, 4000), num_classes=2, num_bits=1
        )
        compiled_model.use_bitpacking = False
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    all_logit_sums, all_labels = [], []
    
    for inputs, labels in tqdm(val_loader, desc="Collecting logits", leave=False):
        batch_size = inputs.shape[0]
        inputs_np = inputs.numpy()
        
        if inputs_np.shape[2] >= 4000:
            n_windows = (inputs_np.shape[2] - 4000) // 4000 + 1
            logit_sum = np.zeros((batch_size, 2), dtype=np.float32)
            
            for w in range(n_windows):
                window = inputs_np[:, 0, w*4000 : (w+1)*4000]
                bool_input = (window > 0).astype(bool).reshape(batch_size, -1)
                logit_sum += np.asarray(compiled_model(bool_input), dtype=np.float32)
            
            # NOTE: Average the results! Divide by the number of windows.
            # This makes the offset ideal for single 0.25s window execution on a microcontroller.
            logit_mean = logit_sum / n_windows
            all_logit_sums.append(logit_mean)
        else:
            bool_input = (inputs_np > 0).astype(bool).reshape(batch_size, -1)
            logits = np.asarray(compiled_model(bool_input), dtype=np.float32)
            all_logit_sums.append(logits)
            
        all_labels.extend(labels.numpy())

    all_logit_sums = np.concatenate(all_logit_sums, axis=0)
    all_labels = np.array(all_labels)

    # 1. Calculate scores (Drone - Background)
    y_scores = all_logit_sums[:, 1] - all_logit_sums[:, 0]
    
    # 2. Find the actual range of scores produced by the model
    min_score = np.floor(np.min(y_scores))
    max_score = np.ceil(np.max(y_scores))
    
    # 3. Search for the best offset within this dynamic range
    search_space = np.linspace(min_score, max_score, 1000)
    
    best_bal, best_offset = 0.0, 0.0
    for offset in search_space:
        preds = (y_scores > offset).astype(int)
        bal = balanced_accuracy_score(all_labels, preds) * 100
        
        if bal > best_bal:
            best_bal, best_offset = bal, offset
            
    print(f"Found ideal Offset: {best_offset:.3f} (Val Bal.Acc will increase to {best_bal:.2f}%)")
    print(f"Search range: from {min_score} to {max_score}")
    
    # --- SAVE OFFSET DIRECTLY TO MODEL FOLDER ---
    offset_path = os.path.join(folder_path, "offset.txt")
    with open(offset_path, "w") as f:
        f.write(str(best_offset))
        
    print(f"Calibration offset saved to: {offset_path}")

if __name__ == "__main__":
    # Get model folder from CONFIG
    target_dir = CONFIG["COMMON"]["SAVE_DIR"]
    
    if not os.path.exists(target_dir):
        print(f"Model directory not found: {target_dir}")
        exit()
        
    val_loader = get_val_loader()
    calibrate_model(target_dir, val_loader)
    
    print(f"\nCalibration complete! Offset added to {target_dir}")
