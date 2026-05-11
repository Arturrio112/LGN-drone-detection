import torch
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import random
import subprocess
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from datasets import load_dataset
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

# Optimize memory allocation for large models
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# --- LIBRARY IMPORTS ---
try:
    import torchlogix
    from torchlogix.layers import LogicConv2d, GroupSum, OrPooling
    print("'torchlogix' imported successfully.")
except ImportError:
    print("ERROR: 'torchlogix' not found. Please install the torchlogix library.")
    exit()

# ==========================================
# DYNAMIC SEED LOGIC
# ==========================================
def get_or_create_seed(save_dir, is_training=False):
    seed_file = os.path.join(save_dir, "model_seed.txt")
    if is_training:
        seed = random.randint(1, 100000)
        with open(seed_file, "w") as f:
            f.write(str(seed))
        print(f"New random seed generated for this run (Seed: {seed})")
        return seed
    else:
        if os.path.exists(seed_file):
            with open(seed_file, "r") as f:
                seed = int(f.read().strip())
            print(f"Loaded training seed from file (Seed: {seed})")
            return seed
        else:
            print("No seed file found — falling back to 42.")
            return 42

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    "COMMON": {
        "TARGET_SR": 16000,
        "N_CLASSES": 2,
        "EPOCHS": 15,
        "MIN_SAVE_THRESHOLD": 55.0,
        "LR": 0.01,
        "LR_PATIENCE": 2,
        "COLLAPSE_PATIENCE": 3,
        "SAVE_DIR": f"./models/saved_models_3layer_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "NUM_WORKERS": 0,
        # LOCAL NON-DRONE AUDIO FOLDER
        # Used to balance the dataset. Point to Google Speech Commands or similar.
        "EXTRA_NONDRONE_LOCAL_DIR": "../LGN/speech_commands",
    },
    "1d": {
        "BATCH_SIZE": 64,
        "NUM_SAMPLES": 4000,    # 0.25s training window
        "TEST_SAMPLES": 16000,  # 1.0s test window (Sliding window voting)
        "K_FACTOR": 32,
        "NUM_BITS": 1,
    }
}

os.makedirs(CONFIG["COMMON"]["SAVE_DIR"], exist_ok=True)

if torch.cuda.is_available():
    DEVICE_STR = "cuda"
    DEVICE = torch.device("cuda")
else:
    DEVICE_STR = "cpu"
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ==========================================
# 3-LAYER ARCHITECTURE
# ==========================================
# Matemātika dimensijām:
# L1: in=(1,4000) rf=(1,64) stride=(1,32) pad=(0,16) -> W=125
# P1: kernel=(1,4) stride=(1,4)                      -> W=31
# L2: in=(1,31)   rf=(1,7)  stride=(1,2)  pad=(0,3)  -> W=16
# P2: kernel=(1,2) stride=(1,2)                      -> W=8
# L3: in=(1,8)    rf=(1,3)  stride=(1,1)  pad=(0,1)  -> W=8
# P3: kernel=(1,2) stride=(1,2)                      -> W=4
# Flatten: 4 * (K*4) = 16 * K  |  GroupSum -> 2 logits

P1_OUT   = 31
P2_OUT   = 8
K        = CONFIG["1d"]["K_FACTOR"]
FLAT_DIM = 16 * K  # Ar K=32, tas būs 512

class DroneLGN1D(nn.Module):
    def __init__(self, device_str, k=None):
        super().__init__()
        if k is None:
            k = CONFIG["1d"]["K_FACTOR"]
        self.features = nn.Sequential(
            LogicConv2d(in_dim=(1, 4000), device=device_str, channels=1, num_kernels=k,
                        receptive_field_size=(1, 64), tree_depth=6,
                        stride=(1, 32), padding=(0, 16)),
            OrPooling(kernel_size=(1, 4), stride=(1, 4)),
            LogicConv2d(in_dim=(1, P1_OUT), device=device_str, channels=k, num_kernels=k * 2,
                        receptive_field_size=(1, 7), tree_depth=5,
                        stride=(1, 2), padding=(0, 3)),
            OrPooling(kernel_size=(1, 2), stride=(1, 2)),
            LogicConv2d(in_dim=(1, P2_OUT), device=device_str, channels=k * 2, num_kernels=k * 4,
                        receptive_field_size=(1, 3), tree_depth=4,
                        stride=(1, 1), padding=(0, 1)),
            OrPooling(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Flatten()
        )
        self.classifier = GroupSum(k=CONFIG["COMMON"]["N_CLASSES"], tau=1.0, device=device_str)
        assert FLAT_DIM % CONFIG["COMMON"]["N_CLASSES"] == 0

    def forward(self, x):
        x = torch.gt(x, 0).float()
        if x.ndim == 2:   x = x.unsqueeze(1).unsqueeze(1)
        elif x.ndim == 3: x = x.unsqueeze(2)
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# DATASETS
# ==========================================
class DADS_LGN_Dataset(Dataset):
    """Primary drone/non-drone dataset from HuggingFace."""
    def __init__(self, hf_dataset, is_training=False):
        self.dataset = hf_dataset
        self.is_training = is_training
        self.target_sr = CONFIG["COMMON"]["TARGET_SR"]
        self.train_len = CONFIG["1d"]["NUM_SAMPLES"]
        self.test_len = CONFIG["1d"]["TEST_SAMPLES"]

    def _process_signal(self, audio_array, original_sr):
        signal = torch.tensor(audio_array, dtype=torch.float32)
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        elif signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        if original_sr != self.target_sr:
            signal = T.Resample(original_sr, self.target_sr)(signal)

        target_len = self.train_len if self.is_training else self.test_len
        current_len = signal.shape[1]
        if self.is_training:
            if current_len > target_len:
                start = np.random.randint(0, current_len - target_len)
                signal = signal[:, start: start + target_len]
            else:
                signal = nn.functional.pad(signal, (0, target_len - current_len))
        else:
            if current_len > target_len:
                signal = signal[:, :target_len]
            else:
                signal = nn.functional.pad(signal, (0, target_len - current_len))
        return signal

    def __getitem__(self, idx):
        item = self.dataset[idx]
        signal = self._process_signal(item['audio']['array'], item['audio']['sampling_rate'])
        max_val = torch.abs(signal).max()
        if max_val > 0:
            signal = signal / max_val
        return signal, torch.tensor(item['label'], dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

class LocalNonDroneDataset(Dataset):
    """Dataset for balancing classes using local audio files."""
    EXTENSIONS = {'.wav', '.mp3', '.flac'}

    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.target_sr = CONFIG["COMMON"]["TARGET_SR"]
        self.target_len = CONFIG["1d"]["NUM_SAMPLES"]

    def _process_signal(self, signal, original_sr):
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        if original_sr != self.target_sr:
            signal = T.Resample(original_sr, self.target_sr)(signal)
        current_len = signal.shape[1]
        if current_len > self.target_len:
            start = np.random.randint(0, current_len - self.target_len)
            signal = signal[:, start: start + self.target_len]
        else:
            signal = nn.functional.pad(signal, (0, self.target_len - current_len))
        return signal

    def __getitem__(self, idx):
        try:
            signal, sr = torchaudio.load(self.file_paths[idx])
            signal = self._process_signal(signal.float(), sr)
            max_val = signal.abs().max()
            if max_val > 0:
                signal = signal / max_val
            return signal, torch.tensor(0, dtype=torch.long)
        except Exception:
            return torch.zeros(1, self.target_len), torch.tensor(0, dtype=torch.long)

    def __len__(self):
        return len(self.file_paths)

def get_dataloaders():
    """Load and prepare balanced dataloaders."""
    print("Loading datasets...")
    full = load_dataset("geronimobasso/drone-audio-detection-samples", split='train')
    
    # Stratified-style split
    split1 = full.train_test_split(test_size=0.15, seed=42)
    test_ds = split1['test']
    split2 = split1['train'].train_test_split(test_size=0.176, seed=42)

    primary_train_ds = DADS_LGN_Dataset(split2['train'], is_training=True)
    primary_labels = np.array(split2['train']['label'])

    native_count_0 = (primary_labels == 0).sum()
    native_count_1 = (primary_labels == 1).sum()

    n_extra_needed = max(0, native_count_1 - native_count_0)
    extra_ds = None
    folder = CONFIG["COMMON"].get("EXTRA_NONDRONE_LOCAL_DIR")
    
    if folder and Path(folder).exists():
        print(f"Scanning local audio in: {folder}")
        all_files = [p for p in Path(folder).rglob("*") 
                     if p.suffix.lower() in LocalNonDroneDataset.EXTENSIONS 
                     and "_background_noise_" not in str(p)]
        if all_files:
            take_count = min(n_extra_needed, len(all_files))
            selected = random.sample(all_files, take_count)
            extra_ds = LocalNonDroneDataset(selected)
            print(f"Added {len(extra_ds)} local non-drone samples.")

    if extra_ds is not None:
        combined_train_ds = ConcatDataset([primary_train_ds, extra_ds])
        combined_labels = np.concatenate([primary_labels, np.zeros(len(extra_ds), dtype=int)])
    else:
        combined_train_ds, combined_labels = primary_train_ds, primary_labels

    count_0 = (combined_labels == 0).sum()
    count_1 = (combined_labels == 1).sum()

    # WeightedRandomSampler handles any remaining imbalance perfectly
    weight_per_sample = np.where(combined_labels == 0, 1.0 / count_0, 1.0 / count_1)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weight_per_sample, dtype=torch.float32),
        num_samples=len(combined_train_ds),
        replacement=True,
    )

    bs = CONFIG["1d"]["BATCH_SIZE"]
    nw = CONFIG["COMMON"]["NUM_WORKERS"]

    train_loader = DataLoader(combined_train_ds, batch_size=bs, sampler=sampler, num_workers=nw)
    val_loader = DataLoader(DADS_LGN_Dataset(split2['test'], is_training=False), batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(DADS_LGN_Dataset(test_ds, is_training=False), batch_size=bs, shuffle=False, num_workers=nw)

    return train_loader, val_loader, test_loader

# ==========================================
# TRAINING
# ==========================================
def train_and_save():
    """Execute the training loop."""
    print("\n=== STARTING TRAINING ===")
    seed = get_or_create_seed(CONFIG["COMMON"]["SAVE_DIR"], is_training=True)
    set_global_seed(seed)

    train_loader, val_loader, _ = get_dataloaders()
    
    model = DroneLGN1D(device_str=DEVICE_STR).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["COMMON"]["LR"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=CONFIG["COMMON"]["LR_PATIENCE"], min_lr=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    best_bal_acc = 0.0
    collapse_streak = 0
    save_path = os.path.join(CONFIG["COMMON"]["SAVE_DIR"], "best_1d.pth")

    for epoch in range(CONFIG["COMMON"]["EPOCHS"]):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Sliding window logit summation for robust validation
                windows = inputs.unfold(2, CONFIG["1d"]["NUM_SAMPLES"], CONFIG["1d"]["NUM_SAMPLES"])
                logit_sum = None
                for w in range(windows.shape[2]):
                    out = model(windows[:, 0, w, :].to(DEVICE))
                    logit_sum = out if logit_sum is None else logit_sum + out
                preds.extend(logit_sum.argmax(1).cpu().numpy())
                targets.extend(labels.numpy())

        bal_acc = balanced_accuracy_score(targets, preds) * 100
        scheduler.step(bal_acc)
        
        cm = confusion_matrix(targets, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
        else:
            spec = 0.0
        
        print(f"Epoch {epoch + 1} | LR: {optimizer.param_groups[0]['lr']:.6f} | Val Bal.Acc: {bal_acc:.2f}% (Spec: {spec:.1f}%)")

        if bal_acc > best_bal_acc and bal_acc >= CONFIG["COMMON"]["MIN_SAVE_THRESHOLD"]:
            best_bal_acc = bal_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model ({bal_acc:.2f}%)")

        # Collapse guard: Stop if specificity drops too low (model forgets non-drones)
        if spec < 5.0:
            collapse_streak += 1
            if collapse_streak >= CONFIG["COMMON"]["COLLAPSE_PATIENCE"]:
                print(f"Early Stop: Model collapse detected. Best saved: {best_bal_acc:.2f}%")
                break
        else:
            collapse_streak = 0

    print(f"\nTraining complete. Best Accuracy: {best_bal_acc:.2f}%")
    return save_path

# ==========================================
# C COMPILATION
# ==========================================
def compile_model():
    """Compile the trained PyTorch model into a C shared library."""
    print("\n=== COMPILING TO C ===")
    save_path = os.path.join(CONFIG["COMMON"]["SAVE_DIR"], "best_1d.pth")
    if not os.path.exists(save_path):
        print("Error: Trained model not found. Run training first.")
        return None

    seed = get_or_create_seed(CONFIG["COMMON"]["SAVE_DIR"], is_training=False)
    set_global_seed(seed)

    model = DroneLGN1D(device_str=DEVICE_STR).to(DEVICE)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    model.eval()
    model = model.cpu()

    # Reset device to CPU for all Logic modules before compilation
    for m in model.modules():
        if isinstance(m, LogicConv2d):
            m.device = 'cpu'
            for attr in vars(m):
                val = getattr(m, attr)
                if isinstance(val, torch.Tensor): setattr(m, attr, val.cpu())
                elif isinstance(val, tuple) and all(isinstance(t, torch.Tensor) for t in val):
                    setattr(m, attr, tuple(t.cpu() for t in val))

    try:
        for layer in model.features:
            if isinstance(layer, LogicConv2d):
                layer.indices = layer._get_flat_indices_for_compiler()

        full_model = nn.Sequential(*list(model.features.children()), model.classifier)
        compiled = torchlogix.CompiledLogicNet(
            model=full_model, input_shape=(1, 1, CONFIG["1d"]["NUM_SAMPLES"]),
            num_bits=1, use_bitpacking=False, cpu_compiler='gcc', verbose=True
        )

        c_path = os.path.join(CONFIG["COMMON"]["SAVE_DIR"], "model_1d.c")
        so_path = os.path.join(CONFIG["COMMON"]["SAVE_DIR"], "compiled_1d.so")

        print("Generating C code...")
        with open(c_path, "w", encoding="utf-8") as f:
            f.write(compiled.get_c_code())

        print("Compiling with GCC (-O3)...")
        subprocess.run(['gcc', '-shared', '-fPIC', '-O3', '-o', so_path, c_path], check=True)
        print("Compilation successful!")
        return so_path

    except Exception as e:
        print(f"Compilation failed: {e}")
        return None

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    import threading
    
    def main():
        train_and_save()
        so_path = compile_model()
        if so_path:
            print(f"\nC Model is ready at: {so_path}")

    try:
        # LGN models often require larger stack sizes due to deep logic trees
        threading.stack_size(64 * 1024 * 1024)
        t = threading.Thread(target=main)
        t.start()
        t.join()
    except Exception:
        print("Could not set stack size, running normally...")
        main()
