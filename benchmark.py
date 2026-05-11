import os
import time
import subprocess
import numpy as np
import torch
import torch.nn as nn
import json
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from tqdm import tqdm

try:
    import torchlogix
    print("Successfully imported 'torchlogix'.")
except ImportError:
    print("Warning: 'torchlogix' not found. LGN models will not work.")

# ==========================================
# 0. CONFIGURATION & MODEL LIST
# ==========================================
RESULTS_FILE = "benchmark_final_results.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

MODELS_TO_TEST = []
models_dir = "models"
if os.path.exists(models_dir):
    for folder_name in sorted(os.listdir(models_dir)):
        path = os.path.join(models_dir, folder_name)
        if os.path.isdir(path):
            name = folder_name
            if folder_name.startswith("saved_models_train_"):
                name = "M_" + folder_name.replace("saved_models_train_", "")
            elif folder_name.startswith("saved_models_lgn_"):
                name = "M_" + folder_name.replace("saved_models_lgn_", "")
            
            # Detect CNN models vs LGN models
            if "cnn" in folder_name.lower():
                pth_files = [f for f in os.listdir(path) if f.endswith('.pth')]
                if pth_files:
                    pth_path = os.path.join(path, pth_files[0])
                    model_class = "StandardDroneCNN1D_3Layer" if "3layer" in folder_name.lower() else "StandardDroneCNN1D"
                    MODELS_TO_TEST.append({
                        "path": pth_path, 
                        "name": name, 
                        "desc": f"CNN Model from {folder_name}", 
                        "type": "pytorch_cnn",
                        "model_class": model_class
                    })
            else:
                MODELS_TO_TEST.append({
                    "path": path, 
                    "name": name, 
                    "desc": f"LGN Model from {folder_name}", 
                    "use_offset": True, 
                    "type": "lgn"
                })

# ==========================================
# 1. PYTORCH ARCHITECTURE DEFINITIONS
# ==========================================
class StandardDroneCNN1D(nn.Module):
    def __init__(self, k=32, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, k, kernel_size=64, stride=32, padding=16), nn.BatchNorm1d(k), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(k, k * 2, kernel_size=16, stride=4, padding=8), nn.BatchNorm1d(k * 2), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.classifier = nn.Linear(k * 2 * 4, num_classes)

    def forward(self, x):
        if x.ndim == 4: x = x.squeeze(2)
        elif x.ndim == 2: x = x.unsqueeze(1)
        return self.classifier(self.features(x))

class StandardDroneCNN1D_3Layer(nn.Module):
    def __init__(self, k=32, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, k, kernel_size=32, stride=16, padding=8), nn.BatchNorm1d(k), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(k, k * 2, kernel_size=16, stride=4, padding=8), nn.BatchNorm1d(k * 2), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(k * 2, k * 2, kernel_size=8, stride=2, padding=4), nn.BatchNorm1d(k * 2), nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Flatten()
        )
        self.classifier = nn.Linear(k * 2 * 1, num_classes)

    def forward(self, x):
        if x.ndim == 4: x = x.squeeze(2)
        elif x.ndim == 2: x = x.unsqueeze(1)
        return self.classifier(self.features(x))

# ==========================================
# 2. DATA LOADER & HELPERS
# ==========================================
class DADS_Benchmark_Dataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    def __getitem__(self, idx):
        item = self.dataset[idx]
        signal = torch.tensor(item['audio']['array'], dtype=torch.float32)
        if signal.ndim == 1: signal = signal.unsqueeze(0)
        elif signal.shape[0] > 1: signal = torch.mean(signal, dim=0, keepdim=True)
        if item['audio']['sampling_rate'] != 16000:
            signal = T.Resample(item['audio']['sampling_rate'], 16000)(signal)
        target_len = 16000
        current_len = signal.shape[1]
        if current_len > target_len: signal = signal[:, :target_len]
        else: signal = torch.nn.functional.pad(signal, (0, target_len - current_len))
        max_val = torch.abs(signal).max()
        if max_val > 0: signal = signal / max_val
        return signal, torch.tensor(item['label'], dtype=torch.long)
    def __len__(self): return len(self.dataset)

def get_test_loader():
    print("Loading Test data (15% holdout)...")
    full = load_dataset("geronimobasso/drone-audio-detection-samples", split='train')
    test_ds = full.train_test_split(test_size=0.15, seed=42)['test']
    loader = DataLoader(DADS_Benchmark_Dataset(test_ds), batch_size=32, shuffle=False)
    return loader

def count_lgn_gates(c_path):
    try:
        with open(c_path, 'r', encoding='utf-8') as f:
            content = f.read()
            gates = content.count('&') + content.count('|') + content.count('^')
            return gates / 1_000_000
    except: return 0

def get_valid_path(base_path):
    paths = [os.path.join(".", base_path), os.path.join(".", "calibrated_old", base_path)]
    for p in paths:
        if os.path.exists(p) or os.path.exists(os.path.join(p, "model_1d.c")): return p
    return None

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000

# ==========================================
# 3. BENCHMARK FUNCTIONS
# ==========================================
def run_lgn_benchmark(model_info, test_loader):
    folder = get_valid_path(model_info["path"])
    if not folder: return None
    
    c_path = os.path.join(folder, "model_1d.c")
    so_path = os.path.join(folder, "compiled_1d_O3.so")
    
    print(f"\nTesting LGN (C-Compiled): {model_info['name']}")
    if not os.path.exists(so_path) and os.path.exists(c_path):
        try:
            subprocess.run(['gcc', '-shared', '-fPIC', '-O3', '-o', so_path, c_path], check=True)
        except Exception:
            pass

    if not os.path.exists(so_path):
        fallback_so = os.path.join(folder, "compiled_1d.so")
        if os.path.exists(fallback_so):
            so_path = fallback_so
        
    offset = 0.0
    if model_info.get("use_offset"):
        try:
            with open(os.path.join(folder, "offset.txt"), "r") as f: offset = float(f.read().strip())
        except: offset = 6.0

    compiled_model = torchlogix.CompiledLogicNet.load(save_lib_path=so_path, input_shape=(1, 1, 4000), num_classes=2, num_bits=1)
    compiled_model.use_bitpacking = False
    
    params_m = count_lgn_gates(c_path)
    size_mb = os.path.getsize(so_path) / (1024 * 1024)

    # --- LATENCY (Isolated, Batch=1) ---
    dummy_lat = np.zeros((1, 4000), dtype=bool)
    for _ in range(10): compiled_model(dummy_lat) 
    
    start_lat = time.perf_counter()
    for _ in range(100): compiled_model(dummy_lat)
    latency_ms = ((time.perf_counter() - start_lat) / 100) * 1000

    # --- RTFx / THROUGHPUT (Mass, Batch=5) ---
    dummy_mass = np.zeros((5, 4, 4000), dtype=bool) 
    for _ in range(10): 
        for w in range(4): compiled_model(dummy_mass[:, w, :])
        
    start_rtfx = time.perf_counter()
    runs = 50
    for _ in range(runs):
        for w in range(4): compiled_model(dummy_mass[:, w, :])
    elapsed_rtfx = time.perf_counter() - start_rtfx
    
    total_audio_sec = 5 * runs # 5 audio clips * 1 second each * runs
    rtfx = total_audio_sec / elapsed_rtfx

    # --- ACCURACY ---
    all_means, all_labels = [], []
    for inputs, labels in tqdm(test_loader, desc="   Calculating accuracy", leave=False):
        batch_size = inputs.shape[0]
        windows = inputs.unfold(2, 4000, 4000)
        w_logits = []
        for w in range(windows.shape[2]):
            bool_in = (windows[:, 0, w, :].numpy() > 0).astype(bool).reshape(batch_size, -1)
            w_logits.append(np.asarray(compiled_model(bool_in)))
        all_means.extend(np.mean(np.stack(w_logits, axis=0), axis=0))
        all_labels.extend(labels.numpy())

    logit_means = np.array(all_means)
    scores = logit_means[:, 1] - logit_means[:, 0]
    preds = (scores > offset).astype(int)
    bal_acc = balanced_accuracy_score(all_labels, preds) * 100
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

    print(f"   Result: Bal.Acc = {bal_acc:.2f}%, RTFx = {rtfx:.2f}")
    print(f"   Latency: {latency_ms:.2f} ms")
    if size_mb > 0: print(f"   Disk size: {size_mb:.3f} MB")
    if params_m > 0: print(f"   Parameters/Gates: {params_m:.2f} M")
    print(f"   RAM/VRAM: {size_mb:.1f} MB")
    
    return {
        "name": model_info["name"], "desc": model_info["desc"],
        "acc": float(bal_acc), "rtfx": float(rtfx),
        "latency_ms": float(latency_ms),
        "size_mb": float(size_mb),
        "params_m": float(params_m),
        "vram_mb": float(size_mb), 
        "matrix": np.array([[int(tn), int(fp)], [int(fn), int(tp)]])
    }

def run_pytorch_our_cnn(model_info, test_loader):
    path = model_info["path"]
    if not os.path.exists(path): 
        print(f"Cannot find model: {path}")
        return None
    
    print(f"\nTesting CNN: {model_info['name']} on {DEVICE}")
    model_class = globals()[model_info["model_class"]]
    model = model_class().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    
    params_m = count_parameters(model)
    
    # --- LATENCY (Isolated, Batch=1) ---
    dummy_lat = torch.randn(1, 1, 4000, device=DEVICE)
    with torch.no_grad():
        for _ in range(10): model(dummy_lat)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        start_lat = time.perf_counter()
        for _ in range(100): model(dummy_lat)
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        latency_ms = ((time.perf_counter() - start_lat) / 100) * 1000

    # --- RTFx / THROUGHPUT (Mass, Batch=5) ---
    dummy_mass = torch.randn(5, 1, 16000, device=DEVICE)
    windows_mass = dummy_mass.unfold(2, 4000, 4000)
    with torch.no_grad():
        for _ in range(10):
            for w in range(4): model(windows_mass[:, 0, w, :])
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        
        start_rtfx = time.perf_counter()
        runs = 50
        for _ in range(runs):
            for w in range(4): model(windows_mass[:, 0, w, :])
        if DEVICE.type == "cuda": torch.cuda.synchronize()
        elapsed_rtfx = time.perf_counter() - start_rtfx
        
    total_audio_sec = 5 * runs # 5 clips * 1 second each * runs
    rtfx = total_audio_sec / elapsed_rtfx

    # --- MEMORY ---
    if DEVICE.type == "cuda": 
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        if vram_mb == 0: vram_mb = torch.cuda.memory_reserved() / (1024 * 1024)
    else:
        vram_mb = params_m * 4

    # --- ACCURACY ---
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"   Accuracy {model_info['name']}", leave=False):
            inputs_dev = inputs.to(DEVICE)
            windows = inputs_dev.unfold(2, 4000, 4000)
            w_logits = []
            for w in range(windows.shape[2]): w_logits.append(model(windows[:, 0, w, :]).cpu().numpy())
            mean_logits = np.mean(np.stack(w_logits, axis=0), axis=0)
            all_preds.extend(mean_logits.argmax(axis=1))
            all_labels.extend(labels.numpy())

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    bal_acc = balanced_accuracy_score(all_labels, all_preds) * 100
    
    print(f"   Result: Bal.Acc = {bal_acc:.2f}%, RTFx = {rtfx:.2f}")
    print(f"   Latency: {latency_ms:.2f} ms")
    if params_m > 0: print(f"   Parameters: {params_m:.2f} M")
    if vram_mb > 0: print(f"   RAM/VRAM: {vram_mb:.1f} MB")
    
    return {
        "name": model_info["name"], "desc": model_info["desc"],
        "acc": float(bal_acc), "rtfx": float(rtfx),
        "latency_ms": float(latency_ms),
        "size_mb": 0.0,
        "params_m": float(params_m),
        "vram_mb": float(vram_mb), 
        "matrix": np.array([[int(tn), int(fp)], [int(fn), int(tp)]])
    }

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    if not MODELS_TO_TEST:
        print("No models found in the 'models' directory.")
    else:
        loader = get_test_loader()
        results = []
        for model_info in MODELS_TO_TEST:
            if model_info["type"] == "lgn":
                res = run_lgn_benchmark(model_info, loader)
            elif model_info["type"] == "pytorch_cnn":
                res = run_pytorch_our_cnn(model_info, loader)
            else:
                res = None
            if res: results.append(res)
                
        if results:
            save_data = [ {**r, "matrix": r["matrix"].tolist()} for r in results ]
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=4)
            print(f"\nBenchmarks completed. Results saved to {RESULTS_FILE}")