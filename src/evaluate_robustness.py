import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from dataset import get_dataloaders
from models import ALS_ResNet18, ALS_DenseNet121 # Import both architectures
from config import Config

# --- CONFIGURATION: Match each muscle to its best model ---
MODEL_CONFIG = {
    "Bicep":      {"arch": "resnet", "path": "best_models/resnet18_bicep_best.pth"},
    "Antebrazo":  {"arch": "densenet", "path": "best_models/densenet121_antebrazo_best.pth"},
    "Quadriceps": {"arch": "resnet",   "path": "best_models/resnet18_quadriceps_best.pth"},
    "Tibial":     {"arch": "resnet",   "path": "best_models/resnet18_tibial_best.pth"}
}

def get_model_instance(arch_type):
    """Factory function to return the correct model architecture."""
    if arch_type == "resnet":
        return ALS_ResNet18()
    elif arch_type == "densenet":
        return ALS_DenseNet121()
    else:
        raise ValueError(f"Unknown architecture: {arch_type}")

def evaluate_robustness(muscle_name):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    conf = MODEL_CONFIG[muscle_name]
    
    print(f"\n{'='*60}\nEVALUATING {muscle_name.upper()} ({conf['arch'].upper()})\n{'='*60}")

    if not os.path.exists(conf['path']):
        print(f"[ERROR] Model file not found at: {conf['path']}")
        return None

    # 1. Initialize the correct architecture and load weights safely
    model = get_model_instance(conf['arch']).to(device)
    model.load_state_dict(torch.load(conf['path'], weights_only=True))
    model.eval()

    # 2. Load data
    _, val_loader, class_names = get_dataloaders(muscle_name=muscle_name)

    y_true, y_pred, y_probs = [], [], []

    # 3. Inference
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    # 4. Final Metrics
    auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0][0], 0, 0, 0)
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"Results for {muscle_name}:")
    print(f" > AUC: {auc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f}")

    return {"muscle": muscle_name, "auc": auc, "sens": sens, "spec": spec}

if __name__ == "__main__":
    results = []
    for muscle in MODEL_CONFIG.keys():
        res = evaluate_robustness(muscle)
        if res: results.append(res)
    
    # Summary Table
    print("\n" + "#"*40 + "\nFINAL ROBUSTNESS REPORT (MIXED MODELS)\n" + "#"*40)
    for r in results:
        print(f"{r['muscle']:<12} -> AUC: {r['auc']:.4f} | Sens: {r['sens']:.4f} | Spec: {r['spec']:.4f}")