"""
explainability.py
─────────────────
Explainability maps (Grad-CAM, Guided Grad-CAM, Saliency, Occlusion)
for the FINAL SYSTEM of the thesis: ResNet-50 trained separately on each
of the 4 muscles with 5-fold StratifiedGroupKFold.

For each muscle:
  1. Selects the fold with the highest validation AUC (best checkpoint).
  2. Loads the corresponding weights from best_models_kfold/.
  3. Rebuilds the validation set of that fold (same seed and partitions
     as during training).
  4. Takes 2 ALS + 2 Control images from the val set (honest out-of-fold:
     the model did not see them during training).
  5. Generates the 4 types of maps and saves them.

Output
------
models/resultados_kfold/explainability_resnet50/
    gradcam/
    guided_gradcam/
    saliency/
    occlusion/

Usage: python3 explainability.py
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedGroupKFold

from models import get_model
from config import Config
from patient_level_fusion import patient_uid


# ── Configuration ─────────────────────────────────────────────
# Final system: ResNet-50 on the 4 muscles. It is the architecture adopted
# as the final system in the patient-level fusion analysis (informe_fusion.txt);
# the five architectures are statistically equivalent at the fusion level.
MODEL_NAME = "resnet50"
MUSCLES    = ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]

WEIGHTS_DIR = os.path.join(Config.BASE_DIR, "best_models_kfold")
PRED_PATH   = os.path.join(Config.BASE_DIR, "models", "resultados_kfold",
                           "kfold_predictions.json")
RESULTS_DIR = os.path.join(Config.BASE_DIR, "models", "resultados_kfold",
                           "explainability_resnet50")

# How many ALS and Control images per muscle
N_ALS     = 2
N_CONTROL = 2

TRANSFORM_NORM = transforms.Compose([
    transforms.Resize(Config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
TRANSFORM_RAW = transforms.Compose([
    transforms.Resize(Config.IMAGE_SIZE),
    transforms.ToTensor(),
])


# ── Best-fold-per-muscle selection ────────────────────────────
def best_fold_for(model_name, muscle, predictions):
    """From kfold_predictions.json, the fold with the highest AUC for (model, muscle)."""
    candidates = [r for r in predictions
                  if r["model"] == model_name and r["muscle"] == muscle]
    if not candidates:
        raise RuntimeError(f"No predictions for {model_name}/{muscle}")
    best = max(candidates, key=lambda r: r["auc"])
    return best["fold"], best["auc"]


# ── Rebuild the fold's validation set ─────────────────────────
def get_val_samples(muscle, fold_idx, n_als=N_ALS, n_ctrl=N_CONTROL):
    """
    Rebuilds exactly the validation set of the requested fold and returns
    n_als ALS images + n_ctrl Control images.
    """
    data_path = os.path.join(Config.PROCESSED_DATA_PATH, muscle)
    ds_norm   = datasets.ImageFolder(root=data_path, transform=TRANSFORM_NORM)
    ds_raw    = datasets.ImageFolder(root=data_path, transform=TRANSFORM_RAW)

    labels, groups = [], []
    for fp, lbl in ds_norm.samples:
        cls = ds_norm.classes[lbl]
        labels.append(lbl)
        groups.append(patient_uid(os.path.basename(fp), cls))

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                random_state=Config.SEED)
    folds = list(sgkf.split(X=range(len(ds_norm)), y=labels, groups=groups))
    if fold_idx < 1 or fold_idx > len(folds):
        raise ValueError(f"Fold {fold_idx} out of range (1..{len(folds)})")
    _, val_idx = folds[fold_idx - 1]

    # Separate indices by class (label 0=Control, 1=ELA according to ImageFolder)
    by_class = {0: [], 1: []}
    for i in val_idx:
        by_class[labels[i]].append(i)

    samples = []
    # 2 ALS first, then 2 Control (stable order: by file name)
    for cls_lbl, n, label in [(1, n_als, "ELA"), (0, n_ctrl, "Control")]:
        chosen = sorted(by_class[cls_lbl],
                        key=lambda i: os.path.basename(ds_norm.samples[i][0]))[:n]
        for i in chosen:
            fp, _ = ds_norm.samples[i]
            tensor_norm = ds_norm[i][0].unsqueeze(0)
            tensor_raw  = ds_raw[i][0].unsqueeze(0)
            samples.append((tensor_norm, tensor_raw, label, fp))

    return samples, val_idx, len(by_class[0]), len(by_class[1])


# ── Helpers ────────────────────────────────────────────────────
def tensor_to_np(tensor_raw):
    img = tensor_raw.squeeze(0).permute(1, 2, 0).numpy()
    return (img * 255).clip(0, 255).astype(np.uint8)


def make_leaf(tensor, device):
    arr = tensor.squeeze(0).detach().cpu().numpy()
    leaf = torch.tensor(arr, dtype=torch.float32, device=device)
    leaf.requires_grad_(True)
    return leaf


def save_figure(fig, technique, muscle, label, idx):
    folder = os.path.join(RESULTS_DIR, technique)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{muscle}_{MODEL_NAME}_{label}_{idx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {path}")


def plot_three_panels(img_np, heatmap, title, technique, muscle, label, idx):
    hm = heatmap.astype(float)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    colored = cm.jet(hm)[:, :, :3]
    overlay = (0.5 * img_np / 255.0 + 0.5 * colored).clip(0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np);          axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(hm, cmap="jet");  axes[1].set_title("Heatmap");  axes[1].axis("off")
    axes[2].imshow(overlay);         axes[2].set_title("Overlay");  axes[2].axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, technique, muscle, label, idx)


def _get_target_layer(model, model_name):
    if model_name in ("resnet18", "resnet50"):  return model.layer4[-1]
    if model_name == "vgg16":                   return model.features[-1]
    if model_name == "densenet121":             return model.features.denseblock4
    if model_name == "efficientnet_b0":         return model.features[-1]
    if model_name == "mobilenet_v3":            return model.features[-1]
    if model_name == "convnext_tiny":           return model.features[-1]
    raise ValueError(f"Unsupported architecture: {model_name}")


def _set_relu_inplace(model, value: bool):
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = value


# ══════════════════════════════════════════════════════════════
#  1. GRAD-CAM
# ══════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model, model_name):
        self.model       = model
        self.activations = None
        self.gradients   = None
        layer = _get_target_layer(model, model_name)
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):   self.activations = o.detach()
    def _bwd(self, m, gi, go): self.gradients   = go[0].detach()

    def generate(self, leaf_inp):
        out = self.model(leaf_inp.unsqueeze(0))
        cls = out.argmax(dim=1).item()
        self.model.zero_grad()
        out[0, cls].backward(retain_graph=True)
        w  = self.gradients.mean(dim=(2, 3), keepdim=True)
        hm = torch.relu((w * self.activations).sum(1)).squeeze().cpu().numpy()
        H, W = Config.IMAGE_SIZE
        return np.array(Image.fromarray(hm).resize((W, H), Image.BILINEAR))


def run_gradcam(model, samples, muscle):
    print(f"  · Grad-CAM for {muscle}...")
    device = next(model.parameters()).device
    gc     = GradCAM(model, MODEL_NAME)
    for idx, (tn, tr, label, _) in enumerate(samples):
        hm = gc.generate(make_leaf(tn, device))
        plot_three_panels(tensor_to_np(tr), hm,
                          f"Grad-CAM | {muscle} | {MODEL_NAME.upper()} | {label}",
                          "gradcam", muscle, label, idx + 1)


# ══════════════════════════════════════════════════════════════
#  2. GUIDED GRAD-CAM
# ══════════════════════════════════════════════════════════════
class GuidedBackprop:
    def __init__(self, model):
        self.model   = model
        self.handles = []
        _set_relu_inplace(model, False)
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                self.handles.append(
                    m.register_full_backward_hook(
                        lambda mod, gin, gout: (torch.clamp(gin[0], min=0),)
                    )
                )

    def generate(self, leaf_inp):
        out = self.model(leaf_inp.unsqueeze(0))
        cls = out.argmax(dim=1).item()
        self.model.zero_grad()
        out[0, cls].backward()
        gbp = leaf_inp.grad.detach().cpu().numpy()
        return np.abs(gbp).mean(axis=0)

    def remove(self):
        for h in self.handles:
            h.remove()
        _set_relu_inplace(self.model, True)


def run_guided_gradcam(model, samples, muscle):
    print(f"  · Guided Grad-CAM for {muscle}...")
    device = next(model.parameters()).device
    gbp    = GuidedBackprop(model)
    gc     = GradCAM(model, MODEL_NAME)

    for idx, (tn, tr, label, _) in enumerate(samples):
        gc_map  = gc.generate(make_leaf(tn, device))
        gc_norm = (gc_map - gc_map.min()) / (gc_map.max() - gc_map.min() + 1e-8)

        gbp_map  = gbp.generate(make_leaf(tn, device))
        gbp_norm = (gbp_map - gbp_map.min()) / (gbp_map.max() - gbp_map.min() + 1e-8)

        guided = gbp_norm * gc_norm
        plot_three_panels(tensor_to_np(tr), guided,
                          f"Guided Grad-CAM | {muscle} | {MODEL_NAME.upper()} | {label}",
                          "guided_gradcam", muscle, label, idx + 1)
    gbp.remove()


# ══════════════════════════════════════════════════════════════
#  3. SALIENCY MAPS
# ══════════════════════════════════════════════════════════════
def run_saliency(model, samples, muscle):
    print(f"  · Saliency Maps for {muscle}...")
    device = next(model.parameters()).device
    for idx, (tn, tr, label, _) in enumerate(samples):
        leaf = make_leaf(tn, device)
        out  = model(leaf.unsqueeze(0))
        model.zero_grad()
        out[0, out.argmax(dim=1).item()].backward()
        sal = leaf.grad.detach().abs().cpu().numpy().max(axis=0)
        plot_three_panels(tensor_to_np(tr), sal,
                          f"Saliency Map | {muscle} | {MODEL_NAME.upper()} | {label}",
                          "saliency", muscle, label, idx + 1)


# ══════════════════════════════════════════════════════════════
#  4. OCCLUSION
# ══════════════════════════════════════════════════════════════
def run_occlusion(model, samples, muscle, patch_size=32, stride=16):
    print(f"  · Occlusion for {muscle}...")
    device = next(model.parameters()).device
    H, W   = Config.IMAGE_SIZE
    for idx, (tn, tr, label, _) in enumerate(samples):
        t = tn.to(device)
        with torch.no_grad():
            probs     = torch.softmax(model(t), dim=1)
            cls       = probs.argmax(dim=1).item()
            base_conf = probs[0, cls].item()

        heatmap = np.zeros((H, W))
        counts  = np.zeros((H, W))
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                occ = t.clone()
                occ[:, :, y:y+patch_size, x:x+patch_size] = 0.5
                with torch.no_grad():
                    conf = torch.softmax(model(occ), dim=1)[0, cls].item()
                drop = base_conf - conf
                heatmap[y:y+patch_size, x:x+patch_size] += drop
                counts[y:y+patch_size,  x:x+patch_size] += 1
        heatmap /= np.where(counts == 0, 1, counts)
        plot_three_panels(tensor_to_np(tr), heatmap,
                          f"Occlusion | {muscle} | {MODEL_NAME.upper()} | {label}",
                          "occlusion", muscle, label, idx + 1)


# ══════════════════════════════════════════════════════════════
#  Summary figure 4×4 (Grad-CAM of the 4 muscles)
# ══════════════════════════════════════════════════════════════
def make_summary_figure(model_per_muscle_samples_hms):
    """
    Generates a single figure with all the Grad-CAM maps in a 4x4 grid
    (4 muscles × 4 images: 2 ALS + 2 Control).
    """
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for row, muscle in enumerate(MUSCLES):
        entries = model_per_muscle_samples_hms.get(muscle, [])
        for col, (tr, hm, label) in enumerate(entries[:4]):
            img_np = tensor_to_np(tr)
            hm_n = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
            colored = cm.jet(hm_n)[:, :, :3]
            overlay = (0.5 * img_np / 255.0 + 0.5 * colored).clip(0, 1)
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f"{muscle} — {label}", fontsize=11)
            axes[row, col].axis("off")
    fig.suptitle(f"Grad-CAM | ResNet-50 | Multi-muscle fusion system",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "summary_gradcam_4x4.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary 4x4 -> {path}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def run_all():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")
    print(f"Results in: {RESULTS_DIR}\n")

    if not os.path.exists(PRED_PATH):
        raise FileNotFoundError(
            f"Missing {PRED_PATH}. Run `python3 train_kfold.py` first."
        )
    with open(PRED_PATH) as f:
        predictions = json.load(f)

    summary_data = {}  # muscle -> [(tensor_raw, hm, label), ...]

    for muscle in MUSCLES:
        fold, auc = best_fold_for(MODEL_NAME, muscle, predictions)
        pth = os.path.join(WEIGHTS_DIR,
                           f"{MODEL_NAME}_{muscle.lower()}_fold{fold}_best.pth")
        print(f"\n{'='*70}")
        print(f"  {muscle.upper()}  ·  fold={fold}  ·  AUC val={auc:.2f}%")
        print(f"  Weights: {pth}")
        print(f"{'='*70}")

        if not os.path.exists(pth):
            print(f"  [SKIP] .pth does not exist")
            continue

        model = get_model(MODEL_NAME).to(device)
        model.load_state_dict(torch.load(pth, map_location=device,
                                         weights_only=True))
        model.eval()

        samples, _, n0, n1 = get_val_samples(muscle, fold)
        print(f"  Selected images: {len(samples)} from val "
              f"(val total: {n0} Control + {n1} ELA)")

        run_gradcam(model, samples, muscle)
        run_guided_gradcam(model, samples, muscle)
        run_saliency(model, samples, muscle)
        run_occlusion(model, samples, muscle)

        # For the summary figure: reuse the first GradCAM per image
        gc = GradCAM(model, MODEL_NAME)
        per_muscle = []
        for tn, tr, label, _ in samples:
            hm = gc.generate(make_leaf(tn, device))
            per_muscle.append((tr, hm, label))
        summary_data[muscle] = per_muscle

    make_summary_figure(summary_data)

    print(f"\n{'='*70}")
    print(f"  EXPLAINABILITY DONE  ·  {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_all()
