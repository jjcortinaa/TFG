"""
explainability.py
─────────────────
Mapas de explicabilidad (Grad-CAM, Guided Grad-CAM, Saliency, Occlusion)
para el SISTEMA FINAL del TFG: ResNet-50 entrenado por separado en cada
uno de los 4 músculos con 5-fold StratifiedGroupKFold.

Para cada músculo:
  1. Selecciona el fold con mayor AUC en val (mejor checkpoint).
  2. Carga los pesos correspondientes desde best_models_kfold/.
  3. Reconstruye el conjunto de validación de ese fold (mismas seed y
     particiones que durante entrenamiento).
  4. Toma 2 imágenes ALS + 2 Control de la val set (out-of-fold honesto:
     el modelo no las vio en entrenamiento).
  5. Genera los 4 tipos de mapas y los guarda.

Salida
------
models/resultados_kfold/explainability_resnet50/
    gradcam/
    guided_gradcam/
    saliency/
    occlusion/

Uso: python3 explainability.py
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


# ── Configuración ─────────────────────────────────────────────
# Sistema final: ResNet-50 en los 4 músculos. Esta selección viene del
# análisis de fusión a nivel paciente (informe_fusion.txt): ResNet-50
# fusionado da 99.15% AUC y 98.11% Acc, ganador entre las 5 arquitecturas.
MODEL_NAME = "resnet50"
MUSCLES    = ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]

WEIGHTS_DIR = os.path.join(Config.BASE_DIR, "best_models_kfold")
PRED_PATH   = os.path.join(Config.BASE_DIR, "models", "resultados_kfold",
                           "kfold_predictions.json")
RESULTS_DIR = os.path.join(Config.BASE_DIR, "models", "resultados_kfold",
                           "explainability_resnet50")

# Cuántas imágenes ALS y Control por músculo
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


# ── Selección del mejor fold por músculo ──────────────────────
def best_fold_for(model_name, muscle, predictions):
    """De kfold_predictions.json, fold con mayor AUC para (model, muscle)."""
    candidates = [r for r in predictions
                  if r["model"] == model_name and r["muscle"] == muscle]
    if not candidates:
        raise RuntimeError(f"No hay predicciones para {model_name}/{muscle}")
    best = max(candidates, key=lambda r: r["auc"])
    return best["fold"], best["auc"]


# ── Reconstrucción de la val set del fold ─────────────────────
def get_val_samples(muscle, fold_idx, n_als=N_ALS, n_ctrl=N_CONTROL):
    """
    Reconstruye exactamente la val set del fold pedido y devuelve
    n_als imágenes ELA + n_ctrl Control.
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
        raise ValueError(f"Fold {fold_idx} fuera de rango (1..{len(folds)})")
    _, val_idx = folds[fold_idx - 1]

    # Separar índices por clase (label 0=Control, 1=ELA según ImageFolder)
    by_class = {0: [], 1: []}
    for i in val_idx:
        by_class[labels[i]].append(i)

    samples = []
    # 2 ELA primero, después 2 Control (orden estable: por nombre de archivo)
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
    print(f"    Guardado -> {path}")


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
    raise ValueError(f"Arquitectura no soportada: {model_name}")


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
    print(f"  · Grad-CAM para {muscle}...")
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
    print(f"  · Guided Grad-CAM para {muscle}...")
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
    print(f"  · Saliency Maps para {muscle}...")
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
    print(f"  · Occlusion para {muscle}...")
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
#  Figura resumen 4×4 (Grad-CAM de los 4 músculos)
# ══════════════════════════════════════════════════════════════
def make_summary_figure(model_per_muscle_samples_hms):
    """
    Genera una única figura con todos los Grad-CAM en grid 4x4
    (4 músculos × 4 imágenes: 2 ELA + 2 Control).
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
    fig.suptitle(f"Grad-CAM | ResNet-50 | Sistema fusión multi-músculo",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "summary_gradcam_4x4.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nResumen 4×4 -> {path}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════
def run_all():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Dispositivo: {device}")
    print(f"Resultados en: {RESULTS_DIR}\n")

    if not os.path.exists(PRED_PATH):
        raise FileNotFoundError(
            f"Falta {PRED_PATH}. Ejecuta primero `python3 train_kfold.py`."
        )
    with open(PRED_PATH) as f:
        predictions = json.load(f)

    summary_data = {}  # músculo -> [(tensor_raw, hm, label), ...]

    for muscle in MUSCLES:
        fold, auc = best_fold_for(MODEL_NAME, muscle, predictions)
        pth = os.path.join(WEIGHTS_DIR,
                           f"{MODEL_NAME}_{muscle.lower()}_fold{fold}_best.pth")
        print(f"\n{'='*70}")
        print(f"  {muscle.upper()}  ·  fold={fold}  ·  AUC val={auc:.2f}%")
        print(f"  Pesos: {pth}")
        print(f"{'='*70}")

        if not os.path.exists(pth):
            print(f"  [SKIP] No existe el .pth")
            continue

        model = get_model(MODEL_NAME).to(device)
        model.load_state_dict(torch.load(pth, map_location=device,
                                         weights_only=True))
        model.eval()

        samples, _, n0, n1 = get_val_samples(muscle, fold)
        print(f"  Imágenes seleccionadas: {len(samples)} de val "
              f"(val total: {n0} Control + {n1} ELA)")

        run_gradcam(model, samples, muscle)
        run_guided_gradcam(model, samples, muscle)
        run_saliency(model, samples, muscle)
        run_occlusion(model, samples, muscle)

        # Para la figura resumen: reusa el primer GradCAM por imagen
        gc = GradCAM(model, MODEL_NAME)
        per_muscle = []
        for tn, tr, label, _ in samples:
            hm = gc.generate(make_leaf(tn, device))
            per_muscle.append((tr, hm, label))
        summary_data[muscle] = per_muscle

    make_summary_figure(summary_data)

    print(f"\n{'='*70}")
    print(f"  EXPLICABILIDAD COMPLETADA  ·  {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_all()
