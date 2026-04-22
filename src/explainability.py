"""
explainability.py
─────────────────
Grad-CAM, Guided Grad-CAM, Saliency Maps y Occlusion
para los mejores modelos por músculo.

Uso: python3 explainability.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob
from PIL import Image
from torchvision import transforms
from models import get_model
from config import Config

# ── Configuración ─────────────────────────────────────────────
BEST_MODELS = {
    "Quadriceps": "resnet50",
    "Tibial":     "densenet121",
    "Antebrazo":  "densenet121",
    "Bicep":      "densenet121",
}

MODELS_DIR  = os.path.join(Config.BASE_DIR, "best_models")
DATA_DIR    = os.path.join(Config.BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(Config.BASE_DIR, "models",
                           "resultados_comparativa", "explainability")
N_IMAGES = 2

TRANSFORM = transforms.Compose([
    transforms.Resize(Config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

TRANSFORM_RAW = transforms.Compose([
    transforms.Resize(Config.IMAGE_SIZE),
    transforms.ToTensor(),
])


# ══════════════════════════════════════════════════════════════
#  Utilidades comunes
# ══════════════════════════════════════════════════════════════

def load_model(model_name, muscle_name, device):
    pth = os.path.join(MODELS_DIR, f"{model_name}_{muscle_name.lower()}_best.pth")
    if not os.path.exists(pth):
        raise FileNotFoundError(f"No encontrado: {pth}")
    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(pth, map_location=device, weights_only=True))
    model.eval()
    return model


def get_sample_images(muscle_name, n=2):
    samples = []
    for folder_name, label in [("0_Control", "Control"), ("1_ELA", "ELA")]:
        folder = os.path.join(DATA_DIR, muscle_name, folder_name)
        if not os.path.exists(folder):
            print(f"  [WARN] No existe: {folder}")
            continue
        paths = sorted(
            glob.glob(os.path.join(folder, "*.jpg")) +
            glob.glob(os.path.join(folder, "*.png"))
        )[:n]
        for path in paths:
            img = Image.open(path).convert("RGB")
            samples.append((
                TRANSFORM(img).unsqueeze(0),
                TRANSFORM_RAW(img).unsqueeze(0),
                label, path
            ))
    return samples


def tensor_to_np(tensor_raw):
    img = tensor_raw.squeeze(0).permute(1, 2, 0).numpy()
    return (img * 255).clip(0, 255).astype(np.uint8)


def make_leaf(tensor, device):
    """Crea un tensor leaf con requires_grad=True para garantizar .grad"""
    arr = tensor.squeeze(0).detach().cpu().numpy()
    leaf = torch.tensor(arr, dtype=torch.float32, device=device)
    leaf.requires_grad_(True)
    return leaf


def save_figure(fig, technique, muscle, model_name, label, idx):
    folder = os.path.join(RESULTS_DIR, technique)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{muscle}_{model_name}_{label}_{idx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado -> {path}")


def plot_three_panels(img_np, heatmap, title, technique,
                      muscle, model_name, label, idx):
    hm      = heatmap.astype(float)
    hm      = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    colored = cm.jet(hm)[:, :, :3]
    overlay = (0.5 * img_np / 255.0 + 0.5 * colored).clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np);          axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(hm, cmap="jet");  axes[1].set_title("Heatmap");  axes[1].axis("off")
    axes[2].imshow(overlay);         axes[2].set_title("Overlay");  axes[2].axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, technique, muscle, model_name, label, idx)


def _get_target_layer(model, model_name):
    if model_name in ("resnet18", "resnet50"):  return model.layer4[-1]
    if model_name == "vgg16":                   return model.features[-1]
    if model_name == "densenet121":             return model.features.denseblock4
    if model_name == "efficientnet_b0":         return model.features[-1]
    if model_name == "mobilenet_v3":            return model.features[-1]
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


def run_gradcam(model, model_name, samples, muscle):
    print(f"  Generando Grad-CAM para {muscle} ({model_name})...")
    device = next(model.parameters()).device
    gc     = GradCAM(model, model_name)
    for idx, (tensor, tensor_raw, label, _) in enumerate(samples):
        hm = gc.generate(make_leaf(tensor, device))
        plot_three_panels(tensor_to_np(tensor_raw), hm,
                          f"Grad-CAM | {muscle} | {model_name.upper()} | {label}",
                          "gradcam", muscle, model_name, label, idx + 1)


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


def run_guided_gradcam(model, model_name, samples, muscle):
    print(f"  Generando Guided Grad-CAM para {muscle} ({model_name})...")
    device = next(model.parameters()).device
    gbp    = GuidedBackprop(model)
    gc     = GradCAM(model, model_name)

    for idx, (tensor, tensor_raw, label, _) in enumerate(samples):
        gc_map   = gc.generate(make_leaf(tensor, device))
        gc_norm  = (gc_map - gc_map.min()) / (gc_map.max() - gc_map.min() + 1e-8)

        gbp_map  = gbp.generate(make_leaf(tensor, device))
        gbp_norm = (gbp_map - gbp_map.min()) / (gbp_map.max() - gbp_map.min() + 1e-8)

        guided_gc = gbp_norm * gc_norm
        plot_three_panels(tensor_to_np(tensor_raw), guided_gc,
                          f"Guided Grad-CAM | {muscle} | {model_name.upper()} | {label}",
                          "guided_gradcam", muscle, model_name, label, idx + 1)

    gbp.remove()


# ══════════════════════════════════════════════════════════════
#  3. SALIENCY MAPS
# ══════════════════════════════════════════════════════════════

def run_saliency(model, model_name, samples, muscle):
    print(f"  Generando Saliency Maps para {muscle} ({model_name})...")
    device = next(model.parameters()).device
    for idx, (tensor, tensor_raw, label, _) in enumerate(samples):
        leaf = make_leaf(tensor, device)
        out  = model(leaf.unsqueeze(0))
        model.zero_grad()
        out[0, out.argmax(dim=1).item()].backward()
        sal = leaf.grad.detach().abs().cpu().numpy().max(axis=0)
        plot_three_panels(tensor_to_np(tensor_raw), sal,
                          f"Saliency Map | {muscle} | {model_name.upper()} | {label}",
                          "saliency", muscle, model_name, label, idx + 1)


# ══════════════════════════════════════════════════════════════
#  4. OCCLUSION
# ══════════════════════════════════════════════════════════════

def run_occlusion(model, model_name, samples, muscle,
                  patch_size=32, stride=16):
    print(f"  Generando Occlusion para {muscle} ({model_name})...")
    device = next(model.parameters()).device
    H, W   = Config.IMAGE_SIZE

    for idx, (tensor, tensor_raw, label, _) in enumerate(samples):
        t = tensor.to(device)
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
        plot_three_panels(tensor_to_np(tensor_raw), heatmap,
                          f"Occlusion | {muscle} | {model_name.upper()} | {label}",
                          "occlusion", muscle, model_name, label, idx + 1)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def run_all():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDispositivo: {device}")
    print(f"Resultados en: {RESULTS_DIR}\n")

    for muscle, model_name in BEST_MODELS.items():
        print(f"\n{'='*60}")
        print(f"  MÚSCULO: {muscle.upper()}  |  MODELO: {model_name.upper()}")
        print(f"{'='*60}")
        try:
            model   = load_model(model_name, muscle, device)
            samples = get_sample_images(muscle, n=N_IMAGES)
            if not samples:
                print(f"  [SKIP] Sin imágenes en {muscle}")
                continue
            ela  = sum(1 for s in samples if s[2] == "ELA")
            ctrl = sum(1 for s in samples if s[2] == "Control")
            print(f"  Imágenes cargadas: {len(samples)} ({ela} ELA + {ctrl} Control)")

            run_gradcam(model, model_name, samples, muscle)
            run_guided_gradcam(model, model_name, samples, muscle)
            run_saliency(model, model_name, samples, muscle)
            run_occlusion(model, model_name, samples, muscle)

        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
        except Exception as e:
            import traceback
            print(f"  [ERROR] {muscle}/{model_name}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("  EXPLICABILIDAD COMPLETADA")
    print(f"  Resultados en: {RESULTS_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_all()