"""
evaluate_saved.py
─────────────────
Carga los modelos ya entrenados desde TFG/best_models/ y calcula
todas las métricas sin volver a entrenar nada.

Uso desde TFG/src/:
    python3 evaluate_saved.py
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import csv

from sklearn.metrics import confusion_matrix, roc_auc_score
from dataset import get_dataloaders
from models import get_model
from config import Config

# Donde están los .pth guardados por train.py / evaluation.py
MODELS_DIR  = os.path.join(Config.BASE_DIR, "best_models")

# Donde se guardan los resultados
RESULTS_DIR = os.path.join(Config.BASE_DIR, "models", "resultados_comparativa")

MUSCLES     = ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]
MODEL_NAMES = ["resnet18", "resnet50", "vgg16", "densenet121", "efficientnet_b0", "mobilenet_v3"]


def evaluate_one(model_name: str, muscle_name: str, device):
    """
    Carga el .pth de un (modelo, músculo) y devuelve sus métricas.
    Si no existe el fichero .pth lo indica y devuelve None.
    """
    pth_path = os.path.join(MODELS_DIR, f"{model_name}_{muscle_name.lower()}_best.pth")

    if not os.path.exists(pth_path):
        print(f"  [SKIP] No encontrado: {pth_path}")
        return None

    print(f"  Cargando {pth_path}")

    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
    model.eval()

    _, val_loader, class_names = get_dataloaders(muscle_name=muscle_name)

    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    cm  = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs) * 100 if len(set(y_true)) > 1 else 0

    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = tp = fp = fn = 0

    total = tp + tn + fp + fn
    acc  = 100 * (tp + tn) / total if total > 0 else 0
    sens = 100 * tp / (tp + fn)    if (tp + fn) > 0 else 0
    spec = 100 * tn / (tn + fp)    if (tn + fp) > 0 else 0

    print(f"  -> Acc:{acc:.1f}%  Sens:{sens:.1f}%  Spec:{spec:.1f}%  AUC:{auc:.1f}%")

    return {
        "model": model_name, "muscle": muscle_name,
        "acc": acc, "sens": sens, "spec": spec, "auc": auc,
        "cm": cm, "class_names": class_names
    }


# ── Plots ──────────────────────────────────────────────────────────────

def save_cm_plot(r):
    plots_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=r["class_names"], yticklabels=r["class_names"])
    plt.title(f"Confusion Matrix\n{r['model'].upper()} - {r['muscle']}", fontsize=11)
    plt.ylabel("True Label (Gold Standard)")
    plt.xlabel("Predicted Label (AI)")
    plt.tight_layout()
    path = os.path.join(plots_dir, f"cm_{r['model']}_{r['muscle'].lower()}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  CM plot -> {path}")


def save_bar_chart(results, metric="auc"):
    muscles     = ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]
    model_names = list(dict.fromkeys(r["model"] for r in results))

    data = {m: [0.0] * len(muscles) for m in model_names}
    for r in results:
        if r["muscle"] in muscles:
            data[r["model"]][muscles.index(r["muscle"])] = r[metric]

    x      = np.arange(len(muscles))
    width  = 0.8 / len(model_names)
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (mname, color) in enumerate(zip(model_names, colors)):
        bars = ax.bar(x + i * width - (len(model_names) - 1) * width / 2,
                      data[mname], width * 0.9,
                      label=mname.upper(), color=color)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Muscle Group", fontsize=12)
    ax.set_ylabel(f"{metric.upper()} (%)", fontsize=12)
    ax.set_title(f"Deep Learning Comparison - {metric.upper()} by Muscle",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(muscles, fontsize=11)
    ax.set_ylim(0, 115)
    ax.axhline(90, color="gray", linestyle="--", linewidth=1,
               label="90% target (Martinez-Paya 2017)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plots_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"bar_{metric}_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Bar chart ({metric}) -> {path}")


def save_radar_chart(results):
    muscles     = ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]
    model_names = list(dict.fromkeys(r["model"] for r in results))
    auc_matrix  = {(r["model"], r["muscle"]): r["auc"] for r in results}

    angles = np.linspace(0, 2 * np.pi, len(muscles), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    for mname, color in zip(model_names, colors):
        values = [auc_matrix.get((mname, m), 0) for m in muscles]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=mname.upper(), color=color)
        ax.fill(angles, values, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(muscles, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title("AUC (%) by Model and Muscle Group", y=1.08,
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.tight_layout()
    plots_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "radar_auc_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Radar chart -> {path}")


# ── Reports ────────────────────────────────────────────────────────────

def save_reports(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, "resultados_comparativa.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "muscle", "acc", "sens", "spec", "auc"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow({k: round(r[k], 2) if isinstance(r[k], float) else r[k]
                             for k in ["model", "muscle", "acc", "sens", "spec", "auc"]})
    print(f"  CSV -> {csv_path}")

    txt_path = os.path.join(RESULTS_DIR, "final_report.txt")
    with open(txt_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("  COMPARATIVE DEEP LEARNING REPORT - ALS MUSCLE ULTRASOUND\n")
        f.write("="*70 + "\n\n")
        f.write("  Reference baseline (Martinez-Paya 2017):\n")
        f.write("  Biceps    EV+MTh    -> AUC 92.6% | Sens 88% | Spec 83%\n")
        f.write("  Antebrazo GLCM+MTh  -> AUC 90.5% | Sens 81% | Spec 79%\n")
        f.write("  Quadriceps GLCM+MTh -> AUC 98.3% | Sens 94% | Spec 96%\n")
        f.write("  Tibialis  EV+MTh    -> AUC 95.3% | Sens 85% | Spec 92%\n\n")


        for muscle in ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]:
            muscle_results = sorted(
                [r for r in results if r["muscle"] == muscle],
                key=lambda r: r["auc"], reverse=True
            )
            if not muscle_results:
                continue
            f.write(f"  -- {muscle.upper()} --\n")
            f.write(f"  {'MODEL':<20} {'ACC':>8} {'SENS':>8} {'SPEC':>8} {'AUC':>8}\n")
            f.write("  " + "-"*52 + "\n")
            for r in muscle_results:
                f.write(f"  {r['model']:<20} "
                        f"{r['acc']:>7.1f}% "
                        f"{r['sens']:>7.1f}% "
                        f"{r['spec']:>7.1f}% "
                        f"{r['auc']:>7.1f}%\n")
            f.write("\n")
    print(f"  TXT report -> {txt_path}")


# ── Main ───────────────────────────────────────────────────────────────

def evaluate_all(muscles=None, model_names=None):
    """
    Recorre todos los (modelo x músculo), carga los .pth que existan
    en TFG/best_models/ y genera todas las métricas y gráficas.
    Los modelos no entrenados aún se saltan automáticamente.
    """
    if muscles     is None: muscles     = MUSCLES
    if model_names is None: model_names = MODEL_NAMES

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDispositivo : {device}")
    print(f"Buscando modelos en: {MODELS_DIR}")
    print(f"Guardando resultados en: {RESULTS_DIR}\n")

    results = []
    for muscle in muscles:
        print(f"\n{'='*60}")
        print(f"  MÚSCULO: {muscle.upper()}")
        print(f"{'='*60}")
        for model_name in model_names:
            r = evaluate_one(model_name, muscle, device)
            if r is not None:
                results.append(r)
                save_cm_plot(r)

    if not results:
        print("\n[AVISO] No se encontró ningún modelo entrenado en:", MODELS_DIR)
        print("Ejecuta primero: python3 train.py")
        return

    # Gráficas comparativas (solo si hay más de un resultado)
    if len(results) > 1:
        save_bar_chart(results, metric="auc")
        save_bar_chart(results, metric="acc")
        save_bar_chart(results, metric="sens")
        save_radar_chart(results)

    save_reports(results)

    # Tabla resumen en consola
    print("\n" + "="*70)
    print("  RESULTADOS FINALES (ordenados por AUC)")
    print("="*70)
    print(f"  {'MODEL':<20} {'MUSCLE':<14} {'ACC':>8} {'SENS':>8} {'SPEC':>8} {'AUC':>8}")
    print("  " + "-"*66)
    for r in sorted(results, key=lambda r: r["auc"], reverse=True):
        print(f"  {r['model']:<20} {r['muscle']:<14} "
              f"{r['acc']:>7.1f}% {r['sens']:>7.1f}% "
              f"{r['spec']:>7.1f}% {r['auc']:>7.1f}%")
    print("="*70)
    return results


if __name__ == "__main__":
    evaluate_all()
