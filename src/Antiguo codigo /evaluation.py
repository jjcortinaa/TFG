import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import csv

from sklearn.metrics import confusion_matrix, roc_auc_score
from dataset import get_dataloaders
from models import get_model, get_all_model_names
from config import Config

# All results go to: TFG/models/resultados_comparativa/
RESULTS_DIR = os.path.join(Config.BASE_DIR, "models", "resultados_comparativa")


# ══════════════════════════════════════════════════════════════════════
#  Core: train + evaluate one (model, muscle) combination
# ══════════════════════════════════════════════════════════════════════

def train_and_evaluate(model_name: str, muscle_name: str, device):
    """Best = highest AUC on the validation set."""
    train_loader, val_loader, class_names = get_dataloaders(muscle_name=muscle_name)

    model     = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best = {"acc": 0, "sens": 0, "spec": 0, "auc": 0, "cm": None}
    best_weights = None

    for epoch in range(Config.EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
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
        auc = roc_auc_score(y_true, y_probs) if len(set(y_true)) > 1 else 0

        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = tp = fp = fn = 0

        total = tp + tn + fp + fn
        acc  = 100 * (tp + tn) / total if total > 0 else 0
        sens = 100 * tp / (tp + fn)    if (tp + fn) > 0 else 0
        spec = 100 * tn / (tn + fp)    if (tn + fp) > 0 else 0

        scheduler.step(auc * 100)
        print(f"    Epoch [{epoch+1:>3}/{Config.EPOCHS}]  "
              f"Acc:{acc:.1f}%  Sens:{sens:.1f}%  Spec:{spec:.1f}%  AUC:{auc:.3f}")

        if auc > best["auc"]:
            best = {"acc": acc, "sens": sens, "spec": spec,
                    "auc": auc * 100, "cm": cm}
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save to TFG/best_models/
    save_dir = os.path.join(Config.BASE_DIR, "best_models")
    os.makedirs(save_dir, exist_ok=True)
    if best_weights:
        torch.save(
            best_weights,
            os.path.join(save_dir, f"{model_name}_{muscle_name.lower()}_best.pth")
        )

    best["model"]       = model_name
    best["muscle"]      = muscle_name
    best["class_names"] = class_names
    return best


# ══════════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════════

def save_cm_plot(cm, model_name, muscle_name, class_names):
    plots_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix\n{model_name.upper()} - {muscle_name}", fontsize=11)
    plt.ylabel("True Label (Gold Standard)")
    plt.xlabel("Predicted Label (AI)")
    plt.tight_layout()
    path = os.path.join(plots_dir, f"cm_{model_name}_{muscle_name.lower()}.png")
    plt.savefig(path, dpi=150)
    plt.close()


def save_radar_chart(summary_results):
    muscles     = ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]
    model_names = list(dict.fromkeys(r["model"] for r in summary_results))
    auc_matrix  = {(r["model"], r["muscle"]): r["auc"] for r in summary_results}

    angles = np.linspace(0, 2 * np.pi, len(muscles), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    for model_name, color in zip(model_names, colors):
        values = [auc_matrix.get((model_name, m), 0) for m in muscles]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model_name.upper(), color=color)
        ax.fill(angles, values, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(muscles, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title("AUC (%) by Model and Muscle Group", y=1.08, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plots", "radar_auc_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Radar chart -> {path}")


def save_bar_chart(summary_results, metric="auc"):
    muscles     = ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]
    model_names = list(dict.fromkeys(r["model"] for r in summary_results))

    data = {m: [0.0] * len(muscles) for m in model_names}
    for r in summary_results:
        if r["muscle"] in muscles:
            data[r["model"]][muscles.index(r["muscle"])] = r[metric]

    x      = np.arange(len(muscles))
    width  = 0.8 / len(model_names)
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        bars = ax.bar(x + i * width - (len(model_names) - 1) * width / 2,
                      data[model_name], width * 0.9,
                      label=model_name.upper(), color=color)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Muscle Group", fontsize=12)
    ax.set_ylabel(f"{metric.upper()} (%)", fontsize=12)
    ax.set_title(f"Deep Learning Model Comparison - {metric.upper()} by Muscle",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(muscles, fontsize=11)
    ax.set_ylim(0, 115)
    ax.axhline(90, color="gray", linestyle="--", linewidth=1,
               label="90% target (Martinez-Paya 2017)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plots", f"bar_{metric}_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Bar chart ({metric}) -> {path}")


# ══════════════════════════════════════════════════════════════════════
#  Reports
# ══════════════════════════════════════════════════════════════════════

def save_reports(summary_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # CSV (easy to open in Excel)
    csv_path = os.path.join(RESULTS_DIR, "resultados_comparativa.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "muscle", "acc", "sens", "spec", "auc"]
        )
        writer.writeheader()
        for r in summary_results:
            writer.writerow({k: round(r[k], 2) if isinstance(r[k], float) else r[k]
                             for k in ["model", "muscle", "acc", "sens", "spec", "auc"]})
    print(f"  CSV -> {csv_path}")

    # TXT human-readable report
    txt_path = os.path.join(RESULTS_DIR, "final_report.txt")
    with open(txt_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("  COMPARATIVE DEEP LEARNING REPORT - ALS MUSCLE ULTRASOUND\n")
        f.write("="*70 + "\n\n")
        f.write("  Reference baseline (Martinez-Paya 2017):\n")
        f.write("  Quadriceps GLCM+MTh -> AUC 0.983 | Sens 94% | Spec 96%\n")
        f.write("  Tibialis  EV+MTh    -> AUC 0.953 | Sens 85% | Spec 92%\n\n")

        for muscle in ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]:
            f.write(f"  -- {muscle.upper()} --\n")
            f.write(f"  {'MODEL':<20} {'ACC':>8} {'SENS':>8} {'SPEC':>8} {'AUC':>8}\n")
            f.write("  " + "-"*52 + "\n")
            for r in sorted(
                [r for r in summary_results if r["muscle"] == muscle],
                key=lambda r: r["auc"], reverse=True
            ):
                f.write(f"  {r['model']:<20} "
                        f"{r['acc']:>7.1f}% "
                        f"{r['sens']:>7.1f}% "
                        f"{r['spec']:>7.1f}% "
                        f"{r['auc']:>7.1f}%\n")
            f.write("\n")
    print(f"  TXT report -> {txt_path}")


# ══════════════════════════════════════════════════════════════════════
#  Main orchestrator
# ══════════════════════════════════════════════════════════════════════

def run_full_comparison(muscles=None, model_names=None):
    """
    Trains and evaluates every (model x muscle) combination.
    Generates CM plots, bar charts, radar chart, CSV and TXT report.
    All output goes to TFG/models/resultados_comparativa/
    """
    if muscles is None:
        muscles = ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]
    if model_names is None:
        model_names = get_all_model_names()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice : {device}")
    print(f"Models : {model_names}")
    print(f"Muscles: {muscles}")
    print(f"Output : {RESULTS_DIR}\n")

    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

    summary_results = []
    for muscle in muscles:
        for model_name in model_names:
            print(f"\n{'─'*60}")
            print(f"  >> {model_name.upper()}  x  {muscle.upper()}")
            print(f"{'─'*60}")
            try:
                result = train_and_evaluate(model_name, muscle, device)
                summary_results.append(result)
                if result["cm"] is not None:
                    save_cm_plot(result["cm"], model_name,
                                 muscle, result["class_names"])
            except Exception as e:
                print(f"  [ERROR] {model_name} / {muscle}: {e}")

    save_bar_chart(summary_results, metric="auc")
    save_bar_chart(summary_results, metric="acc")
    save_bar_chart(summary_results, metric="sens")
    save_radar_chart(summary_results)
    save_reports(summary_results)

    print("\n" + "="*70)
    print("  FINAL COMPARISON TABLE (sorted by AUC)")
    print("="*70)
    print(f"  {'MODEL':<20} {'MUSCLE':<14} {'ACC':>8} {'SENS':>8} {'SPEC':>8} {'AUC':>8}")
    print("  " + "-"*66)
    for r in sorted(summary_results, key=lambda r: r["auc"], reverse=True):
        print(f"  {r['model']:<20} {r['muscle']:<14} "
              f"{r['acc']:>7.1f}% {r['sens']:>7.1f}% "
              f"{r['spec']:>7.1f}% {r['auc']:>7.1f}%")
    print("="*70)
    return summary_results


if __name__ == "__main__":
    # OPTION A: Full comparison (all models x all muscles)
    run_full_comparison()

    # OPTION B: Quick test — one muscle, two models
    # run_full_comparison(
    #     muscles=["Quadriceps"],
    #     model_names=["vgg16", "efficientnet_b0"]
    # )
