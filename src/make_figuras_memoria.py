#!/usr/bin/env python3
"""
Genera las figuras de resultados que aparecen en la memoria del TFG a partir
de los ficheros de resultados (no se vuelve a entrenar ni a inferir nada):

  - memoria/images/confusion_final.png      Matriz de confusión del sistema final.
  - memoria/images/plots/boxplot_auc_combined.png   Boxplots de AUC por fold (2x2).
  - memoria/images/roc/roc_{musculo}.png    Curvas ROC out-of-fold por músculo.

Entradas:
  - models/resultados_kfold/kfold_predictions.json   (AUC por fold + y_true/y_probs)
  - models/resultados_kfold/fusion_per_architecture.csv  (matriz de confusión final)

Uso:
    cd src
    python3 make_figuras_memoria.py
"""
import os
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score

# ── Rutas (relativas a la raíz del repo) ────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "models", "resultados_kfold")
IMG_DIR     = os.path.join(BASE_DIR, "memoria", "images")
PRED_PATH   = os.path.join(RESULTS_DIR, "kfold_predictions.json")
FUSION_CSV  = os.path.join(RESULTS_DIR, "fusion_per_architecture.csv")

# ── Configuración de presentación ───────────────────────────────────────
ARCHS = ["resnet18", "resnet50", "densenet121", "efficientnet_b0", "convnext_tiny"]
DISP  = {"resnet18": "ResNet-18", "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
         "efficientnet_b0": "EfficientNet-B0", "convnext_tiny": "ConvNeXt-Tiny"}
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]  # azul, rojo, verde, naranja, morado
MUSCLES = [("Bicep", "(a) Bíceps"), ("Antebrazo", "(b) Antebrazo"),
           ("Quadriceps", "(c) Cuádriceps"), ("Tibial", "(d) Tibial")]
ROC_FILE = {"Bicep": "roc_bicep", "Antebrazo": "roc_antebrazo",
            "Quadriceps": "roc_quadriceps", "Tibial": "roc_tibial"}

# Sistema final que se ilustra en la matriz de confusión
FINAL_MODEL = "resnet50"
FINAL_RULE  = "mean_youden"   # fusión por media simple + umbral de Youden


# ── Carga de datos ──────────────────────────────────────────────────────
def _load_predictions():
    with open(PRED_PATH) as f:
        return json.load(f)


def _fold_aucs(pred, muscle, arch):
    rows = sorted((r for r in pred if r["muscle"] == muscle and r["model"] == arch),
                  key=lambda r: r["fold"])
    return [r["auc"] for r in rows]


def _oof(pred, muscle, arch):
    rows = [r for r in pred if r["muscle"] == muscle and r["model"] == arch]
    y_true, y_prob = [], []
    for r in rows:
        y_true += r["y_true"]
        y_prob += r["y_probs"]
    return np.array(y_true), np.array(y_prob)


# ── Figura 1: matriz de confusión del sistema final ─────────────────────
def make_confusion():
    row = None
    with open(FUSION_CSV) as f:
        for r in csv.DictReader(f):
            if r["model"] == FINAL_MODEL and r["fusion"] == FINAL_RULE:
                row = r
                break
    if row is None:
        raise SystemExit(f"No encuentro {FINAL_MODEL}/{FINAL_RULE} en {FUSION_CSV}")

    TN, FP, FN, TP = int(row["tn"]), int(row["fp"]), int(row["fn"]), int(row["tp"])
    AUC, ACC, SENS, SPEC = (float(row["auc"]), float(row["acc"]),
                            float(row["sens"]), float(row["spec"]))
    N = TN + FP + FN + TP
    cm = np.array([[TN, FP], [FN, TP]])

    dark, light = "#1f5fa6", "#eef3f9"
    fig, ax = plt.subplots(figsize=(11, 5), dpi=120)
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(3.1, 4.78, f"Sistema final: {DISP[FINAL_MODEL]} + fusión multi-músculo "
            f"({N} pacientes)", fontsize=15, fontweight="bold", ha="center", va="center")

    labels = [["Verdaderos\nnegativos (TN)", "Falsos\npositivos (FP)"],
              ["Falsos\nnegativos (FN)", "Verdaderos\npositivos (TP)"]]
    xs, ys = [1.6, 3.05], [3.55, 2.1]
    for i in range(2):
        for j in range(2):
            cx, cy, diag = xs[j], ys[i], (i == j)
            ax.add_patch(plt.Rectangle((cx - 0.725, cy - 0.725), 1.45, 1.45,
                         facecolor=dark if diag else light, edgecolor="white", linewidth=3))
            ax.text(cx, cy + 0.18, str(cm[i, j]), fontsize=34, fontweight="bold",
                    ha="center", va="center", color="white" if diag else "#1a1a1a")
            ax.text(cx, cy - 0.45, labels[i][j], fontsize=10.5, ha="center", va="center",
                    color="#dce6f2" if diag else "#5a6b7b")
    ax.text(0.45, ys[0], "Real Control", fontsize=12, ha="right", va="center")
    ax.text(0.45, ys[1], "Real ELA", fontsize=12, ha="right", va="center")
    ax.text(xs[0], ys[1] - 1.05, "Predicho Control", fontsize=12, ha="center", va="center")
    ax.text(xs[1], ys[1] - 1.05, "Predicho ELA", fontsize=12, ha="center", va="center")

    box = FancyBboxPatch((6.7, 1.9), 3.7, 1.5, boxstyle="round,pad=0.1,rounding_size=0.12",
                         facecolor="#f5f6f7", edgecolor="#9aa3ab", linewidth=1.3)
    ax.add_patch(box)
    txt = (f"AUC = {AUC:.2f} %\nExactitud = {ACC:.2f} %\n"
           f"Sensibilidad = {SENS:.2f} %\nEspecificidad = {SPEC:.2f} %").replace(".", ",")
    ax.text(6.95, 2.65, txt, fontsize=13.5, ha="left", va="center")

    out = os.path.join(IMG_DIR, "confusion_final.png")
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight", facecolor="white"); plt.close()
    print("  ->", out)


# ── Figura 2: boxplots de AUC por fold (2x2) ────────────────────────────
def make_boxplots(pred):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), dpi=150)
    for ax, (mk, title) in zip(axes.flat, MUSCLES):
        data = [_fold_aucs(pred, mk, a) for a in ARCHS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=2),
                        flierprops=dict(marker="o", markersize=6,
                                        markerfacecolor="white", markeredgecolor="black"))
        for patch, c in zip(bp["boxes"], COLORS):
            patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor("black")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_ylabel("AUC (%)", fontsize=12)
        ax.set_ylim(65, 101); ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(range(1, len(ARCHS) + 1))
        ax.set_xticklabels([DISP[a] for a in ARCHS], rotation=20, ha="right", fontsize=10)
    out = os.path.join(IMG_DIR, "plots", "boxplot_auc_combined.png")
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight", facecolor="white"); plt.close()
    print("  ->", out)


# ── Figura 3: curvas ROC out-of-fold por músculo ────────────────────────
def make_roc(pred):
    for mk, _ in MUSCLES:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
        for a, c in zip(ARCHS, COLORS):
            yt, yp = _oof(pred, mk, a)
            fpr, tpr, _ = roc_curve(yt, yp)
            auc = roc_auc_score(yt, yp) * 100
            ax.plot(fpr, tpr, color=c, lw=1.8, label=f"{DISP[a]} (AUC = {auc:.1f}%)")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_xlabel("Tasa de falsos positivos (1 - Especificidad)", fontsize=11)
        ax.set_ylabel("Tasa de verdaderos positivos (Sensibilidad)", fontsize=11)
        ax.set_title(f"Curvas ROC -- {mk}", fontsize=13)
        ax.legend(loc="lower right", fontsize=10); ax.grid(alpha=0.3)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
        out = os.path.join(IMG_DIR, "roc", ROC_FILE[mk] + ".png")
        plt.tight_layout(); plt.savefig(out, bbox_inches="tight", facecolor="white"); plt.close()
        print("  ->", out)


def main():
    os.makedirs(os.path.join(IMG_DIR, "plots"), exist_ok=True)
    os.makedirs(os.path.join(IMG_DIR, "roc"), exist_ok=True)
    pred = _load_predictions()
    print("Matriz de confusión:"); make_confusion()
    print("Boxplots por fold:");   make_boxplots(pred)
    print("Curvas ROC:");          make_roc(pred)
    print("Hecho.")


if __name__ == "__main__":
    main()
