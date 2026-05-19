"""
train_kfold.py
──────────────
Entrenamiento con 5-fold StratifiedGroupKFold para cada (modelo, músculo).

Por qué
-------
El split único 80/20 con ~22 imágenes en validación es frágil: 1 imagen
mal clasificada ≈ 4.5 puntos de accuracy. Con 5 folds obtenemos una
media y una desviación estándar por cada (modelo, músculo), lo cual:
  (1) reduce la dependencia del split concreto,
  (2) permite calcular un intervalo de confianza al 95% para el AUC,
  (3) habilita tests estadísticos pareados entre arquitecturas.

Salida
------
- best_models_kfold/{modelo}_{musculo}_fold{k}_best.pth  (pesos de cada fold)
- models/resultados_kfold/kfold_predictions.json          (y_true, y_probs, y_pred por fold)
- models/resultados_kfold/kfold_summary.csv               (media ± std por (modelo, músculo))

Uso
---
    cd src
    python3 train_kfold.py
    # opcional: una sola arquitectura
    # python3 -c "from train_kfold import train_kfold_all; train_kfold_all(model_names=['resnet18'])"
"""
import os
import json
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from dataset import get_kfold_splits
from models import get_model, get_all_model_names
from config import Config


# Directorios de salida
SAVE_DIR     = os.path.join(Config.BASE_DIR, "best_models_kfold")
RESULTS_DIR  = os.path.join(Config.BASE_DIR, "models", "resultados_kfold")
PRED_PATH    = os.path.join(RESULTS_DIR, "kfold_predictions.json")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "kfold_summary.csv")

MUSCLES     = ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]

# Arquitecturas seleccionadas para la comparativa del TFG.
# Se descartan VGG-16 (138 M params → overfitting esperado en dataset pequeño)
# y MobileNet-V3 Small (redundante con EfficientNet-B0 en la narrativa de
# "arquitectura moderna eficiente"). Se incluye ConvNeXt-Tiny como
# representante de la convergencia arquitectónica post-Transformer.
MODEL_NAMES = [
    "resnet18",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "convnext_tiny",
]


# ── Entrenamiento de un único fold ─────────────────────────────────────
def _train_single_fold(model_name, muscle_name, fold_idx,
                       train_loader, val_loader, device):
    """
    Entrena una combinación (modelo, músculo) en UN fold concreto.
    Selecciona el checkpoint por mejor AUC en validación
    (más estable que accuracy con val pequeño).

    Devuelve un dict con métricas finales y predicciones del mejor epoch.
    """
    model     = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best = {
        "auc": -1.0, "acc": 0.0, "sens": 0.0, "spec": 0.0,
        "epoch": 0, "y_true": [], "y_pred": [], "y_probs": [],
    }
    best_weights = None

    for epoch in range(Config.EPOCHS):
        # -- Training --
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # -- Validation --
        model.eval()
        y_true, y_pred, y_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out   = model(images)
                probs = torch.softmax(out, dim=1)[:, 1]
                _, pred = torch.max(out, 1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())
                y_probs.extend(probs.cpu().tolist())

        # Métricas
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        total = tp + tn + fp + fn
        acc  = 100 * (tp + tn) / total if total else 0.0
        sens = 100 * tp / (tp + fn) if (tp + fn) else 0.0
        spec = 100 * tn / (tn + fp) if (tn + fp) else 0.0
        auc  = (roc_auc_score(y_true, y_probs) * 100
                if len(set(y_true)) > 1 else 0.0)

        scheduler.step(auc)
        print(f"    [Fold {fold_idx}] Ep {epoch+1:>3}/{Config.EPOCHS} "
              f"Loss={running_loss/len(train_loader):.4f} "
              f"Acc={acc:.1f}% Sens={sens:.1f}% Spec={spec:.1f}% AUC={auc:.1f}%")

        if auc > best["auc"]:
            best = {
                "auc": auc, "acc": acc, "sens": sens, "spec": spec,
                "epoch": epoch + 1,
                "y_true": y_true, "y_pred": y_pred, "y_probs": y_probs,
            }
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Guardar pesos del fold
    os.makedirs(SAVE_DIR, exist_ok=True)
    pth_path = os.path.join(
        SAVE_DIR, f"{model_name}_{muscle_name.lower()}_fold{fold_idx}_best.pth"
    )
    if best_weights is not None:
        torch.save(best_weights, pth_path)

    print(f"    [Fold {fold_idx}] BEST (ep {best['epoch']}): "
          f"Acc={best['acc']:.1f}% Sens={best['sens']:.1f}% "
          f"Spec={best['spec']:.1f}% AUC={best['auc']:.1f}%  ->  {pth_path}")

    return best


# ── Bucle sobre todos los folds de un (modelo, músculo) ────────────────
def train_kfold_combination(model_name, muscle_name, n_splits=5, device=None):
    """
    Entrena los `n_splits` folds de una combinación. Devuelve lista de
    dicts (uno por fold) con métricas y predicciones.
    """
    if device is None:
        device = (torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))

    print(f"\n{'='*70}")
    print(f"  MODEL: {model_name.upper()}  |  MUSCLE: {muscle_name.upper()}  "
          f"|  {n_splits}-FOLD CV")
    print(f"  Device: {device}")
    print(f"{'='*70}")

    results = []
    for k, (train_loader, val_loader, _) in enumerate(
        get_kfold_splits(muscle_name=muscle_name, n_splits=n_splits, verbose=True),
        start=1,
    ):
        t0 = time.time()
        r = _train_single_fold(
            model_name, muscle_name, k, train_loader, val_loader, device
        )
        r["fold"]   = k
        r["model"]  = model_name
        r["muscle"] = muscle_name
        r["time_s"] = round(time.time() - t0, 1)
        results.append(r)

    # Resumen de la combinación
    aucs  = [r["auc"]  for r in results]
    accs  = [r["acc"]  for r in results]
    sens_ = [r["sens"] for r in results]
    specs = [r["spec"] for r in results]
    print(f"\n  [{model_name} / {muscle_name}] resumen k-fold:")
    print(f"    AUC  = {np.mean(aucs):.2f}% ± {np.std(aucs):.2f}%   "
          f"(folds: {['%.1f' % a for a in aucs]})")
    print(f"    Acc  = {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
    print(f"    Sens = {np.mean(sens_):.2f}% ± {np.std(sens_):.2f}%")
    print(f"    Spec = {np.mean(specs):.2f}% ± {np.std(specs):.2f}%")
    return results


# ── Entrada principal: todas las combinaciones ─────────────────────────
def train_kfold_all(muscles=None, model_names=None, n_splits=5):
    """
    Entrena todas las combinaciones (modelo × músculo) con k-fold CV.
    Persiste predicciones y resumen en `models/resultados_kfold/`.
    """
    if muscles     is None: muscles     = MUSCLES
    if model_names is None: model_names = MODEL_NAMES

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    total_combinations = len(muscles) * len(model_names)
    done = 0
    t_start = time.time()

    for muscle in muscles:
        for model_name in model_names:
            done += 1
            print(f"\n\n###### {done}/{total_combinations} "
                  f"({model_name} x {muscle}) ######")
            try:
                res = train_kfold_combination(
                    model_name, muscle, n_splits=n_splits, device=device
                )
                all_results.extend(res)
                # Guardar JSON incremental — así si algo peta no perdemos nada
                _save_predictions_json(all_results)
                _save_summary_csv(all_results)
            except Exception as e:
                print(f"[ERROR] {model_name} / {muscle}: {e}")

    elapsed = (time.time() - t_start) / 60.0
    print(f"\n\n===== FINALIZADO en {elapsed:.1f} min =====")
    print(f"  Predicciones -> {PRED_PATH}")
    print(f"  Resumen      -> {SUMMARY_PATH}")
    return all_results


# ── Persistencia ───────────────────────────────────────────────────────
def _save_predictions_json(results):
    """
    Guarda TODAS las predicciones por fold. Este fichero es la entrada
    de `statistical_tests.py`. Lo reescribimos cada vez para que sea
    seguro ante caídas.
    """
    payload = []
    for r in results:
        payload.append({
            "model":   r["model"],
            "muscle":  r["muscle"],
            "fold":    r["fold"],
            "epoch":   r["epoch"],
            "acc":     round(r["acc"],  3),
            "sens":    round(r["sens"], 3),
            "spec":    round(r["spec"], 3),
            "auc":     round(r["auc"],  3),
            "y_true":  r["y_true"],
            "y_pred":  r["y_pred"],
            "y_probs": [round(p, 6) for p in r["y_probs"]],
        })
    with open(PRED_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def _save_summary_csv(results):
    """
    Agrega por (modelo, músculo) con media ± std de cada métrica.
    """
    by_combo = {}
    for r in results:
        key = (r["model"], r["muscle"])
        by_combo.setdefault(key, []).append(r)

    rows = []
    for (model, muscle), folds in by_combo.items():
        aucs  = [f["auc"]  for f in folds]
        accs  = [f["acc"]  for f in folds]
        sens_ = [f["sens"] for f in folds]
        specs = [f["spec"] for f in folds]
        rows.append({
            "model":    model,
            "muscle":   muscle,
            "n_folds":  len(folds),
            "auc_mean":  round(float(np.mean(aucs)),  3),
            "auc_std":   round(float(np.std(aucs)),   3),
            "acc_mean":  round(float(np.mean(accs)),  3),
            "acc_std":   round(float(np.std(accs)),   3),
            "sens_mean": round(float(np.mean(sens_)), 3),
            "sens_std":  round(float(np.std(sens_)),  3),
            "spec_mean": round(float(np.mean(specs)), 3),
            "spec_std":  round(float(np.std(specs)),  3),
        })

    rows.sort(key=lambda r: (r["muscle"], -r["auc_mean"]))

    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # OPCIÓN A: todas las combinaciones (largo — ~10x el tiempo de un entrenamiento normal)
    train_kfold_all()

    # OPCIÓN B: prueba rápida con una arquitectura ligera
    # train_kfold_all(model_names=["mobilenet_v3"], muscles=["Bicep"])
