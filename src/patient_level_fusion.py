"""
patient_level_fusion.py
───────────────────────
Patient-level fusion: combines the predictions of the 4 muscles into a
single ALS / Control verdict per patient.

Why
---
Per-muscle CV gives image/fold-level metrics. In clinical practice we want
a system that, given a patient, produces ONE final decision. This is done
by aggregating in two steps:
    1) For each patient and each muscle: mean of P(ALS) over ALL their
       images in that muscle (out-of-fold; each patient is in val exactly
       once in each per-muscle CV).
    2) For each patient: fusion of the up-to-4 muscle P(ALS) values with
       different rules (mean, vote, AUC-weighted mean).

No overfitting: no new model is trained. The per-muscle predictions are
OOF (out-of-fold); the model never saw the patient in its fold.

"Per-architecture" fusion (five rows — one per architecture):
    densenet121 fused across its 4 muscles, etc.

"Per-muscle champions" fusion:
    densenet121-bicep + efficientnet_b0-antebrazo + resnet18-quadriceps
    + resnet18-tibial.

Output
------
- models/resultados_kfold/oof_predictions.json
- models/resultados_kfold/fusion_per_architecture.csv
- models/resultados_kfold/fusion_champions.csv
- models/resultados_kfold/informe_fusion.txt
"""
import os
import re
import json
import csv
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

from config import Config
from models import get_model

# ── Constants ──────────────────────────────────────────────────────────
RESULTS_DIR  = os.path.join(Config.BASE_DIR, "models", "resultados_kfold")
WEIGHTS_DIR  = os.path.join(Config.BASE_DIR, "best_models_kfold")
OOF_PATH     = os.path.join(RESULTS_DIR, "oof_predictions.json")
ARCH_CSV     = os.path.join(RESULTS_DIR, "fusion_per_architecture.csv")
CHAMP_CSV    = os.path.join(RESULTS_DIR, "fusion_champions.csv")
REPORT_PATH  = os.path.join(RESULTS_DIR, "informe_fusion.txt")

MUSCLES   = ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]
MODELS    = ["resnet18", "resnet50", "densenet121", "efficientnet_b0", "convnext_tiny"]

# Per-muscle champions chosen from the statistical report
CHAMPIONS = {
    "Bicep":      "densenet121",
    "Antebrazo":  "efficientnet_b0",
    "Quadriceps": "resnet18",
    "Tibial":     "resnet18",
}

# Mean AUC per (model, muscle) — for the "AUC-weighted mean" rule.
# Taken from informe_estadistico.txt section 1.
AUC_TABLE = {
    ("resnet18",        "Bicep"): 86.25,
    ("resnet50",        "Bicep"): 86.35,
    ("densenet121",     "Bicep"): 90.30,
    ("efficientnet_b0", "Bicep"): 85.86,
    ("convnext_tiny",   "Bicep"): 86.28,
    ("resnet18",        "Antebrazo"):  91.38,
    ("resnet50",        "Antebrazo"):  89.76,
    ("densenet121",     "Antebrazo"):  88.73,
    ("efficientnet_b0", "Antebrazo"):  91.71,
    ("convnext_tiny",   "Antebrazo"):  91.03,
    ("resnet18",        "Quadriceps"): 99.33,
    ("resnet50",        "Quadriceps"): 99.33,
    ("densenet121",     "Quadriceps"): 99.33,
    ("efficientnet_b0", "Quadriceps"): 99.13,
    ("convnext_tiny",   "Quadriceps"): 99.33,
    ("resnet18",        "Tibial"):     93.96,
    ("resnet50",        "Tibial"):     92.37,
    ("densenet121",     "Tibial"):     93.17,
    ("efficientnet_b0", "Tibial"):     90.98,
    ("convnext_tiny",   "Tibial"):     92.38,
}


# ── Patient identification normalized across muscles ───────────────────
# Dataset naming convention (confirmed with the supervisor):
# the same control patient appears as "C001" in Bicep/Antebrazo/Quadriceps
# and as "RC001" in Tibial. Here we normalize to the common ID "C001" + class.
_NUMERIC_RE = re.compile(r"^R?(C?\d+)")

def patient_uid(filename: str, class_name: str) -> str:
    """Returns a unique patient ID normalized across muscles."""
    m = _NUMERIC_RE.match(filename)
    if not m:
        return f"{class_name}/UNPARSED-{filename}"
    return f"{class_name}/{m.group(1)}"


# ── Split reconstruction + OOF inference ───────────────────────────────
def _val_transform():
    return transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _build_muscle_dataset(muscle: str):
    """Returns (dataset, labels, groups). Same seed/order as dataset.py."""
    path = os.path.join(Config.PROCESSED_DATA_PATH, muscle)
    ds = datasets.ImageFolder(root=path, transform=_val_transform())
    labels, groups = [], []
    for fp, lbl in ds.samples:
        cls = ds.classes[lbl]
        labels.append(lbl)
        groups.append(patient_uid(os.path.basename(fp), cls))
    return ds, labels, groups


def generate_oof_predictions(device=None):
    """
    For each (model, muscle, fold) loads the .pth, rebuilds the split with
    the same seed and runs inference over val. Saves P(ALS) per image
    together with its patient_id, label and fold.
    """
    if device is None:
        device = (torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))
    print(f"[device] {device}")

    out = []  # each element: one image dict
    for muscle in MUSCLES:
        ds, labels, groups = _build_muscle_dataset(muscle)
        sgkf = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=Config.SEED,
        )
        # Pre-compute val_idx per fold (same order as training)
        folds = list(sgkf.split(X=range(len(ds)), y=labels, groups=groups))

        for model_name in MODELS:
            print(f"  · OOF {model_name:<18} {muscle:<10}", end=" ", flush=True)
            for k, (_, val_idx) in enumerate(folds, start=1):
                pth = os.path.join(
                    WEIGHTS_DIR,
                    f"{model_name}_{muscle.lower()}_fold{k}_best.pth",
                )
                if not os.path.exists(pth):
                    print(f"[skip fold {k}: no .pth]", end=" ")
                    continue

                model = get_model(model_name).to(device)
                state = torch.load(pth, map_location=device)
                model.load_state_dict(state)
                model.eval()

                loader = DataLoader(
                    Subset(ds, val_idx),
                    batch_size=Config.BATCH_SIZE, shuffle=False,
                )
                idx_iter = iter(val_idx)  # to pair with paths
                with torch.no_grad():
                    for images, lbls in loader:
                        images = images.to(device)
                        out_logits = model(images)
                        probs = torch.softmax(out_logits, dim=1)[:, 1].cpu().numpy()
                        preds = out_logits.argmax(dim=1).cpu().numpy()
                        for j, (p, pr, lb) in enumerate(zip(probs, preds, lbls.tolist())):
                            global_idx = next(idx_iter)
                            fp, _ = ds.samples[global_idx]
                            pid   = groups[global_idx]
                            out.append({
                                "model":   model_name,
                                "muscle":  muscle,
                                "fold":    k,
                                "patient": pid,
                                "label":   int(lb),
                                "prob":    float(p),
                                "pred":    int(pr),
                                "image":   os.path.basename(fp),
                            })
                # free GPU/MPS memory before the next checkpoint
                del model, state
            print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OOF_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nOOF predictions -> {OOF_PATH}  ({len(out)} rows)")
    return out


# ── Patient × muscle aggregation ────────────────────────────────────────
def aggregate_per_patient_muscle(oof):
    """
    Groups by (model, muscle, patient). Returns a dict:
        agg[(model, muscle)][patient] = {prob: mean, label: int}
    """
    bucket = defaultdict(lambda: defaultdict(list))
    label_of = {}
    for r in oof:
        key = (r["model"], r["muscle"])
        bucket[key][r["patient"]].append(r["prob"])
        label_of[r["patient"]] = r["label"]
    agg = {}
    for key, by_pat in bucket.items():
        agg[key] = {pid: {"prob": float(np.mean(probs)),
                          "label": label_of[pid]}
                    for pid, probs in by_pat.items()}
    return agg


# ── Fusion rules ─────────────────────────────────────────────────────────
def fuse_mean(probs_per_muscle):
    return float(np.mean(probs_per_muscle))

def fuse_weighted(probs_per_muscle, weights):
    w = np.array(weights, dtype=float)
    p = np.array(probs_per_muscle, dtype=float)
    return float((w * p).sum() / w.sum())

def fuse_majority_vote(probs_per_muscle, threshold=0.5):
    votes = sum(1 for p in probs_per_muscle if p >= threshold)
    return 1 if votes > len(probs_per_muscle) / 2 else 0


# ── Patient-level metrics ───────────────────────────────────────────────
def _metrics_from_pred(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = max(1, tp + tn + fp + fn)
    return {
        "n":    tp + tn + fp + fn,
        "acc":  100 * (tp + tn) / n,
        "sens": 100 * tp / max(1, tp + fn),
        "spec": 100 * tn / max(1, tn + fp),
        "tp":   tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _best_youden_threshold(y_true, y_score):
    cand = np.unique(np.concatenate([[0.0, 1.0], y_score]))
    best = (-1.0, 0.5)
    for t in cand:
        y_pred = (y_score >= t).astype(int)
        m = _metrics_from_pred(y_true, y_pred)
        j = (m["sens"] + m["spec"]) / 100 - 1
        if j > best[0]:
            best = (j, float(t))
    return best[1]


def _bootstrap_patient_ci(y_true, scores, threshold, n_boot=1000, seed=42):
    """
    Percentile bootstrap 95% CI at the patient level for Acc/Sens/Spec/AUC.

    Resamples the set of patients N times with replacement. In each
    resample it recomputes the 4 metrics using the fixed threshold. Returns
    the 2.5 and 97.5 percentiles. Resamples with a single class are skipped
    (Sens/Spec/AUC cannot be computed).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    accs, senss, specs, aucs = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, sc = y_true[idx], scores[idx]
        if len(set(yt)) < 2:
            continue
        pred = (sc >= threshold).astype(int)
        m = _metrics_from_pred(yt, pred)
        accs.append(m["acc"]); senss.append(m["sens"]); specs.append(m["spec"])
        try:
            aucs.append(100 * roc_auc_score(yt, sc))
        except ValueError:
            pass

    def _q(x):
        return (round(float(np.quantile(x, 0.025)), 2),
                round(float(np.quantile(x, 0.975)), 2)) if x else (None, None)

    return {
        "acc_ci_low":  _q(accs)[0],  "acc_ci_high":  _q(accs)[1],
        "sens_ci_low": _q(senss)[0], "sens_ci_high": _q(senss)[1],
        "spec_ci_low": _q(specs)[0], "spec_ci_high": _q(specs)[1],
        "auc_ci_low":  _q(aucs)[0],  "auc_ci_high":  _q(aucs)[1],
    }


# ── Per-architecture fusion ─────────────────────────────────────────────
def fusion_per_architecture(agg):
    rows = []
    for model in MODELS:
        # For this architecture, collect predictions per patient×muscle
        by_pat = defaultdict(dict)  # patient -> {muscle: prob, "label": lbl}
        for muscle in MUSCLES:
            patients = agg.get((model, muscle), {})
            for pid, info in patients.items():
                by_pat[pid][muscle] = info["prob"]
                by_pat[pid]["label"] = info["label"]

        if not by_pat:
            continue

        # Build per-patient vectors
        y_true, mean_probs, weighted_probs, votes = [], [], [], []
        n_muscles_seen = []
        for pid, info in by_pat.items():
            probs  = [info[m] for m in MUSCLES if m in info]
            wts    = [AUC_TABLE[(model, m)] for m in MUSCLES if m in info]
            if not probs:
                continue
            y_true.append(info["label"])
            mean_probs.append(fuse_mean(probs))
            weighted_probs.append(fuse_weighted(probs, wts))
            votes.append(fuse_majority_vote(probs))
            n_muscles_seen.append(len(probs))

        y_true        = np.array(y_true)
        mean_probs    = np.array(mean_probs)
        weighted_probs= np.array(weighted_probs)

        # Metrics at threshold 0.5 and at the optimal (Youden) threshold over the fusion
        for label, score in [("mean_youden",     mean_probs),
                             ("weighted_youden", weighted_probs)]:
            thr = _best_youden_threshold(y_true, score)
            pred = (score >= thr).astype(int)
            m = _metrics_from_pred(y_true, pred)
            try:
                auc = 100 * roc_auc_score(y_true, score)
            except ValueError:
                auc = float("nan")
            ci = _bootstrap_patient_ci(y_true, score, thr)
            rows.append({
                "model":      model,
                "fusion":     label,
                "n_patients": int(len(y_true)),
                "muscles_avg":round(float(np.mean(n_muscles_seen)), 2),
                "thr":        round(thr, 3),
                "auc":        round(auc, 2),
                "auc_ci_low": ci["auc_ci_low"],   "auc_ci_high": ci["auc_ci_high"],
                "acc":        round(m["acc"], 2),
                "acc_ci_low": ci["acc_ci_low"],   "acc_ci_high": ci["acc_ci_high"],
                "sens":       round(m["sens"], 2),
                "sens_ci_low":ci["sens_ci_low"],  "sens_ci_high":ci["sens_ci_high"],
                "spec":       round(m["spec"], 2),
                "spec_ci_low":ci["spec_ci_low"],  "spec_ci_high":ci["spec_ci_high"],
                "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
            })

        # Majority vote (binary decision, no meaningful continuous AUC)
        votes_arr = np.array(votes)
        m = _metrics_from_pred(y_true, votes_arr)
        rows.append({
            "model":      model,
            "fusion":     "majority_vote",
            "n_patients": int(len(y_true)),
            "muscles_avg":round(float(np.mean(n_muscles_seen)), 2),
            "thr":        0.5,
            "auc":        None,
            "auc_ci_low": None, "auc_ci_high": None,
            "acc":        round(m["acc"], 2),
            "acc_ci_low": None, "acc_ci_high": None,
            "sens":       round(m["sens"], 2),
            "sens_ci_low":None, "sens_ci_high":None,
            "spec":       round(m["spec"], 2),
            "spec_ci_low":None, "spec_ci_high":None,
            "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
        })

    rows.sort(key=lambda r: (r["model"], r["fusion"]))
    _write_csv(ARCH_CSV, rows)
    return rows


# ── "Per-muscle champions" fusion ───────────────────────────────────────
def fusion_champions(agg):
    """Combines the 4 champion models (one per muscle) -> a single row per rule."""
    by_pat = defaultdict(dict)
    for muscle in MUSCLES:
        model = CHAMPIONS[muscle]
        for pid, info in agg.get((model, muscle), {}).items():
            by_pat[pid][muscle] = info["prob"]
            by_pat[pid]["label"] = info["label"]

    y_true, mean_probs, weighted_probs, votes = [], [], [], []
    n_muscles_seen = []
    for pid, info in by_pat.items():
        probs = [info[m] for m in MUSCLES if m in info]
        wts   = [AUC_TABLE[(CHAMPIONS[m], m)] for m in MUSCLES if m in info]
        if not probs:
            continue
        y_true.append(info["label"])
        mean_probs.append(fuse_mean(probs))
        weighted_probs.append(fuse_weighted(probs, wts))
        votes.append(fuse_majority_vote(probs))
        n_muscles_seen.append(len(probs))

    y_true         = np.array(y_true)
    mean_probs     = np.array(mean_probs)
    weighted_probs = np.array(weighted_probs)

    rows = []
    for label, score in [("mean_youden", mean_probs),
                         ("weighted_youden", weighted_probs)]:
        thr = _best_youden_threshold(y_true, score)
        pred = (score >= thr).astype(int)
        m = _metrics_from_pred(y_true, pred)
        try:
            auc = 100 * roc_auc_score(y_true, score)
        except ValueError:
            auc = float("nan")
        ci = _bootstrap_patient_ci(y_true, score, thr)
        rows.append({
            "fusion":     label,
            "n_patients": int(len(y_true)),
            "muscles_avg":round(float(np.mean(n_muscles_seen)), 2),
            "thr":        round(thr, 3),
            "auc":        round(auc, 2),
            "auc_ci_low": ci["auc_ci_low"],   "auc_ci_high": ci["auc_ci_high"],
            "acc":        round(m["acc"], 2),
            "acc_ci_low": ci["acc_ci_low"],   "acc_ci_high": ci["acc_ci_high"],
            "sens":       round(m["sens"], 2),
            "sens_ci_low":ci["sens_ci_low"],  "sens_ci_high":ci["sens_ci_high"],
            "spec":       round(m["spec"], 2),
            "spec_ci_low":ci["spec_ci_low"],  "spec_ci_high":ci["spec_ci_high"],
            "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
        })

    votes_arr = np.array(votes)
    m = _metrics_from_pred(y_true, votes_arr)
    rows.append({
        "fusion":     "majority_vote",
        "n_patients": int(len(y_true)),
        "muscles_avg":round(float(np.mean(n_muscles_seen)), 2),
        "thr":        0.5,
        "auc":        None,
        "auc_ci_low": None, "auc_ci_high": None,
        "acc":        round(m["acc"], 2),
        "acc_ci_low": None, "acc_ci_high": None,
        "sens":       round(m["sens"], 2),
        "sens_ci_low":None, "sens_ci_high":None,
        "spec":       round(m["spec"], 2),
        "spec_ci_low":None, "spec_ci_high":None,
        "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
    })

    _write_csv(CHAMP_CSV, rows)
    return rows


# ── Report (kept in Spanish: it ships with the Spanish thesis) ──────────
def write_report(arch_rows, champ_rows):
    with open(REPORT_PATH, "w") as f:
        f.write("="*78 + "\n")
        f.write("  INFORME DE FUSIÓN A NIVEL PACIENTE — TFG ELA\n")
        f.write("="*78 + "\n\n")

        f.write("MOTIVACIÓN\n")
        f.write("-"*78 + "\n")
        f.write("Cada CV-músculo da P(ALS) out-of-fold por imagen. Agregamos a paciente\n")
        f.write("por la media de sus imágenes en ese músculo, y fusionamos los 4 músculos\n")
        f.write("con tres reglas (media simple, media ponderada por AUC, voto mayoritario).\n\n")

        f.write("CONVENCIÓN DE NOMBRADO (confirmada con el tutor): 'Cnnn' en\n")
        f.write("Bicep/Antebrazo/Quadriceps y 'RCnnn' en Tibial designan al MISMO\n")
        f.write("sujeto control. El ID se normaliza en patient_uid().\n\n")

        def _fmt_ci(lo, hi, suffix="%"):
            if lo is None or hi is None:
                return "    --    "
            return f"[{lo:>5.1f},{hi:>6.1f}]{suffix}"

        f.write("\n1) FUSIÓN POR ARQUITECTURA (cada modelo combina sus 4 músculos)\n")
        f.write("    n=52 pacientes; IC95% por bootstrap (N=1000) sobre pacientes.\n")
        f.write("-"*78 + "\n")
        for r in arch_rows:
            auc = f"{r['auc']:>5.2f}" if r['auc'] is not None else "  -- "
            tnfp = f"{r['tp']}/{r['tn']}/{r['fp']}/{r['fn']}"
            f.write(f"\n  {r['model'].upper()}  ·  {r['fusion']}  ·  thr={r['thr']:.3f}  ·  TP/TN/FP/FN={tnfp}\n")
            if r.get('fusion') == 'majority_vote':
                f.write(f"     Acc={r['acc']:>5.2f}%  Sens={r['sens']:>5.2f}%  Spec={r['spec']:>5.2f}%\n")
            else:
                f.write(f"     AUC ={auc}%  IC95% {_fmt_ci(r.get('auc_ci_low'),  r.get('auc_ci_high'))}\n")
                f.write(f"     Acc ={r['acc']:>5.2f}%  IC95% {_fmt_ci(r.get('acc_ci_low'),  r.get('acc_ci_high'))}\n")
                f.write(f"     Sens={r['sens']:>5.2f}%  IC95% {_fmt_ci(r.get('sens_ci_low'), r.get('sens_ci_high'))}\n")
                f.write(f"     Spec={r['spec']:>5.2f}%  IC95% {_fmt_ci(r.get('spec_ci_low'), r.get('spec_ci_high'))}\n")

        f.write("\n\n2) FUSIÓN DE LOS CAMPEONES POR MÚSCULO\n")
        f.write("    Bicep:densenet121  Antebrazo:efficientnet_b0  "
                "Quadriceps:resnet18  Tibial:resnet18\n")
        f.write("-"*78 + "\n")
        for r in champ_rows:
            auc = f"{r['auc']:>5.2f}" if r['auc'] is not None else "  -- "
            tnfp = f"{r['tp']}/{r['tn']}/{r['fp']}/{r['fn']}"
            f.write(f"\n  {r['fusion']}  ·  thr={r['thr']:.3f}  ·  TP/TN/FP/FN={tnfp}\n")
            if r.get('fusion') == 'majority_vote':
                f.write(f"     Acc={r['acc']:>5.2f}%  Sens={r['sens']:>5.2f}%  Spec={r['spec']:>5.2f}%\n")
            else:
                f.write(f"     AUC ={auc}%  IC95% {_fmt_ci(r.get('auc_ci_low'),  r.get('auc_ci_high'))}\n")
                f.write(f"     Acc ={r['acc']:>5.2f}%  IC95% {_fmt_ci(r.get('acc_ci_low'),  r.get('acc_ci_high'))}\n")
                f.write(f"     Sens={r['sens']:>5.2f}%  IC95% {_fmt_ci(r.get('sens_ci_low'), r.get('sens_ci_high'))}\n")
                f.write(f"     Spec={r['spec']:>5.2f}%  IC95% {_fmt_ci(r.get('spec_ci_low'), r.get('spec_ci_high'))}\n")

        f.write("\n\nNotas\n")
        f.write("-----\n")
        f.write("- P(ALS) por (paciente, músculo) = media de las imágenes de ese\n")
        f.write("  paciente en ese músculo, usando el modelo del fold en el que el\n")
        f.write("  paciente quedó en validación (out-of-fold honesto).\n")
        f.write("- 'mean_youden' / 'weighted_youden' aplican Youden's J sobre la\n")
        f.write("  probabilidad fusionada para encontrar el umbral óptimo.\n")
        f.write("- 'majority_vote' usa umbral 0.5 por músculo y declara ALS si ≥3 de 4\n")
        f.write("  músculos votan ALS (≥2 si solo hay 4 disponibles → estricto).\n")
        f.write("- n_patients=número de pacientes con al menos un músculo. muscles_avg\n")
        f.write("  es el promedio de músculos disponibles por paciente.\n")
    print(f"  Report -> {REPORT_PATH}")


# ── Utilities ──────────────────────────────────────────────────────────
def _write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"  CSV -> {path}")


# ── Main ───────────────────────────────────────────────────────────────
def main(skip_inference=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if skip_inference and os.path.exists(OOF_PATH):
        with open(OOF_PATH) as f:
            oof = json.load(f)
        print(f"[loaded] {OOF_PATH}  ({len(oof)} rows)")
    else:
        print("[1/3] Generating OOF predictions...")
        oof = generate_oof_predictions()

    print("\n[2/3] Patient×muscle aggregation...")
    agg = aggregate_per_patient_muscle(oof)
    n_combos = sum(len(v) for v in agg.values())
    print(f"    {len(agg)} (model, muscle) — {n_combos} patient×muscle preds")

    print("\n[3/3] Fusion and report...")
    arch_rows  = fusion_per_architecture(agg)
    champ_rows = fusion_champions(agg)
    write_report(arch_rows, champ_rows)

    print("\nDone. Results in:", RESULTS_DIR)


if __name__ == "__main__":
    import sys
    skip = "--skip-inference" in sys.argv
    main(skip_inference=skip)
