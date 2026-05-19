"""
statistical_tests.py
────────────────────
Análisis estadístico sobre los resultados de `train_kfold.py`.

Qué calcula
-----------
1) IC 95 % del AUC medio por (modelo, músculo) con dos métodos:
     - t-Student sobre los 5 AUCs por fold  (principal)
     - bootstrap (N=1000) sobre los 5 AUCs  (complementario, no paramétrico)
   Nota: NO se usa bootstrap sobre predicciones concatenadas porque cada
   fold proviene de un modelo distinto con calibración propia y mezclar
   sus probabilidades deteriora el ranking global.
2) Comparación vs. baseline clínico de Martínez-Payá 2017:
   - Si el AUC publicado cae FUERA del IC95% t-Student, declaramos
     "diferencia estadísticamente significativa" (p<0.05).
   - Si cae DENTRO, el resultado es "equivalente / no concluyente".
3) Métricas clínicas recalibradas con Youden's J:
   - Búsqueda del umbral óptimo por fold que maximiza Sens+Spec-1.
   - Comparación frente al umbral fijo 0.5 que usa el modelo por defecto.
4) Comparación pareada entre arquitecturas dentro de un mismo músculo:
   - Test de Wilcoxon signed-rank sobre los 5 AUCs por fold (pareado).
   - Test de DeLong sobre predicciones concatenadas (ROC real).
   - Test de McNemar sobre aciertos/fallos binarios.

Entrada
-------
`models/resultados_kfold/kfold_predictions.json` (generado por train_kfold.py)

Salida
------
- models/resultados_kfold/ci_bootstrap.csv           (IC95% por modelo/músculo)
- models/resultados_kfold/vs_baseline.csv            (comparación con Martínez-Payá)
- models/resultados_kfold/recalibrated_youden.csv    (Sens/Spec a umbral óptimo)
- models/resultados_kfold/pairwise_models.csv        (tests entre arquitecturas)
- models/resultados_kfold/plots/boxplot_auc_{músculo}.png
- models/resultados_kfold/informe_estadistico.txt

Uso
---
    cd src
    python3 statistical_tests.py
"""
import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations

from sklearn.metrics import roc_auc_score
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

from config import Config


# ── Constantes ─────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(Config.BASE_DIR, "models", "resultados_kfold")
PRED_PATH   = os.path.join(RESULTS_DIR, "kfold_predictions.json")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")

# Baseline publicado (Martínez-Payá 2017). AUC en porcentaje.
BASELINE = {
    "Bicep":      {"auc": 92.6, "sens": 88.0, "spec": 83.0, "tech": "EV+MTh"},
    "Antebrazo":  {"auc": 90.5, "sens": 81.0, "spec": 79.0, "tech": "GLCM+MTh"},
    "Quadriceps": {"auc": 98.3, "sens": 94.0, "spec": 96.0, "tech": "GLCM+MTh"},
    "Tibial":     {"auc": 95.3, "sens": 85.0, "spec": 92.0, "tech": "EV+MTh"},
}

N_BOOTSTRAP = 1000
RNG_SEED    = Config.SEED


# ── Carga de datos ─────────────────────────────────────────────────────
def load_predictions():
    if not os.path.exists(PRED_PATH):
        raise FileNotFoundError(
            f"No encuentro {PRED_PATH}. Ejecuta primero `python3 train_kfold.py`."
        )
    with open(PRED_PATH) as f:
        data = json.load(f)
    # Agrupar por (modelo, músculo)
    grouped = {}
    for r in data:
        key = (r["model"], r["muscle"])
        grouped.setdefault(key, []).append(r)
    return grouped


def _concat_predictions(folds):
    """Concatena y_true, y_probs, y_pred de los 5 folds en un único vector."""
    y_true  = np.array([y for f in folds for y in f["y_true"]])
    y_pred  = np.array([y for f in folds for y in f["y_pred"]])
    y_probs = np.array([p for f in folds for p in f["y_probs"]])
    return y_true, y_pred, y_probs


# ── 1. Intervalo de confianza del AUC medio por fold ──────────────────
# Nota metodológica: el bootstrap "por concatenación" (muestrear sobre las
# predicciones de los 5 folds fusionadas) es incorrecto cuando cada fold
# viene de un modelo distinto con su propia calibración de probabilidades,
# porque al mezclar probabilidades de modelos distintos el ranking global
# se deteriora artificialmente y el IC colapsa.
# Lo correcto para k-fold CV con n pequeño es el IC de Student sobre los
# k AUCs por fold: mean ± t(n-1, α/2) · std / √n .
# Complementamos con un bootstrap *de los propios AUCs por fold* (muestreo
# con reemplazo de los 5 valores) para tener también un IC no paramétrico.
def auc_ci_from_folds(fold_aucs, alpha=0.05, n_boot=N_BOOTSTRAP):
    """
    IC para la media del AUC sobre los k folds.
    Devuelve: (mean, t_low, t_high, boot_low, boot_high) en porcentaje.
    """
    arr = np.asarray(fold_aucs, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    # IC t-Student (principal)
    if n > 1:
        std = float(arr.std(ddof=1))
        t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
        half = t_crit * std / np.sqrt(n)
    else:
        half = 0.0
    t_low  = max(0.0, mean - half)
    t_high = min(100.0, mean + half)
    # IC bootstrap sobre los k AUCs (secundario, no paramétrico)
    rng = np.random.default_rng(RNG_SEED)
    means = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    b_low  = float(np.quantile(means, alpha / 2))
    b_high = float(np.quantile(means, 1 - alpha / 2))
    return mean, t_low, t_high, b_low, b_high


def run_ci_per_model_muscle(grouped):
    rows = []
    for (model, muscle), folds in grouped.items():
        per_fold_auc = [f["auc"] for f in folds]   # ya en porcentaje
        mean, t_low, t_high, b_low, b_high = auc_ci_from_folds(per_fold_auc)
        rows.append({
            "model":    model,
            "muscle":   muscle,
            "n_folds":  len(folds),
            "auc_fold_mean":  round(mean,   2),
            "auc_fold_std":   round(float(np.std(per_fold_auc, ddof=1)), 2),
            "auc_t_ci_low":   round(t_low,  2),
            "auc_t_ci_high":  round(t_high, 2),
            "auc_boot_ci_low":  round(b_low,  2),
            "auc_boot_ci_high": round(b_high, 2),
        })
    rows.sort(key=lambda r: (r["muscle"], -r["auc_fold_mean"]))
    _write_csv(os.path.join(RESULTS_DIR, "ci_bootstrap.csv"), rows)
    return rows


# ── 2. Comparación vs baseline ─────────────────────────────────────────
def compare_vs_baseline(ci_rows):
    """
    Para cada (modelo, músculo): ¿el AUC del baseline cae DENTRO o FUERA
    del IC95% (t-Student) de nuestro modelo? Si fuera y por encima → el
    baseline es mejor. Si fuera y por debajo → nuestro modelo es mejor.

    Usamos el IC t-Student como principal (apropiado para la media de k
    AUCs por fold con k=5). El IC bootstrap se guarda como complemento.
    """
    rows = []
    for r in ci_rows:
        base = BASELINE[r["muscle"]]["auc"]
        low, high = r["auc_t_ci_low"], r["auc_t_ci_high"]
        if base < low:
            verdict = "NUESTRO MEJOR (p<0.05)"
        elif base > high:
            verdict = "BASELINE MEJOR (p<0.05)"
        else:
            verdict = "equivalente / no concluyente"
        rows.append({
            "model":            r["model"],
            "muscle":           r["muscle"],
            "our_auc":          r["auc_fold_mean"],
            "our_t_ci_low":     r["auc_t_ci_low"],
            "our_t_ci_high":    r["auc_t_ci_high"],
            "our_boot_ci_low":  r["auc_boot_ci_low"],
            "our_boot_ci_high": r["auc_boot_ci_high"],
            "baseline_auc":     base,
            "baseline_tech":    BASELINE[r["muscle"]]["tech"],
            "verdict":          verdict,
        })
    rows.sort(key=lambda r: (r["muscle"], -r["our_auc"]))
    _write_csv(os.path.join(RESULTS_DIR, "vs_baseline.csv"), rows)
    return rows


# ── 2b. Recalibración clínica con Youden's J por fold ──────────────────
# El modelo se guarda con el umbral 0.5, que no es óptimo si las clases
# están desbalanceadas o el modelo está mal calibrado. Para reportar
# Sens/Spec honestos, buscamos en CADA fold el umbral que maximiza el
# índice de Youden J = Sens + Spec - 1 sobre sus propias predicciones, y
# promediamos las métricas resultantes entre folds.
def _metrics_at_threshold(y_true, y_probs, thr):
    y_pred = (np.asarray(y_probs) >= thr).astype(int)
    y_true = np.asarray(y_true)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc  = (tp + tn) / max(1, (tp + tn + fp + fn))
    return sens * 100, spec * 100, acc * 100


def _best_youden_threshold(y_true, y_probs):
    """Busca el umbral que maximiza Sens+Spec-1 barriendo los probs únicos."""
    y_true  = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    # Incluimos 0 y 1 para extremos; puntos candidatos = puntos medios entre probs
    cand = np.unique(np.concatenate([[0.0, 1.0], y_probs]))
    best = (-1.0, 0.5, 0.0, 0.0)  # J, thr, sens, spec
    for t in cand:
        s, sp, _ = _metrics_at_threshold(y_true, y_probs, t)
        j = (s + sp) / 100 - 1
        if j > best[0]:
            best = (j, float(t), s, sp)
    return best  # (J, thr, sens, spec)


def run_recalibrated_metrics(grouped):
    """
    Para cada (modelo, músculo) recalibra Sens/Spec/Acc por fold usando
    Youden's J, y devuelve la media ± std entre folds.
    """
    rows = []
    for (model, muscle), folds in grouped.items():
        sens_l, spec_l, acc_l, thr_l = [], [], [], []
        sens05_l, spec05_l = [], []
        for f in folds:
            yt, yp, pp = f["y_true"], f["y_pred"], f["y_probs"]
            # Youden por fold sobre SUS propios probs (calibración local)
            _, thr, s, sp = _best_youden_threshold(yt, pp)
            s05, sp05, acc05 = _metrics_at_threshold(yt, pp, 0.5)
            _, _, accJ = _metrics_at_threshold(yt, pp, thr)
            sens_l.append(s);   spec_l.append(sp);   acc_l.append(accJ); thr_l.append(thr)
            sens05_l.append(s05); spec05_l.append(sp05)
        rows.append({
            "model":             model,
            "muscle":            muscle,
            "n_folds":           len(folds),
            # Umbral 0.5 (el que usa el modelo tal cual)
            "sens_thr05_mean":   round(float(np.mean(sens05_l)), 2),
            "spec_thr05_mean":   round(float(np.mean(spec05_l)), 2),
            # Youden's J
            "sens_youden_mean":  round(float(np.mean(sens_l)), 2),
            "sens_youden_std":   round(float(np.std(sens_l, ddof=1)) if len(sens_l) > 1 else 0.0, 2),
            "spec_youden_mean":  round(float(np.mean(spec_l)), 2),
            "spec_youden_std":   round(float(np.std(spec_l, ddof=1)) if len(spec_l) > 1 else 0.0, 2),
            "acc_youden_mean":   round(float(np.mean(acc_l)), 2),
            "thr_youden_mean":   round(float(np.mean(thr_l)), 3),
            "thr_youden_std":    round(float(np.std(thr_l, ddof=1)) if len(thr_l) > 1 else 0.0, 3),
        })
    rows.sort(key=lambda r: (r["muscle"], -r["acc_youden_mean"]))
    _write_csv(os.path.join(RESULTS_DIR, "recalibrated_youden.csv"), rows)
    return rows


# ── 3a. Tests pareados entre arquitecturas ─────────────────────────────
def _wilcoxon_auc(folds_a, folds_b):
    """Test pareado sobre AUC por fold (misma partición StratifiedGroupKFold)."""
    auc_a = [f["auc"] for f in sorted(folds_a, key=lambda r: r["fold"])]
    auc_b = [f["auc"] for f in sorted(folds_b, key=lambda r: r["fold"])]
    if len(auc_a) != len(auc_b):
        return None, None
    try:
        stat, p = stats.wilcoxon(auc_a, auc_b)
        return float(stat), float(p)
    except ValueError:
        return None, None  # todos los folds iguales


def _delong_test(y_true, probs_a, probs_b):
    """
    Test de DeLong para comparar dos AUCs correlacionados (mismo y_true).
    Implementación basada en Sun & Xu (2014), O(n log n). Devuelve (z, p).
    """
    y_true   = np.asarray(y_true)
    probs_a  = np.asarray(probs_a)
    probs_b  = np.asarray(probs_b)

    pos = y_true == 1
    neg = y_true == 0
    xa, ya = probs_a[pos], probs_a[neg]
    xb, yb = probs_b[pos], probs_b[neg]
    m, n = len(xa), len(ya)
    if m == 0 or n == 0:
        return None, None

    # Componentes V (placement values)
    def placement(x, y):
        v10 = np.array([(np.sum(y < xi) + 0.5 * np.sum(y == xi)) / len(y) for xi in x])
        v01 = np.array([(np.sum(x > yj) + 0.5 * np.sum(x == yj)) / len(x) for yj in y])
        return v10, v01

    v10_a, v01_a = placement(xa, ya)
    v10_b, v01_b = placement(xb, yb)

    auc_a = v10_a.mean()
    auc_b = v10_b.mean()

    # Matriz de covarianza 2x2 entre los dos AUCs
    s10 = np.cov(np.vstack([v10_a, v10_b])) / m
    s01 = np.cov(np.vstack([v01_a, v01_b])) / n
    S = s10 + s01

    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff <= 0:
        return 0.0, 1.0
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def _mcnemar_test(y_true, pred_a, pred_b):
    """Test de McNemar sobre aciertos/fallos binarios."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    b01 = int(np.sum( correct_a & ~correct_b))  # A bien, B mal
    b10 = int(np.sum(~correct_a &  correct_b))  # A mal,  B bien
    table = [[0, b01], [b10, 0]]
    try:
        res = mcnemar(table, exact=True)
        return float(res.statistic), float(res.pvalue), b01, b10
    except Exception:
        return None, None, b01, b10


def run_pairwise_between_models(grouped):
    rows = []
    muscles = sorted({m for (_, m) in grouped.keys()})

    for muscle in muscles:
        combos = sorted([(mn, folds) for (mn, mu), folds in grouped.items()
                         if mu == muscle])
        for (mn_a, folds_a), (mn_b, folds_b) in combinations(combos, 2):
            ya, pa, ppa = _concat_predictions(folds_a)
            yb, pb, ppb = _concat_predictions(folds_b)
            # Sanity: mismo y_true (misma partición StratifiedGroupKFold)
            same_y = (len(ya) == len(yb) and np.array_equal(ya, yb))

            w_stat, w_p = _wilcoxon_auc(folds_a, folds_b)
            if same_y:
                d_z, d_p                  = _delong_test(ya, ppa, ppb)
                m_stat, m_p, b01, b10     = _mcnemar_test(ya, pa, pb)
            else:
                d_z = d_p = m_stat = m_p = None
                b01 = b10 = None

            rows.append({
                "muscle":       muscle,
                "model_a":      mn_a,
                "model_b":      mn_b,
                "auc_a_mean":   round(float(np.mean([f['auc'] for f in folds_a])), 2),
                "auc_b_mean":   round(float(np.mean([f['auc'] for f in folds_b])), 2),
                "wilcoxon_p":   _round(w_p),
                "delong_p":     _round(d_p),
                "mcnemar_p":    _round(m_p),
                "mcnemar_b01":  b01,
                "mcnemar_b10":  b10,
            })
    rows.sort(key=lambda r: (r["muscle"],
                             r["wilcoxon_p"] if r["wilcoxon_p"] is not None else 1))
    _write_csv(os.path.join(RESULTS_DIR, "pairwise_models.csv"), rows)
    return rows


# ── 4. Boxplots del AUC por músculo ────────────────────────────────────
def plot_boxplots_per_muscle(grouped):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    muscles = sorted({m for (_, m) in grouped.keys()})
    for muscle in muscles:
        combos = sorted([(mn, folds) for (mn, mu), folds in grouped.items()
                         if mu == muscle])
        if not combos:
            continue
        names = [mn for mn, _ in combos]
        aucs  = [[f["auc"] for f in folds] for _, folds in combos]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        bp = ax.boxplot(aucs, labels=[n.upper() for n in names],
                        patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="black"))
        # Baseline horizontal
        ax.axhline(BASELINE[muscle]["auc"], color="red", linestyle="--",
                   linewidth=1.2,
                   label=f"Baseline Martínez-Payá 2017: {BASELINE[muscle]['auc']}%")
        # Colores
        cmap = plt.cm.tab10(np.linspace(0, 1, len(names)))
        for patch, color in zip(bp["boxes"], cmap):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_ylabel("AUC (%)")
        ax.set_title(f"AUC por fold — {muscle} (5-fold StratifiedGroupKFold)")
        ax.set_ylim(40, 105)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        plt.xticks(rotation=20)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f"boxplot_auc_{muscle.lower()}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Boxplot -> {path}")


# ── 5. Informe final legible ───────────────────────────────────────────
def write_report(ci_rows, vs_base_rows, pairwise_rows, recal_rows):
    path = os.path.join(RESULTS_DIR, "informe_estadistico.txt")
    with open(path, "w") as f:
        f.write("="*78 + "\n")
        f.write("  INFORME ESTADÍSTICO — TFG ELA (5-fold StratifiedGroupKFold)\n")
        f.write("="*78 + "\n\n")

        f.write("1) AUC MEDIO POR (MODELO, MÚSCULO) CON IC 95 %\n")
        f.write("   IC principal: t-Student sobre los 5 AUCs por fold.\n")
        f.write("   IC secundario: bootstrap (N=1000) sobre los 5 AUCs.\n")
        f.write("-"*78 + "\n")
        for muscle in ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]:
            f.write(f"\n  -- {muscle.upper()} --\n")
            f.write(f"  {'MODEL':<18} {'AUC fold μ±σ':<18} "
                    f"{'IC95% t-Stud':<22} {'IC95% boot':<22}\n")
            f.write("  " + "-"*78 + "\n")
            subset = [r for r in ci_rows if r["muscle"] == muscle]
            for r in subset:
                f.write(f"  {r['model']:<18} "
                        f"{r['auc_fold_mean']:>6.2f} ± {r['auc_fold_std']:<6.2f}  "
                        f"[{r['auc_t_ci_low']:>5.2f}; {r['auc_t_ci_high']:>5.2f}]   "
                        f"[{r['auc_boot_ci_low']:>5.2f}; {r['auc_boot_ci_high']:>5.2f}]\n")

        f.write("\n\n2) COMPARACIÓN VS MARTÍNEZ-PAYÁ 2017 (IC t-Student)\n")
        f.write("-"*78 + "\n")
        for muscle in ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]:
            base = BASELINE[muscle]
            f.write(f"\n  -- {muscle.upper()} --  "
                    f"baseline AUC={base['auc']}% ({base['tech']})\n")
            subset = [r for r in vs_base_rows if r["muscle"] == muscle]
            for r in subset:
                f.write(f"    {r['model']:<18} "
                        f"AUC={r['our_auc']:>6.2f}% "
                        f"[{r['our_t_ci_low']:>5.2f}, {r['our_t_ci_high']:>5.2f}]  "
                        f"-> {r['verdict']}\n")

        f.write("\n\n3) MÉTRICAS CLÍNICAS RECALIBRADAS (umbral óptimo de Youden's J)\n")
        f.write("    Comparar Sens/Spec a umbral fijo 0.5 vs. umbral óptimo por fold.\n")
        f.write("-"*78 + "\n")
        for muscle in ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]:
            f.write(f"\n  -- {muscle.upper()} --  "
                    f"(baseline Sens={BASELINE[muscle]['sens']}%, "
                    f"Spec={BASELINE[muscle]['spec']}%)\n")
            f.write(f"  {'MODEL':<18} {'Sens@0.5':>9} {'Spec@0.5':>9} "
                    f"{'Sens@J':>10} {'Spec@J':>10} {'Acc@J':>8} {'thr@J':>8}\n")
            f.write("  " + "-"*78 + "\n")
            subset = [r for r in recal_rows if r["muscle"] == muscle]
            for r in subset:
                f.write(f"  {r['model']:<18} "
                        f"{r['sens_thr05_mean']:>8.2f}% "
                        f"{r['spec_thr05_mean']:>8.2f}%  "
                        f"{r['sens_youden_mean']:>7.2f}±{r['sens_youden_std']:<4.1f}  "
                        f"{r['spec_youden_mean']:>7.2f}±{r['spec_youden_std']:<4.1f}  "
                        f"{r['acc_youden_mean']:>6.2f}% "
                        f"{r['thr_youden_mean']:>6.3f}\n")

        f.write("\n\n4) COMPARACIÓN PAREADA ENTRE ARQUITECTURAS\n")
        f.write("    (por músculo; p<0.05 indica diferencia significativa)\n")
        f.write("-"*78 + "\n")
        for muscle in ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]:
            f.write(f"\n  -- {muscle.upper()} --\n")
            f.write(f"  {'A':<18} {'B':<18} "
                    f"{'Wilcoxon p':>12} {'DeLong p':>12} {'McNemar p':>12}\n")
            f.write("  " + "-"*72 + "\n")
            subset = [r for r in pairwise_rows if r["muscle"] == muscle]
            for r in subset:
                f.write(f"  {r['model_a']:<18} {r['model_b']:<18} "
                        f"{_fmt_p(r['wilcoxon_p']):>12} "
                        f"{_fmt_p(r['delong_p']):>12} "
                        f"{_fmt_p(r['mcnemar_p']):>12}\n")

        f.write("\n\nNotas metodológicas:\n")
        f.write("- IC t-Student sobre los 5 AUCs por fold es el método estándar en\n")
        f.write("  k-fold CV con k pequeño. El bootstrap sobre predicciones concatenadas\n")
        f.write("  NO es válido aquí porque cada fold viene de un modelo distinto con su\n")
        f.write("  propia calibración de probabilidades.\n")
        f.write("- 'Wilcoxon p' es un test pareado sobre los 5 AUCs por fold.\n")
        f.write("- 'DeLong p' es un test sobre los AUCs agregados (predicciones concatenadas).\n")
        f.write("- 'McNemar p' compara aciertos/fallos binarios sobre predicciones concatenadas.\n")
        f.write("- El umbral de Youden (J = Sens+Spec-1) se busca en CADA fold sobre sus\n")
        f.write("  propias probabilidades. Las métricas reportadas son la media entre folds.\n")
        f.write("- Frente al baseline de Martínez-Payá no podemos aplicar DeLong/McNemar\n")
        f.write("  porque no están disponibles sus predicciones individuales, por eso\n")
        f.write("  usamos el IC t-Student del AUC y verificamos si el valor publicado cae\n")
        f.write("  dentro del intervalo.\n")
    print(f"  Informe -> {path}")


# ── Utilidades ─────────────────────────────────────────────────────────
def _round(x, n=4):
    return None if x is None else round(x, n)


def _fmt_p(p):
    if p is None:
        return "n/a"
    if p < 0.001:
        return "<0.001 ***"
    if p < 0.01:
        return f"{p:.3f} **"
    if p < 0.05:
        return f"{p:.3f} *"
    return f"{p:.3f}"


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
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Cargando predicciones desde: {PRED_PATH}")
    grouped = load_predictions()
    n_combos = len(grouped)
    print(f"Combinaciones (modelo x músculo): {n_combos}")

    print("\n[1/5] IC95% del AUC (t-Student + bootstrap sobre folds)...")
    ci_rows = run_ci_per_model_muscle(grouped)

    print("\n[2/5] Comparación vs Martínez-Payá 2017...")
    vs_rows = compare_vs_baseline(ci_rows)

    print("\n[3/5] Recalibración Sens/Spec con Youden's J...")
    recal_rows = run_recalibrated_metrics(grouped)

    print("\n[4/5] Tests pareados entre arquitecturas...")
    pairwise_rows = run_pairwise_between_models(grouped)

    print("\n[5/5] Boxplots y informe final...")
    plot_boxplots_per_muscle(grouped)
    write_report(ci_rows, vs_rows, pairwise_rows, recal_rows)

    print("\nHecho. Resultados en:", RESULTS_DIR)


if __name__ == "__main__":
    main()
