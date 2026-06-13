# Bachelor's Thesis — ALS detection via deep learning on muscle ultrasound

Automatic classification system (healthy control vs. patient with amyotrophic
lateral sclerosis, ALS) from muscle ultrasound images, using convolutional
neural networks with *transfer learning* on ImageNet.

The work compares five architectures (ResNet-18, ResNet-50, DenseNet-121,
EfficientNet-B0 and ConvNeXt-Tiny) on four muscle groups (biceps brachii,
forearm flexors, quadriceps and tibialis anterior), validates the results with
5-fold *Stratified Group K-Fold* and compares them statistically against the
clinical baseline of Martinez-Paya et al. (2017) based on GLCM textural
analysis. As the main contribution, a **patient-level decision system** is
built that fuses the predictions of the four muscles.

**Author:** Jose Juan Cortina
**Degree:** IMAT — Mathematical Engineering and Artificial Intelligence

---

## Results summary

Final system: ResNet-50 trained independently on each muscle and fused at the
patient level by averaging probabilities + recalibration with *Youden's J*.
Metrics with a 1000-sample bootstrap over the 52 patients:

| Metric        | Value       | 95% CI (bootstrap)        |
|---------------|-------------|---------------------------|
| AUC           | 98.67 %     | (see `informe_fusion.txt`) |
| Accuracy      | 96.15 %     | "                          |
| Sensitivity   | 100.00 %    | "                          |
| Specificity   | 92.31 %     | "                          |
| Confusion     | 26 TP / 24 TN / 2 FP / 0 FN |               |

Comparison with the clinical baseline (Martinez-Paya 2017, image-level metrics;
our metrics are patient-level):

| Metric (mean of 4 muscles) | Martinez-Paya 2017 | Final system  |
|----------------------------|--------------------|---------------|
| AUC                        | ≈94 %              | **98.67 %**   |
| Sensitivity                | ≈87 %              | **100 %**     |
| Specificity                | ≈87 %              | **92.31 %**   |

At the muscle level, the t-Student 95% CIs of the five architectures overlap
with the AUC published by Martinez-Paya in all four muscles
(*equivalent / inconclusive*), which is consistent with a moderate *n*
(52 patients) and supports that the real contribution comes from the
multi-muscle integration at the patient level.

---

## Repository structure

```
TFG/
├── src/                              Source code (12 modules)
│   ├── config.py                     Global configuration (paths, seeds, hyperparameters)
│   ├── preprocessing.py              TIFF → JPG 224×224
│   ├── organise_data.py              Organisation of the images by muscle
│   ├── dataset.py                    Dataloaders with PER-PATIENT split + StratifiedGroupKFold
│   ├── models.py                     Factory of the 5 architectures
│   ├── train.py                      Simple 80/20 training (exploratory phase)
│   ├── train_kfold.py                Training with 5-fold StratifiedGroupKFold
│   ├── evaluate_saved.py             Evaluation of the simple split (exploratory phase)
│   ├── statistical_tests.py          CI, vs baseline, Youden, Wilcoxon/DeLong/McNemar
│   ├── patient_level_fusion.py       Multi-muscle OOF fusion at patient level + bootstrap
│   ├── explainability.py             Grad-CAM, Guided Grad-CAM, Saliency, Occlusion (ResNet-50)
│   └── create_figures.py            Regenerates the thesis result figures
│
├── memoria/                          LaTeX source of the thesis
│   ├── TFG_spanish.tex               Main document (bilingual ES/EN)
│   ├── Anexo_I.tex                   Authorship declaration
│   ├── ref_tfg.bib                   Main bibliography
│   ├── ref_executive_summary.bib     Executive-summary bibliography
│   ├── images/                       Figures (confusion matrix, boxplots, ROC, Grad-CAM, logos)
│   ├── .latexmkrc                    Build configuration (latexmk + biber)
│   └── TFG_spanish.pdf               Compiled thesis
│
├── .vscode/                          LaTeX Workshop configuration (for reviewers)
├── README.md
├── requirements.txt
└── .gitignore
```

> **What is NOT included in this repository** (excluded via `.gitignore`):
> - **Patient data** (`data/`): property of the collaborating medical centre, not redistributable.
> - **Trained weights** (`best_models_kfold/`, ~5.5 GB): available on Google Drive (see the "Trained weights" section below).
> - **Intermediate results** (`models/resultados_kfold/`: predictions, OOF, fusion CSVs and reports).
> - **Reference papers** (`docs/`): due to copyright.
>
> This public repository is the deliverable for the source code and the thesis.
> To reproduce the results without retraining, download the trained weights from
> the Google Drive link below and place them as `best_models_kfold/`; the patient
> data are not redistributed for confidentiality reasons.

---

## Trained weights (not included in the repository)

The 100 checkpoints in `best_models_kfold/` (~5.5 GB) are not versioned due to
their size. They are available on Google Drive:

**https://drive.google.com/drive/folders/1hDHp9OzT3U6jzT9ogTiLIab_jd5rGU96?usp=drive_link**

To reproduce the patient-level fusion and the results **without retraining**,
download the folder and place it as `best_models_kfold/` at the project root;
then run `python3 src/patient_level_fusion.py` (and `python3 src/statistical_tests.py`).
Alternatively, the whole experiment can be regenerated from scratch with
`python3 src/train_kfold.py` (~6–10 h on Apple Silicon with MPS).

---

## Dataset

The dataset comes from the collaborating medical centre. Each subject provides
two images per muscle, corresponding to the right (`d`) and left (`i`) sides.
The images are **ROIs** already cropped by the centre (they do not include
skin, bone or ultrasound-scanner annotations).

**Naming:**

```
{patient_ID}{d|i}_{Muscle}_clean.jpg
```

Examples: `C001d_BBr_clean.jpg`, `1001i_Cdr_clean.jpg`, `RC001d_TbA_clean.jpg`.
Controls appear as `Cnnn` in the Bicep / Antebrazo / Quadriceps sets and as
`RCnnn` in Tibial: both prefixes refer to the same patient under a different
acquisition convention.

**Composition:**

| Muscle      | Control | ALS | Total images | Unique subjects |
|-------------|---------|-----|--------------|-----------------|
| Bicep       | 52      | 52  | 104          | 52              |
| Antebrazo   | 52      | 52  | 104          | 52              |
| Quadriceps  | 52      | 52  | 104          | 52              |
| Tibial      | 52      | 52  | 104          | 52              |

Total: **52 unique patients** (26 ALS + 26 Control), each with their 4 muscles
(2 images per muscle: right and left sides).

---

## Methodology

### Preprocessing
Conversion of the `.tif` ROIs to 224×224 RGB JPG (ImageNet standard) and
normalisation with the statistics `mean=[0.485, 0.456, 0.406]`,
`std=[0.229, 0.224, 0.225]`. Training-time *data augmentation*: horizontal
flip (p=0.5), vertical flip (p=0.2), ±10° rotation, color jitter
(brightness/contrast 0.2), ±5% affine translation. The augmentation is
clinically motivated (probe and protocol variability across acquisitions).

### Per-patient partitioning (leakage-free)
All images of the same subject go to the same subset. The patient identity is
extracted from the file name with a robust regular expression. The main
cross-validation uses `StratifiedGroupKFold` with **5 folds**, combining the
per-subject constraint (`groups`) with class stratification. A simple version
(80/20 split with `GroupShuffleSplit`) was used in the exploratory phase.

### Architectures
Five CNNs pre-trained on ImageNet (`torchvision.models`), with the last layer
replaced for binary classification:

| Architecture       | Approx. parameters | Notes                              |
|--------------------|--------------------|-------------------------------------|
| ResNet-18          | 11 M               | *Best all-rounder*                  |
| ResNet-50          | 26 M               | **Final system after fusion**       |
| DenseNet-121       | 8 M                |                                     |
| EfficientNet-B0    | 5 M                |                                     |
| ConvNeXt-Tiny      | 28 M               | Modern CNN inspired by Transformers |

VGG-16 and MobileNet-V3 Small were dropped from the final comparison due to
capacity and redundancy reasons respectively.

### Training
- `CrossEntropyLoss` loss, `Adam` optimizer (LR `1e-4`).
- `ReduceLROnPlateau` scheduler (factor 0.5, patience 5) on validation AUC.
- Batch size 16, 50 epochs, fixed seed (42).
- Checkpoint selection by **validation AUC** (more stable than accuracy with
  small *n*).

### Statistical analysis (`statistical_tests.py`)
- 95% CI of the mean AUC per (model, muscle) with **t-Student** over the 5
  per-fold AUCs and, complementarily, bootstrap over those same AUCs. It is
  documented why bootstrap over concatenated predictions is *not* used (it
  mixes calibrations from different models).
- Comparison against the clinical baseline (Martinez-Paya 2017): if the
  published AUC falls inside the 95% CI, "equivalent"; if outside, a
  statistical advantage for one side or the other.
- Recalibration with the **optimal Youden threshold** (J = Sens + Spec − 1)
  searched per fold, to report honest Sens/Spec at the muscle level.
- Paired tests between architectures: Wilcoxon signed-rank (over per-fold
  AUCs), DeLong (over concatenated predictions) and McNemar (over binary
  hits/misses).

### Patient-level fusion (`patient_level_fusion.py`)
- **Out-of-fold** (OOF) predictions: each patient gets a per-muscle
  probability from the fold in which they were in validation (the model never
  saw them during training).
- Patient aggregation: mean of the probabilities of their images in each
  muscle.
- Three multi-muscle fusion rules: simple mean, AUC-weighted mean, majority
  vote.
- Recalibration with Youden over the fused probability.
- 95% bootstrap CI (N=1000) over the 52 patients for Acc/Sens/Spec/AUC.
- Results are reported per architecture **and** for a "per-muscle champions"
  variant (best model of each muscle). The champions variant performs
  statistically equivalently to the single architectures (the fused AUCs fall
  in overlapping confidence intervals); the final system uses a single
  architecture (ResNet-50) for operational simplicity and maintainability, not
  for an AUC advantage, and because it offers 100% sensitivity (no ALS case
  missed).

### Explainability (`explainability.py`)
Four techniques on ResNet-50 (final system), using the weights of the
best-AUC fold per muscle and two ALS + two Control images taken from that
fold's *val set* (honest out-of-fold):

- Grad-CAM
- Guided Grad-CAM
- Saliency maps
- Occlusion (32-px patches, *stride* 16)

A 4×4 summary figure (4 muscles × 4 images) ready for the thesis is included.

---

## Metrics

The metrics relevant in medical diagnosis are reported:

- **Sensitivity** (TPR): fraction of ALS patients correctly identified.
- **Specificity** (TNR): fraction of controls correctly identified.
- **AUC-ROC**: insensitive to threshold and class imbalance.
- At the patient level, all of the above come with a 95% bootstrap CI.

---

## Installation

Requires Python 3.10 or higher. Tested on macOS (Apple Silicon, MPS) and
Linux (CUDA).

```bash
git clone <repo-url>
cd TFG
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

---

## Usage (full pipeline)

All commands are run from `src/`.

```bash
# 1) Preprocess the original images (only the first time)
python3 preprocessing.py
python3 organise_data.py

# 2) Verify the per-patient split (no-overlap assertion)
python3 dataset.py

# 3) Main training: 5 models × 4 muscles × 5 folds = 100 trainings
#    Generates best_models_kfold/ and models/resultados_kfold/kfold_predictions.json
python3 train_kfold.py

# 4) Statistical analysis (t-Student CI, vs baseline, Youden, paired tests)
python3 statistical_tests.py

# 5) Patient-level fusion (out-of-fold + bootstrap 95% CI)
python3 patient_level_fusion.py
#  To reuse oof_predictions.json without re-inferring:
#  python3 patient_level_fusion.py --skip-inference

# 6) Explainability maps (ResNet-50, the 4 muscles)
python3 explainability.py

# 7) Regenerate the thesis figures from the results
#    (confusion matrix, AUC boxplots and ROC curves)
python3 create_figures.py
```

The exploratory phase (80/20 split + `evaluate_saved.py`) is kept in the
repository for traceability but is **not part of the main experiment**.

---

## Hardware

- **Recommended:** macOS with Apple Silicon (MPS) or Linux with an NVIDIA GPU (CUDA).
- **Approximate time of the main experiment on M-series with MPS:**
  ~6–10 h for the 100 trainings + minutes for the analyses.
- Accelerator detection is automatic (`config.py`).

---

## References

1. Martinez-Paya, J. J., del Bano-Aledo, M. E., Rios-Diaz, J., et al. (2017).
   *Quantitative muscle ultrasonography using textural analysis in amyotrophic
   lateral sclerosis.* Ultrasonic Imaging, 39(6):357–368.
   DOI: 10.1177/0161734617711370
2. Martinez-Paya, J. J., Rios-Diaz, J., Medina-Mirapeix, F., et al. (2018).
   *Monitoring progression of amyotrophic lateral sclerosis using ultrasound
   morpho-textural muscle biomarkers: a pilot study.* Ultrasound in Medicine &
   Biology, 44(1):102–109.

The papers are not redistributed in this repository for copyright reasons; they
can be accessed through the DOI / journal above.

---

## License and use

Code developed as a Bachelor's Thesis (Trabajo Fin de Grado). The clinical data
are property of the collaborating medical centre and are not included in this
repository.
