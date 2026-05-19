# TFG — Detección de ELA mediante deep learning sobre ecografía muscular

Sistema de clasificación automática (control sano vs. paciente con esclerosis
lateral amiotrófica, ELA) a partir de imágenes de ecografía muscular usando
redes neuronales convolucionales con *transfer learning* sobre ImageNet.

El trabajo compara cinco arquitecturas (ResNet-18, ResNet-50, DenseNet-121,
EfficientNet-B0 y ConvNeXt-Tiny) en cuatro grupos musculares (bíceps braquial,
flexor del carpo, cuádriceps y tibial anterior), valida los resultados mediante
5-fold *Stratified Group K-Fold* y los compara estadísticamente contra la
línea base clínica de Martínez-Payá y col. (2017) basada en análisis textural
GLCM. Como aportación principal, se construye un **sistema de decisión a
nivel paciente** que fusiona las predicciones de los cuatro músculos.

**Autor:** Jose Juan Cortina
**Grado:** IMAT — Ingeniería Matemática

---

## Resumen de resultados

Sistema final: ResNet-50 entrenado de forma independiente en cada músculo y
fusionado a nivel paciente mediante media de probabilidades + recalibración con
*Youden's J*. Métricas con bootstrap de 1000 muestras sobre los 53 pacientes:

| Métrica       | Valor       | IC95 % (bootstrap)        |
|---------------|-------------|---------------------------|
| AUC           | 99.15 %     | (ver `informe_fusion.txt`) |
| Accuracy      | 98.11 %     | "                          |
| Sensibilidad  | 96.30 %     | "                          |
| Especificidad | 100.00 %    | "                          |
| Confusión     | 26 TP / 26 TN / 0 FP / 1 FN |               |

Comparación con la línea base clínica (Martínez-Payá 2017, métricas a nivel
imagen; nuestras métricas son a nivel paciente):

| Métrica (media 4 músculos) | Martínez-Payá 2017 | Sistema final |
|----------------------------|--------------------|---------------|
| AUC                        | ≈94 %              | **99.15 %**   |
| Sensibilidad               | ≈87 %              | **96.30 %**   |
| Especificidad              | ≈87 %              | **100 %**     |

A nivel músculo, los IC95 % t-Student de las cinco arquitecturas se solapan con
el AUC publicado por Martínez-Payá en los cuatro músculos
(*equivalente / no concluyente*), lo que es consistente con un *n* moderado
(53 pacientes) y respalda que la mejora real proviene de la integración
multi-músculo a nivel paciente.

---

## Estructura del repositorio

```
TFG/
├── src/                                Código fuente
│   ├── config.py                       Configuración global (rutas, seeds, hiperparámetros)
│   ├── preprocessing.py                TIFF → JPG 224×224
│   ├── organise_data.py                Reorganización por músculo
│   ├── dataset.py                      Dataloaders con split POR PACIENTE + StratifiedGroupKFold
│   ├── models.py                       Factory de las 5 arquitecturas
│   ├── train.py                        Entrenamiento simple (split 80/20) — fase exploratoria
│   ├── train_kfold.py                  Entrenamiento con 5-fold StratifiedGroupKFold
│   ├── evaluate_saved.py               Evaluación + métricas + gráficas (split simple)
│   ├── statistical_tests.py            IC t-Student, vs baseline, recalibración Youden, Wilcoxon/DeLong/McNemar
│   ├── patient_level_fusion.py         Fusión OOF multi-músculo a nivel paciente + bootstrap IC95 %
│   └── explainability.py               Grad-CAM, Guided Grad-CAM, Saliency, Occlusion (ResNet-50)
│
├── data/
│   ├── classified_data/                Imágenes originales (.tif) facilitadas por el centro
│   └── processed/                      Dataset listo para entrenar (JPG 224×224)
│       ├── Bicep/
│       ├── Antebrazo/
│       ├── Quadriceps/
│       └── Tibial/
│
├── best_models_kfold/                  Pesos (.pth) de los 100 modelos del CV (5×4×5)
├── best_models/                        Pesos del split 80/20 (fase exploratoria)
├── best_models_leaky_backup/           Backup de la versión inicial con leakage por paciente
│
├── models/resultados_kfold/            Salidas del experimento principal
│   ├── kfold_predictions.json          Predicciones por fold de cada (modelo, músculo)
│   ├── kfold_summary.csv               Media ± std por (modelo, músculo)
│   ├── ci_bootstrap.csv                IC95 % t-Student y bootstrap por (modelo, músculo)
│   ├── vs_baseline.csv                 Veredicto vs Martínez-Payá 2017
│   ├── recalibrated_youden.csv         Sens/Spec con umbral óptimo por fold
│   ├── pairwise_models.csv             Wilcoxon / DeLong / McNemar entre arquitecturas
│   ├── informe_estadistico.txt         Informe legible de los tests
│   ├── oof_predictions.json            Predicciones out-of-fold con paciente_id
│   ├── fusion_per_architecture.csv     Fusión multi-músculo por arquitectura
│   ├── fusion_champions.csv            Fusión "campeones por músculo"
│   ├── informe_fusion.txt              Informe legible de la fusión paciente
│   ├── plots/
│   │   └── boxplot_auc_<musculo>.png   Boxplots por músculo
│   └── explainability_resnet50/
│       ├── gradcam/                    Grad-CAM por músculo, ELA y Control
│       ├── guided_gradcam/
│       ├── saliency/
│       ├── occlusion/
│       └── summary_gradcam_4x4.png     Figura resumen 4×4
│
├── docs/                               Memoria, papers de referencia (Martínez-Payá 2017, 2018)
├── requirements.txt
└── README.md
```

---

## Dataset

El dataset proviene del centro médico colaborador. Cada sujeto aporta dos
imágenes por músculo correspondientes a las lateralidades derecha (`d`) e
izquierda (`i`). Las imágenes son **ROIs** ya recortadas por el centro (no
incluyen piel, hueso ni anotaciones del ecógrafo).

**Nomenclatura:**

```
{ID_paciente}{d|i}_{Músculo}_clean.jpg
```

Ejemplos: `C001d_BBr_clean.jpg`, `1001i_Cdr_clean.jpg`, `RC001d_TbA_clean.jpg`.
Los controles aparecen como `Cnnn` en los conjuntos de Bicep / Antebrazo /
Quadriceps y como `RCnnn` en Tibial: ambos prefijos refieren al mismo paciente
con distinta convención de adquisición.

**Composición:**

| Músculo     | Control | ELA | Total imágenes | Sujetos únicos |
|-------------|---------|-----|----------------|----------------|
| Bicep       | 52      | 52  | 104            | 52             |
| Antebrazo   | 52      | 52  | 104            | 52             |
| Quadriceps  | 52      | 52  | 104            | 52             |
| Tibial      | 52      | 52  | 104            | 52             |

Total: **53 pacientes únicos** (27 ELA + 26 Control), cada uno con sus 4 músculos.

---

## Metodología

### Preprocesado
Conversión de las ROIs `.tif` a JPG 224×224 RGB (estándar de ImageNet) y
normalización con los estadísticos `mean=[0.485, 0.456, 0.406]`,
`std=[0.229, 0.224, 0.225]`. *Data augmentation* en entrenamiento: flip
horizontal (p=0.5), flip vertical (p=0.2), rotación ±10°, color jitter
(brightness/contrast 0.2), traslación afín ±5 %. La augmentación está motivada
clínicamente (variabilidad de sonda y protocolos entre adquisiciones).

### Particionado por paciente (sin fuga)
Todas las imágenes de un mismo sujeto van al mismo subconjunto. La identidad
del paciente se extrae del nombre de fichero mediante una expresión regular
robusta. La validación cruzada principal usa `StratifiedGroupKFold` con
**5 folds**, que combina la restricción por sujeto (`groups`) con la
estratificación por clase. Una versión simple (split 80/20 con
`GroupShuffleSplit`) se usó en la fase exploratoria.

### Arquitecturas
Cinco CNNs pre-entrenadas en ImageNet (`torchvision.models`), con la última
capa reemplazada para clasificación binaria:

| Arquitectura       | Parámetros aprox. | Notas                              |
|--------------------|-------------------|-------------------------------------|
| ResNet-18          | 11 M              | *Best all-rounder*                  |
| ResNet-50          | 26 M              | **Sistema final tras fusión**       |
| DenseNet-121       | 7 M               |                                     |
| EfficientNet-B0    | 5 M               |                                     |
| ConvNeXt-Tiny      | 28 M              | CNN moderna inspirada en Transformers |

Se descartaron VGG-16 y MobileNet-V3 Small de la comparativa final por
motivos de capacidad y redundancia respectivamente.

### Entrenamiento
- Pérdida `CrossEntropyLoss`, optimizador `Adam` (LR `1e-4`).
- *Scheduler* `ReduceLROnPlateau` (factor 0.5, patience 5) sobre AUC en val.
- Batch size 16, 50 epochs, semilla fija (42).
- Selección de checkpoint por **AUC en val** (más estable que accuracy con
  *n* pequeño).

### Análisis estadístico (`statistical_tests.py`)
- IC95 % de la media del AUC por (modelo, músculo) con **t-Student** sobre
  los 5 AUCs por fold y, complementariamente, bootstrap sobre esos mismos
  AUCs. Se documenta por qué *no* se usa bootstrap sobre predicciones
  concatenadas (mezcla calibraciones de modelos distintos).
- Comparación frente al baseline clínico (Martínez-Payá 2017): si el AUC
  publicado cae dentro del IC95 %, "equivalente"; si fuera, ventaja
  estadística para una u otra parte.
- Recalibración con **umbral óptimo de Youden** (J = Sens + Spec − 1)
  buscado por fold, para reportar Sens/Spec honestos a nivel músculo.
- Tests pareados entre arquitecturas: Wilcoxon signed-rank (sobre AUCs por
  fold), DeLong (sobre predicciones concatenadas) y McNemar (sobre
  aciertos binarios).

### Fusión a nivel paciente (`patient_level_fusion.py`)
- Predicciones **out-of-fold** (OOF): cada paciente recibe una probabilidad
  por músculo proveniente del fold en el que estuvo en validación
  (nunca lo vio el modelo en entrenamiento).
- Agregación a paciente: media de las probabilidades de sus imágenes en
  cada músculo.
- Tres reglas de fusión multi-músculo: media simple, media ponderada por
  AUC del músculo, voto mayoritario.
- Recalibración con Youden sobre la probabilidad fusionada.
- IC95 % por bootstrap (N=1000) sobre los 53 pacientes para Acc/Sens/Spec/AUC.
- Se reporta por arquitectura **y** una variante "campeones por músculo"
  (mejor modelo de cada músculo). El sistema final descarta esta variante
  porque mezclar arquitecturas distintas descalibra las probabilidades en
  la fusión y rinde peor que ResNet-50 aplicado uniformemente.

### Explicabilidad (`explainability.py`)
Cuatro técnicas sobre ResNet-50 (sistema final), usando los pesos del fold
con mejor AUC en cada músculo y dos imágenes ELA + dos Control extraídas de
la *val set* de ese fold (out-of-fold honesto):

- Grad-CAM
- Guided Grad-CAM
- Saliency maps
- Occlusion (parches 32, *stride* 16)

Se incluye una figura resumen 4×4 (4 músculos × 4 imágenes) lista para la
memoria.

---

## Métricas

Se reportan las relevantes en diagnóstico médico:

- **Sensibilidad** (TPR): fracción de pacientes ELA correctamente
  identificados.
- **Especificidad** (TNR): fracción de controles correctamente
  identificados.
- **AUC-ROC**: insensible al umbral y al desbalance.
- A nivel paciente, todas las anteriores se acompañan de IC95 % por
  bootstrap.

---

## Instalación

Requiere Python 3.10 o superior. Probado en macOS (Apple Silicon, MPS) y
Linux (CUDA).

```bash
git clone <url-del-repo>
cd TFG
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

---

## Uso (pipeline completo)

Todos los comandos se ejecutan desde `src/`.

```bash
# 1) Preprocesar las imágenes originales (solo la primera vez)
python3 preprocessing.py
python3 organise_data.py

# 2) Verificar el split por paciente (assert de no-overlap)
python3 dataset.py

# 3) Entrenamiento principal: 5 modelos × 4 músculos × 5 folds = 100 trainings
#    Genera best_models_kfold/ y models/resultados_kfold/kfold_predictions.json
python3 train_kfold.py

# 4) Análisis estadístico (IC t-Student, vs baseline, Youden, tests pareados)
python3 statistical_tests.py

# 5) Fusión a nivel paciente (out-of-fold + bootstrap IC95%)
python3 patient_level_fusion.py
#  Para reusar oof_predictions.json sin re-inferir:
#  python3 patient_level_fusion.py --skip-inference

# 6) Mapas de explicabilidad (ResNet-50, los 4 músculos)
python3 explainability.py
```

La fase exploratoria (split 80/20 + `evaluate_saved.py`) se mantiene en el
repositorio por trazabilidad pero **no es parte del experimento principal**.

---

## Hardware

- **Recomendado:** macOS con Apple Silicon (MPS) o Linux con GPU NVIDIA (CUDA).
- **Tiempo aproximado del experimento principal en M-series con MPS:**
  ~6–10 h para los 100 trainings + minutos para los análisis.
- La detección del acelerador es automática (`config.py`).

---

## Referencias

1. Martínez-Payá, J. J., del Baño-Aledo, M. E., Ríos-Díaz, J., et al. (2017).
   *Quantitative muscle ultrasonography using textural analysis in amyotrophic
   lateral sclerosis.*
2. Martínez-Payá, J. J., Ríos-Díaz, J., Medina-Mirapeix, F., et al. (2018).
   *Monitoring the progression of amyotrophic lateral sclerosis through muscle
   ultrasound: a longitudinal study.*

Ambos artículos se encuentran en `docs/`.

---

## Licencia y uso

Código desarrollado como Trabajo Fin de Grado. Los datos clínicos son
propiedad del centro médico colaborador y no están incluidos en este
repositorio.
