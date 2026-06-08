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
*Youden's J*. Métricas con bootstrap de 1000 muestras sobre los 52 pacientes:

| Métrica       | Valor       | IC95 % (bootstrap)        |
|---------------|-------------|---------------------------|
| AUC           | 98.67 %     | (ver `informe_fusion.txt`) |
| Accuracy      | 96.15 %     | "                          |
| Sensibilidad  | 100.00 %    | "                          |
| Especificidad | 92.31 %     | "                          |
| Confusión     | 26 TP / 24 TN / 2 FP / 0 FN |               |

Comparación con la línea base clínica (Martínez-Payá 2017, métricas a nivel
imagen; nuestras métricas son a nivel paciente):

| Métrica (media 4 músculos) | Martínez-Payá 2017 | Sistema final |
|----------------------------|--------------------|---------------|
| AUC                        | ≈94 %              | **98.67 %**   |
| Sensibilidad               | ≈87 %              | **100 %**     |
| Especificidad              | ≈87 %              | **92.31 %**   |

A nivel músculo, los IC95 % t-Student de las cinco arquitecturas se solapan con
el AUC publicado por Martínez-Payá en los cuatro músculos
(*equivalente / no concluyente*), lo que es consistente con un *n* moderado
(52 pacientes) y respalda que la aportación real proviene de la integración
multi-músculo a nivel paciente.

---

## Estructura del repositorio

```
TFG/
├── src/                              Código fuente (12 módulos)
│   ├── config.py                     Configuración global (rutas, semillas, hiperparámetros)
│   ├── preprocessing.py              TIFF → JPG 224×224
│   ├── organise_data.py              Organización de las imágenes por músculo
│   ├── dataset.py                    Dataloaders con split POR PACIENTE + StratifiedGroupKFold
│   ├── models.py                     Factoría de las 5 arquitecturas
│   ├── train.py                      Entrenamiento simple 80/20 (fase exploratoria)
│   ├── train_kfold.py                Entrenamiento con 5-fold StratifiedGroupKFold
│   ├── evaluate_saved.py             Evaluación del split simple (fase exploratoria)
│   ├── statistical_tests.py          IC, vs baseline, Youden, Wilcoxon/DeLong/McNemar
│   ├── patient_level_fusion.py       Fusión OOF multi-músculo a nivel paciente + bootstrap
│   ├── explainability.py             Grad-CAM, Guided Grad-CAM, Saliency, Occlusion (ResNet-50)
│   └── make_figuras_memoria.py       Regenera las figuras de resultados de la memoria
│
├── memoria/                          Fuente LaTeX de la memoria
│   ├── TFG_spanish.tex               Documento principal (bilingüe ES/EN)
│   ├── Anexo_I.tex                   Declaración de autoría
│   ├── ref_tfg.bib                   Bibliografía principal
│   ├── ref_executive_summary.bib     Bibliografía del resumen ejecutivo
│   ├── images/                       Figuras (matriz de confusión, boxplots, ROC, Grad-CAM, logos)
│   ├── .latexmkrc, compilar.command, COMO_COMPILAR.md   Ayudas de compilación
│   └── TFG_spanish.pdf               Memoria compilada
│
├── .vscode/                          Configuración de LaTeX Workshop (para revisores)
├── TFG_spanish.pdf                   Memoria compilada (copia en la raíz)
├── README.md
├── requirements.txt
└── .gitignore
```

> **Qué NO se incluye en este repositorio** (excluido por `.gitignore`):
> - **Datos de pacientes** (`data/`): propiedad del centro médico colaborador, no redistribuibles.
> - **Pesos entrenados** (`best_models_kfold/`, ~5,5 GB): disponibles en Google Drive (ver la sección «Pesos entrenados» más abajo).
> - **Resultados intermedios** (`models/resultados_kfold/`: predicciones, OOF, CSV de fusión e informes).
> - **Artículos de referencia** (`docs/`): por derechos de autor.
>
> El paquete completo y autocontenido para reproducir el trabajo sin reentrenar (código + `data/processed/` + resultados + memoria) se entrega por separado al tribunal en un ZIP.

---

## Pesos entrenados (no incluidos en el repositorio)

Los 100 checkpoints de `best_models_kfold/` (~5,5 GB) no se versionan por tamaño.
Están disponibles en Google Drive:

**https://drive.google.com/drive/folders/1hDHp9OzT3U6jzT9ogTiLIab_jd5rGU96?usp=drive_link**

Para reproducir la fusión a nivel paciente y los resultados **sin reentrenar**,
descarga la carpeta y colócala como `best_models_kfold/` en la raíz del proyecto;
después ejecuta `python3 src/patient_level_fusion.py` (y `python3 src/statistical_tests.py`).
Alternativamente, todo el experimento puede regenerarse desde cero con
`python3 src/train_kfold.py` (~6–10 h en Apple Silicon con MPS).

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

Total: **52 pacientes únicos** (26 ELA + 26 Control), cada uno con sus 4 músculos
(2 imágenes por músculo: lados derecho e izquierdo).

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
- IC95 % por bootstrap (N=1000) sobre los 52 pacientes para Acc/Sens/Spec/AUC.
- Se reporta por arquitectura **y** una variante "campeones por músculo"
  (mejor modelo de cada músculo). La variante de campeones rinde de forma
  estadísticamente equivalente a las arquitecturas únicas (los AUC fusionados
  caen en intervalos de confianza solapados); el sistema final emplea una sola
  arquitectura (ResNet-50) por simplicidad operativa y mantenibilidad, no por
  una ventaja de AUC, y por ofrecer sensibilidad del 100 % (ningún ELA sin
  detectar).

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

# 7) Regenerar las figuras de la memoria desde los resultados
#    (matriz de confusión, boxplots de AUC y curvas ROC)
python3 make_figuras_memoria.py
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
