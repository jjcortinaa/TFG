import os
import re
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from config import Config


# -----------------------------------------------------------------------------
# Subject identification
# -----------------------------------------------------------------------------
# Captura el prefijo alfanumérico del sujeto: "C001", "C026", "RC001", "1001",
# "1018", etc. Se queda con opcionalmente R, opcionalmente C, y 1-4 dígitos.
# Es robusto a variaciones de nombre que sí aparecen en el dataset:
#   "C001d_BBr_clean.jpg"  -> "C001"
#   "C001i_BBr_clean.jpg"  -> "C001"    (mismo sujeto, otro lado)
#   "RC001d_TbA_clean.jpg" -> "RC001"
#   "1001d_BBr_clean.jpg"  -> "1001"
#   "1018_Cdr_clean.jpg"   -> "1018"    (nombre sin letra de lado; Cuádriceps)
#   "1018d(b)_clean.jpg"   -> "1018"    (segunda toma con paréntesis; Cuádriceps)
#   "C026_Cdr_clean.jpg"   -> "C026"    (nombre sin letra de lado; Cuádriceps)
_SUBJECT_RE = re.compile(r"^(R?C?\d+)")


def _extract_subject_id(image_path, class_name):
    """
    Extrae un identificador de sujeto único a partir del nombre de fichero.

    El subject_id se prefija con la clase para garantizar que no pueda
    colisionar entre Control y ELA (defensa en profundidad: aunque la
    numeración actual ya no colisiona, el código queda más robusto así).
    """
    filename = os.path.basename(image_path)
    m = _SUBJECT_RE.match(filename)
    if not m:
        # Fallback: si nunca deberíamos llegar aquí, preservamos el stem
        # entero para que el assert de no-solape lo detecte.
        stem = filename.split("_", 1)[0]
        return f"{class_name}/UNPARSED-{stem}"
    return f"{class_name}/{m.group(1)}"


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
def _get_transforms():
    """
    Training: augmentation médicamente motivada (horizontal/vertical flip,
    ligera rotación y color jitter para simular variabilidad de sonda
    entre hospitales).
    Validation: solo resize + normalize.
    """
    train_transforms = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, val_transforms


# -----------------------------------------------------------------------------
# Dataset construction + group vector
# -----------------------------------------------------------------------------
def _build_dataset(muscle_name):
    """
    Crea dos ImageFolder sobre el mismo directorio (misma lista de samples,
    distintos transforms) y devuelve además los vectores `labels` y
    `groups` alineados por índice con `samples`.
    """
    data_path = Config.PROCESSED_DATA_PATH
    if muscle_name:
        data_path = os.path.join(data_path, muscle_name)

    train_tfm, val_tfm = _get_transforms()
    train_ds = datasets.ImageFolder(root=data_path, transform=train_tfm)
    val_ds   = datasets.ImageFolder(root=data_path, transform=val_tfm)

    labels, groups = [], []
    for path, lbl in train_ds.samples:
        class_name = train_ds.classes[lbl]
        labels.append(lbl)
        groups.append(_extract_subject_id(path, class_name))

    return train_ds, val_ds, labels, groups


def _assert_no_overlap(stage, train_groups, val_groups):
    """Sanity check: ningún sujeto puede estar simultáneamente en train y val."""
    train_s, val_s = set(train_groups), set(val_groups)
    overlap = train_s & val_s
    assert not overlap, f"[{stage}] OVERLAP de sujetos train/val: {overlap}"
    print(f"  [{stage}] sujetos  train={len(train_s):>3}  val={len(val_s):>3}  "
          f"overlap={len(overlap)}  OK")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def get_dataloaders(muscle_name=None, train_split=0.8, verbose=True):
    """
    Loader con split 80/20 **A NIVEL DE SUJETO** (no de imagen).

    Antes usábamos `torch.randperm` sobre las imágenes individuales. Eso
    provocaba fuga de datos: un paciente con imagen izquierda y derecha
    podía tener un lado en train y el otro en val, de forma que el
    modelo aprendía a reconocer al sujeto en vez del patrón de ELA.
    `GroupShuffleSplit` fija como grupo el `subject_id`, garantizando
    que todas las imágenes de un mismo paciente caigan juntas.

    Parameters
    ----------
    muscle_name : "Bicep" | "Antebrazo" | "Quadriceps" | "Tibial" | None
    train_split : fracción de sujetos que van a entrenamiento (default 0.8)

    Returns
    -------
    train_loader, val_loader, class_names
    """
    train_ds, val_ds, labels, groups = _build_dataset(muscle_name)
    n_images = len(train_ds)

    gss = GroupShuffleSplit(
        n_splits=1,
        train_size=train_split,
        random_state=Config.SEED,
    )
    train_idx, val_idx = next(gss.split(
        X=range(n_images), y=labels, groups=groups,
    ))

    if verbose:
        tr_g = [groups[i] for i in train_idx]
        vl_g = [groups[i] for i in val_idx]
        print(f"  [dataset] {muscle_name or 'ALL'}: {n_images} imgs | "
              f"Train={len(train_idx)} Val={len(val_idx)} | "
              f"Classes={train_ds.classes}")
        _assert_no_overlap("split", tr_g, vl_g)

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=Config.BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=Config.BATCH_SIZE, shuffle=False,
    )
    return train_loader, val_loader, train_ds.classes


def get_kfold_splits(muscle_name=None, n_splits=5, verbose=True):
    """
    Generador de folds con `StratifiedGroupKFold`:
      - conserva balance de clases (0_Control vs 1_ELA) en cada fold
      - mantiene los sujetos separados entre train y val de cada fold

    Para la fase 2 del TFG (cross-validation). Uso:

        for k, (tr, vl, classes) in enumerate(get_kfold_splits("Bicep")):
            train_model(..., train_loader=tr, val_loader=vl)
    """
    train_ds, val_ds, labels, groups = _build_dataset(muscle_name)
    n_images = len(train_ds)

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=Config.SEED,
    )
    for k, (train_idx, val_idx) in enumerate(
        sgkf.split(X=range(n_images), y=labels, groups=groups)
    ):
        if verbose:
            tr_g = [groups[i] for i in train_idx]
            vl_g = [groups[i] for i in val_idx]
            print(f"  [dataset] Fold {k+1}/{n_splits} | {muscle_name or 'ALL'} | "
                  f"Train={len(train_idx)} Val={len(val_idx)}")
            _assert_no_overlap(f"fold-{k+1}", tr_g, vl_g)

        train_loader = DataLoader(
            Subset(train_ds, train_idx),
            batch_size=Config.BATCH_SIZE, shuffle=True,
        )
        val_loader = DataLoader(
            Subset(val_ds, val_idx),
            batch_size=Config.BATCH_SIZE, shuffle=False,
        )
        yield train_loader, val_loader, train_ds.classes


# -----------------------------------------------------------------------------
# Self-check: python dataset.py  ->  verifica integridad del split
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    for muscle in ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]:
        print(f"\n=== {muscle} ===")
        get_dataloaders(muscle_name=muscle)
