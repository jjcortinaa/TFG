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
# Captures the subject's alphanumeric prefix: "C001", "C026", "RC001", "1001",
# "1018", etc. It keeps an optional R, an optional C, and 1-4 digits.
# It is robust to the naming variations that appear in the dataset:
#   "C001d_BBr_clean.jpg"  -> "C001"
#   "C001i_BBr_clean.jpg"  -> "C001"    (same subject, other side)
#   "RC001d_TbA_clean.jpg" -> "RC001"
#   "1001d_BBr_clean.jpg"  -> "1001"
#   "1018_Cdr_clean.jpg"   -> "1018"    (name without a side letter; Quadriceps)
#   "1018d(b)_clean.jpg"   -> "1018"    (second take with parentheses; Quadriceps)
#   "C026_Cdr_clean.jpg"   -> "C026"    (name without a side letter; Quadriceps)
_SUBJECT_RE = re.compile(r"^(R?C?\d+)")


def _extract_subject_id(image_path, class_name):
    """
    Extracts a unique subject identifier from the file name.

    The subject_id is prefixed with the class to guarantee that it cannot
    collide between Control and ALS (defence in depth: although the current
    numbering no longer collides, this keeps the code more robust).
    """
    filename = os.path.basename(image_path)
    m = _SUBJECT_RE.match(filename)
    if not m:
        # Fallback: if we should never reach here, keep the whole stem
        # so that the no-overlap assertion can detect it.
        stem = filename.split("_", 1)[0]
        return f"{class_name}/UNPARSED-{stem}"
    return f"{class_name}/{m.group(1)}"


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
def _get_transforms():
    """
    Training: medically motivated augmentation (horizontal/vertical flip,
    slight rotation and color jitter to simulate probe variability across
    hospitals).
    Validation: only resize + normalize.
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
    Builds two ImageFolder objects over the same directory (same sample list,
    different transforms) and also returns the `labels` and `groups` vectors
    aligned by index with `samples`.
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
    """Sanity check: no subject may be in train and val at the same time."""
    train_s, val_s = set(train_groups), set(val_groups)
    overlap = train_s & val_s
    assert not overlap, f"[{stage}] train/val subject OVERLAP: {overlap}"
    print(f"  [{stage}] subjects  train={len(train_s):>3}  val={len(val_s):>3}  "
          f"overlap={len(overlap)}  OK")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def get_dataloaders(muscle_name=None, train_split=0.8, verbose=True):
    """
    Loader with an 80/20 split **AT THE SUBJECT LEVEL** (not per image).

    We used to apply `torch.randperm` over individual images. That caused
    data leakage: a patient with a left and a right image could have one
    side in train and the other in val, so the model learned to recognise
    the subject instead of the ALS pattern. `GroupShuffleSplit` fixes the
    `subject_id` as the group, guaranteeing that all images of the same
    patient fall together.

    Parameters
    ----------
    muscle_name : "Bicep" | "Antebrazo" | "Quadriceps" | "Tibial" | None
    train_split : fraction of subjects assigned to training (default 0.8)

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
    Fold generator based on `StratifiedGroupKFold`:
      - preserves the class balance (0_Control vs 1_ELA) in each fold
      - keeps subjects separated between train and val in every fold

    For phase 2 of the thesis (cross-validation). Usage:

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
# Self-check: python dataset.py  ->  verifies the integrity of the split
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    for muscle in ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]:
        print(f"\n=== {muscle} ===")
        get_dataloaders(muscle_name=muscle)
