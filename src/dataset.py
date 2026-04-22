import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from config import Config


def get_dataloaders(muscle_name=None, train_split=0.8):
    """
    Loads images for a specific muscle group (or the whole dataset).

    Parameters
    ----------
    muscle_name : "Bicep" | "Antebrazo" | "Quadriceps" | "Tibial" | None
    train_split : fraction for training (default 0.8)

    Structure expected on disk:
        TFG/data/processed/<muscle>/<0_Control|1_ELA>/*.jpg
    """
    # Config.PROCESSED_DATA_PATH = TFG/data/processed
    data_path = Config.PROCESSED_DATA_PATH
    if muscle_name:
        data_path = os.path.join(data_path, muscle_name)

    # Training transforms: medical-specific augmentation
    # ColorJitter simulates scanner variability between hospitals
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

    # Validation transforms: no augmentation, only normalise
    val_transforms = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Two dataset instances: same files, different transforms
    train_dataset = datasets.ImageFolder(root=data_path, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(root=data_path, transform=val_transforms)

    # Reproducible split (same indices every run thanks to fixed seed)
    num_images = len(train_dataset)
    indices = torch.randperm(
        num_images,
        generator=torch.Generator().manual_seed(Config.SEED)
    ).tolist()

    split         = int(train_split * num_images)
    train_indices = indices[:split]
    val_indices   = indices[split:]

    train_data = Subset(train_dataset, train_indices)
    val_data   = Subset(val_dataset,   val_indices)

    train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=Config.BATCH_SIZE, shuffle=False)

    print(f"  [dataset] {muscle_name or 'ALL'}: {num_images} imgs | "
          f"Train={len(train_indices)} Val={len(val_indices)} | "
          f"Classes={train_dataset.classes}")

    return train_loader, val_loader, train_dataset.classes
