import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from config import Config


def get_dataloaders(muscle_name=None, train_split=0.8):
    """
    Loads images for a specific muscle or the whole dataset.
    Uses two ImageFolder instances to handle different transforms for train and val.
    """

    # 1. SET THE PATH
    # We navigate to the specific muscle folder (e.g., 'Quadriceps')
    data_path = Config.PROCESSED_DATA_PATH
    if muscle_name:
        data_path = os.path.join(data_path, muscle_name)

    # 2. DEFINE TRANSFORMS
    # Augmentation for the training phase
    train_transforms = transforms.Compose(
        [
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Only normalization for the validation phase
    val_transforms = transforms.Compose(
        [
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 3. INITIALIZE TWO IDENTICAL DATASETS WITH DIFFERENT TRANSFORMS
    # This is the cleanest way to avoid the shared-transform bug
    train_dataset = datasets.ImageFolder(root=data_path, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=data_path, transform=val_transforms)

    # 4. GENERATE INDICES FOR THE SPLIT
    num_images = len(train_dataset)
    indices = torch.randperm(
        num_images, generator=torch.Generator().manual_seed(Config.SEED)
    ).tolist()

    split = int(train_split * num_images)
    train_indices = indices[:split]
    val_indices = indices[split:]

    # 5. CREATE SUBSETS
    # Subset points to the indices of the specific dataset object [cite: 140, 414]
    train_data = Subset(train_dataset, train_indices)
    val_data = Subset(val_dataset, val_indices)

    # 6. CREATE DATALOADERS
    train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, train_dataset.classes
