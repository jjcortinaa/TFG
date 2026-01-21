import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from config import Config

def get_dataloaders(train_split=0.8):
    """
    Loads images, applies Data Augmentation to the training set,
    normalization to both, and creates DataLoaders for the M4 GPU.
    """
    
    # 1. AUGMENTATION FOR TRAINING
    # We add random transformations to make the model more robust
    train_transforms = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5), # 50% chance to flip
        transforms.RandomRotation(10),           # Rotate by +/- 10 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Slight shift
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  #Normalization based on ImageNet stats
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. SIMPLE TRANSFORMS FOR VALIDATION
    # No augmentation here, just standard resizing and normalization
    val_transforms = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. LOAD DATASET
    # We load the whole folder first
    # Create a dataset object that automatically maps folder names to numerical labels.
    # It scans the directory, identifies '0_Control' and '1_ELA' as classes, 
    # and labels all image files found within those subfolders (0 and 1).
    full_dataset = datasets.ImageFolder(root=Config.PROCESSED_DATA_PATH)
    
    # 4. TRAIN / VALIDATION SPLIT
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(Config.SEED)
    train_data, val_data = random_split(full_dataset, [train_size, val_size], generator=generator)

    # 5. APPLY INDIVIDUAL TRANSFORMS
    # We wrap the subsets to apply different transforms to each
    train_data.dataset.transform = train_transforms
    val_data.dataset.transform = val_transforms

    # 6. CREATE DATALOADERS
    train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, full_dataset.classes

if __name__ == "__main__":
    t_loader, v_loader, classes = get_dataloaders()
    print(f"[SUCCESS] Data Augmentation active for training.")
    print(f" -> Training batches: {len(t_loader)}")
    print(f" -> Validation batches: {len(v_loader)}")