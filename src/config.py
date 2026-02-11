import os
import torch
import random
import numpy as np


class Config:
    """
    Centralized configuration for the ALS Detection project.
    All paths, hyperparameters, and hardware settings are defined here.
    """

    # 1. DIRECTORY STRUCTURE
    # Locate the root directory of the project
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Data paths
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

    # Model storage
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models")

    # 2. HARDWARE ACCELERATION (Apple Silicon M4)
    # Checks if Metal Performance Shaders (MPS) is available for GPU acceleration
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    # 3. IMAGE SPECIFICATIONS
    # Standard size for pre-trained CNNs (ResNet, VGG, etc.)
    IMAGE_SIZE = (224, 224)
    CHANNELS = 3  # RGB format required by most Transfer Learning models

    # 4. TRAINING HYPERPARAMETERS
    BATCH_SIZE = 16  # Number of images processed before the model updates
    LEARNING_RATE = 1e-4  # Initial step size for the optimizer ($10^{-4}$)
    EPOCHS = 50  # Total number of iterations over the full dataset

    # 5. REPRODUCIBILTY
    SEED = 42

    @classmethod
    def setup_project(cls):
        """
        Creates the necessary folder structure and sets random seeds for reproducibility.
        """
        # Create directories if they don't exist
        directories = [cls.RAW_DATA_PATH, cls.PROCESSED_DATA_PATH, cls.MODEL_SAVE_PATH]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"[INFO] Created directory: {directory}")

        # Fix seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        print(f"[INFO] Project set up. Using device: {cls.DEVICE}")


if __name__ == "__main__":
    # Execute setup when running the script directly
    Config.setup_project()
