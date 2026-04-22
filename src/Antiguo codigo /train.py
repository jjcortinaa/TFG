import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from models import ALS_ResNet18
from config import Config
import os


def train(muscle_name):
    # 1. Setup the M4 Chip (MPS) or CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Starting training for [{muscle_name}] on: {device} ---")

    # 2. Load data for the specific muscle
    # This now uses your updated dataset.py logic
    train_loader, val_loader, _ = get_dataloaders(muscle_name=muscle_name)

    # 3. Model, Loss, and Optimizer
    model = ALS_ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    # Learning rate is kept low to preserve ImageNet features while adapting to ultrasound textures
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 4. Training Loop
    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 5. Metrics and Evaluation (Validation Phase)
        model.eval()
        correct, total = 0, 0
        tp, tn, fp, fn = 0, 0, 0, 0  # Metrics for Sensitivity and Specificity

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Manual calculation of confusion matrix components
                for p, l in zip(predicted, labels):
                    if l == 1 and p == 1:
                        tp += 1  # True Positive (ALS correctly identified)
                    if l == 0 and p == 0:
                        tn += 1  # True Negative (Control correctly identified)
                    if l == 0 and p == 1:
                        fp += 1  # False Positive (Sano predicted as ELA)
                    if l == 1 and p == 0:
                        fn += 1  # False Negative (ELA predicted as Sano)

        # 6. Calculate Final Epoch Metrics
        acc = 100 * correct / total
        sens = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity (Recall)
        spec = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity

        print(
            f"Epoch [{epoch+1}/{Config.EPOCHS}] -> Loss: {running_loss/len(train_loader):.4f} | "
            f"Acc: {acc:.2f}% | Sens: {sens:.2f}% | Spec: {spec:.2f}%"
        )

    # 7. Save the specialized model
    # We use a unique name for each muscle to compare results later
    os.makedirs("models", exist_ok=True)
    save_path = f"models/resnet18_{muscle_name.lower()}_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n--- Training finished. Model saved at: {save_path} ---")


if __name__ == "__main__":
    # You can change this to: "Bicep", "Antebrazo", "Quadriceps", or "Tibial"
    # Martinez-Paya (2017) found best results in Quadriceps (AUC 0.985)
    TARGET_MUSCLE = "Quadriceps"
    train(TARGET_MUSCLE)
