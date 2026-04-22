import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataset import get_dataloaders
from models import get_model, get_all_model_names
from config import Config

# Saved models go to:  TFG/best_models/<model>_<muscle>_best.pth
SAVE_DIR = os.path.join(Config.BASE_DIR, "best_models")


def train_model(model_name: str, muscle_name: str):
    """
    Trains a single model on a single muscle group.
    Returns a dict with the best metrics achieved during training.

    Parameters
    ----------
    model_name  : one of get_all_model_names()
    muscle_name : "Bicep" | "Antebrazo" | "Quadriceps" | "Tibial"
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name.upper()}  |  MUSCLE: {muscle_name.upper()}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    train_loader, val_loader, class_names = get_dataloaders(muscle_name=muscle_name)

    model     = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # ReduceLROnPlateau: important for small medical datasets
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    best = {"acc": 0, "sens": 0, "spec": 0, "epoch": 0}
    best_weights = None

    for epoch in range(Config.EPOCHS):
        # -- Training --
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

        # -- Validation --
        model.eval()
        tp = tn = fp = fn = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images), 1)
                for p, l in zip(predicted, labels):
                    if l == 1 and p == 1: tp += 1
                    if l == 0 and p == 0: tn += 1
                    if l == 0 and p == 1: fp += 1
                    if l == 1 and p == 0: fn += 1

        total = tp + tn + fp + fn
        acc  = 100 * (tp + tn) / total if total > 0 else 0
        sens = 100 * tp / (tp + fn)    if (tp + fn) > 0 else 0
        spec = 100 * tn / (tn + fp)    if (tn + fp) > 0 else 0

        scheduler.step(acc)
        print(f"  Epoch [{epoch+1:>3}/{Config.EPOCHS}] "
              f"Loss: {running_loss/len(train_loader):.4f} | "
              f"Acc: {acc:.1f}% | Sens: {sens:.1f}% | Spec: {spec:.1f}%")

        if acc > best["acc"]:
            best = {"acc": acc, "sens": sens, "spec": spec, "epoch": epoch + 1}
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # -- Save to TFG/best_models/ --
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{model_name}_{muscle_name.lower()}_best.pth")
    if best_weights:
        torch.save(best_weights, save_path)

    print(f"\n  Best epoch {best['epoch']}: Acc={best['acc']:.1f}% | "
          f"Sens={best['sens']:.1f}% | Spec={best['spec']:.1f}%")
    print(f"  Model saved -> {save_path}")

    best["model"]  = model_name
    best["muscle"] = muscle_name
    return best


def train_all(muscles=None, model_names=None):
    """
    Trains every (model x muscle) combination and prints a summary table.
    """
    if muscles is None:
        muscles = ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]
    if model_names is None:
        model_names = get_all_model_names()

    all_results = []
    for muscle in muscles:
        for model_name in model_names:
            try:
                result = train_model(model_name, muscle)
                all_results.append(result)
            except Exception as e:
                print(f"[ERROR] {model_name} / {muscle}: {e}")

    print("\n" + "="*70)
    print("  TRAINING SUMMARY")
    print("="*70)
    print(f"  {'MODEL':<20} {'MUSCLE':<14} {'ACC':>8} {'SENS':>8} {'SPEC':>8}")
    print("  " + "-"*66)
    for r in all_results:
        print(f"  {r['model']:<20} {r['muscle']:<14} "
              f"{r['acc']:>7.1f}% {r['sens']:>7.1f}% {r['spec']:>7.1f}%")
    print("="*70)
    return all_results


if __name__ == "__main__":
    # OPTION A: train ONE model on ONE muscle (quick test)
    # train_model("vgg16", "Quadriceps")

    # OPTION B: train ALL models on ALL muscles
    train_all()
