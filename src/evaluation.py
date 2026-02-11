# import torch
# import torch.nn as nn
# import torch.optim as optim
# from dataset import get_dataloaders
# from models import ALS_ResNet18
# from config import Config
# import os

# DIR_NAME = "resultados2"

# def train_muscle(muscle_name):
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#     print(f"\n{'='*50}")
#     print(f"STARTING TRAINING FOR: {muscle_name}")
#     print(f"{'='*50}")

#     # 2. Load data for this specific muscle
#     train_loader, val_loader, _ = get_dataloaders(muscle_name=muscle_name)

#     # 3. Model, Loss, and Optimizer
#     model = ALS_ResNet18().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.0001) 

#     best_metrics = {"acc": 0, "sens": 0, "spec": 0}

#     # 4. Training Loop
#     for epoch in range(Config.EPOCHS):
#         model.train()
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#         # 5. Validation Phase
#         model.eval()
#         correct, total = 0, 0
#         tp, tn, fp, fn = 0, 0, 0, 0 

#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)
                
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#                 for p, l in zip(predicted, labels):
#                     if l == 1 and p == 1: tp += 1 
#                     if l == 0 and p == 0: tn += 1 
#                     if l == 0 and p == 1: fp += 1
#                     if l == 1 and p == 0: fn += 1

#         # Calculate epoch metrics
#         acc = 100 * correct / total
#         sens = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
#         spec = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
        
#         # Keep track of the best results
#         if acc > best_metrics["acc"]:
#             best_metrics = {"acc": acc, "sens": sens, "spec": spec}

#         print(f"Epoch [{epoch+1}/{Config.EPOCHS}] -> Acc: {acc:.2f}% | Sens: {sens:.2f}% | Spec: {spec:.2f}%")

#     # 6. Save the specialized model weights
    
#     os.makedirs(DIR_NAME, exist_ok=True)
#     save_path = f"{DIR_NAME}/resnet18_{muscle_name.lower()}_best.pth"
#     torch.save(model.state_dict(), save_path)
    
#     return best_metrics

# if __name__ == "__main__":
#     # The four muscle groups from Martinez-Paya (2017) [cite: 11]
#     muscle_list = ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]
#     summary_results = {}

#     for muscle in muscle_list:
#         try:
#             metrics = train_muscle(muscle)
#             summary_results[muscle] = metrics
#         except Exception as e:
#             print(f"Error training {muscle}: {e}")

#     # 7. Generate Final Summary Report for your Thesis (TFG)
#     print("\n\n" + "="*50)
#     print("FINAL SUMMARY FOR TFG COMPARISON")
#     print("="*50)
#     print(f"{'MUSCLE':<15} | {'ACC':<10} | {'SENS':<10} | {'SPEC':<10}")
#     print("-" * 50)
    
#     with open(f"{DIR_NAME}/final_report.txt", "w") as f:
#         f.write("ALS Deep Learning Results Summary\n")
#         f.write("-" * 50 + "\n")
#         for muscle, m in summary_results.items():
#             line = f"{muscle:<15} | {m['acc']:.2f}% | {m['sens']:.2f}% | {m['spec']:.2f}%"
#             print(line)
#             f.write(line + "\n")
    
#     print(f"\nReport saved in {DIR_NAME}/final_report.txt")

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from dataset import get_dataloaders
from models import ALS_DenseNet121
from config import Config
import os

DIR_NAME = "resultados_dense"

def train_muscle(muscle_name):
    # Setup for M4 (MPS) or CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*50}\nTRAINING & EVALUATING: {muscle_name}\n{'='*50}")

    train_loader, val_loader, class_names = get_dataloaders(muscle_name=muscle_name)
    model = ALS_DenseNet121().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_metrics = {"acc": 0, "sens": 0, "spec": 0, "auc": 0, "cm": None}

    for epoch in range(Config.EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation with Advanced Metrics
        model.eval()
        y_true, y_pred, y_probs = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, predicted = torch.max(outputs, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_probs.extend(probs.cpu().numpy())

        # Metric Calculations
        cm = confusion_matrix(y_true, y_pred)
        # Handle cases where the confusion matrix might not be 2x2
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            # Fallback for single-class validation edge cases
            tn = cm[0][0] if y_true[0] == 0 else 0
            tp = cm[0][0] if y_true[0] == 1 else 0
            fp = fn = 0
        
        acc = 100 * (tp + tn) / len(y_true)
        sens = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
        auc = 100 * roc_auc_score(y_true, y_probs) if len(set(y_true)) > 1 else 0

        if acc >= best_metrics["acc"]:
            best_metrics = {"acc": acc, "sens": sens, "spec": spec, "auc": auc, "cm": cm}

        print(f"Epoch [{epoch+1}/{Config.EPOCHS}] -> Acc: {acc:.2f}% | Sens: {sens:.2f}% | Spec: {spec:.2f}% | AUC: {auc:.2f}%")

    # Plot and Save Confusion Matrix
    save_cm_plot(best_metrics["cm"], muscle_name, class_names)
    
    os.makedirs(DIR_NAME, exist_ok=True)
    torch.save(model.state_dict(), f"{DIR_NAME}/densenet121_{muscle_name.lower()}_best.pth")
    return best_metrics

def save_cm_plot(cm, muscle_name, class_names):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {muscle_name}')
    plt.ylabel('True Label (Gold Standard)')
    plt.xlabel('Predicted Label (AI)')
    os.makedirs(f"{DIR_NAME}/plots", exist_ok=True)
    plt.savefig(f"{DIR_NAME}/plots/cm_{muscle_name.lower()}.png")
    plt.close()

if __name__ == "__main__":
    muscle_list = ["Bicep", "Antebrazo", "Quadriceps", "Tibial"]
    summary_results = {}

    for muscle in muscle_list:
        try:
            summary_results[muscle] = train_muscle(muscle)
        except Exception as e:
            print(f"Error in {muscle}: {e}")

    # --- PRINT TO CONSOLE ---
    report_header = "\n" + "="*60 + "\nFINAL MEDICAL SUMMARY FOR TFG\n" + "="*60
    table_header = f"{'MUSCLE':<15} | {'ACC (%)':<8} | {'SENS (%)':<8} | {'SPEC (%)':<8} | {'AUC (%)':<8}"
    
    print(report_header)
    print(table_header)
    print("-" * 60)
    
    # --- SAVE TO FILE ---
    report_path = os.path.join(DIR_NAME, "final_report.txt")
    with open(report_path, "w") as f:
        f.write(report_header + "\n")
        f.write(table_header + "\n")
        f.write("-" * 60 + "\n")
        
        for muscle, m in summary_results.items():
            row = f"{muscle:<15} | {m['acc']:<8.2f} | {m['sens']:<8.2f} | {m['spec']:<8.2f} | {m['auc']:<8.2f}"
            print(row)
            f.write(row + "\n")
            
    print(f"\n[INFO] Report successfully saved at: {report_path}")