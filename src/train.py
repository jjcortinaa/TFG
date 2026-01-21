import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from models import ALS_ResNet18
from config import Config
import os

def train():
    # 1. Configurar dispositivo (Optimizado para Mac M4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Iniciando entrenamiento en: {device} ---")

    # 2. Cargar Datos
    train_loader, val_loader, _ = get_dataloaders()

    # 3. Instanciar Modelo, Pérdida y Optimizador
    model = ALS_ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Un LR bajo para Transfer Learning

    # 4. Bucle de entrenamiento
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

        # 5. Evaluación (Métricas médicas)
        model.eval()
        correct, total = 0, 0
        tp, tn, fp, fn = 0, 0, 0, 0 # Para Sensibilidad y Especificidad

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Cálculo manual de métricas para el TFG
                for p, l in zip(predicted, labels):
                    if l == 1 and p == 1: tp += 1
                    if l == 0 and p == 0: tn += 1
                    if l == 0 and p == 1: fp += 1
                    if l == 1 and p == 0: fn += 1

        # Cálculos finales de la época
        acc = 100 * correct / total
        sens = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"Época [{epoch+1}/{Config.EPOCHS}] -> Loss: {running_loss/len(train_loader):.4f} | Acc: {acc:.2f}% | Sens: {sens:.2f}% | Spec: {spec:.2f}%")

    # 6. Guardar el modelo
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "resultados/resnet18_best.pth")
    print("\n--- Entrenamiento finalizado y modelo guardado ---")

if __name__ == "__main__":
    train()