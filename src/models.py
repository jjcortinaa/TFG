import torch.nn as nn
from torchvision import models

class ALS_ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ALS_ResNet18, self).__init__()
        # Cargamos los pesos de ImageNet (Transfer Learning)
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Sustituimos la capa final (Fully Connected) para tus 2 clases
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)