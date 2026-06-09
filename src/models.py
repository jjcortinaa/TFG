import torch.nn as nn
from torchvision import models


def _get_model_registry():
    return {
        "resnet18":        _build_resnet18,
        "resnet50":        _build_resnet50,
        "vgg16":           _build_vgg16,
        "densenet121":     _build_densenet121,
        "efficientnet_b0": _build_efficientnet_b0,
        "mobilenet_v3":    _build_mobilenet_v3,
        "convnext_tiny":   _build_convnext_tiny,
    }


def get_model(model_name: str, num_classes: int = 2) -> nn.Module:
    """
    Factory function. Returns the requested model with Transfer Learning weights.
    Usage:
        model = get_model("vgg16")
        model = get_model("efficientnet_b0")
    """
    registry = _get_model_registry()
    if model_name not in registry:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {list(registry.keys())}"
        )
    return registry[model_name](num_classes)


def get_all_model_names():
    return list(_get_model_registry().keys())


# ── Individual builders ────────────────────────────────────────────────

def _build_resnet18(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def _build_resnet50(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def _build_vgg16(num_classes):
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model

def _build_densenet121(num_classes):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def _build_efficientnet_b0(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def _build_mobilenet_v3(num_classes):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

def _build_convnext_tiny(num_classes):
    """
    ConvNeXt-Tiny (Liu et al., 2022). Modern CNN inspired by Transformers
    (LayerNorm, GELU, stage-based blocks) but keeping the convolutional
    inductive bias -- suitable for small datasets like this one.
    ~28M parameters.
    """
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


# ── Backward-compatible aliases (evaluate_robustness.py still works) ──

class ALS_ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = _build_resnet18(num_classes)
    def forward(self, x):
        return self.model(x)

class ALS_DenseNet121(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = _build_densenet121(num_classes)
    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    print("Available models:", get_all_model_names())
    for name in get_all_model_names():
        m = get_model(name)
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:<20} -> Trainable params: {params:,}")
