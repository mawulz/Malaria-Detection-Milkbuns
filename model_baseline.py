import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class BaselineEfficientNet(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.4):
        super(BaselineEfficientNet, self).__init__()
        
        # Load Pretrained EfficientNetB0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        base_model = efficientnet_b0(weights=weights)
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Head modification for binary classification with dropout
        in_features = self.features[8][0].out_channels # should be 1280
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate/2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def build_model():
    return BaselineEfficientNet(dropout_rate=0.4)
