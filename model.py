import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        reduced_planes = max(4, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return out  # we will multiply afterwards in CBAM module

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x  # we will multiply afterwards

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention Map * X
        ca_out = self.sigmoid(self.ca(x))
        x = x * ca_out
        # Spatial Attention Map * X
        sa_out = self.sigmoid(self.sa(x))
        x = x * sa_out
        return x

class HybridEfficientNetCBAM(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.4):
        super(HybridEfficientNetCBAM, self).__init__()
        
        # Load Pretrained EfficientNetB0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        base_model = efficientnet_b0(weights=weights)
        self.features = base_model.features
        
        # Determine the number of output channels for each stage and inject CBAM
        # torchvision's EfficientNetB0 features indices:
        # 0: ConvNormActivation (stem)
        # 1: MBConv (Stage 1)
        # 2: MBConv (Stage 2) -> We start inserting CBAM sequentially after block 2 (which is index 2) up to 7
        
        for idx in [5, 6, 7]:
            mbconv_block = self.features[idx]
            
            out_channels = {
                5: 112,
                6: 192,
                7: 320
            }[idx]
            self.features[idx] = nn.Sequential(
                mbconv_block,
                CBAM(out_channels)
            )

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
    return HybridEfficientNetCBAM(dropout_rate=0.4)
