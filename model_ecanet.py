import math
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# =============================================================================
# CBAM modules (kept here as reference — used in model.py)
# =============================================================================
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         reduced_planes = max(4, in_planes // ratio)
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_planes, reduced_planes, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return out
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return x
#
# class CBAM(nn.Module):
#     def __init__(self, in_planes, ratio=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.ca = ChannelAttention(in_planes, ratio)
#         self.sa = SpatialAttention(kernel_size)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         ca_out = self.sigmoid(self.ca(x))
#         x = x * ca_out
#         sa_out = self.sigmoid(self.sa(x))
#         x = x * sa_out
#         return x
# =============================================================================


class ECALayer(nn.Module):
    """Efficient Channel Attention (ECA-Net).

    Uses adaptive 1D convolution over channel descriptors instead of
    MLP-based dimensionality reduction, avoiding the information bottleneck.

    Paper: ECA-Net: Efficient Channel Attention for Deep Convolutional
           Neural Networks (https://arxiv.org/abs/1910.03151)

    Args:
        channels (int): Number of input channels.
        gamma (int): Denominator scaling factor for adaptive kernel size.
        b (int): Additive factor for adaptive kernel size.
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        # Adaptive kernel size: k = |log2(C)/gamma + b/gamma|_odd
        k = int(abs(math.log2(channels) / gamma + b / gamma))
        k = k if k % 2 else k + 1  # ensure odd

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling: (B, C, H, W) -> (B, C, 1, 1)
        y = self.avg_pool(x)
        # Reshape for 1D conv: (B, C, 1, 1) -> (B, 1, C)
        y = y.squeeze(-1).transpose(-1, -2)
        # 1D convolution across channels
        y = self.conv(y)
        # Reshape back: (B, 1, C) -> (B, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)
        # Sigmoid gating
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class HybridEfficientNetECA(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.4):
        super(HybridEfficientNetECA, self).__init__()

        # Load Pretrained EfficientNetB0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        base_model = efficientnet_b0(weights=weights)
        self.features = base_model.features

        # Inject ECA-Net after MBConv blocks 5, 6, 7
        # (same injection points as CBAM in model.py)
        for idx in [5, 6, 7]:
            mbconv_block = self.features[idx]

            out_channels = {
                5: 112,
                6: 192,
                7: 320
            }[idx]
            self.features[idx] = nn.Sequential(
                mbconv_block,
                ECALayer(out_channels)
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Head modification for binary classification with dropout
        in_features = self.features[8][0].out_channels  # should be 1280
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_model():
    return HybridEfficientNetECA(dropout_rate=0.4)
