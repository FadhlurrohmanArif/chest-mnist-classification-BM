import torch
import torch.nn as nn
import torchvision.models as models

class RegNetY(nn.Module):
    """
    Wrapper for torchvision RegNet-Y backbones.
    - in_channels: input channels (will replace first conv if != 3)
    - num_classes: if 2 -> single-logit output (for BCEWithLogitsLoss), else multi-class output
    - backbone_name: choose regnet variant available in torchvision, e.g. 'regnet_y_400mf'
    - pretrained: whether to load pretrained weights (default False)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 backbone_name: str = "regnet_y_400mf", pretrained: bool = False):
        super().__init__()
        self.backbone = getattr(models, backbone_name)(pretrained=pretrained)
        # adapt first conv if input channels differ
        if in_channels != 3:
            replaced = False
            # try common locations for first conv
            try:
                stem = getattr(self.backbone, "stem", None)
                if stem is not None and isinstance(stem, nn.Sequential) and isinstance(stem[0], nn.Conv2d):
                    first_conv = stem[0]
                    new_conv = nn.Conv2d(in_channels, first_conv.out_channels,
                                         kernel_size=first_conv.kernel_size,
                                         stride=first_conv.stride,
                                         padding=first_conv.padding,
                                         bias=(first_conv.bias is not None))
                    stem[0] = new_conv
                    replaced = True
            except Exception:
                replaced = False
            if not replaced:
                try:
                    first_conv = getattr(self.backbone, "conv1", None)
                    if isinstance(first_conv, nn.Conv2d):
                        new_conv = nn.Conv2d(in_channels, first_conv.out_channels,
                                             kernel_size=first_conv.kernel_size,
                                             stride=first_conv.stride,
                                             padding=first_conv.padding,
                                             bias=(first_conv.bias is not None))
                        self.backbone.conv1 = new_conv
                        replaced = True
                except Exception:
                    pass
            # if still not replaced, leave as-is (will error at runtime if channels mismatch)

        # adjust classifier head
        out_features = 1 if num_classes == 2 else num_classes
        if hasattr(self.backbone, "fc"):
            in_feats = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_feats, out_features)
        elif hasattr(self.backbone, "classifier"):
            in_feats = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(in_feats, out_features)
        else:
            # fallback: attach a simple head
            self.backbone.head = nn.Linear(1000, out_features)

    def forward(self, x):
        return self.backbone(x)