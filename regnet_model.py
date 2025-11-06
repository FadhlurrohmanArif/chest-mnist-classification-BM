import torch
import torch.nn as nn

class RegNetY16GF(nn.Module):
    """
    Wrapper for torchvision RegNet Y_16GF (v1 if available).
    Usage: RegNetY16GF(in_channels=1, num_classes=2)
    - If in_channels != 3, a small conv maps input channels -> 3.
    - For binary classification (num_classes==2) final head outputs a single logit.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Try to get regnet factory from torchvision with fallback names
        regnet_factory = None
        try:
            # torchvision >= 0.14+ often exposes regnet_y_16gf_v1
            from torchvision.models import regnet_y_16gf_v1 as regnet_factory
        except Exception:
            try:
                from torchvision.models import regnet_y_16gf as regnet_factory
            except Exception:
                regnet_factory = None

        if regnet_factory is None:
            raise ImportError(
                "RegNet factory not found in torchvision. "
                "Install a torchvision version that provides regnet_y_16gf[_v1]."
            )

        # Instantiate backbone (no pretrained weights here)
        backbone = regnet_factory(weights=None)

        # Replace final head
        in_features = backbone.fc.in_features
        out_features = 1 if num_classes == 2 else num_classes
        backbone.fc = nn.Linear(in_features, out_features)

        self.backbone = backbone

        # If input channels are not 3, prepend conv to map to 3 channels
        if in_channels != 3:
            self.conv_in = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=False)
            # simple normalization layer can help; keep identity if not needed
            self.bn_in = nn.BatchNorm2d(3)
        else:
            self.conv_in = None
            self.bn_in = None

    def forward(self, x):
        if self.conv_in is not None:
            x = self.conv_in(x)
            x = self.bn_in(x)
            x = torch.relu(x)
        x = self.backbone(x)
        # For binary, return (N,1) or (N,) is both handled by training code; return (N,1) here
        if self.num_classes == 2:
            return x.view(-1, 1)
        return x