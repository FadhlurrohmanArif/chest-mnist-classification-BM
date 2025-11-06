import torch
import torch.nn as nn

def _set_module_by_name(model, module_name, new_module):
    parts = module_name.split('.')
    parent = model
    for p in parts[:-1]:
        if p.isdigit() and isinstance(parent, (list, nn.Sequential)):
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit() and isinstance(parent, (list, nn.Sequential)):
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)

def _replace_first_conv(model, in_channels):
    # cari conv pertama dengan in_channels == 3 dan ganti
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            new_conv = nn.Conv2d(in_channels, module.out_channels,
                                 kernel_size=module.kernel_size,
                                 stride=module.stride,
                                 padding=module.padding,
                                 bias=(module.bias is not None))
            _set_module_by_name(model, name, new_conv)
            return True
    return False

def _replace_classifier(model, out_features):
    # cari Linear terakhir dan ganti dengan Linear(out_features)
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            new_linear = nn.Linear(module.in_features, out_features)
            _set_module_by_name(model, name, new_linear)
            return True
    return False

class EfficientNetV1(nn.Module):
    """
    Wrapper: coba gunakan torchvision EfficientNet-V1 (variant 's' default).
    Jika torchvision tidak tersedia atau tidak kompatibel, fallback ke SimpleCNN dari model.py.
    Output dim: 1 jika num_classes == 2 (sesuai BCEWithLogitsLoss di train.py), else num_classes.
    """
    def __init__(self, in_channels=1, num_classes=2, variant='s'):
        super().__init__()
        out_features = 1 if num_classes == 2 else num_classes
        self.model = None

        # Coba import EfficientNet v1 dari torchvision (jika tersedia)
        try:
            from torchvision.models import efficientnet_v1_s, efficientnet_v1_m, efficientnet_v1_l
            variant_map = {
                's': efficientnet_v1_s,
                'm': efficientnet_v1_m,
                'l': efficientnet_v1_l
            }
            fn = variant_map.get(variant, efficientnet_v1_s)
            base = fn(weights=None)  # tanpa pretrained
            # ganti conv pertama jika input channel != 3
            if in_channels != 3:
                _replace_first_conv(base, in_channels)
            # ganti classifier terakhir agar sesuai out_features
            if not _replace_classifier(base, out_features):
                # jika gagal, coba ganti attribute classifier jika ada
                if hasattr(base, 'classifier'):
                    try:
                        if isinstance(base.classifier, nn.Sequential):
                            for i, m in enumerate(base.classifier):
                                if isinstance(m, nn.Linear):
                                    base.classifier[i] = nn.Linear(m.in_features, out_features)
                                    break
                        elif isinstance(base.classifier, nn.Linear):
                            base.classifier = nn.Linear(base.classifier.in_features, out_features)
                    except Exception:
                        pass
            self.model = base
        except Exception as e:
            # fallback: gunakan SimpleCNN dari model.py agar kode tetap berjalan
            print("Warning: torchvision EfficientNet-V1 not available or incompatible. Falling back to SimpleCNN. (", e, ")")
            try:
                from model import SimpleCNN
                self.model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
            except Exception as e2:
                raise RuntimeError("Cannot create EfficientNetV1: torchvision not available and fallback failed: " + str(e2))

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # quick test
    m = EfficientNetV1(in_channels=1, num_classes=2, variant='s')
    print(m)
    x = torch.randn(4, 1, 28, 28)
    y = m(x)
    print("output shape:", y.shape)