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
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and getattr(module, 'in_channels', None) == 3:
            new_conv = nn.Conv2d(in_channels, module.out_channels,
                                 kernel_size=module.kernel_size,
                                 stride=module.stride,
                                 padding=module.padding,
                                 bias=(module.bias is not None))
            _set_module_by_name(model, name, new_conv)
            return True
    return False

def _replace_classifiers(model, out_features):
    replaced = False
    # main classifier
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, out_features)
        replaced = True
    # aux classifiers (if model constructed with aux_logits)
    if hasattr(model, 'aux1') and hasattr(model, 'aux2'):
        try:
            if hasattr(model.aux1, 'fc') and isinstance(model.aux1.fc, nn.Linear):
                model.aux1.fc = nn.Linear(model.aux1.fc.in_features, out_features)
                replaced = True
            if hasattr(model.aux2, 'fc') and isinstance(model.aux2.fc, nn.Linear):
                model.aux2.fc = nn.Linear(model.aux2.fc.in_features, out_features)
                replaced = True
        except Exception:
            pass
    return replaced

class GoogLeNetWrapper(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, pretrained_weights=True, aux_logits=False):
        super().__init__()
        out_features = 1 if num_classes == 2 else num_classes
        self.model = None

        try:
            from torchvision.models import googlenet, GoogLeNet_Weights
            weights = GoogLeNet_Weights.IMAGENET1K_V1 if pretrained_weights else None
            # disable aux_logits for simplicity unless user wants them
            base = googlenet(weights=weights, aux_logits=aux_logits)
            if in_channels != 3:
                _replace_first_conv(base, in_channels)
            _replace_classifiers(base, out_features)
            self.model = base
        except Exception as e:
            # fallback to SimpleCNN
            print("Warning: torchvision googlenet not available or failed. Falling back to SimpleCNN. (", e, ")")
            try:
                from model import SimpleCNN
                self.model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
            except Exception as e2:
                raise RuntimeError("Cannot create GoogLeNet wrapper and fallback failed: " + str(e2))

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    m = GoogLeNetWrapper(in_channels=1, num_classes=2, pretrained_weights=False)
    print(m)
    x = torch.randn(2, 1, 224, 224)
    y = m(x)
    print("output shape:", y.shape)