# train.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from torch.nn.utils import clip_grad_norm_
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import SimpleCNN
from utils import plot_training_history, visualize_random_val_predictions

# --- Default Hyperparameter (bisa override via CLI) ---
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 0.01
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_CLIP_NORM = 1.0

def _estimate_pos_weight_from_loader(train_loader):
    """Hitung pos_weight untuk BCEWithLogitsLoss jika dataset tidak seimbang.
    pos_weight = N_neg / N_pos (format tensor [1])"""
    try:
        counts = torch.zeros(2, dtype=torch.long)
        for _, labels in train_loader:
            # labels bisa shape (N,) atau (N,1)
            labels = labels.view(-1).long()
            binc = torch.bincount(labels, minlength=2)
            counts[:2] += binc[:2]
        pos = counts[1].item()
        neg = counts[0].item()
        if pos == 0:
            return None
        pw = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
        return pw
    except Exception:
        return None

def build_optimizer(params, name: str, lr: float, weight_decay: float, momentum: float,
                    beta1: float, beta2: float, nesterov: bool):
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    elif name == "rmsprop":
        return optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:  # "adam" (default)
        return optim.Adam(params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay, amsgrad=True)

def build_scheduler(name: str, optimizer, epochs: int, steps_per_epoch: int,
                    step_size: int, gamma: float, t_max: int, plateau_patience: int,
                    plateau_factor: float, max_lr: float):
    name = (name or "none").lower()
    if name == "steplr":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=t_max if t_max > 0 else epochs)
    elif name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=plateau_patience, factor=plateau_factor)
    elif name == "onecycle":
        total_steps = epochs * steps_per_epoch
        total_steps = max(total_steps, 10)  # guard
        return OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)
    else:
        return None

def train(model_type: str = "simple",
          epochs: int = DEFAULT_EPOCHS,
          batch_size: int = DEFAULT_BATCH_SIZE,
          lr: float = DEFAULT_LR,
          optimizer_name: str = "adam",
          momentum: float = 0.9,
          beta1: float = 0.9,
          beta2: float = 0.999,
          weight_decay: float = DEFAULT_WEIGHT_DECAY,
          scheduler_name: str = "none",
          step_size: int = 20,
          gamma: float = 0.1,
          t_max: int = 50,
          plateau_patience: int = 5,
          plateau_factor: float = 0.5,
          onecycle_max_lr: float = 0.05,
          clip_norm: float = DEFAULT_CLIP_NORM,
          manual_pos_weight: float = None,
          nesterov: bool = True):

    # 1. Memuat Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(batch_size)
    
    # 2. Pilih Model
    if model_type == "efficient":
        try:
            from model_efficientnet import EfficientNetV1 as ModelClass
        except Exception as e:
            print("Warning: model_efficientnet import failed, falling back to SimpleCNN:", e)
            ModelClass = SimpleCNN
    elif model_type == "regnet":
        try:
            from regnet_model import RegNetY16GF as ModelClass
        except Exception as e:
            print("Warning: regnet_model import failed, falling back to SimpleCNN:", e)
            ModelClass = SimpleCNN
    else:
        ModelClass = SimpleCNN

    # Inisialisasi Model
    model = ModelClass(in_channels=in_channels, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model)
    print("Using device:", device)
    
    # 3. Loss + Optimizer (+ pos_weight bila binary & imbalance)
    criterion = None
    pos_weight_tensor = None
    if num_classes == 2:
        if manual_pos_weight is not None and manual_pos_weight > 0:
            pos_weight_tensor = torch.tensor([manual_pos_weight], dtype=torch.float32).to(device)
        else:
            est = _estimate_pos_weight_from_loader(train_loader)
            if est is not None:
                pos_weight_tensor = est.to(device)
                print(f"[INFO] Estimated pos_weight = {pos_weight_tensor.item():.4f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(
        model.parameters(), optimizer_name, lr, weight_decay, momentum, beta1, beta2, nesterov
    )

    # Scheduler (opsional)
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = build_scheduler(
        scheduler_name, optimizer, epochs, steps_per_epoch,
        step_size, gamma, t_max, plateau_patience, plateau_factor, onecycle_max_lr
    )
    
    # Inisialisasi history
    train_losses_history, val_losses_history = [], []
    train_accs_history, val_accs_history = [], []
    
    print("\n--- Memulai Training ---")
    
    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)

            # handle labels per task type
            if num_classes == 2:
                labels = labels.float().to(device)
                if labels.dim() == 1:
                    labels = labels.view(-1, 1)
            else:
                labels = labels.long().to(device).squeeze()
            
            outputs = model(images)
            
            # ensure outputs shape matches loss expectation
            if num_classes == 2 and outputs.dim() == 1:
                outputs = outputs.view(-1, 1)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (stabilitas training)
            if clip_norm and clip_norm > 0:
                clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            optimizer.step()

            # OneCycleLR butuh step per batch
            if isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            running_loss += loss.item()
            
            # Hitung training accuracy
            if num_classes == 2:
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            else:
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / max(len(train_loader), 1)
        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0.0
        
        # --- Fase Validasi ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                if num_classes == 2:
                    labels = labels.float().to(device)
                    if labels.dim() == 1:
                        labels = labels.view(-1, 1)
                else:
                    labels = labels.long().to(device).squeeze()
                
                outputs = model(images)
                if num_classes == 2 and outputs.dim() == 1:
                    outputs = outputs.view(-1, 1)
                
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                
                if num_classes == 2:
                    probs = torch.sigmoid(outputs)
                    predicted = (probs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                else:
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_running_loss / max(len(val_loader), 1)
        val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0.0
        
        # Scheduler per-epoch (kecuali OneCycle yang sudah per-batch)
        if scheduler and not isinstance(scheduler, OneCycleLR):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # Simpan history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        # Tampilkan lr saat ini
        try:
            current_lr = optimizer.param_groups[0]["lr"]
        except Exception:
            current_lr = lr

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"LR: {current_lr:.5f} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    print("--- Training Selesai ---")
    
    # Tampilkan plot
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)

    # Visualisasi prediksi pada 10 gambar random dari validation set
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["simple", "efficient", "regnet"], default="simple",
                        help="Model to use: 'simple' for SimpleCNN, 'efficient' for EfficientNetV1, 'regnet' for RegNet_Y_16GF_V1")

    # Hyperparameter via CLI
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--clip-norm", type=float, default=DEFAULT_CLIP_NORM)

    # Optimizer options
    parser.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--nesterov", action="store_true", help="Use Nesterov (SGD only)")

    # Scheduler options
    parser.add_argument("--scheduler", choices=["none", "steplr", "cosine", "plateau", "onecycle"], default="none")
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--t-max", type=int, default=50)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--onecycle-max-lr", type=float, default=0.05)

    # Imbalance handling
    parser.add_argument("--pos-weight", type=float, default=None,
                        help="Set manual pos_weight for BCEWithLogitsLoss (if binary). If not set, it will be estimated.")

    args = parser.parse_args()

    train(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer_name=args.optimizer,
        momentum=args.momentum,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        scheduler_name=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        t_max=args.t_max,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        onecycle_max_lr=args.onecycle_max_lr,
        clip_norm=args.clip_norm,
        manual_pos_weight=args.pos_weight,
        nesterov=args.nesterov
    )
