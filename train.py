# train.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import SimpleCNN
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter ---
EPOCHS = 16
BATCH_SIZE = 16
LEARNING_RATE = 0.000001

def train(model_type: str = "simple"):
    # 1. Memuat Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Pilih Model
    if model_type == "efficient":
        try:
            from model_efficientnet import EfficientNetV1 as ModelClass
        except Exception as e:
            print("Warning: model_efficientnet import failed, falling back to SimpleCNN:", e)
            ModelClass = SimpleCNN
    elif model_type == "googlenet":
        try:
            from model_googlenet import GoogLeNetWrapper as ModelClass
        except Exception as e:
            print("Warning: model_googlenet import failed, falling back to SimpleCNN:", e)
            ModelClass = SimpleCNN
    else:
        ModelClass = SimpleCNN

    # Inisialisasi Model
    model = ModelClass(in_channels=in_channels, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(model)
    print("Using device:", device)
    
    # 3. Mendefinisikan Loss Function dan Optimizer
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Inisialisasi list untuk menyimpan history
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    print("\n--- Memulai Training ---")
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
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
            if num_classes == 2:
                if outputs.dim() == 1:
                    outputs = outputs.view(-1, 1)
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
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
        
        avg_train_loss = running_loss / len(train_loader)
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
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0.0
        
        # Simpan history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

    print("--- Training SelesAI ---")
    
    # Tampilkan plot
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)

    # Visualisasi prediksi pada 10 gambar random dari validation set
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["simple", "efficient", "googlenet"], default="simple",
                        help="Model to use: 'simple', 'efficient', or 'googlenet'")
    args = parser.parse_args()
    train(model_type=args.model)
