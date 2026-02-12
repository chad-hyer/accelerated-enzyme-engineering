import os, json
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

# --- Imports from your provided files ---
# Ensure these files are in the same folder or PYTHONPATH
from meta_tensor_builder import ProteinTilingDataset

# ==========================================
# 1. Early Stopping Helper
# ==========================================
class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=15, min_delta=1e-4, path='checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        torch.save(model.state_dict(), self.path)
        # print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')

# ==========================================
# 2. Model Architectures
# ==========================================
class DynamicMetaLearnerExpansion(nn.Module):
    """Original Expansion Style: Input -> Wide -> Wider -> ... -> 1"""
    def __init__(self, input_channels=4, hidden_channels=32, kernel_size=5, num_layers=3, dropout=0.0):
        super().__init__()
        layers = []
        padding_h = (kernel_size - 1) // 2 
        
        # Input Layer
        layers.append(nn.Conv2d(input_channels, hidden_channels, 
                                kernel_size=(kernel_size, 1), padding=(padding_h, 0)))
        layers.append(nn.ReLU())
        if dropout > 0: layers.append(nn.Dropout2d(p=dropout))
        
        # Hidden Layers (Expanding)
        current_channels = hidden_channels
        for _ in range(num_layers - 2):
            next_channels = current_channels * 2
            layers.append(nn.Conv2d(current_channels, next_channels, 
                                    kernel_size=(kernel_size, 1), padding=(padding_h, 0)))
            layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout2d(p=dropout))
            current_channels = next_channels

        # Final Projection
        layers.append(nn.Conv2d(current_channels, 1, kernel_size=(1, 1)))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class DynamicMetaLearnerHourglass(nn.Module):
    """New Hourglass Style: Input -> Wide -> Narrow -> 1"""
    def __init__(self, input_channels=4, hidden_channels=32, kernel_size=5, num_layers=3, dropout=0.0):
        super().__init__()
        layers = []
        padding_h = (kernel_size - 1) // 2 
        MAX_CHANNELS = 64
        
        # --- ENCODER (Expansion) ---
        encoder_channels = [] 
        current = input_channels
        next_ch = hidden_channels
        
        for i in range(num_layers):
            if next_ch > MAX_CHANNELS: next_ch = MAX_CHANNELS
            
            layers.append(nn.Conv2d(current, next_ch, kernel_size=(kernel_size, 1), padding=(padding_h, 0)))
            layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout2d(p=dropout))
            
            encoder_channels.append(next_ch)
            current = next_ch
            next_ch = current * 2 

        # --- DECODER (Contraction) ---
        # Reverse encoder outputs (excluding the peak)
        decoder_targets = encoder_channels[:-1][::-1]
        
        for target_ch in decoder_targets:
            layers.append(nn.Conv2d(current, target_ch, kernel_size=(kernel_size, 1), padding=(padding_h, 0)))
            layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout2d(p=dropout))
            current = target_ch

        # Final Projection
        layers.append(nn.Conv2d(current, 1, kernel_size=(1, 1)))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. Main Cross-Validation Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, required=True, choices=['expansion', 'hourglass'], help='Model architecture style')
    parser.add_argument('--config_idx', type=int, required=True, help='Index from param_grid')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum epochs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    args = parser.parse_args()

    # --- 1. Recreate Configuration Grid ---
    param_grid = {
        'kernel_size': [3, 5, 7, 9],
        'num_layers': [2, 3, 4],
        'hidden_channels': [16, 32],
        'dropout': [0.0, 0.2]
    }
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    if args.config_idx >= len(combinations):
        print(f"Error: Config ID {args.config_idx} out of range.")
        return

    config = combinations[args.config_idx]
    model_name = f"{args.style}_config_{args.config_idx}"

    # --- 2. Initialize Model for Verification ---
    if args.style == 'expansion':
        model = DynamicMetaLearnerExpansion(**config)
    else:
        model = DynamicMetaLearnerHourglass(**config)

    # --- VERIFICATION BLOCK ------------------------------------------------
    print(f"\n{'='*50}")
    print(f"ARCHITECTURE CHECK: {model_name}")
    print(f"{'='*50}")
    print(f"Hyperparameters:")
    print(json.dumps(config, indent=4))
    print(f"\nModel Structure (Verify Layer Sizes):")
    print(model)
    print(f"{'='*50}\n")
    # -----------------------------------------------------------------------

    print(f"--- Running {args.folds}-Fold CV for {model_name} ---")

    # --- 3. Prepare Dataset ---
    CA_DIR = "/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/model_architecture_optimization/CA_training_files"
    full_dataset = ProteinTilingDataset(
        pred_dir=os.path.join(CA_DIR, 'predictions'),
        error_dir=os.path.join(CA_DIR, 'errors'),
        evc_path=os.path.join(CA_DIR, 'CA_single_mutant_matrix.csv'),
        global_start=1, global_end=243 
    )

    # --- 4. Cross-Validation Loop ---
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_best_losses = []
    
    save_dir = "cv_models"
    os.makedirs(save_dir, exist_ok=True)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
        print(f"\n--- FOLD {fold+1}/{args.folds} ---")
        
        # Subsamplers
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(full_dataset, batch_size=args.batch_size, 
                                  sampler=train_subsampler, num_workers=args.workers)
        val_loader = DataLoader(full_dataset, batch_size=args.batch_size, 
                                sampler=val_subsampler, num_workers=args.workers)
        
        # Re-Initialize Model for Training (Reset weights)
        if args.style == 'expansion':
            model = DynamicMetaLearnerExpansion(**config)
        else:
            model = DynamicMetaLearnerHourglass(**config)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss(reduction='none')
        
        fold_save_path = os.path.join(save_dir, f"{model_name}_fold{fold+1}.pth")
        early_stopping = EarlyStopping(patience=15, min_delta=0.0001, path=fold_save_path)
        
        for epoch in range(args.epochs):
            # TRAIN
            model.train()
            train_loss = 0.0
            train_batches = 0
            for inp, tgt, mask in train_loader:
                optimizer.zero_grad()
                output = model(inp)
                loss = (criterion(output, tgt) * mask).sum() / (mask.sum() + 1e-6)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_batches += 1
            avg_train_loss = train_loss / (train_batches + 1e-6)

            # VALIDATE
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for inp, tgt, mask in val_loader:
                    output = model(inp)
                    loss = (criterion(output, tgt) * mask).sum() / (mask.sum() + 1e-6)
                    val_loss += loss.item()
                    val_batches += 1
            avg_val_loss = val_loss / (val_batches + 1e-6)
            
            if epoch % 10 == 0:
                print(f"  Ep {epoch}: Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f}")

            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        print(f"  Fold {fold+1} Best Val Loss: {early_stopping.best_loss:.5f}")
        fold_best_losses.append(early_stopping.best_loss)

    mean_loss = np.mean(fold_best_losses)
    std_loss = np.std(fold_best_losses)
    
    print(f"\n==========================================")
    print(f"RESULTS FOR {model_name}")
    print(f"Mean CV Loss: {mean_loss:.5f} (+/- {std_loss:.5f})")
    print(f"==========================================")
    
    with open("cv_results_summary.txt", "a") as f:
        f.write(f"{model_name}\t{mean_loss:.5f}\t{std_loss:.5f}\n")

if __name__ == "__main__":
    main()