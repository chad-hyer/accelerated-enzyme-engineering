import os
import argparse
import itertools
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- Imports from your provided files ---
from meta_tensor_builder import ProteinTilingDataset

# ==========================================
# Model Architectures (Same as deep_train.py)
# ==========================================
class DynamicMetaLearnerExpansion(nn.Module):
    def __init__(self, input_channels=4, hidden_channels=32, kernel_size=5, num_layers=3, dropout=0.0):
        super().__init__()
        layers = []
        padding_h = (kernel_size - 1) // 2 
        
        layers.append(nn.Conv2d(input_channels, hidden_channels, kernel_size=(kernel_size, 1), padding=(padding_h, 0)))
        layers.append(nn.ReLU())
        if dropout > 0: layers.append(nn.Dropout2d(p=dropout))
        
        current_channels = hidden_channels
        for _ in range(num_layers - 2):
            next_channels = current_channels * 2
            layers.append(nn.Conv2d(current_channels, next_channels, kernel_size=(kernel_size, 1), padding=(padding_h, 0)))
            layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout2d(p=dropout))
            current_channels = next_channels

        layers.append(nn.Conv2d(current_channels, 1, kernel_size=(1, 1)))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class DynamicMetaLearnerHourglass(nn.Module):
    def __init__(self, input_channels=4, hidden_channels=32, kernel_size=5, num_layers=3, dropout=0.0):
        super().__init__()
        layers = []
        padding_h = (kernel_size - 1) // 2 
        MAX_CHANNELS = 64
        
        # Encoder
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

        # Decoder
        decoder_targets = encoder_channels[:-1][::-1]
        for target_ch in decoder_targets:
            layers.append(nn.Conv2d(current, target_ch, kernel_size=(kernel_size, 1), padding=(padding_h, 0)))
            layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout2d(p=dropout))
            current = target_ch

        layers.append(nn.Conv2d(current, 1, kernel_size=(1, 1)))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# Main Production Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, required=True, choices=['expansion', 'hourglass'])
    parser.add_argument('--config_idx', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True, help='Fixed number of epochs (Average from CV results)')
    parser.add_argument('--batch_size', type=int, default=32, help='Use larger batch size for production if possible')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    # --- 1. Get Config ---
    param_grid = {
        'kernel_size': [3, 5, 7, 9],
        'num_layers': [2, 3, 4],
        'hidden_channels': [16, 32],
        'dropout': [0.0, 0.2]
    }
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    config = combinations[args.config_idx]
    
    print(f"--- TRAINING PRODUCTION MODEL ---")
    print(f"Style: {args.style} | Config: {args.config_idx}")
    print(f"Params: {config}")
    print(f"Epochs: {args.epochs} (Fixed)")

    # --- 2. Initialize Model ---
    if args.style == 'expansion':
        model = DynamicMetaLearnerExpansion(**config)
    else:
        model = DynamicMetaLearnerHourglass(**config)

    # --- 3. Full Dataset ---
    CA_DIR = "/scratch/groups/mjewett/Chad_Hyer_Tiling_Experiment/src/model_architecture_optimization/CA_training_files"
    full_dataset = ProteinTilingDataset(
        pred_dir=os.path.join(CA_DIR, 'predictions'),
        error_dir=os.path.join(CA_DIR, 'errors'),
        evc_path=os.path.join(CA_DIR, 'CA_single_mutant_matrix.csv'),
        global_start=1, global_end=243 
    )
    
    # Train on EVERYTHING (shuffle=True is essential)
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='none')

    # --- 4. Training Loop ---
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        batches = 0
        
        for inp, tgt, mask in train_loader:
            optimizer.zero_grad()
            output = model(inp)
            loss = (criterion(output, tgt) * mask).sum() / (mask.sum() + 1e-6)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss/batches:.5f}")

    # --- 5. Save Final Model ---
    save_dir = "production_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.style}_config_{args.config_idx}_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"SUCCESS. Model saved to: {save_path}")

if __name__ == "__main__":
    main()