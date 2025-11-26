# --- 1. Model Definition (Fully Convolutional) ---
import torch
import torch.nn as nn
import numpy as np
from meta_tensor_builder import ProteinTilingDataset, DataLoader, build_inference_tensor, AA_ORDER

class TilingMetaLearner(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 4 Channels -> Output: 1 Channel (Predicted Median Error)
        # Using Padding to keep Length dimension same
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=(1, 1)) # Final projection
        )
        
    def forward(self, x):
        return self.net(x)

# --- 2. Training Phase (Using your Tiling Experiment Data) ---
def train_model():
    # Setup Dataset
    # Adjust global_start/end to match your SSM experiment range (1-243)
    dataset = ProteinTilingDataset(
        pred_dir='./tiling_data/predictions',
        error_dir='./tiling_data/errors',
        evc_path='./tiling_data/CA_single_mutant_matrix.csv',
        global_start=1,
        global_end=243
    )
    
    # Use Batch Size 1 because of variable protein lengths (if mixing proteins)
    # If training on just one protein's history, you can increase batch size
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = TilingMetaLearner()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='none') # Use 'none' to apply mask later
    
    print("Starting Training...")
    for epoch in range(10): # Example epochs
        total_loss = 0
        for batch_idx, (inp, tgt, mask) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward Pass
            output = model(inp) # (B, 1, L, 20)
            
            # Calculate Loss ONLY on valid targets (Mask = 1)
            loss_raw = criterion(output, tgt)
            loss = (loss_raw * mask).sum() / (mask.sum() + 1e-6)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")
        
    # Save Model
    torch.save(model.state_dict(), "meta_learner.pth")
    print("Model Saved.")
    return model

# --- 3. Inference Phase (New Protein) ---
def evaluate_new_protein(model_path):
    # Load Model
    model = TilingMetaLearner()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Inputs for NEW Protein (e.g. Length 500)
    # Notice: Different L (500) and Start (1) than training
    new_protein_evc = './new_protein/evc_couplings.csv'
    new_protein_preds = './new_protein/initial_ridge_preds.csv'
    
    tensor_input = build_inference_tensor(
        new_protein_evc, 
        new_protein_preds, 
        global_start=1, 
        global_end=500
    )
    
    # Predict
    with torch.no_grad():
        pred_error_map = model(tensor_input) # (1, 1, 500, 20)
        
    # Selection Logic
    score_grid = pred_error_map.squeeze().numpy() # (500, 20)
    
    # Mask out knowns (Channel 3 of input)
    known_mask = tensor_input[0, 3, :, :].numpy()
    score_grid[known_mask == 1] = np.inf # Set knowns to infinity
    
    # Find Best Mutation (Lowest Predicted Median Error)
    best_idx = np.unravel_index(np.argmin(score_grid), score_grid.shape)
    pos_idx, aa_idx = best_idx
    
    # Map back to real world coordinates
    real_pos = pos_idx + 1 # + global_start
    real_aa = AA_ORDER[aa_idx]
    
    print(f"Recommended Next Mutation: {real_pos}{real_aa}")
    print(f"Predicted Median Error: {score_grid.min():.4f}")

# --- Run Demo ---
if __name__ == "__main__":
    # Uncomment to run
    # trained_model = train_model()
    # evaluate_new_protein("meta_learner.pth")
    pass