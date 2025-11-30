# --- 1. Model Definition (Fully Convolutional) ---
import torch
import torch.nn as nn
import numpy as np
from meta_tensor_builder import ProteinTilingDataset, DataLoader, build_inference_tensor, AA_ORDER, AA_TO_IDX
import pandas as pd
import re

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
        pred_dir='D:/Downloads/Files to work with/meta_model/predictions',
        error_dir='D:/Downloads/Files to work with/meta_model/errors',
        evc_path='D:/Downloads/Files to work with/meta_model/CA_single_mutant_matrix.csv',
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
    new_protein_evc = 'D:/Downloads/Files to work with/meta_model/CA_single_mutant_matrix.csv'
    new_protein_preds = 'D:/Downloads/Files to work with/meta_model/Iteration 0 Prediction.csv'
    
    tensor_input = build_inference_tensor(
        new_protein_evc, 
        new_protein_preds, 
        global_start=1, 
        global_end=243
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
    return score_grid

def recommend_top_n(model_path, evc_path, prediction_path, global_start, global_end, n=10):
    """
    Predicts and returns the top N mutations that minimize the median error.
    STRICTLY limits recommendations to mutations present in the 'prediction_path' CSV.
    """
    # 1. Load Model
    model = TilingMetaLearner()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # 2. Build Input Tensor
    tensor_input = build_inference_tensor(
        evc_path, 
        prediction_path, 
        global_start=global_start, 
        global_end=global_end
    )
    
    # 3. Create 'Allow List' and WT Lookup from the Prediction File
    pred_df = pd.read_csv(prediction_path)
    
    # Initialize a mask of Valid Candidates (Default = False/0)
    # This grid matches the dimensions of the model output
    length = global_end - global_start + 1
    valid_candidate_mask = np.zeros((length, 20), dtype=bool)
    
    wt_map = {} # To store Wild Type identity per position
    
    def parse_mut_local(s):
        m = re.match(r"([A-Z])(\d+)([A-Z])", str(s))
        if m: return m.groups()
        return None, None, None

    # Scan the CSV to build the Allow List
    for val in pred_df['AminoAcid'].unique():
        wt, pos_str, mut = parse_mut_local(val)
        
        if pos_str is not None:
            pos = int(pos_str)
            
            # Save WT info for formatting later
            wt_map[pos] = wt
            
            # Mark this specific mutation as VALID in our mask
            # Check bounds first
            if global_start <= pos <= global_end:
                pos_idx = pos - global_start
                if mut in AA_TO_IDX:
                    aa_idx = AA_TO_IDX[mut]
                    valid_candidate_mask[pos_idx, aa_idx] = True

    # 4. Predict Error Map (Full Grid)
    with torch.no_grad():
        pred_error_map = model(tensor_input) 
        
    # 5. Process Scores with Strict Masking
    score_grid = pred_error_map.squeeze().numpy() # Shape: (Length, 20)
    
    # --- STEP 5a: Apply the "Allow List" ---
    # Any mutation NOT in the CSV becomes Infinity
    score_grid[~valid_candidate_mask] = np.inf
    
    # --- STEP 5b: Mask Known Data (Already Trained) ---
    # If the CSV says it's "train", we mask it out so we don't pick it again
    # (Channel 3 in input tensor tracks this)
    known_mask = tensor_input[0, 3, :, :].numpy()
    score_grid[known_mask == 1] = np.inf 
    
    # 6. Find Top N
    flat_indices = np.argsort(score_grid.flatten())
    
    recommendations = []
    
    print(f"--- Top {n} Recommended Mutations ---")
    print(f"{'Rank':<5} {'Mutation':<10} {'Pred. Error':<12}")
    
    count = 0
    for idx in flat_indices:
        if count >= n: break
            
        # Stop if we hit Infinity (means we ran out of valid options)
        if score_grid.flatten()[idx] == np.inf:
            print("No more valid mutations available in the provided list.")
            break
            
        # Convert index back to coordinates
        pos_idx, aa_idx = np.unravel_index(idx, score_grid.shape)
        
        real_pos = pos_idx + global_start
        mut_aa = AA_ORDER[aa_idx]
        wt_aa = wt_map.get(real_pos, "?")
        
        mutation_str = f"{wt_aa}{real_pos}{mut_aa}"
        score = score_grid[pos_idx, aa_idx]
        
        recommendations.append({
            'Rank': count + 1,
            'Mutation': mutation_str,
            'Predicted_Error': score
        })
        
        print(f"{count+1:<5} {mutation_str:<10} {score:.5f}")
        count += 1
        
    return pd.DataFrame(recommendations)


def recommend_worst_n(model_path, evc_path, prediction_path, global_start, global_end, n=10):
    """
    Predicts and returns the 'Bottom N' mutations (Highest Median Error).
    Excludes known training data and invalid mutations.
    """
    # 1. Load Model
    model = TilingMetaLearner()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # 2. Build Input Tensor
    tensor_input = build_inference_tensor(
        evc_path, 
        prediction_path, 
        global_start=global_start, 
        global_end=global_end
    )
    
    # 3. Create 'Allow List' and WT Lookup
    pred_df = pd.read_csv(prediction_path)
    length = global_end - global_start + 1
    valid_candidate_mask = np.zeros((length, 20), dtype=bool)
    wt_map = {} 
    
    def parse_mut_local(s):
        m = re.match(r"([A-Z])(\d+)([A-Z])", str(s))
        if m: return m.groups()
        return None, None, None

    for val in pred_df['AminoAcid'].unique():
        wt, pos_str, mut = parse_mut_local(val)
        if pos_str is not None:
            pos = int(pos_str)
            wt_map[pos] = wt
            if global_start <= pos <= global_end:
                pos_idx = pos - global_start
                if mut in AA_TO_IDX:
                    aa_idx = AA_TO_IDX[mut]
                    valid_candidate_mask[pos_idx, aa_idx] = True

    # 4. Predict Error Map
    with torch.no_grad():
        pred_error_map = model(tensor_input) 
        
    score_grid = pred_error_map.squeeze().numpy()
    
    # 5. Apply Masks (Set Invalid/Known to Infinity)
    score_grid[~valid_candidate_mask] = np.inf
    known_mask = tensor_input[0, 3, :, :].numpy()
    score_grid[known_mask == 1] = np.inf 
    
    # --- CHANGED LOGIC FOR BOTTOM N ---
    
    # 6. Flatten and Filter
    flat_scores = score_grid.flatten()
    
    # Get indices of all mutations that are NOT Infinity (Valid & Unknown)
    valid_indices = np.where(flat_scores != np.inf)[0]
    
    if len(valid_indices) == 0:
        print("No valid candidates found to rank.")
        return pd.DataFrame()
    
    # 7. Sort Descending (Highest Error First)
    # We sort the valid_indices based on their scores
    sorted_indices_desc = valid_indices[np.argsort(-flat_scores[valid_indices])]
    
    recommendations = []
    
    print(f"--- Bottom {n} Mutations (Highest Predicted Error) ---")
    print(f"{'Rank':<5} {'Mutation':<10} {'Pred. Error':<12}")
    
    count = 0
    for idx in sorted_indices_desc:
        if count >= n: break
            
        pos_idx, aa_idx = np.unravel_index(idx, score_grid.shape)
        
        real_pos = pos_idx + global_start
        mut_aa = AA_ORDER[aa_idx]
        wt_aa = wt_map.get(real_pos, "?")
        
        mutation_str = f"{wt_aa}{real_pos}{mut_aa}"
        score = flat_scores[idx]
        
        recommendations.append({
            'Rank': count + 1,
            'Mutation': mutation_str,
            'Predicted_Error': score
        })
        
        print(f"{count+1:<5} {mutation_str:<10} {score:.5f}")
        count += 1
        
    return pd.DataFrame(recommendations)

# --- Run Demo ---

# Uncomment to run
#trained_model = train_model()
next_muts = evaluate_new_protein("meta_learner.pth")

model_path = "meta_learner.pth"
evc_path = 'D:/Downloads/Files to work with/meta_model/CA_single_mutant_matrix.csv'
prediction_path = 'D:/Downloads/Files to work with/meta_model/Reversed Iteration 6 Prediction.csv'
global_start = 1
global_end = 243
meta_tiling_path = 'D:/Downloads/Files to work with/meta_model/reversed_meta_tiling_path.tsv'
try:
    metaDF = pd.read_csv(meta_tiling_path, delimiter='\t')
    last_it = metaDF['Iteration'].max()
except FileNotFoundError:
    last_it = 0
    with open(meta_tiling_path, 'w') as f:
        f.write("Iteration\tBest_Mutation\tMedian_Error\n")

recommendations = recommend_worst_n(model_path, evc_path, prediction_path, global_start, global_end, n=100)
recommendations.rename(columns={'Rank':'Iteration','Mutation':'Best_Mutation','Predicted_Error':'Median_Error'},inplace=True)
recommendations['Iteration'] = recommendations['Iteration'] + last_it

try:
    with open(meta_tiling_path, 'a') as f: # 'a' = append mode
        for index, row in recommendations.iterrows():
            iteration_count = row['Iteration']
            best_mutation_name = row['Best_Mutation']
            best_median_error = row['Median_Error']
            f.write(f"{iteration_count}\t{best_mutation_name}\t{best_median_error}\n")
except Exception as e:
    print(f"Warning: Could not write to log file {meta_tiling_path}. {e}")


'''
OLD META

model_path = "meta_learner.pth"
evc_path = 'D:/Downloads/Files to work with/meta_model/CA_single_mutant_matrix.csv'
prediction_path = 'D:/Downloads/Files to work with/meta_model/Iteration 6 Prediction.csv'
global_start = 1
global_end = 243
meta_tiling_path = 'D:/Downloads/Files to work with/meta_model/meta_tiling_path.tsv'
try:
    metaDF = pd.read_csv(meta_tiling_path, delimiter='\t')
    last_it = metaDF['Iteration'].max()
except FileNotFoundError:
    last_it = 0
    with open(meta_tiling_path, 'w') as f:
        f.write("Iteration\tBest_Mutation\tMedian_Error\n")

recommendations = recommend_top_n(model_path, evc_path, prediction_path, global_start, global_end, n=100)
recommendations.rename(columns={'Rank':'Iteration','Mutation':'Best_Mutation','Predicted_Error':'Median_Error'},inplace=True)
recommendations['Iteration'] = recommendations['Iteration'] + last_it

try:
    with open(meta_tiling_path, 'a') as f: # 'a' = append mode
        for index, row in recommendations.iterrows():
            iteration_count = row['Iteration']
            best_mutation_name = row['Best_Mutation']
            best_median_error = row['Median_Error']
            f.write(f"{iteration_count}\t{best_mutation_name}\t{best_median_error}\n")
except Exception as e:
    print(f"Warning: Could not write to log file {meta_tiling_path}. {e}")
'''