import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# --- Constants ---
AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

def parse_mutation(mut_str):
    """Parses 'A114C' into ('A', 114, 'C')."""
    match = re.match(r"([A-Z])(\d+)([A-Z])", str(mut_str))
    if match:
        wt, pos, mut = match.groups()
        return wt, int(pos), mut
    return None, None, None

def load_evc_map(evc_path, global_start, global_end):
    """
    Loads EVC data and aligns it to the global canvas (start to end).
    """
    df = pd.read_csv(evc_path)
    length = global_end - global_start + 1
    evc_map = np.zeros((length, 20)) # (L, 20)
    
    for _, row in df.iterrows():
        p = int(row['pos'])
        if p < global_start or p > global_end: continue
        
        aa = row['subs']
        if aa in AA_TO_IDX:
            evc_map[p - global_start, AA_TO_IDX[aa]] = row['prediction_epistatic']
            
    return evc_map

class ProteinTilingDataset(Dataset):
    def __init__(self, pred_dir, error_dir, evc_path, global_start, global_end):
        """
        Args:
            pred_dir: Directory containing 'prediction_it_X.csv'
            error_dir: Directory containing 'median_error_it_X.csv'
            evc_path: Path to 'CA_single_mutant_matrix.csv'
            global_start, global_end: The residue range of the PROTEIN (e.g. 1 to 243)
        """
        self.pred_dir = pred_dir
        self.error_dir = error_dir
        self.global_start = global_start
        self.length = global_end - global_start + 1
        
        # 1. Load Static EVC Map once
        self.evc_map = load_evc_map(evc_path, global_start, global_end)
        
        # 2. Index Files
        # Assumes files are named 'prediction_it_X.csv' and 'median_error_it_X.csv'
        files = os.listdir(pred_dir)
        self.iterations = sorted([int(re.search(r'it_(\d+)', f).group(1)) 
                                  for f in files if 'prediction_it_' in f])
        
    def __len__(self):
        return len(self.iterations)
    
    def __getitem__(self, idx):
        it_num = self.iterations[idx]
        
        # Load CSVs
        pred_path = os.path.join(self.pred_dir, f'prediction_it_{it_num}.csv')
        error_path = os.path.join(self.error_dir, f'median_error_it_{it_num}.csv')
        
        pred_df = pd.read_csv(pred_path)
        error_df = pd.read_csv(error_path)
        
        # --- Build Input Tensor (4, L, 20) ---
        # Channels: 0:EVC, 1:Preds, 2:Activity, 3:Mask
        input_tensor = np.zeros((self.length, 20, 4))
        input_tensor[:, :, 0] = self.evc_map # Fill Static EVC
        
        # Fill Dynamic Channels (Preds, Activity, Mask)
        pred_df['parsed'] = pred_df['AminoAcid'].map(parse_mutation)
        for _, row in pred_df.iterrows():
            wt, p, mut = row['parsed']
            if p is None or p < self.global_start or p >= (self.global_start + self.length): continue
            
            if mut in AA_TO_IDX:
                aa_idx = AA_TO_IDX[mut]
                pos_idx = p - self.global_start
                
                input_tensor[pos_idx, aa_idx, 1] = row['Prediction']
                
                if row['Class'] == 'train':
                    input_tensor[pos_idx, aa_idx, 2] = row['Activity']
                    input_tensor[pos_idx, aa_idx, 3] = 1.0 # Known/Train
                else:
                    input_tensor[pos_idx, aa_idx, 2] = 0.0
                    input_tensor[pos_idx, aa_idx, 3] = 0.0 # Unknown/Test

        # --- Build Target Tensor (Median Error) ---
        target_tensor = np.zeros((self.length, 20))
        target_mask = np.zeros((self.length, 20))
        
        error_df['parsed'] = error_df['Mutation'].map(parse_mutation)
        for _, row in error_df.iterrows():
            wt, p, mut = row['parsed']
            if p is None or p < self.global_start or p >= (self.global_start + self.length): continue
            
            if mut in AA_TO_IDX:
                aa_idx = AA_TO_IDX[mut]
                pos_idx = p - self.global_start
                
                target_tensor[pos_idx, aa_idx] = row['Median_Error']
                target_mask[pos_idx, aa_idx] = 1.0
                
        # Convert to PyTorch (Channels First)
        # Input: (4, L, 20) | Target: (1, L, 20)
        inp = torch.FloatTensor(input_tensor).permute(2, 0, 1)
        tgt = torch.FloatTensor(target_tensor).unsqueeze(0)
        msk = torch.FloatTensor(target_mask).unsqueeze(0)
        
        return inp, tgt, msk
    
def build_inference_tensor(evc_path, prediction_path, global_start, global_end):
    """
    Prepares data for a NEW protein to be fed into the model.
    """
    # 1. Load Static EVC
    evc_map = load_evc_map(evc_path, global_start, global_end)
    length = global_end - global_start + 1
    
    # 2. Initialize Input Tensor
    input_tensor = np.zeros((length, 20, 4))
    input_tensor[:, :, 0] = evc_map
    
    # 3. Load Ridge Predictions
    pred_df = pd.read_csv(prediction_path)
    pred_df['parsed'] = pred_df['AminoAcid'].map(parse_mutation)
    
    for _, row in pred_df.iterrows():
        wt, p, mut = row['parsed']
        if p is None or p < global_start or p >= (global_start + length): continue
        
        if mut in AA_TO_IDX:
            aa_idx = AA_TO_IDX[mut]
            pos_idx = p - global_start
            
            # Fill Prediction
            input_tensor[pos_idx, aa_idx, 1] = row['Prediction']
            
            # Fill Activity/Mask if known (e.g. from initial training set)
            if row['Class'] == 'train':
                input_tensor[pos_idx, aa_idx, 2] = row['Activity']
                input_tensor[pos_idx, aa_idx, 3] = 1.0
            else:
                input_tensor[pos_idx, aa_idx, 2] = 0.0
                input_tensor[pos_idx, aa_idx, 3] = 0.0
                
    # Convert to PyTorch Batch (1, 4, L, 20)
    inp_torch = torch.FloatTensor(input_tensor).permute(2, 0, 1).unsqueeze(0)
    
    return inp_torch