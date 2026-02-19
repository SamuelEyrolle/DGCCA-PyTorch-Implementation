import torch
import numpy as np
import os
import pandas as pd
import re

def set_seed(seed):
    """
    Sets the seed for all relevant libraries to ensure experiment reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_views(data_dir):
    """
    Loads all files named view[n].csv from a directory and ensures they 
    are aligned for DGCCA training.
    """
    # 1. Gather and sort files numerically (view1, view2, ..., view10)
    view_files = [f for f in os.listdir(data_dir) if f.startswith('view') and f.endswith('.csv')]
    
    if not view_files:
        raise FileNotFoundError(f"No files starting with 'view' and ending in '.csv' found in {data_dir}")

    view_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
    
    views = []
    expected_samples = None
    
    # 2. Load data and validate row consistency
    for f in view_files:
        path = os.path.join(data_dir, f)
        df = pd.read_csv(path, header=None)
        
        current_samples, current_features = df.shape
        
        if expected_samples is None:
            expected_samples = current_samples
        elif current_samples != expected_samples:
            raise ValueError(
                f"Row mismatch in {f}: Found {current_samples} samples, "
                f"but previous views had {expected_samples}. All views must be aligned."
            )
            
        views.append(torch.tensor(df.values, dtype=torch.float32))
        print(f"Loaded {f}: {current_samples} samples x {current_features} features")

    return views