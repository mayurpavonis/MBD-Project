# convert_pt_to_csv.py
import torch
import numpy as np
import pandas as pd
import sys

def convert_pt_to_csv(pt_file, csv_file):
    # Load PyTorch data
    data = torch.load(pt_file)
    
    # Convert to numpy if it's a tensor
    if torch.is_tensor(data):
        data = data.numpy()
    
    # Create DataFrame and save as CSV
    if data.ndim == 1:
        df = pd.DataFrame(data, columns=['value'])
    elif data.ndim == 2:
        df = pd.DataFrame(data, columns=[f'channel_{i}' for i in range(data.shape[1])])
    else:
        # For higher dimensional data, we might need to flatten
        df = pd.DataFrame(data.reshape(data.shape[0], -1))
    
    df.to_csv(csv_file, index=False)
    print(f"Converted {pt_file} to {csv_file}")

if __name__ == "__main__":
    convert_pt_to_csv(sys.argv[1], sys.argv[2])