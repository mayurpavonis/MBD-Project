# File: clip_pt_to_mat.py

import torch
import numpy as np
import scipy.io as sio

def load_pt(pt_path):
    """Load the .pt, expects a tensor of shape [num_windows, window_length, 3]."""
    t = torch.load(pt_path, map_location='cpu')
    if not isinstance(t, torch.Tensor):
        raise TypeError("Expected torch.Tensor in .pt file")
    arr = t.numpy()
    return arr

def extract_gx_first_K(arr, K=25):
    """Take first K windows; extract gx (axis 0). Returns array shape (K, window_length)."""
    num_windows, window_length, num_axes = arr.shape
    if K > num_windows:
        raise ValueError(f"K={K} more than available windows {num_windows}")
    # axis_index = 0 for gx
    axis_index = 0
    clipped = arr[:K, :, axis_index]   # shape (K, window_length)
    return clipped, window_length

def infer_stride_and_overlap(clipped_windows, tol=1e-6):
    """
    From clipped windows array of shape (K, window_length),
    attempt to infer stride (hop between windows) by comparing tails and heads.
    Returns inferred stride (integer between 1 and window_length), or
    returns window_length if no overlap detected.
    """
    K, L = clipped_windows.shape
    # If K < 2, no overlap info
    if K < 2:
        return L
    
    # For windows i and i+1, check for maximum overlap by testing possible strides
    # For s in 1..L, test whether window[i][s:] ≈ window[i+1][:L-s]
    # We'll find the largest “match” (smallest s) that gives good similarity
    for s in range(1, L):
        matches = True
        for i in range(K - 1):
            tail = clipped_windows[i, s:]
            head = clipped_windows[i + 1, :L - s]
            if not np.allclose(tail, head, atol=tol, rtol=0):
                matches = False
                break
        if matches:
            # stride = s  => overlap = L - s
            return s
    # If no s < L found, assume no overlap => stride = L
    return L

def remove_overlaps_and_flatten(clipped_windows, stride):
    """
    Given clipped windows (shape K x L) and stride,
    flatten into a 1D signal, removing overlapping tail-head duplicates.
    """
    K, L = clipped_windows.shape
    if stride >= L:
        # no overlap
        flat = clipped_windows.reshape(K * L)
        return flat
    else:
        # overlap amount = L - stride
        overlap = L - stride
        # build list
        parts = []
        parts.append(clipped_windows[0])  # full first window
        for i in range(1, K):
            # skip the first 'overlap' samples of window i, because they duplicate end of window i-1
            parts.append(clipped_windows[i, overlap:])
        # concatenate
        flat = np.concatenate(parts, axis=0)
        return flat

def build_time_vector(n_samples):
    """Build a time vector of sample indices [0,1,2,...,n_samples-1] as float32."""
    return np.arange(n_samples, dtype='float32')

def save_mat(mat_path, varname, signal, time_vec):
    """Save .mat with variables (signal, time) in float32."""
    signal_f32 = signal.astype('float32')
    time_f32 = time_vec.astype('float32')
    mdict = {
        varname: signal_f32,
        'time': time_f32
    }
    sio.savemat(mat_path, mdict, do_compression=True)
    print(f"Saved {mat_path}: var '{varname}' length {signal_f32.shape[0]}")

def process_one(pt_path, out_mat_path, K=25):
    arr = load_pt(pt_path)
    clipped, L = extract_gx_first_K(arr, K=K)
    print(f"Clipped shape: {clipped.shape}")  # (K, L)
    stride = infer_stride_and_overlap(clipped_windows=clipped)
    print(f"Inferred stride = {stride} (window_length = {L})")
    flat_signal = remove_overlaps_and_flatten(clipped_windows=clipped, stride=stride)
    print(f"Flattened signal length = {flat_signal.shape[0]}")
    time_vec = build_time_vector(flat_signal.shape[0])
    # choose varname
    if 'noisy' in pt_path.lower():
        varname = 'gx_noisy'
    else:
        varname = 'gx_GT'
    save_mat(out_mat_path, varname, flat_signal, time_vec)

if __name__ == "__main__":
    # Example usage
    process_one('X_train_noisy.pt', 'X_train_noisy_clip25.mat', K=25)
    process_one('X_train_GT.pt',    'X_train_GT_clip25.mat',    K=25)
