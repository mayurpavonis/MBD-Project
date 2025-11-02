# File: pt2mat.py

import torch
import numpy as np
import scipy.io as sio
import os

def load_pt(pt_path):
    """Load the .pt file (should be tracked by LFS)."""
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f".pt file not found: {pt_path}")
    t = torch.load(pt_path, map_location='cpu')
    if not isinstance(t, torch.Tensor):
        raise TypeError("Expected a torch.Tensor in the .pt file.")
    arr = t.numpy()
    return arr

def extract_gx_first_K(arr, K=25):
    """Take first K windows; extract gx (axis index 0)."""
    num_windows, window_length, num_axes = arr.shape
    if num_axes < 1:
        raise ValueError("Expected last dimension to have 3 axes; found fewer.")
    if K > num_windows:
        K = num_windows  # gracefully degrade
    clipped = arr[:K, :, 0]  # axis 0 = gx
    return clipped, window_length

def infer_stride(arr2d, tol=1e-6):
    """
    Infer stride between windows by checking overlap.
    Returns inferred stride; equals window_length if no overlap detected.
    """
    K, L = arr2d.shape
    if K < 2:
        return L
    # Test possible strides from 1 to L-1
    for s in range(1, L):
        all_match = True
        for i in range(K - 1):
            tail = arr2d[i, s:]
            head = arr2d[i+1, :L - s]
            if not np.allclose(tail, head, atol=tol, rtol=0):
                all_match = False
                break
        if all_match:
            return s
    return L

def flatten_without_overlap(arr2d, stride):
    """Flatten windows sequentially, removing overlaps if stride < window_length."""
    K, L = arr2d.shape
    if stride >= L:
        # No overlap case
        flat = arr2d.reshape(K * L)
    else:
        overlap = L - stride
        pieces = []
        pieces.append(arr2d[0])
        for i in range(1, K):
            pieces.append(arr2d[i, overlap:])
        flat = np.concatenate(pieces, axis=0)
    return flat

def build_time(num_samples):
    """Build time vector with sample-indices (0,1,...)."""
    return np.arange(num_samples, dtype=np.float32)

def save_mat(mat_path, varname, signal, time_vec):
    """Save .mat with float32 signal & time."""
    signal_f32 = signal.astype(np.float32)
    time_f32 = time_vec.astype(np.float32)
    mdict = {
        varname: signal_f32,
        'time': time_f32
    }
    # Use compression to lower size
    sio.savemat(mat_path, mdict, do_compression=True)
    print(f"Saved {mat_path}: variable '{varname}', sample count {signal_f32.shape[0]}")

def process(pt_noisy_path, pt_GT_path, out_noisy_mat, out_GT_mat, K=25):
    # Process noisy
    arr_noisy = load_pt(pt_noisy_path)
    clipped_noisy, Ln = extract_gx_first_K(arr_noisy, K=K)
    stride_n = infer_stride(clipped_noisy)
    flat_noisy = flatten_without_overlap(clipped_noisy, stride_n)
    time_n = build_time(flat_noisy.shape[0])
    save_mat(out_noisy_mat, 'gx_noisy', flat_noisy, time_n)

    # Process GT
    arr_GT = load_pt(pt_GT_path)
    clipped_GT, LG = extract_gx_first_K(arr_GT, K=K)
    # optional check: LG should equal Ln; but they might be same
    stride_GT = infer_stride(clipped_GT)
    flat_GT = flatten_without_overlap(clipped_GT, stride_GT)
    time_GT = build_time(flat_GT.shape[0])
    save_mat(out_GT_mat, 'gx_GT', flat_GT, time_GT)

if __name__ == "__main__":
    process(
        pt_noisy_path = "X_train_noisy.pt",
        pt_GT_path    = "X_train_GT.pt",
        out_noisy_mat = "X_train_noisy_clip25.mat",
        out_GT_mat    = "X_train_GT_clip25.mat",
        K = 25
    )
