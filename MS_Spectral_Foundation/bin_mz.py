
import numpy as np
import torch

"""
Module 3: Discretize / tokenize the spectrum (m/z binning)
"""

def bin_mz(npz_path, bin_size=0.5, mz_min=50.0, mz_max=2500.0, is_normalized=False):
    """
    Fixed-width binning for spectra_filtered.npz
        Uses floor for binning
        Clamps m/z to [mz_min, mz_max] or [0, 1] if normalized
        Bins limited to [0, n_bins-1]
        Invalid (padded) positions (m/z <= 0) set to 0
    Args:
        npz_path: path to spectra_filtered.npz
        bin_size: bin width
        mz_min: minimum m/z
        mz_max: maximum m/z
        is_normalized: True if m/z already normalized to [0,1]
    Returns:
        bins: torch.LongTensor, shape (N, L)
    """
    d = np.load(npz_path, allow_pickle=True)
    peaks_mz = d['peaks_mz']
    S = len(peaks_mz)
    max_len = max(len(mz) for mz in peaks_mz)
    # Pad to max_len
    mzs = np.zeros((S, max_len), dtype=float)
    for i, arr in enumerate(peaks_mz):
        mzs[i, :len(arr)] = arr
    mzs = torch.tensor(mzs, dtype=torch.float)
    # number of bins and valid mask
    n_bins = int(np.ceil((mz_max - mz_min) / bin_size))
    valid_mask = mzs > 0
    if not valid_mask.any():
        return torch.zeros_like(mzs, dtype=torch.long)
    # binning calculations
    if is_normalized:
        clamped = torch.clamp(mzs, min=0.0, max=1.0)
        bins = torch.floor(clamped * n_bins).long()
    else:
        clamped = torch.clamp(mzs, min=mz_min, max=mz_max)
        bins = torch.floor((clamped - mz_min) / bin_size).long()
    bins = bins.clamp(min=0, max=n_bins - 1)
    bins[~valid_mask] = 0
    return bins.detach()


def bin_mz_tensor(mzs: torch.Tensor, *, bin_size: float, mz_min: float, mz_max: float, n_bins: int, is_normalized: bool = False) -> torch.Tensor:
    """
    Bin a tensor of m/z values directly (no file I/O).
    Args:
        mzs: torch.Tensor of m/z values, shape (N, L)
        bin_size: bin width
        mz_min: minimum m/z
        mz_max: maximum m/z
        n_bins: total number of bins
        is_normalized: True if m/z already normalized to [0, 1]
    Returns:
        bins: torch.LongTensor, same shape as mzs
    """
    valid_mask = mzs > 0
    if not valid_mask.any():
        return torch.zeros_like(mzs, dtype=torch.long)
    if is_normalized:
        clamped = torch.clamp(mzs, min=0.0, max=1.0)
        bins = torch.floor(clamped * n_bins).long()
    else:
        clamped = torch.clamp(mzs, min=mz_min, max=mz_max)
        bins = torch.floor((clamped - mz_min) / bin_size).long()
    bins = bins.clamp(min=0, max=n_bins - 1)
    bins[~valid_mask] = 0
    return bins.detach()


if __name__ == "__main__":
    import sys
    npz_path = "output/spectra_filtered.npz"
    bin_size = 0.5
    mz_min = 50.0
    mz_max = 2500.0
    is_normalized = False
    # Optional: override with command line arguments
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
    if len(sys.argv) > 2:
        bin_size = float(sys.argv[2])
    if len(sys.argv) > 3:
        mz_min = float(sys.argv[3])
    if len(sys.argv) > 4:
        mz_max = float(sys.argv[4])
    if len(sys.argv) > 5:
        is_normalized = bool(int(sys.argv[5]))
    bins = bin_mz(npz_path, bin_size, mz_min, mz_max, is_normalized)
    print(f"bins shape: {bins.shape}")
    print(bins[:5])  # Output the binning results of the first 5 spectra

    # Save binning results
    out_bins_path = "output/bins.npz"
    import os
    os.makedirs(os.path.dirname(out_bins_path) or ".", exist_ok=True)
    np.savez_compressed(out_bins_path, bins=bins.numpy())
    print(f"Saved bins to {out_bins_path}")

    # Output the first 100 rows to txt
    out_txt = "output/bins_first100.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        for i, row in enumerate(bins[:100]):
            f.write(f"# Bin row {i+1}\n")
            f.write(str(row.tolist()) + "\n\n")
    print(f"Wrote first 100 bins to {out_txt}")