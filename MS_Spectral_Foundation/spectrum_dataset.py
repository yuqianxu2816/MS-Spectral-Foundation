"""
Dataset class connecting Module 1 (mgf_parse), Module 2 (peak_filter), and Module 3 (bin_mz)
Data Flow: MGF -> parse -> filter -> binning -> training
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Optional, Callable
from .mgf_parse import parse_mgf, Spectrum, save_spectra_npz
from .peak_filter import (
    set_mz_range,
    remove_precursor_peak,
    scale_intensity,
    filter_intensity,
    discard_low_quality,
    _scale_to_unit_norm,
    apply_preprocessing_pipeline
)
from .bin_mz import bin_mz


class SpectrumDataset(Dataset):
    """
    PyTorch Dataset for mass spectrometry data.
    
    Complete Data Flow (follows DDS.md module order):
    1. MGF file -> parse_mgf (Module 1) 
    2. peak_filter (Module 2) 
    3. bin_mz_tensor (Module 3, pre-computed) 
    4. model training (Module 4-6)
    
    Key Design:
    1. Binning is performed ONCE during dataset initialization (Module 3)
    2. Training directly uses pre-computed bin labels (no dynamic binning)
    3. Improves efficiency and follows the modular design in DDS.md
    """
    
    def __init__(
        self,
        mgf_path: str,
        max_peaks: int = 150,
        mz_min: float = 50.0,
        mz_max: float = 2500.0,
        bin_size: float = 0.5,
        # Peak filtering parameters (Module 2)
        remove_precursor_tol: float = 2.0,
        min_intensity: float = 0.01,
        min_peaks: int = 5,
        preprocessing_fn: Optional[Callable] = None,
    ):

        self.max_peaks = max_peaks
        self.mz_min = mz_min
        self.mz_max = mz_max
        self.bin_size = bin_size
        self.n_bins = int(np.ceil((mz_max - mz_min) / bin_size))
        
        print(f"[Dataset Init] Bin configuration: n_bins={self.n_bins}, bin_size={bin_size}")
        
        # Module 1: Parse MGF
        print(f"[Module 1] Parsing MGF file: {mgf_path}")
        all_spectra = parse_mgf(mgf_path)
        print(f"[Module 1] Parsed {len(all_spectra)} spectra")
        
        # Module 2: Build preprocessing pipeline
        if preprocessing_fn is None:
            pipeline = [
                set_mz_range(mz_min, mz_max),
                remove_precursor_peak(tol=remove_precursor_tol, unit="Da"),
                scale_intensity(scaling="root", degree=2, max_intensity=1.0),
                filter_intensity(min_intensity=min_intensity, max_peaks=max_peaks),
                discard_low_quality(min_peaks=min_peaks),
                _scale_to_unit_norm,
            ]

            def preprocessing_fn(peaks, precursor_mz=None):
                return apply_preprocessing_pipeline(
                    peaks=peaks,
                    pipeline=pipeline,
                    precursor_mz=precursor_mz,
                )
        # Apply Module 2 preprocessing 
        print(f"[Module 2] Applying preprocessing pipeline...")
        module2_processed = []
        valid_indices = []
        dropped = 0
        
        for orig_idx, sp in enumerate(all_spectra):
            precursor_mz = sp["meta"].get("PEPMASS", None)
            peaks = np.array(sp["peaks"], dtype=float)
            
            # Module 2: Apply preprocessing
            out = preprocessing_fn(peaks, precursor_mz=precursor_mz)
            if out is None:
                dropped += 1
                continue
            
            valid_indices.append(orig_idx)
            sp2 = {
                "meta": sp["meta"],
                "peaks": list(map(tuple, out.tolist())),  # list[(mz,inten)]
            }
            module2_processed.append(sp2)
        
        # Store valid indices so callers can align external metadata
        self.valid_indices = valid_indices
        
        print(f"[Module 2] After preprocessing: {len(module2_processed)} valid spectra (dropped {dropped})")
        
        # Module 3: Binning using bin_mz from bin_mz.py =====
        print(f"[Module 3] Applying binning using bin_mz()...")
        
        # Save Module 2 output to temporary npz file using save_spectra_npz
        import tempfile
        import os
        
        temp_dir = tempfile.mkdtemp()
        temp_npz = os.path.join(temp_dir, "temp_spectra.npz")
        
        # Save in standard format using save_spectra_npz from mgf_parse
        save_spectra_npz(module2_processed, temp_npz)
        print(f"[Module 3] Saved preprocessed spectra to temporary npz: {temp_npz}")
        
        # Call bin_mz from Module 3 (it will handle padding internally)
        bins = bin_mz(temp_npz, bin_size=bin_size, mz_min=mz_min, mz_max=mz_max, is_normalized=False)
        
        # Clean up temporary file
        os.remove(temp_npz)
        os.rmdir(temp_dir)
        
        print(f"[Module 3] Binning complete: {len(bins)} spectra, bins shape: {bins.shape}")
        
        # Post-processing: Format data with binning results
        print(f"[Post-processing] Formatting data for training...")
        self.processed_data: List[Dict] = []
        
        for i, sp2 in enumerate(module2_processed):
            peaks_array = np.array(sp2["peaks"], dtype=float)
            
            # Extract metadata
            precursor_mz = sp2["meta"].get("PEPMASS", 0.0) or 0.0
            precursor_charge = sp2["meta"].get("CHARGE", 2) or 2
            
            # Separate m/z and intensity
            mz_values = peaks_array[:, 0]
            intensity_values = peaks_array[:, 1]
            
            # Get bin_labels from Module 3 output
            bin_labels = bins[i]
            
            # Pad or truncate to max_peaks for consistency
            n_peaks = len(mz_values)
            max_len = bins.shape[1]  # Length determined by bin_mz
            
            if n_peaks < max_len:
                mz_padded = np.zeros(max_len, dtype=np.float32)
                intensity_padded = np.zeros(max_len, dtype=np.float32)
                mz_padded[:n_peaks] = mz_values
                intensity_padded[:n_peaks] = intensity_values
            else:
                mz_padded = mz_values[:max_len]
                intensity_padded = intensity_values[:max_len]
            
            # Normalize m/z to [0,1]
            mz_normalized = (mz_padded - mz_min) / (mz_max - mz_min)
            mz_normalized[mz_padded == 0] = 0  # Keep padding as 0
            
            self.processed_data.append({
                "precursor_mz": precursor_mz,
                "precursor_charge": precursor_charge,
                "mz_array": mz_padded,
                "intensity_array": intensity_padded,
                "mz_normalized": mz_normalized,
                "bin_labels": bin_labels.numpy(),  # Bins from Module 3
            })
        
        print(f"[Post-processing] Formatted {len(self.processed_data)} spectra for training")
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary with pre-computed bin labels from Module 3.
        """
        data = self.processed_data[idx]
        
        return {
            "precursor_mz": torch.tensor([data["precursor_mz"]], dtype=torch.float32),
            "precursor_charge": torch.tensor([data["precursor_charge"]], dtype=torch.float32),
            "mz_array": torch.tensor(data["mz_array"], dtype=torch.float32),
            "intensity_array": torch.tensor(data["intensity_array"], dtype=torch.float32),
            "mz_normalized": torch.tensor(data["mz_normalized"], dtype=torch.float32),
            "bin_labels": torch.tensor(data["bin_labels"], dtype=torch.long),  # ✅ Pre-computed
        }


class SpectrumDataModule:
    """
    Convenience wrapper for train/val dataloaders.
    """
    def __init__(
        self,
        train_mgf_path: str,
        val_mgf_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        **dataset_kwargs
    ):
        self.train_dataset = SpectrumDataset(train_mgf_path, **dataset_kwargs)
        self.val_dataset = SpectrumDataset(val_mgf_path, **dataset_kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
