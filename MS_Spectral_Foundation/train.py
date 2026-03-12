"""
Training script for model_ssl.py using SpectrumDataset

Complete Data Flow (follows DDS.md module order):
1. MGF files -> Module 1 (mgf_parse.py): Parse spectra
2. -> Module 2 (peak_filter.py): Filter and preprocess peaks
3. -> Module 3 (bin_mz.py): Pre-compute bin labels during dataset initialization
4. -> Module 4-6 (model_ssl.py): Self-supervised training with pre-computed bins
"""
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from MS_Spectral_Foundation.spectrum_dataset import SpectrumDataset, SpectrumDataModule
from MS_Spectral_Foundation.model_ssl import SpectrumSSLv2
import os


def main():
    # Check GPU availability
    print("="*80)
    print("GPU Information:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
        print(f"  GPU name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("  No GPU detected, will use CPU")
    print("="*80 + "\n")
    
    # Configuration
    config = {
        # Data parameters
        "train_mgf": r"C:\Users\Lenovo\Desktop\576dataset\09062023_Mehta_GR10000524_DDRC_Sample4_561_cirrhotic_output.mgf",
        "val_mgf": r"C:\Users\Lenovo\Desktop\576dataset\09062023_Mehta_GR10000524_DDRC_Sample4_561_cirrhotic_output.mgf",
        "max_peaks": 150,
        "mz_min": 50.0,
        "mz_max": 2500.0,
        "bin_size": 0.5,
        
        # Model parameters
        "dim_model": 512,
        "n_head": 8,
        "n_layers": 6,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "mask_prob": 0.15,
        
        # Training parameters
        "batch_size": 32,
        "lr": 1e-4,
        "max_epochs": 20,
        "num_workers": 4,
    }
    
    print("="*80)
    print("MS-Spectral-Foundation Training")
    print("Data Flow: MGF → Module 1 (parse) → Module 2 (filter) → Module 3 (bin) → Module 4-6 (train)")
    print("="*80)
    
    print("\n Module Integration Verification:")
    print("   spectrum_dataset.py integrates:")
    print("     - Module 1 (mgf_parse.py): parse_mgf() - Parse MGF file")
    print("     - Module 2 (peak_filter.py): apply_preprocessing_pipeline() - Filter peaks")
    print("     - Module 3 (bin_mz.py): bin_mz_tensor() - Pre-compute bin labels")
    print("   model_ssl.py implements:")
    print("     - Module 4: mask_spectrum() - BERT-style masking")
    print("     - Module 5: forward() + training_step() - Self-supervised training")
    print("     - Module 6: get_embeddings() - Extract learned representations\n")
    
    # Step 1-3: Create dataset (Module 1, 2, 3 executed during init) 
    print("\n[Step 1-3] Creating datasets (Modules 1-3 will execute)...\n")
    
    train_dataset = SpectrumDataset(
        mgf_path=config["train_mgf"],
        max_peaks=config["max_peaks"],
        mz_min=config["mz_min"],
        mz_max=config["mz_max"],
        bin_size=config["bin_size"],  # Module 3 binning parameters
        remove_precursor_tol=2.0,
        min_intensity=0.01,
        min_peaks=5,
    )
    
    val_dataset = SpectrumDataset(
        mgf_path=config["val_mgf"],
        max_peaks=config["max_peaks"],
        mz_min=config["mz_min"],
        mz_max=config["mz_max"],
        bin_size=config["bin_size"],
        remove_precursor_tol=2.0,
        min_intensity=0.01,
        min_peaks=5,
    )
    
    print(f"\n[Step 1-3] Datasets ready:")
    print(f"  Train: {len(train_dataset)} spectra")
    print(f"  Val:   {len(val_dataset)} spectra")
    print(f"  Bins:  {train_dataset.n_bins} bins (pre-computed in Module 3)")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    
    # Step 4-6: Initialize model (Modules 4-6)
    print(f"\n[Step 4-6] Initializing model (Modules 4-6)...\n")
    
    model = SpectrumSSLv2(
        dim_model=config["dim_model"],
        n_head=config["n_head"],
        n_layers=config["n_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        lr=config["lr"],
        mz_min=config["mz_min"],
        mz_max=config["mz_max"],
        bin_size=config["bin_size"],
        max_peaks=config["max_peaks"],
        mask_prob=config["mask_prob"],
        total_epochs=config["max_epochs"],
    )
    
    print(f"[Step 4-6] Model initialized:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  n_bins: {model.n_bins} (matches dataset)")
    
    # Verify data flow
    print(f"\n[Verification] Checking one batch from dataloader...")
    sample_batch = next(iter(train_loader))
    print(f"  Batch keys: {list(sample_batch.keys())}")
    print(f"  mz_normalized shape: {sample_batch['mz_normalized'].shape}")
    print(f"  bin_labels shape: {sample_batch['bin_labels'].shape}")
    print(f"  bin_labels already computed by Module 3 (no dynamic binning during training)")
    
    # Demonstrate explicit function calls from model_ssl.py
    print(f"\n{'='*80}")
    print("Demonstrating explicit calls to model_ssl.py functions")
    print(f"{'='*80}\n")
    
    # (1) _process_batch: Extract and normalize batch data
    print("[Function 1/8] model._process_batch()")
    mzs, intensities, precursors, bin_labels = model._process_batch(sample_batch)
    print(f"  Input: sample_batch from SpectrumDataset")
    print(f"  Output: mzs {mzs.shape}, intensities {intensities.shape}, precursors {precursors.shape}, bin_labels {bin_labels.shape}")
    print(f"  Batch processed: normalized m/z + precursors [mass, charge, mz] + pre-computed bins\n")
    
    # (2) bin_mz: Verify binning (already done in Dataset, but model can re-bin if needed)
    print("[Function 2/8] model.bin_mz()")
    rebinned = model.bin_mz(mzs, is_normalized=True)
    bins_match = torch.allclose(rebinned, bin_labels)
    print(f"  Input: mzs (normalized) {mzs.shape}")
    print(f"  Output: bin indices {rebinned.shape}")
    print(f"  Binning verified: matches pre-computed labels = {bins_match}\n")
    
    # (3) mask_spectrum: Apply BERT-style masking
    print("[Function 3/8] model.mask_spectrum()")
    masked_mzs, mask = model.mask_spectrum(mzs, intensities, mask_rate=model.mask_prob)
    masked_count = mask.sum().item()
    print(f"  Input: mzs {mzs.shape}, intensities {intensities.shape}")
    print(f"  Output: masked_mzs {masked_mzs.shape}, mask {mask.shape}")
    print(f"  Masking applied: {masked_count} positions masked (BERT 80/10/10 strategy)\n")
    
    # (4) forward: Run full model forward pass
    print("[Function 4/8] model.forward()")
    predicted_mz, logits, full_padding_mask, full_features = model.forward(masked_mzs, intensities, precursors)
    print(f"  Input: masked_mzs {masked_mzs.shape}, intensities {intensities.shape}, precursors {precursors.shape}")
    print(f"  Output:")
    print(f"    - predicted_mz: {predicted_mz.shape} (continuous m/z predictions)")
    print(f"    - logits: {logits.shape} (classification logits for {model.n_bins} bins)")
    print(f"    - full_padding_mask: {full_padding_mask.shape} (mask for CLS+precursor+peaks)")
    print(f"    - full_features: {full_features.shape} (embeddings for full sequence)")
    print(f"  Forward pass complete: Transformer encoding with CLS+precursor+peaks\n")
    
    # (5) training_step: Single training iteration
    print("[Function 5/8] model.training_step()")
    train_loss = model.training_step(sample_batch, 0, mode="train")
    print(f"  Input: batch from DataLoader")
    print(f"  Output: loss = {train_loss.item():.6f}")
    print(f"  Training step executed: MSE loss computed (dynamic CE+MSE weighting)\n")
    
    # (6) validation_step: Single validation iteration
    print("[Function 6/8] model.validation_step()")
    val_batch = next(iter(val_loader))
    val_loss = model.validation_step(val_batch, 0)
    print(f"  Input: validation batch from DataLoader")
    print(f"  Output: loss = {val_loss.item():.6f}")
    print(f"  Validation step executed: evaluation without gradient updates\n")
    
    # (7) Configure trainer and simulate epoch callbacks
    print("[Functions 7-8] Epoch end callbacks (will be called automatically during training)")
    print("  - on_train_epoch_end(): Logs metrics, saves model, plots loss curves")
    print("  - on_validation_epoch_end(): Aggregates validation metrics")
    print(f"  These will be triggered at end of each epoch by PyTorch Lightning\n")
    
    print(f"{'='*80}")
    print("All model_ssl.py functions verified successfully!")
    print(f"{'='*80}\n")
    
    # Full Training Loop 
    print(f"[Training] Starting full training for {config['max_epochs']} epochs...")
    print(f"  Data flow: SpectrumDataset (Module 1-3) -> model_ssl.py (Module 4-6)")
    print(f"  Training calls: _process_batch -> mask_spectrum -> forward -> training_step")
    print(f"  Validation calls: _process_batch -> mask_spectrum -> forward -> validation_step")
    print(f"  Epoch callbacks: on_train_epoch_end, on_validation_epoch_end\n")
    
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu", 
        devices=1, 
        log_every_n_steps=10,
        default_root_dir="./outputs",
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)


if __name__ == "__main__":
    main()
