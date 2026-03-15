"""
Complete pipeline for Module 7 & 8: Embedding Analysis

This script demonstrates the full workflow:
1. Load pretrained model
2. Extract spectrum-level embeddings (Module 6)
3. Perform sample-level aggregation (Module 7.1)
4. Analyze embedding distributions (Module 7.2)
5. Discover exemplar spectra (Module 7.3)
6. Visualize and evaluate (Module 8)
"""
import os
import hashlib
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from MS_Spectral_Foundation.analyze_embeddings import EmbeddingAnalyzer, create_sample_metadata
from MS_Spectral_Foundation.spectrum_dataset import SpectrumDataset
from MS_Spectral_Foundation.metadata_utils import (
    parse_metadata_from_mgf,
    prepare_analysis_data
)


def _cache_key(mgf_paths: list, model_path: str, embedding_type: str) -> str:
    """Generate an 8-char MD5 cache key from input files, model path, and embedding type."""
    raw = "|".join(sorted(mgf_paths)) + "|" + model_path + "|" + embedding_type
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def _save_cache(cache_dir: str, cache_id: str,
                embeddings: np.ndarray, precursor_mz: np.ndarray,
                precursor_charge: np.ndarray, metadata: pd.DataFrame):
    """Save embeddings and metadata to the cache directory."""
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, f"{cache_id}_embeddings.npy"), embeddings)
    np.save(os.path.join(cache_dir, f"{cache_id}_precursor_mz.npy"), precursor_mz)
    np.save(os.path.join(cache_dir, f"{cache_id}_precursor_charge.npy"), precursor_charge)
    metadata.to_parquet(os.path.join(cache_dir, f"{cache_id}_metadata.parquet"), index=False)
    print(f"[Cache] Saved to {cache_dir}  (id={cache_id})")


def _load_cache(cache_dir: str, cache_id: str):
    """Try to load cached results. Returns None if the cache does not exist."""
    emb_path  = os.path.join(cache_dir, f"{cache_id}_embeddings.npy")
    mz_path   = os.path.join(cache_dir, f"{cache_id}_precursor_mz.npy")
    chg_path  = os.path.join(cache_dir, f"{cache_id}_precursor_charge.npy")
    meta_path = os.path.join(cache_dir, f"{cache_id}_metadata.parquet")
    if all(os.path.exists(p) for p in [emb_path, mz_path, chg_path, meta_path]):
        print(f"[Cache] Cache hit (id={cache_id}), loading directly and skipping Modules 1-6")
        return (
            np.load(emb_path),
            np.load(mz_path),
            np.load(chg_path),
            pd.read_parquet(meta_path),
        )
    return None


def main(config_override=None):
    # Configuration
    config = {
        # Model
        "model_path": r"C:\Users\Lenovo\Desktop\576dataset\Github_project\MS-Spectral-Foundation\outputs\model_ssl_v4_epoch019.pt",
        
        # Data - Support multiple MGF files for multi-group analysis
        # Option 1: Single MGF with multiple groups in TITLE field
        # "mgf_paths": [
        #     r"C:\Users\Lenovo\Desktop\576dataset\combined_data.mgf",
        # ],
        
        # Option 2: Multiple MGF files, one per disease group
        "mgf_paths": [
            r"C:\Users\Lenovo\Desktop\576dataset\09062023_Mehta_GR10000524_DDRC_Sample4_561_cirrhotic_output.mgf",
            r"C:\Users\Lenovo\Desktop\576dataset\09062023_Mehta_GR10000524_DDRC_Sample9_0206_HCC_output.mgf",
        ],
        
        # Manual group labels (only needed if MGF TITLE doesn't contain disease info)
        # If None, will try to extract from MGF TITLE field
        "manual_group_labels": ["Cirrhosis", "HCC"],  # One label per MGF file, or None
        
        # Dataset parameters (must match training config)
        "max_peaks": 150,
        "mz_min": 50.0,
        "mz_max": 2500.0,
        "bin_size": 0.5,
        
        # Analysis parameters
        "batch_size": 32,
        "num_workers": 4,
        "embedding_type": "cls",  # "cls", "precursor", or "mean_peaks"
        "n_exemplars": 5,
        
        # Output directory
        "output_dir": r"C:\Users\Lenovo\Desktop\576dataset\Github_project\MS-Spectral-Foundation\logs\embedding_analysis",
        
        # Cache directory (stores Module 1-6 results so subsequent runs skip recomputation)
        # Set to None to disable caching
        "cache_dir": r"C:\Users\Lenovo\Desktop\576dataset\Github_project\MS-Spectral-Foundation\logs\embedding_cache",
    }
    if config_override:
        config.update(config_override)

    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    print("="*80)
    print("MS-Spectral-Foundation: Embedding Analysis Pipeline")
    print("Modules 7 & 8: Distribution Analysis and Representation Evaluation")
    print("="*80)
    
    # Cache Check
    cache_dir = config.get("cache_dir")
    cache_id  = _cache_key(config["mgf_paths"], config["model_path"], config["embedding_type"])
    cached    = _load_cache(cache_dir, cache_id) if cache_dir else None

    if cached is not None:
        embeddings, precursor_mz, precursor_charge, combined_metadata = cached
        print(f"  Shape: {embeddings.shape}")
        print(f"  Metadata rows: {len(combined_metadata)}")
        # Even on a cache hit we still need an analyzer instance (used by aggregation steps etc.)
        model_path = Path(config["model_path"])
        if not model_path.exists():
            ckpt_dir = Path(r"C:\Users\Lenovo\Desktop\576dataset\Github_project\MS-Spectral-Foundation\outputs\lightning_logs")
            checkpoints = list(ckpt_dir.rglob("*.ckpt")) if ckpt_dir.exists() else []
            if checkpoints:
                config["model_path"] = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
            else:
                print("No checkpoints found!")
                return
        analyzer = EmbeddingAnalyzer(
            model_path=config["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        # Step 1: Initialize Analyzer
        print("\n[Step 1] Loading pretrained model...")
        
        # Check if model exists, otherwise use checkpoint
        model_path = Path(config["model_path"])
        if not model_path.exists():
            print(f".pt file not found: {model_path}")
            print("   Looking for Lightning checkpoint...")
            
            # Try to find latest checkpoint
            ckpt_dir = Path(r"C:\Users\Lenovo\Desktop\576dataset\Github_project\MS-Spectral-Foundation\outputs\lightning_logs")
            if ckpt_dir.exists():
                # Find all checkpoints
                checkpoints = list(ckpt_dir.rglob("*.ckpt"))
                if checkpoints:
                    # Use the most recent checkpoint
                    latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    config["model_path"] = str(latest_ckpt)
                    print(f"   Using checkpoint: {latest_ckpt}")
                else:
                    print("No checkpoints found!")
                    return
            else:
                print(" Checkpoint directory not found!")
                return
        
        analyzer = EmbeddingAnalyzer(
            model_path=config["model_path"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Step 2: Load Dataset(s)
        print("\n[Step 2] Loading dataset(s)...")
        
        # Load all MGF files
        all_datasets = []
        all_metadata = []
        
        for mgf_idx, mgf_path in enumerate(config["mgf_paths"]):
            print(f"\n  Loading MGF {mgf_idx + 1}/{len(config['mgf_paths'])}: {Path(mgf_path).name}")
            
            # Load dataset
            dataset = SpectrumDataset(
                mgf_path=mgf_path,
                max_peaks=config["max_peaks"],
                mz_min=config["mz_min"],
                mz_max=config["mz_max"],
                bin_size=config["bin_size"],
                remove_precursor_tol=2.0,
                min_intensity=0.01,
                min_peaks=5,
            )
            
            # Parse metadata from MGF, then filter to only spectra that passed preprocessing
            metadata_df = parse_metadata_from_mgf(mgf_path)
            metadata_df = metadata_df.iloc[dataset.valid_indices].reset_index(drop=True)
            
            # Use filename stem as sample_id so each MGF file is a distinct sample
            file_stem = Path(mgf_path).stem
            metadata_df["sample_id"] = file_stem
            
            # If manual group labels provided, override extracted labels
            if config.get("manual_group_labels") and config["manual_group_labels"][mgf_idx]:
                manual_label = config["manual_group_labels"][mgf_idx]
                metadata_df["disease_group"] = manual_label
                print(f"    Using manual label: {manual_label}")
            
            # Store spectrum offset for later indexing
            metadata_df["dataset_idx"] = mgf_idx
            
            all_datasets.append(dataset)
            all_metadata.append(metadata_df)
            
            print(f"    Loaded: {len(dataset)} spectra")
            print(f"    Groups: {metadata_df['disease_group'].value_counts().to_dict()}")
        
        # Combine all metadata
        combined_metadata = pd.concat(all_metadata, ignore_index=True)
        
        print(f"\n Total datasets: {len(all_datasets)}")
        print(f" Total spectra: {len(combined_metadata)}")
        print(f" Disease groups:")
        for group, count in combined_metadata["disease_group"].value_counts().items():
            print(f"    {group}: {count} spectra")
        
        # Step 3: Extract Embeddings (Module 6)
        print("\n" + "="*80)
        print("Module 6: Extract Spectrum-Level Embeddings")
        print("="*80)
        
        # Extract embeddings from all datasets
        all_embeddings = []
        all_precursor_mz = []
        all_precursor_charge = []
        
        for dataset_idx, dataset in enumerate(all_datasets):
            print(f"\n  Extracting embeddings from dataset {dataset_idx + 1}/{len(all_datasets)}...")
            
            results = analyzer.extract_embeddings(
                dataset=dataset,
                batch_size=config["batch_size"],
                num_workers=config["num_workers"],
                embedding_type=config["embedding_type"]
            )
            
            all_embeddings.append(results["embeddings"])
            all_precursor_mz.append(results["precursor_mz"])
            all_precursor_charge.append(results["precursor_charge"])
        
        # Combine all embeddings
        embeddings      = np.vstack(all_embeddings)
        precursor_mz    = np.concatenate(all_precursor_mz)
        precursor_charge = np.concatenate(all_precursor_charge)
        
        print(f"\n Combined embeddings from all datasets")
        print(f"\nEmbedding statistics:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Mean: {embeddings.mean():.4f}")
        print(f"  Std: {embeddings.std():.4f}")
        print(f"  L2 norm (mean): {np.linalg.norm(embeddings, axis=1).mean():.4f}")
        
        # Save Cache
        if cache_dir:
            _save_cache(cache_dir, cache_id,
                        embeddings, precursor_mz, precursor_charge, combined_metadata)
    
    # Step 4: Prepare Metadata for Analysis
    print("\n" + "="*80)
    print("Preparing Sample Metadata")
    print("="*80)
    
    # Use metadata_utils to prepare analysis data
    sample_ids, labels, label_names = prepare_analysis_data(
        embeddings=embeddings,
        metadata_df=combined_metadata,
        sample_id_col="sample_id",
        group_col="disease_group"
    )
    
    print(f"\n  Sample metadata summary:")
    print(f"    Total spectra: {len(sample_ids)}")
    print(f"    Unique samples: {len(np.unique(sample_ids))}")
    for label_id, label_name in label_names.items():
        print(f"    {label_name}: {(labels == label_id).sum()} spectra")
    
    # Check if we have at least 2 groups for inter-group analysis
    n_groups = len(label_names)
    if n_groups < 2:
        print("\n  WARNING: Only one disease group detected!")
        print("   Inter-group distance analysis requires at least 2 groups.")
        print("   Please provide MGF files from different disease groups.")
        print("\n   Current options:")
        print("   1. Add HCC data: Uncomment the HCC MGF path in config")
        print("   2. Use a combined MGF file with TITLE containing disease labels")
        print("\n   Continuing with single-group analysis (intra-group only)...")
    else:
        print(f"\n Multi-group analysis ready: {n_groups} groups detected")
    
    # Step 5: Sample-Level Aggregation (Module 7.1)
    print("\n" + "="*80)
    print("Module 7.1: Sample-Level Aggregation")
    print("="*80)
    
    sample_embeddings, unique_samples = analyzer.aggregate_to_sample_level(
        embeddings=embeddings,
        sample_ids=sample_ids,
        method="mean"
    )
    
    # Map sample labels
    sample_labels = np.array([labels[sample_ids == s][0] for s in unique_samples])
    
    # Step 6: Distribution Analysis (Module 7.2) 
    print("\n" + "="*80)
    print("Module 7.2: Embedding Distribution Analysis")
    print("="*80)
    
    # Use spectrum-level embeddings for distribution analysis.
    # With only 1 sample per group, sample-level intra-distances are undefined;
    # spectrum-level distances capture real within-group vs between-group variation.
    metrics = analyzer.analyze_embedding_distributions(
        embeddings=embeddings,
        labels=labels,
        label_names=label_names,
        max_samples=2000  # subsample to avoid OOM on 300k spectra
    )
    
    # Step 7: Exemplar Spectra Discovery (Module 7.3)
    print("\n" + "="*80)
    print("Module 7.3: Exemplar Spectra Discovery")
    print("="*80)
    
    # Note: Exemplar indices are global across all datasets
    # To retrieve actual spectra, map indices back to individual datasets
    
    # Find exemplars at spectrum level (not sample level)
    exemplars_centroid = analyzer.find_exemplar_spectra(
        embeddings=embeddings,
        labels=labels,
        dataset=None,  # not used inside the function
        n_exemplars=config["n_exemplars"],
        strategy="centroids"
    )
    
    # Only run extremes strategy if we have 2+ groups
    if len(label_names) >= 2:
        exemplars_extremes = analyzer.find_exemplar_spectra(
            embeddings=embeddings,
            labels=labels,
            dataset=None,  # not used inside the function
            n_exemplars=config["n_exemplars"],
            strategy="extremes"
        )
    else:
        exemplars_extremes = {}
        print(" Skipping 'extremes' strategy (requires 2+ groups)")
    
    print(f"\nExemplar spectra identified:")
    for key, indices in exemplars_centroid.items():
        print(f"  {key}: {len(indices)} spectra")
    for key, indices in exemplars_extremes.items():
        print(f"  {key}: {len(indices)} spectra")
    
    # Save exemplar indices with metadata
    exemplar_output = os.path.join(config["output_dir"], "exemplar_spectra.txt")
    with open(exemplar_output, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Exemplar Spectra Indices\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Note: Indices are global across all loaded MGF files\n")
        f.write(f"Total spectra: {len(embeddings)}\n")
        f.write(f"Disease groups: {label_names}\n\n")
        
        f.write("## Strategy: Centroids\n")
        f.write("-" * 80 + "\n")
        for key, indices in exemplars_centroid.items():
            f.write(f"{key}:\n")
            f.write(f"  Indices: {indices}\n")
            # Add disease group info for each index
            for idx in indices:
                group = label_names[labels[idx]]
                sample = sample_ids[idx]
                f.write(f"    [{idx}] Sample: {sample}, Group: {group}\n")
            f.write("\n")
        
        if exemplars_extremes:
            f.write("\n## Strategy: Extremes\n")
            f.write("-" * 80 + "\n")
            for key, indices in exemplars_extremes.items():
                f.write(f"{key}:\n")
                f.write(f"  Indices: {indices}\n")
                for idx in indices:
                    group = label_names[labels[idx]]
                    sample = sample_ids[idx]
                    f.write(f"    [{idx}] Sample: {sample}, Group: {group}\n")
                f.write("\n")
    
    print(f"  Saved exemplar indices to: {exemplar_output}")
    
    # Step 8: Visualization (Module 8)
    print("\n" + "="*80)
    print("Module 8: Representation Evaluation and Visualization")
    print("="*80)
    
    # Subsample spectrum-level embeddings for visualization (avoids too many points / OOM)
    viz_max = 3000
    rng_viz = np.random.default_rng(0)
    viz_keep = []
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        chosen = rng_viz.choice(idx, size=min(viz_max, len(idx)), replace=False)
        viz_keep.append(chosen)
    viz_keep = np.sort(np.concatenate(viz_keep))
    viz_embeddings = embeddings[viz_keep]
    viz_labels = labels[viz_keep]
    
    # 8.1: PCA visualization
    print("\n[8.1] PCA visualization...")
    analyzer.visualize_embeddings(
        embeddings=viz_embeddings,
        labels=viz_labels,
        method="pca",
        label_names=label_names,
        save_path=os.path.join(config["output_dir"], "embeddings_pca.png"),
        title="Spectrum-Level Embeddings (PCA)"
    )
    
    # 8.2: t-SNE visualization
    print("\n[8.2] t-SNE visualization...")
    analyzer.visualize_embeddings(
        embeddings=viz_embeddings,
        labels=viz_labels,
        method="tsne",
        label_names=label_names,
        save_path=os.path.join(config["output_dir"], "embeddings_tsne.png"),
        title="Spectrum-Level Embeddings (t-SNE)"
    )
    
    # 8.3: UMAP visualization (if available)
    print("\n[8.3] UMAP visualization...")
    analyzer.visualize_embeddings(
        embeddings=viz_embeddings,
        labels=viz_labels,
        method="umap",
        label_names=label_names,
        save_path=os.path.join(config["output_dir"], "embeddings_umap.png"),
        title="Spectrum-Level Embeddings (UMAP)"
    )
    
    # 8.4: Distance distributions
    print("\n[8.4] Distance distribution analysis...")
    analyzer.plot_distance_distributions(
        embeddings=viz_embeddings,
        labels=viz_labels,
        label_names=label_names,
        save_path=os.path.join(config["output_dir"], "distance_distributions.png")
    )
    
    # Step 9: Generate Report (Module 8)
    print("\n" + "="*80)
    print("Module 8: Generate Analysis Report")
    print("="*80)
    
    report_path = os.path.join(config["output_dir"], "embedding_analysis_report.txt")
    analyzer.generate_report(metrics, save_path=report_path)
    
    # Summary 
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nAll results saved to: {config['output_dir']}")
    print("\nGenerated files:")
    print(f"  1. embeddings_pca.png - PCA visualization")
    print(f"  2. embeddings_tsne.png - t-SNE visualization")
    print(f"  3. embeddings_umap.png - UMAP visualization (if available)")
    print(f"  4. distance_distributions.png - Distance analysis")
    print(f"  5. exemplar_spectra.txt - Exemplar spectrum indices")
    print(f"  6. embedding_analysis_report.txt - Analysis report")
    print()
    print("Analysis summary:")
    print(f"  - Total MGF files: {len(config['mgf_paths'])}")
    print(f"  - Total spectra: {len(embeddings)}")
    print(f"  - Unique samples: {len(np.unique(sample_ids))}")
    print(f"  - Disease groups: {len(label_names)} ({', '.join(label_names.values())})")
    print()
    print("Next steps:")
    print("  1. Review visualizations to assess group separation")
    if len(label_names) >= 2:
        print("  2. Check inter-group vs intra-group distances in the report")
    else:
        print("  2.  Add HCC data to enable inter-group distance analysis")
    print("  3. Examine exemplar spectra for biological interpretation")
    print("  4. Apply embeddings to downstream tasks (classification, clustering, etc.)")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
