# Module 7 & 8: Embedding Analysis and Downstream Evaluation

## Overview

Module 7 and 8 implement **non-supervised representation evaluation** for learned spectrum embeddings. These modules analyze whether the self-supervised model (Modules 4-6) captures meaningful biological structure without using disease labels during training.

## Pipeline

```
Module 1-3 (Preprocessing) → Module 4-6 (SSL Training) → Module 7-8 (Analysis)
                              ↓
                    Pretrained Model
                              ↓
                Module 6: Extract Embeddings
                              ↓
      ┌───────────────────────┴───────────────────────┐
      ↓                                               ↓
Module 7.1: Sample-Level Aggregation    Module 7.2: Distribution Analysis
      ↓                                               ↓
Module 7.3: Exemplar Discovery         Module 8: Visualization & Report
```

## Module Descriptions

### Module 6: Extract Spectrum Embeddings
Extract fixed-dimensional representations from pretrained encoder.

**Input**: Spectrum (m/z, intensity pairs)  
**Output**: Embedding vector ∈ R^d

### Module 7.1: Sample-Level Aggregation
Aggregate spectrum embeddings to obtain sample-level representations.

**Methods**:
- Mean pooling (default)
- Median pooling

### Module 7.2: Embedding Distribution Analysis
Compare intra-group vs inter-group distances to assess separation.

**Metrics**:
- Intra-group distances (within HCC, within Cirrhosis)
- Inter-group distances (HCC vs Cirrhosis)
- Separation ratio: inter / mean(intra)
- Statistical tests (Mann-Whitney U)

### Module 7.3: Exemplar Spectra Discovery
Identify representative spectra characterizing group differences.

**Strategies**:
1. **Centroids**: Spectra closest to group centroids
2. **Extremes**: Spectra with max/min cross-group distance
3. **Boundary**: Spectra near decision boundary

### Module 8: Representation Evaluation
Visualize and evaluate learned representations without supervision.

**Visualizations**:
- PCA projection
- t-SNE projection
- UMAP projection
- Distance distribution plots

**Outputs**:
- Embedding visualizations
- Distance analysis plots
- Analysis report

## Installation

### Required Dependencies

```bash
# Core dependencies (already installed for training)
pip install torch pytorch-lightning numpy pandas matplotlib seaborn scikit-learn scipy

# Optional for UMAP
pip install umap-learn
```

## Quick Start

### 1. Basic Usage (Simulated Labels)

```python
from MS_Spectral_Foundation.analyze_embeddings import EmbeddingAnalyzer
from MS_Spectral_Foundation.spectrum_dataset import SpectrumDataset

# Load pretrained model
analyzer = EmbeddingAnalyzer(
    model_path="outputs/model_ssl_v4_epoch099.pt",
    device="cuda"
)

# Load dataset
dataset = SpectrumDataset(
    mgf_path="data.mgf",
    max_peaks=150,
    mz_min=50.0,
    mz_max=2500.0,
    bin_size=0.5
)

# Extract embeddings
results = analyzer.extract_embeddings(dataset, batch_size=64)
embeddings = results["embeddings"]  # (N, d)

# For demonstration: create simulated labels
sample_ids = np.array([f"sample_{i//100}" for i in range(len(embeddings))])
labels = np.array([0 if i < len(embeddings)//2 else 1 for i in range(len(embeddings))])
label_names = {0: "Group_A", 1: "Group_B"}

# Aggregate to sample level
sample_emb, unique_samples = analyzer.aggregate_to_sample_level(
    embeddings, sample_ids, method="mean"
)
sample_labels = np.array([labels[sample_ids == s][0] for s in unique_samples])

# Analyze distributions
metrics = analyzer.analyze_embedding_distributions(
    sample_emb, sample_labels, label_names
)

# Visualize
analyzer.visualize_embeddings(
    sample_emb, sample_labels, method="pca",
    label_names=label_names,
    save_path="embeddings_pca.png"
)

# Generate report
analyzer.generate_report(metrics, save_path="report.txt")
```

### 2. Complete Pipeline Script

Run the complete analysis:

```bash
cd C:\Users\Lenovo\Desktop\576dataset\Github_project\MS-Spectral-Foundation
python -m MS_Spectral_Foundation.run_embedding_analysis
```

This will:
1. Load pretrained model (auto-detects .pt or .ckpt)
2. Extract embeddings from all spectra
3. Perform sample-level aggregation
4. Analyze embedding distributions
5. Discover exemplar spectra
6. Generate visualizations (PCA, t-SNE, UMAP)
7. Create analysis report

**Output files** (saved to `logs/embedding_analysis/`):
```
embeddings_pca.png              # PCA visualization
embeddings_tsne.png             # t-SNE visualization
embeddings_umap.png             # UMAP visualization
distance_distributions.png      # Distance comparison
exemplar_spectra.txt           # Exemplar spectrum indices
embedding_analysis_report.txt  # Full analysis report
```

## Working with Real Metadata

### Option 1: Extract from MGF TITLE Fields

If your MGF file contains disease information in TITLE fields:

```python
from MS_Spectral_Foundation.metadata_utils import parse_metadata_from_mgf

# Parse metadata from MGF
metadata_df = parse_metadata_from_mgf("data.mgf")

# metadata_df contains:
#   - spectrum_idx: spectrum index
#   - sample_id: extracted from TITLE
#   - disease_group: extracted from TITLE (HCC/Cirrhosis/etc)
#   - PEPMASS, CHARGE, RTINSECONDS, etc.

print(metadata_df.head())
print(metadata_df['disease_group'].value_counts())
```

**Supported TITLE formats**:
```
Sample_HCC_001_Scan_12345      → sample_id: Sample_HCC_001, group: HCC
20230906_Cirrhosis_P1_MS2      → sample_id: 20230906_Cirrhosis_P1, group: Cirrhosis
HCC_Patient_042_Spectrum_5678  → sample_id: HCC_Patient_042, group: HCC
```

### Option 2: Load from External CSV

If you have a separate metadata file:

```python
from MS_Spectral_Foundation.metadata_utils import load_metadata_from_csv

# Load from CSV
metadata_df = load_metadata_from_csv(
    csv_path="sample_metadata.csv",
    sample_id_col="sample_id",
    group_col="disease"
)
```

**Expected CSV format**:
```csv
sample_id,disease,age,gender,other_info
Sample_001,HCC,65,M,...
Sample_002,HCC,58,F,...
Sample_003,Cirrhosis,62,M,...
Sample_004,Cirrhosis,70,F,...
```

### Option 3: Combine with Embedding Analysis

```python
from MS_Spectral_Foundation.metadata_utils import (
    parse_metadata_from_mgf,
    prepare_analysis_data
)
from MS_Spectral_Foundation.analyze_embeddings import EmbeddingAnalyzer

# 1. Extract embeddings
analyzer = EmbeddingAnalyzer(model_path="model.pt")
results = analyzer.extract_embeddings(dataset)
embeddings = results["embeddings"]

# 2. Parse metadata
metadata_df = parse_metadata_from_mgf("data.mgf")

# Verify alignment
assert len(embeddings) == len(metadata_df), "Mismatch in spectrum count!"

# 3. Prepare for analysis
sample_ids, numeric_labels, label_names = prepare_analysis_data(
    embeddings=embeddings,
    metadata_df=metadata_df,
    sample_id_col="sample_id",
    group_col="disease_group"
)

# 4. Sample-level aggregation
sample_emb, unique_samples = analyzer.aggregate_to_sample_level(
    embeddings, sample_ids, method="mean"
)
sample_labels = np.array([numeric_labels[sample_ids == s][0] for s in unique_samples])

# 5. Run full analysis
metrics = analyzer.analyze_embedding_distributions(
    sample_emb, sample_labels, label_names
)

exemplars = analyzer.find_exemplar_spectra(
    embeddings, numeric_labels, dataset, n_exemplars=5, strategy="extremes"
)

analyzer.visualize_embeddings(
    sample_emb, sample_labels, method="pca",
    label_names=label_names,
    save_path="embeddings_pca.png"
)

analyzer.generate_report(metrics, save_path="report.txt")
```

## Customizing Analysis

### 1. Change Embedding Type

```python
# Extract different types of embeddings
results_cls = analyzer.extract_embeddings(dataset, embedding_type="cls")
results_prec = analyzer.extract_embeddings(dataset, embedding_type="precursor")
results_peaks = analyzer.extract_embeddings(dataset, embedding_type="mean_peaks")
```

### 2. Change Aggregation Method

```python
# Use median instead of mean for sample aggregation
sample_emb, unique_samples = analyzer.aggregate_to_sample_level(
    embeddings, sample_ids, method="median"
)
```

### 3. Customize Exemplar Discovery

```python
# Find centroids
exemplars_centroid = analyzer.find_exemplar_spectra(
    embeddings, labels, dataset, n_exemplars=10, strategy="centroids"
)

# Find extreme pairs
exemplars_extreme = analyzer.find_exemplar_spectra(
    embeddings, labels, dataset, n_exemplars=5, strategy="extremes"
)

# Find boundary spectra
exemplars_boundary = analyzer.find_exemplar_spectra(
    embeddings, labels, dataset, n_exemplars=20, strategy="boundary"
)
```

### 4. Multi-Group Analysis

The analysis supports arbitrary number of groups:

```python
# Example: 3 disease groups
label_names = {0: "Healthy", 1: "Cirrhosis", 2: "HCC"}
labels = np.array([...])  # 0, 1, or 2

# Analysis will compute:
# - Intra-group distances for each group
# - All pairwise inter-group distances (if 2 groups)
# - Visualization with 3 colors
```

## Interpreting Results

### Separation Ratio

The **separation ratio** measures how well groups are separated in embedding space:

```
separation_ratio = inter_group_distance / mean(intra_group_distances)
```

**Interpretation**:
- **> 1.2**: Strong separation (embeddings capture meaningful differences)
- **1.0 - 1.2**: Moderate separation (some distinguishable structure)
- **< 1.0**: Weak separation (groups overlap significantly)

### Statistical Significance

**Mann-Whitney U test** compares intra-group vs inter-group distance distributions:

- **p < 0.001**: Highly significant difference (strong evidence of separation)
- **p < 0.05**: Significant difference
- **p >= 0.05**: No significant difference

### Visualizations

#### PCA
- Shows linear structure in embedding space
- Check explained variance ratio (higher = more information preserved)
- Look for cluster separation

#### t-SNE / UMAP
- Shows nonlinear manifold structure
- Better for visualizing local neighborhoods
- Check if groups form distinct clusters

#### Distance Distributions
- Overlapping distributions → poor separation
- Distinct distributions → good separation
- Inter-group should be > intra-group for good models

### Exemplar Spectra

Use exemplar indices to:
1. **Retrieve original spectra** from dataset
2. **Visualize m/z patterns** that characterize groups
3. **Identify discriminative peaks** for biological interpretation
4. **Guide downstream targeted analysis**

Example:
```python
# Get exemplar spectra
exemplars = analyzer.find_exemplar_spectra(...)

# Retrieve and visualize
for category, indices in exemplars.items():
    for idx in indices[:3]:  # Plot first 3
        spectrum = dataset[idx]
        plot_spectrum(spectrum["mz_array"], spectrum["intensity_array"])
        plt.title(f"{category} - Spectrum {idx}")
        plt.show()
```

## Troubleshooting

### Model Checkpoint Not Found

If `.pt` file doesn't exist (training didn't reach final epoch):

```python
# Use Lightning checkpoint instead
analyzer = EmbeddingAnalyzer(
    model_path="outputs/lightning_logs/version_1/checkpoints/epoch=5-step=27000.ckpt"
)
```

The script `run_embedding_analysis.py` auto-detects and uses the latest checkpoint.

### UMAP Not Available

```bash
pip install umap-learn
```

Or the script will automatically fall back to PCA.

### Memory Issues

For large datasets:

```python
# Reduce batch size
results = analyzer.extract_embeddings(dataset, batch_size=16)

# Or process in chunks
chunks = []
for i in range(0, len(dataset), 1000):
    subset = Subset(dataset, range(i, min(i+1000, len(dataset))))
    chunk_results = analyzer.extract_embeddings(subset)
    chunks.append(chunk_results["embeddings"])

embeddings = np.vstack(chunks)
```

### Metadata Extraction Issues

If automatic extraction fails:

```python
# Use custom regex pattern
from MS_Spectral_Foundation.metadata_utils import extract_sample_id_from_title

# Define custom pattern
custom_pattern = r'(SampleID_\d+)'

sample_id = extract_sample_id_from_title(
    title="Your_Custom_Format_SampleID_123_Other",
    pattern=custom_pattern
)
```

## Advanced Usage

### Save and Load Embeddings

```python
# Save embeddings for future use
np.savez(
    "embeddings.npz",
    embeddings=embeddings,
    sample_ids=sample_ids,
    labels=labels
)

# Load later
data = np.load("embeddings.npz")
embeddings = data["embeddings"]
sample_ids = data["sample_ids"]
labels = data["labels"]
```

### Batch Processing Multiple Files

```python
from pathlib import Path

mgf_files = list(Path("data/").glob("*.mgf"))

all_embeddings = []
all_metadata = []

for mgf_file in mgf_files:
    # Load dataset
    dataset = SpectrumDataset(mgf_path=str(mgf_file), ...)
    
    # Extract embeddings
    results = analyzer.extract_embeddings(dataset)
    
    # Parse metadata
    metadata = parse_metadata_from_mgf(str(mgf_file))
    
    all_embeddings.append(results["embeddings"])
    all_metadata.append(metadata)

# Combine
embeddings = np.vstack(all_embeddings)
metadata_df = pd.concat(all_metadata, ignore_index=True)
```

## Output Directory Structure

After running `run_embedding_analysis.py`:

```
logs/embedding_analysis/
├── embeddings_pca.png                 # PCA visualization
├── embeddings_tsne.png                # t-SNE visualization
├── embeddings_umap.png                # UMAP visualization
├── distance_distributions.png         # Distance comparison
├── exemplar_spectra.txt              # Exemplar indices
└── embedding_analysis_report.txt     # Full analysis report
```

## Citation

If you use these modules, please cite:

```bibtex
@article{ms-spectral-foundation,
  title={MS-Spectral-Foundation: Self-Supervised Learning for Mass Spectrometry},
  author={Your Name},
  year={2024}
}
```

## References

- Module design follows the DDS (Detailed Design Specification)
- Self-supervised learning inspired by BERT and SimCLR
- Embedding analysis based on representation learning evaluation practices

## Contact

For questions or issues:
- Check `run_embedding_analysis.py` for examples
- See inline docstrings in `analyze_embeddings.py`
- Refer to metadata handling in `metadata_utils.py`
