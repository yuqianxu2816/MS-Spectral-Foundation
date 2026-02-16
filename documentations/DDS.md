### Objective

For each MS/MS spectrum, calculate an embedding vector of a fixed dimension.

### Example

- Input: one MS/MS spectrum (m/z–intensity list)
- Intermediate output: embedding ∈ R^d (e.g., d = 128)
- Final output: sample-level embedding analysis and exemplar spectra revealing distribution-level differences.
---


### Note:
Although the computational modules operate on MS/MS spectra, the biological motivation originates from spatial omics studies where tissue-level molecular heterogeneity motivates representation learning.

## Calculation Steps

### Module 0 — Data Preparation (FTP → mzML → MGF)

This module describes the initial data preparation step before model processing.

#### Input

- Raw LC-MS/MS files downloaded from the public FTP dataset source: https://www.ebi.ac.uk/pride/archive/projects/PXD047546 
- Vendor-specific raw format (`.raw`).

#### Process

1. Download raw files from the FTP dataset repository.
2. Convert `.raw` files to open-format `.mzML` using **ProteoWizard (msconvert)**.
3. Extract MS/MS (MS2) spectra from `.mzML`.
4. Save spectra as `.mgf`, where each `BEGIN IONS … END IONS` block represents one spectrum.

#### Output

- `.mzML` files (intermediate open format)
- `.mgf` files (direct input for downstream modules)

### Module 1 — Read the MGF file

### notify the users if files are bad log the issues
Parse MGF-formatted files and convert each MS/MS spectrum into a standardized in-memory spectrum object containing metadata and peak lists.

**Input**

- File: `*.mgf`
- Format: plain text file containing multiple blocks:
  - BEGIN IONS
  - m/z intensity
  - m/z intensity
  - ...
  - END IONS
 
**Parsing procedure**
- Read the MGF file line by line
- Detect spectrum boundaries using `BEGIN IONS` and `END IONS`
- Parse peak lines into numeric `(m/z, intensity)` pairs
- Store metadata and peaks in a standardized spectrum object

**Output**

- A list of spectrum objects (one per BEGIN IONS ... END IONS block)

**Output format**

- spectra: List[Spectrum]
- Spectrum = {
  - "meta": Dict[str, Any],      # e.g., {"PEPMASS": 512.34, "CHARGE": 2, "RTINSECONDS": 1234.5}
  - peaks": List[Tuple[float, float]]  # [(mz1, inten1), (mz2, inten2), ...]
- }  

### Module 2 — Extract the (m/z, intensity) pairs from each spectrum

Filter raw peak lists from spectrum objects and produce cleaned numeric representations suitable for downstream preprocessing and binning.

**Input**

- One Spectrum object from Step 1
- Spectrum["peaks"] = List[(mz, intensity)]

**Filtering (implemented in `peak_filter.py`)**

The preprocessing pipeline consists of six functions:

- `set_mz_range`
- `remove_precursor_peak`
- `scale_intensity`
- `filter_intensity`
- `discard_low_quality`
- `_scale_to_unit_norm`


**Design inspirations (simplified from external reference implementation)**

External reference: https://github.com/bittremieuxlab/spectrum_utils/blob/main/spectrum_utils/spectrum.py

The first four functions follow the same high-level ideas as an external spectrum preprocessing implementation, but are simplified for this project:

- **set_mz_range**
  - Keeps peaks within a valid m/z range.
  - Simplified to direct array filtering without object-based spectrum manipulation.

- **remove_precursor_peak**
  - Removes peaks around precursor m/z.
  - Simplified to a direct tolerance-based filter (Da / ppm) without isotope modeling.

- **scale_intensity**
  - Supports root / log / rank scaling.
  - adds on an optional "scale relative to max peak" step (`max_intensity`) before scalings

- **filter_intensity**
  - Removes low-intensity peaks and keeps top-N peaks.
  - Implemented as a lightweight NumPy-based filter with relative thresholding.

Overall, `peak_filter.py` keeps the core preprocessing logic but removes complex spectrum object dependencies and advanced annotation-related operations, making the pipeline more concise and easier to integrate into deep learning preprocessing.


**Output**

- Numeric peak list (possibly cleaned or filtered)

**Output format**

- peaks: Array[N, 2]
  - column 0 = m/z
  - column 1 = intensity


### Module 3 — Discretize / tokenize the spectrum (m/z binning)

Convert variable-length peak lists into fixed-length vector representations by discretizing the m/z axis into bins and aggregating intensities.

**Input**

- peaks: Array[N, 2]
- Binning parameters:
  - mz_min, mz_max (e.g., 100–2000)
  - bin_width (e.g., 1.0 or 0.1 Da)
  
**Binning procedure**
- Map variable-length peak lists to fixed-length vectors by m/z binning and intensity aggregation
  
**Output**
- A fixed-length intensity vector x of length B (number of bins)
  
**Output format**
- x: Vector[B]
- aggregated (sum or max) and often normalized intensity per bin

### Module 4 — masked self-supervised learning

Masking is applied only during self-supervised pretraining. During downstream inference and classification, no masking is applied. This module defines the masking strategy and prepares masked inputs and reconstruction targets for self-supervised representation learning.

**Input**

- Binned vector x: Vector[B]
- mask_ratio (e.g., 0.15)

**Mask strategies**
- random bins, where individual bins are independently selected for masking
- structured masking (contiguous ranges), where adjacent m/z bins are masked together to simulate missing spectral regions
  
**Output**
- x_masked: Vector[B] (masked input)
- mask: Vector[B] (binary indicator: 1 = masked bin)
- targets: Vector[B] (ground truth values for masked bins only)


### Module 5 — Train the model to predict masked peaks (most important module)
This training step is fully self-supervised and does not use disease labels (HCC vs cirrhosis). The goal of this module is to train an encoder to learn general spectral representations by reconstructing masked spectral information.

**Model input**
- x_masked (or masked token sequence): Tensor[batch, B]
- mask (to compute loss only on masked locations): Tensor[batch, B]
- targets: Tensor[batch, B]

**Training procedure**
- Feed masked spectral representations into the encoder–decoder model
- Predict the original intensities at masked positions
- Compute reconstruction loss only on masked bins
- Update model parameters to learn general spectral representations without using disease labels
  
**Model output (training)**
- y_pred: reconstructed intensities (or token logits) over the full vector/sequence
- loss: computed on masked positions only

**output format**
- y_pred: Tensor[batch, B]
- loss:   float


### Module 6 — Extract spectrum-level embeddings
After self-supervised pretraining, the pretrained encoder is used to extract spectrum-level embeddings. This module applies the frozen or fine-tuned encoder to unmasked spectra to generate fixed-dimensional embedding vectors for downstream analysis.

**Input**
- unmasked or lightly masked spectrum representation (x or tokenized form)
  - x: Vector[B] or (token_ids, token_values, attention_mask)

**Embedding extraction procedure**
- Apply the pretrained encoder to unmasked spectra
- Extract fixed-dimensional spectrum-level embeddings

**Output**
- A spectrum embedding vector:
  - embedding ∈ R^d (e.g., d = 128)

### Downstream Tasks

### Module 7 — Analyze embedding-space distributions and sample-level structure

This module performs downstream representation analysis based on spectrum-level embeddings to examine whether systematic molecular differences exist between HCC and cirrhosis samples.

**Input**

- Spectrum-level embeddings extracted from Module 6:
  - E = {e1, e2, ..., en},  ei ∈ R^d
- Metadata:
  - sample ID (raw file)
  - disease group (HCC or cirrhosis)


### Analysis procedures

**1) Sample-level aggregation (used for visualization and comparison)**

Spectrum embeddings from the same raw file can be aggregated to obtain sample-level representations:

- Mean pooling (used in this project)
- Median pooling (optional)

Output:

- sample-level embedding:
  - z ∈ R^d


**2) Embedding-space distribution analysis**

Instead of training a supervised classifier, this project evaluates whether embeddings capture disease-related structure by comparing embedding distributions:

- Compare intra-group vs cross-group distances
- Visualize embeddings using PCA / UMAP
- Examine whether HCC and cirrhosis samples exhibit systematic distribution shifts


**3) Exemplar spectra discovery**

Identify representative spectra that characterize similarities or differences between disease groups:

- Spectra with maximal cross-group distance
- Spectra with minimal cross-group distance
- Representative spectra within embedding clusters

These spectra are used for qualitative biological interpretation.


**Output**

- Sample-level embeddings (for visualization)
- Embedding-space comparison results
- Representative exemplar spectra


**Output format**

- sample_embedding: Vector[d]
- distance_matrix or distribution metrics
- exemplar_spectra: List[Spectrum]


### Module 8 — Downstream representation evaluation (non-supervised)

Module 8 summarizes and interprets results produced by Module 7 rather than introducing new computational steps.

This module evaluates the learned representations without supervised disease classification. The goal is to assess whether the learned embeddings reveal systematic molecular structure related to disease groups.

---

**Input**

- Spectrum-level embeddings from Module 6
- Sample-level embeddings from Module 7
- Group labels (used only for comparison and visualization)


### Evaluation strategies

**1) Distribution-level comparison**

Evaluate whether embedding distributions differ between HCC and cirrhosis:

- Compare inter-group and intra-group distances
- Measure embedding-space separation using statistical distance metrics (optional)


**2) Visualization-based evaluation**

- PCA or UMAP projection of embeddings
- Visual inspection of group-level structure


**3) Exemplar-based analysis**

Use exemplar spectra identified in Module 7 to:

- Highlight spectra contributing to cross-group differences
- Examine potential molecular patterns associated with disease states


**Output**

- Embedding-space evaluation results
- Visualization plots
- Exemplar spectra summaries



## What connects the modules

The output of Module 3 serves as the input to Module 4, enabling masked self-supervised training. The pretrained encoder from Modules 4–5 is reused in Module 6 to generate spectrum-level embeddings, which are then analyzed in Module 7–8 through embedding-space distribution comparison and exemplar spectra analysis.
