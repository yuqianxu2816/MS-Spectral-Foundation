### Biological Context:
Input spectra are derived from workflows related to spatial glycan imaging studies of HCC, where spatial molecular heterogeneity motivates downstream representation learning.

## Problem 1B

## Generate MGF Spectra as the First Basic Output

**Objective:**  
Produce MS/MS spectra in MGF format as the first basic computable output for downstream self-supervised learning.

**Tools / Libraries**
- ProteoWizard (`msconvert`)
  - Vendor-independent tool for converting `.raw` files to open formats
  - Used for `.raw` → `.mzML` conversion
- `pyteomics.mzml`
  - Read mzML files and iterate over spectra and filter MS2 scans (`ms level = 2`)
- `pyteomics.mgf`
  - Serialize MS/MS spectra into MGF format and write `BEGIN IONS … END IONS` blocks

---

### Step 1 — Convert raw LC-MS/MS files to mzML

**Task:**  
Convert vendor-specific `.raw` files into open-format `.mzML` files.

**Actions:**
- Install ProteoWizard
- Use `msconvert` to convert `.raw` → `.mzML`
- Retain full MS1 / MS2 information during conversion

**Deliverable:**  
- One or more valid `.mzML` files corresponding to original raw samples

**Completion criteria:**  
- `.mzML` files can be successfully opened and read by downstream tools (e.g., Pyteomics)

---

### Step 2 — Extract MS2 spectra from mzML

**Task:**  
Read `.mzML` files and extract MS/MS (MS2) spectra.

**Actions:**
- Use `pyteomics.mzml` to iterate through spectra
- Filter spectra by `ms level = 2`
- Extract:
  - m/z array
  - intensity array
  - precursor m/z (if available)

**Deliverable:**  
- In-memory representation of MS2 spectra as `(m/z, intensity)` arrays with optional precursor information

**Completion criteria:**  
- Each extracted spectrum contains valid m/z–intensity pairs
- The number of extracted MS2 spectra is greater than zero

---

### Step 3 — Write MS2 spectra to MGF format

**Task:**  
Serialize extracted MS2 spectra into MGF format for model input.

**Actions:**
- Format each spectrum as an MGF block:
  - `BEGIN IONS`
  - precursor information (PEPMASS)
  - `(m/z, intensity)` pairs
  - `END IONS`
- Write spectra using `pyteomics.mgf.write`

**Deliverable:**  
- A `.mgf` file where each `BEGIN IONS … END IONS` block represents one MS/MS spectrum

**Completion criteria:**  
- MGF file is successfully written to disk
- File can be parsed into spectrum objects
- At least one valid spectrum is present in the output MGF


---

## Problem 2B

## Activity 1: MGF Data Inspection and Validation

**Libraries**
- `numpy`
  - Implement masking strategies
  - Prepare binned spectral representations
- `pyteomics.mgf`
  - Iterate over MS/MS spectrum blocks
- `numpy`
  - Validate numeric peak values
  - Compute basic spectrum statistics

### Action 1.1 — Load and validate generated MGF files

**Estimated time:** 2 hours

**Sub-tasks:**
- Load existing `.mgf` files generated in Problem 1B
- Iterate through `BEGIN IONS … END IONS` blocks
- Validate required fields (e.g., presence of peak list, optional PEPMASS)
- Check spectrum count and basic structural consistency

**Deliverable:**
- A validated list of spectrum objects loaded from MGF files

**Completion criteria:**
- MGF files can be successfully loaded without errors
- The number of spectra loaded matches expectations
- Each spectrum contains a non-empty peak list

---

### Action 1.2 — Exploratory spectrum statistics and sanity checks

**Estimated time:** 2-3 hours

**Sub-tasks:**
- Verify that m/z and intensity values are numeric
- Remove peaks with zero or negative intensity
- Compute basic statistics:
  - number of peaks per spectrum
  - m/z range distribution
  - intensity distribution
- Optionally restrict m/z range (e.g., 100–2000)

**Deliverable:**
- Cleaned and validated spectrum objects
- Summary statistics describing the MGF data

**Completion criteria:**
- No invalid peaks remain
- Each spectrum contains ≥1 valid peak
- Summary statistics are successfully computed


**Total time for Activity 1:** **~4–5 hours**


---

## Activity 2: Masked Self-Supervised Model

**Libraries**
- `torch` (PyTorch)
  - Define encoder–decoder model
  - Implement forward and backward passes
  - Perform self-supervised training
- `torch.nn`
  - Build neural network modules
- `torch.optim`
  - Optimize model parameters

### Action 2.1 — Implement random peak masking strategy

**Estimated time:** 2 hours

**Sub-tasks:**
- Define mask ratio (e.g., 15%)
- Randomly select bins or peaks to mask
- Replace masked bins with zeros or a special mask value
- Record mask positions and reconstruction targets
- Use BERT method masking

**Deliverable:**
- A masking function (e.g., `mask_spectrum()`)

**Completion criteria:**
- Masked spectra preserve original shape
- Mask positions and targets are correctly aligned

---

### Action 2.2 — Implement self-supervised reconstruction model

**Estimated time:** 3–4 hours

**Sub-tasks:**
- Define a simple encoder–decoder architecture  
  (e.g., MLP or lightweight Transformer)
- Implement forward pass from masked input to reconstructed output
- Define loss function computed only on masked bins

**Deliverable:**
- A trainable self-supervised model class

**Completion criteria:**
- Model can run forward and backward passes
- Training loss decreases over iterations

---

### Task 2.3 — Self-supervised pretraining

**Estimated time:** 3-4 hours

**Sub-tasks:**
- Prepare mini-batches of masked spectra
- Train the model for a small number of epochs
- Monitor reconstruction loss during training

**Deliverable:**
- A pretrained encoder capable of generating embeddings

**Completion criteria:**
- Loss curve shows a convergent trend
- No runtime or memory errors occur during training

**Total time for Activity 2:** **8–10 hours**

---

## Activity 3: Embedding Evaluation

**Libraries**
- `torch`
  - Extract spectrum-level embeddings from the pretrained encoder
- `numpy`
  - Aggregate embeddings (mean / median pooling)
- `scikit-learn`
  - Perform PCA (`sklearn.decomposition.PCA`)
- `umap-learn` (optional)
  - Perform UMAP dimensionality reduction
- `matplotlib` / `seaborn`
  - Generate embedding visualization plots

### Action 3.1 — Extract spectrum-level embeddings

**Estimated time:** 1–2 hours

**Sub-tasks:**
- Freeze or reuse the pretrained encoder
- Pass unmasked spectra through the encoder
- Collect spectrum-level embedding vectors

**Deliverable:**
- A matrix of spectrum embeddings (`n_spectra × d`)

**Completion criteria:**
- Each spectrum has a corresponding embedding vector
- Embedding dimensionality is consistent (e.g., `d = 128`)

---

### Action 3.2 — Analyze embedding-space distributions

**Estimated time:** 2–3 hours

**Sub-tasks:**
- Group spectrum-level embeddings by raw file (10 in total)
- Aggregate embeddings to sample-level representations using mean pooling
- Compute intra-group and cross-group distances in embedding space
- Compare embedding distributions between HCC and cirrhosis groups
- Optionally compute statistical distance metrics (e.g., centroid distance or distribution similarity)

**Deliverable:**
- Sample-level embeddings
- Embedding-space distance matrix or distribution comparison results

**Completion criteria:**
- One sample-level embedding is produced per raw file (10 in total)
- Distance computation runs without errors
- Embedding-space structure can be quantitatively analyzed

**Total time for Activity 3:** **~3–5 hours**

---

### Action 3.3 — Embedding visualization and structure inspection 

**Estimated time:** 1–2 hours

**Sub-tasks:**
- Apply PCA or UMAP to embedding vectors
- Visualize sample-level or sampled spectrum-level embeddings
- Label points by disease group (HCC vs cirrhosis)
- Inspect whether systematic distribution shifts are visible

**Deliverable:**
- PCA / UMAP visualization plots

**Completion criteria:**
- Visualization is successfully generated
- Embedding structure is interpretable at the group level

---

### Action 3.4 — Exemplar spectra discovery (Task B)

**Estimated time:** 2–3 hours

**Sub-tasks:**
- Compute nearest-neighbor or distance-based relationships between spectra
- Identify spectra with:
  - maximal cross-group distance
  - minimal cross-group distance
  - representative positions within embedding clusters
- Map selected embeddings back to original spectra metadata

**Deliverable:**
- A list of exemplar spectra representing group differences or similarities

**Completion criteria:**
- Exemplar spectra can be traced back to original raw files
- Distance-based selection is reproducible
- Representative spectra are successfully extracted

---

### Action 3.5 — Downstream representation evaluation (Module 8)

**Estimated time:** 1–2 hours

**Sub-tasks:**
- Summarize distribution-level analysis results
- Interpret visualization outcomes
- Analyze exemplar spectra for potential molecular patterns
- Evaluate whether learned embeddings capture systematic disease-related structure

**Deliverable:**
- Representation evaluation summary
- Final plots and exemplar spectra summaries

**Completion criteria:**
- Evaluation results are documented
- Visualization and exemplar analysis are integrated into final interpretation

**Total time for Activity 3:** **~6–10 hours**

## Overall Time Estimate

| Activity | Estimated Time |
|--------|----------------|
| Activity 1: MGF Data Analysis | 4–5 hours |
| Activity 2: Masked Self-Supervised Model | 8–10 hours |
| Activity 3: Embedding Evaluation | 6–10 hours |
| **Total** | **~18–25 hours** |
