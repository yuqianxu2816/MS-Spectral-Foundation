## Link to Original Dataset

https://www.ebi.ac.uk/pride/archive/projects/PXD047546


## Data Overview

The actual input data for this project consists of raw LC-MS/MS mass spectrum files (`.raw`) and their derived chromatogram files (`.mzML` / `.mgf`).  
The remaining `.txt` / `.fasta` files are used for grouping information, validation, or comparison with the results of the original paper, but are not used as direct inputs for the self-supervised model.

The dataset contains 10 raw LC-MS/MS files in total:
- 5 cirrhosis runs
- 5 HCC runs
Each raw file corresponds to one LC-MS/MS run from one serum sample (i.e., one patient-level sample), has around 2 to 5 GB.

In earlier prototype testing, a small subset of files was used for debugging and pipeline validation.  
In the current representation-learning setting, all available raw files are used for embedding-space analysis.

Important distinction:

- **One raw file = one biological sample (one patient serum sample)**
- **One raw file contains thousands to hundreds of thousands of MS/MS spectra**
- **Each MS/MS spectrum (BEGIN IONS ... END IONS in MGF) is treated as a spectrum-level training instance for self-supervised learning**


---

## Core Input


### 1. `.raw` Files (Raw Data Source)

**Cirrhotic samples:**

- 09062023_Mehta_GR10000524_DDRC_Sample1_480_cirrhotic.raw  
- 09062023_Mehta_GR10000524_DDRC_Sample2_491_cirrhotic.raw  
- 09062023_Mehta_GR10000524_DDRC_Sample3_554_cirrhotic.raw  
- 09062023_Mehta_GR10000524_DDRC_Sample4_561_cirrhotic.raw  
- 09062023_Mehta_GR10000524_DDRC_Sample5_654_cirrhotic.raw  

**HCC samples:**

- 09062023_Mehta_GR10000524_DDRC_Sample6_0121_HCC.raw  
- 09062023_Mehta_GR10000524_DDRC_Sample7_0187_HCC.raw  
- 09062023_Mehta_GR10000524_DDRC_Sample8_0203_HCC.raw  
- 09062023_Mehta_GR10000524_DDRC_Sample9_0206_HCC.raw  
- 09062023_Mehta_GR10000524_DDRC_Sample10_0543_HCC.raw  

**Description**
- Raw LC-MS/MS data files generated directly by the mass spectrometer
- Contain the complete acquisition output of an LC-MS/MS run

**Content**
- Full MS1 scans (precursor ion spectra)
- MS/MS (MS2) scans acquired at specific retention times
- Instrument metadata, including:
  - Scan numbers and scan types
  - Retention time for each scan

#### Biological Implications
- LC-MS/MS data from serum sources  
- Two groups:
  - **cirrhotic** (control)  
  - **HCC** (hepatocellular carcinoma + cirrhotic background)  

#### Sample Structure Clarification

In this project:

- A **sample** refers to a single serum LC-MS/MS run.
- Each `.raw` file corresponds to one sample (one patient).
- During acquisition, the instrument performs repeated scans over time, producing many MS/MS spectra from the same sample.

Therefore:

- disease-level groups (HCC vs cirrhosis) exist at the **sample level**
- model training operates at the **spectrum level**

---

### 2. `.mzML` (Intermediate Format)

**Conversion:**
.raw → .mzML using MSConvert

**Content**
- Complete MS1 and MS2 spectra
- For each spectrum:
  - m/z array
  - intensity array
  - retention time
  - precursor information (for MS2 scans)
- Explicit separation of MS1 and MS2 scans

**Role:**
- Open format  
- Retains complete MS1 / MS2 information  

**Role in project:**
- Intermediate conversion format  
- Optional input (if you prefer to read directly from mzML)  


---

### 3. `.mgf` (Direct Input for the Self-supervised Model)

**Content Format**
- BEGIN IONS
- PEPMASS=...
- CHARGE=...
- RTINSECONDS=...
- m/z intensity
- m/z intensity
- ...
- END IONS

In this project, the input to the self-supervised model is MS/MS spectra in MGF format, with each `BEGIN IONS … END IONS` block is treated as a spectrum-level training instance.


#### Data Format Seen by the Model

- A spectrum = a set of `(m/z, intensity)` pairs  
- Can be regarded as: **Sequence / Set / Sparse / Signal**



## Example dataset for using the tool (HW 4)

To demonstrate how the MS-Spectral-Foundation pipeline operates, a small example dataset derived from the PRIDE project **PXD047546** is used.

### Source and preprocessing

The example spectra originate from raw LC-MS/MS serum data comparing **cirrhosis** and **hepatocellular carcinoma (HCC)** samples.  
The preprocessing workflow follows the data preparation pipeline implemented in this project:

1. Raw vendor files (`.raw`) are converted to open format using **ProteoWizard (msconvert)**  
   `.raw → .mzML`

2. MS/MS spectra are extracted from `.mzML` files using **pyteomics**

3. MS2 spectra are serialized into **MGF format**, which serves as the direct input to the self-supervised learning model.

Example files used in the demonstration pipeline:

- 09062023_Mehta_GR10000524_DDRC_Sample4_561_cirrhotic_output.mgf
- 09062023_Mehta_GR10000524_DDRC_Sample9_0206_HCC_output.mgf


### Dataset structure

Each `.mgf` file contains many MS/MS spectra, where each spectrum is represented by a block:

- BEGIN IONS
- PEPMASS=...
- CHARGE=...
- m/z intensity
- m/z intensity
- ...
- END IONS


Each block corresponds to a **single MS/MS spectrum**, which becomes one training instance for the representation learning model.

Typical statistics of the example data:

- File size (MGF): approximately **300–370 MB per sample**
- Number of spectra per file: typically **tens of thousands of MS/MS spectra**
- Each spectrum contains **m/z–intensity peak pairs**
- Peak lists are later filtered and normalized during preprocessing.

### Example analysis scenario

Two example samples are used to illustrate downstream embedding analysis:

| Sample | Condition |
|------|------|
| Sample4_561 | Cirrhosis |
| Sample9_0206 | HCC |

After training the self-supervised spectrum encoder, embeddings are extracted from both datasets and compared using the embedding analysis pipeline. This allows exploration of whether the learned spectral representations capture systematic molecular differences between cirrhosis and HCC samples.

### Why this is a good example dataset

This dataset is suitable for demonstrating the tool for several reasons:

- It contains **real LC-MS/MS spectra from clinical serum samples**, representing realistic mass spectrometry data.
- The dataset includes **two biologically meaningful groups (HCC vs cirrhosis)**, enabling representation comparison in downstream embedding analysis.
- Each file contains **many spectra**, allowing the self-supervised model to learn meaningful spectral patterns.
- The data follows the **full preprocessing pipeline** used by the software (`raw → mzML → mgf`), demonstrating how the tool integrates with standard proteomics workflows.

### Repository example data

Due to the large size of full LC-MS/MS datasets (several GB per raw file, and 300 MB after converted to mgf), only a **small demonstration subset** can be included in the repository.  
If an example file smaller than 20 MB is provided, it can be placed in:

 - data/example_spectra.mgf


This file allows users to quickly test the pipeline without downloading the full dataset.
