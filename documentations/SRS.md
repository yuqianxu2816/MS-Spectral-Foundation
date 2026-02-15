# Software Requirements Specification (SRS)

## 1. Overview

This project aims to develop a deep learning method for mass spectrometry representation based on self-supervised masked learning, utilizing serum mass spectrometry data from hepatocellular carcinoma (HCC) and cirrhosis. The learned representations are used for downstream embedding-space analysis to examine systematic molecular differences between HCC and cirrhosis at the sample level. Due to the limited cohort size, the software emphasizes representation analysis rather than supervised disease classification.

Biological Motivation: This work is inspired by spatial omics studies of HCC, where spatially resolved glycan and proteomic measurements reveal strong molecular heterogeneity across tumor tissues. Rather than directly modeling spatial imaging data, this project focuses on learning transferable representations from derived MS/MS spectra as a computational abstraction of spatially driven molecular variation.

Unlike traditional supervised machine learning approaches that rely on manual feature engineering and explicit labels (e.g., HCC vs. cirrhosis), this project first learns general-purpose representations (spectral embeddings) of MS/MS spectra through self-supervision. These representations are then applied to downstream representation analysis tasks, with the goal of exploring whether robust and transferable molecular structure can be revealed from small-sample, highly heterogeneous glycoproteomics data.


### 1.1 Input

The software accepts the following data formats:

- **`.raw`**  
  Raw LC-MS/MS files from the instrument.

- **`.mzML`**  
  Converted open-format spectra files used for preprocessing.

- **`.mgf`**  
  MS/MS spectra used as the direct model input for training and embedding extraction.


### 1.2 Output

The software generates:

- Processed spectra data
- Spectrum-level embeddings
- Sample-level embeddings
- Embedding-space analysis results
  - Distribution differences and exemplar spectra between HCC and cirrhosis
- Basic embedding visualization results


---

## 2. Intended Use and Use Cases

### 2.1 Intended Use

The software is intended for researchers working with LC-MS/MS proteomics or glycoproteomics data who wish to learn general-purpose spectral representations without relying on manual feature engineering or large labeled datasets.

### 2.2 Use Cases

Typical use cases include:

- Learning spectrum-level embeddings from MS/MS data using masked self-supervised learning
- Aggregating spectrum embeddings to represent biological samples
- Exploring whether learned embeddings capture disease-related molecular structure (e.g., HCC vs. cirrhosis) through embedding-space visualization and distribution analysis

---

## 3. Software Features

The software package provides the following core functionalities:

- Conversion and ingestion of mass spectrometry data in open formats (mzML and MGF)
- Extraction and preprocessing of MS/MS spectra as `(m/z, intensity)` pairs
- Self-supervised training of a masked reconstruction model to learn spectrum embeddings
- Generation of fixed-dimensional spectrum-level and sample-level embeddings
- Basic evaluation of learned embeddings using dimensionality reduction methods (e.g., PCA or UMAP)

### 3.1 Outputs

The output of the software includes:

- Processed MGF files
- Learned spectral embeddings
- Visualization plots for embedding evaluation

---

## 4. Data Access and Execution

### 4.1 Data Access

Input data are obtained from a public proteomics repository and provided as raw LC-MS/MS files. During development, a small representative subset of the data (e.g., one cirrhotic sample and one HCC sample) is used to enable rapid testing and iteration.

The software is designed to operate on these subsets during development and can later be applied to the full dataset for large-scale validation.

### 4.2 Execution Model

Users can run the provided Python scripts to perform:

- Data conversion
- Spectrum preprocessing
- Model training
- Embedding extraction

The software follows a modular design, allowing individual components (e.g., data conversion or embedding extraction) to be executed independently depending on the user’s needs.
