"""
MS-Spectral-Foundation: Self-Supervised Learning for Mass Spectrometry Data

Complete Module Pipeline:
- Module 0: Data Preparation (FTP → mzML → MGF)
- Module 1: MGF Parsing (mgf_parse.py)
- Module 2: Peak Filtering (peak_filter.py)
- Module 3: m/z Binning (bin_mz.py)
- Module 4-6: Self-Supervised Training (model_ssl.py, train.py)
- Module 7-8: Embedding Analysis (analyze_embeddings.py)
"""

__version__ = "1.0.0"

# Core modules (1-3)
from .mgf_parse import parse_mgf, Spectrum, save_spectra_npz, load_spectra_npz
from .peak_filter import (
    set_mz_range,
    remove_precursor_peak,
    scale_intensity,
    filter_intensity,
    discard_low_quality,
    apply_preprocessing_pipeline
)
from .bin_mz import bin_mz, bin_mz_tensor

# Dataset and model (4-6)
from .spectrum_dataset import SpectrumDataset, SpectrumDataModule
from .model_ssl import SpectrumSSLv2

# Analysis modules (7-8)
from .analyze_embeddings import EmbeddingAnalyzer, create_sample_metadata
from .metadata_utils import (
    parse_metadata_from_mgf,
    load_metadata_from_csv,
    prepare_analysis_data,
    extract_sample_id_from_title,
    extract_disease_group_from_title
)

__all__ = [
    # Module 1
    "parse_mgf",
    "Spectrum",
    "save_spectra_npz",
    "load_spectra_npz",
    
    # Module 2
    "set_mz_range",
    "remove_precursor_peak",
    "scale_intensity",
    "filter_intensity",
    "discard_low_quality",
    "apply_preprocessing_pipeline",
    
    # Module 3
    "bin_mz",
    "bin_mz_tensor",
    
    # Module 4-6
    "SpectrumDataset",
    "SpectrumDataModule",
    "SpectrumSSLv2",
    
    # Module 7-8
    "EmbeddingAnalyzer",
    "create_sample_metadata",
    "parse_metadata_from_mgf",
    "load_metadata_from_csv",
    "prepare_analysis_data",
    "extract_sample_id_from_title",
    "extract_disease_group_from_title",
]

