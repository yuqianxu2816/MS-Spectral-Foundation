"""
Utility functions for handling real metadata in embedding analysis.

This script provides tools to:
1. Extract sample information from MGF files
2. Load metadata from CSV files
3. Map spectra to disease groups
4. Prepare data for Module 7 & 8 analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import re


def extract_sample_id_from_title(title: str, pattern: str = None) -> str:
    """
    Extract sample ID from MGF TITLE field.
    
    Args:
        title: TITLE field from MGF spectrum
        pattern: Optional regex pattern to extract sample ID
                 If None, uses common patterns
    
    Returns:
        Extracted sample ID
    
    Examples:
        >>> extract_sample_id_from_title("Sample_HCC_001_Scan_12345")
        'Sample_HCC_001'
        
        >>> extract_sample_id_from_title("20230906_HCC_Patient1_MS2")
        '20230906_HCC_Patient1'
    """
    if pattern is None:
        # Try common patterns
        
        # Pattern 1: Sample_XXX_YYY
        match = re.search(r'(Sample_[^_]+_\d+)', title)
        if match:
            return match.group(1)
        
        # Pattern 2: Date_Group_PatientID
        match = re.search(r'(\d{8}_[^_]+_[^_]+)', title)
        if match:
            return match.group(1)
        
        # Pattern 3: Extract first N characters
        return title[:50]  # Fallback: use first 50 chars
    
    else:
        match = re.search(pattern, title)
        if match:
            return match.group(1) if match.groups() else match.group(0)
        else:
            return title


def extract_disease_group_from_title(title: str) -> str:
    """
    Extract disease group from MGF TITLE field.
    
    Args:
        title: TITLE field from MGF spectrum
    
    Returns:
        Disease group label (e.g., "HCC", "Cirrhosis", "Healthy")
    
    Examples:
        >>> extract_disease_group_from_title("Sample_HCC_001")
        'HCC'
        
        >>> extract_disease_group_from_title("20230906_Cirrhosis_Patient1")
        'Cirrhosis'
    """
    title_upper = title.upper()
    
    # Common disease keywords
    disease_keywords = {
        'HCC': ['HCC', 'HEPATOCELLULAR', 'CARCINOMA'],
        'Cirrhosis': ['CIRRHOSIS', 'CIRRHOTIC', 'FIBROSIS'],
        'Healthy': ['HEALTHY', 'NORMAL', 'CONTROL'],
        'Tumor': ['TUMOR', 'CANCER'],
        'Benign': ['BENIGN'],
    }
    
    for disease, keywords in disease_keywords.items():
        for keyword in keywords:
            if keyword in title_upper:
                return disease
    
    return "Unknown"


def parse_metadata_from_mgf(mgf_path: str) -> pd.DataFrame:
    """
    Parse metadata directly from MGF file.
    
    Extracts:
    - Spectrum index
    - TITLE field
    - Sample ID (extracted from TITLE)
    - Disease group (extracted from TITLE)
    - PEPMASS (precursor m/z)
    - CHARGE
    - RTINSECONDS (retention time)

    Returns dataFrame with metadata for each spectrum
    """
    metadata = []
    spectrum_idx = 0
    
    with open(mgf_path, 'r') as f:
        in_spectrum = False
        current_meta = {}
        
        for line in f:
            line = line.strip()
            
            if line == "BEGIN IONS":
                in_spectrum = True
                current_meta = {"spectrum_idx": spectrum_idx}
            
            elif line == "END IONS":
                in_spectrum = False
                
                # Extract sample ID and disease group from TITLE
                title = current_meta.get("TITLE", "")
                current_meta["sample_id"] = extract_sample_id_from_title(title)
                current_meta["disease_group"] = extract_disease_group_from_title(title)
                
                metadata.append(current_meta)
                spectrum_idx += 1
                current_meta = {}
            
            elif in_spectrum and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                # Store key metadata fields
                if key in ["TITLE", "PEPMASS", "CHARGE", "RTINSECONDS", "SCANS"]:
                    try:
                        if key == "PEPMASS":
                            # PEPMASS might be "mz intensity"
                            value = float(value.split()[0])
                        elif key in ["CHARGE", "SCANS"]:
                            value = int(value.rstrip('+'))
                        elif key == "RTINSECONDS":
                            value = float(value)
                    except (ValueError, IndexError):
                        pass
                    
                    current_meta[key] = value
    
    df = pd.DataFrame(metadata)
    
    print(f"   Parsed metadata from {mgf_path}")
    print(f"   Total spectra: {len(df)}")
    print(f"   Unique samples: {df['sample_id'].nunique()}")
    print(f"   Disease groups: {df['disease_group'].value_counts().to_dict()}")
    
    return df


def load_metadata_from_csv(
    csv_path: str,
    sample_id_col: str = "sample_id",
    group_col: str = "disease_group"
) -> pd.DataFrame:
    """
    Load sample metadata from external CSV file.
    
    Expected CSV format:
        sample_id,disease_group,other_columns...
        Sample_001,HCC,...
        Sample_002,Cirrhosis,...
        ...
    
    Args:
        csv_path: Path to CSV file
        sample_id_col: Column name for sample IDs
        group_col: Column name for disease groups
    
    Returns:
        DataFrame with metadata
    """
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    if sample_id_col not in df.columns:
        raise ValueError(f"Column '{sample_id_col}' not found in CSV")
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in CSV")
    
    print(f"   Loaded metadata from {csv_path}")
    print(f"   Total samples: {len(df)}")
    print(f"   Disease groups: {df[group_col].value_counts().to_dict()}")
    
    return df


def map_labels_to_numeric(
    labels: np.ndarray,
    label_mapping: Dict[str, int] = None
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Convert string labels to numeric labels.
    
    Args:
        labels: Array of string labels
        label_mapping: Optional custom mapping {label: int}
                      If None, creates automatic mapping
    
    Returns:
        - numeric_labels: Array of integer labels
        - label_names: Dict mapping int -> str
    """
    unique_labels = np.unique(labels)
    
    if label_mapping is None:
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
    
    numeric_labels = np.array([label_mapping[label] for label in labels])
    label_names = {v: k for k, v in label_mapping.items()}
    
    return numeric_labels, label_names


def prepare_analysis_data(
    embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
    sample_id_col: str = "sample_id",
    group_col: str = "disease_group"
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Prepare data for embedding analysis.
    
    Returns:
        - sample_ids: Array of sample IDs (length N)
        - numeric_labels: Array of numeric group labels (length N)
        - label_names: Dict mapping int -> disease group name
    """
    # Extract data from DataFrame
    sample_ids = metadata_df[sample_id_col].values
    group_labels = metadata_df[group_col].values
    
    # Convert to numeric
    numeric_labels, label_names = map_labels_to_numeric(group_labels)
    
    print(f"   Prepared analysis data:")
    print(f"   Embeddings: {embeddings.shape}")
    print(f"   Samples: {len(np.unique(sample_ids))} unique")
    print(f"   Groups: {label_names}")
    
    return sample_ids, numeric_labels, label_names

