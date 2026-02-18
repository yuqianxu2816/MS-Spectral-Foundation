"""
Module 0: Convert mzML to MGF
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from pyteomics import mzml, mgf


def mzml_to_mgf(
    mzml_path: str,
    mgf_out_path: str,
    ms_level: int = 2
) -> int: 
    """
    Convert an mzML file to an MGF file by extracting spectra of a given MS level.

    Input:
    1. mzml_path: Path to input .mzML file
    2. mgf_out_path: Path to output .mgf file
    3. ms_level: MS level to extract (default: 2)

    Output:
    1. Number of spectra written to the output MGF.
    """
    spectra: List[Dict[str, Any]] = []

    with mzml.read(mzml_path) as reader:
        for sp in reader:
            if sp.get("ms level") != ms_level:
                continue

            # Extract precursor m/z (pepmass) if present
            pepmass: Optional[float] = None
            try:
                pepmass = (
                    sp["precursorList"]["precursor"][0]
                      ["selectedIonList"]["selectedIon"][0]
                      ["selected ion m/z"]
                )
            except Exception:
                pepmass = None

            spectra.append({
                "m/z array": sp["m/z array"],
                "intensity array": sp["intensity array"],
                "params": {"pepmass": pepmass} if pepmass is not None else {}
            })

    mgf.write(spectra, mgf_out_path)
    return len(spectra)


def main() -> None:
   
    n = mzml_to_mgf(
        "09062023_Mehta_GR10000524_DDRC_Sample4_561_cirrhotic.mzML",
        "output.mgf"
    )


if __name__ == "__main__":
    main()
