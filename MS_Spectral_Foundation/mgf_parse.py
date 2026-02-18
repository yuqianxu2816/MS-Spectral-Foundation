"""
Module 1: Read the MGF file
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

Spectrum = Dict[str, Any]  # {"meta": {...}, "peaks": [(mz, intensity), ...]}

def _parse_charge(val: str) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    tok = s.replace(",", " ").split()[0]
    # take first token before whitespace or comma
    # 2+ -> 2, 3- -> 3
    tok = tok.replace("+", "").replace("-", "")
    if tok.isdigit():
        return int(tok)
    else:
        return None

def _parse_pepmass(val: str) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # take first token: "512.34" -> 512.34, "512.34 9999" -> 512.34
    first = s.split()[0]
    try:
        return float(first)
    except ValueError:
        return None

def parse_mgf(path: str) -> List[Spectrum]:
    spectra: List[Spectrum] = []
    in_block = False
    meta: Dict[str, Any] = {}
    peaks: List[Tuple[float, float]] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.upper() == "BEGIN IONS":
                in_block = True
                meta = {}
                peaks = []
                continue

            if line.upper() == "END IONS":
                if in_block:
                    meta_data = {
                        "PEPMASS": _parse_pepmass(meta.get("PEPMASS")),
                        "CHARGE": _parse_charge(meta.get("CHARGE")),
                        "RTINSECONDS": float(meta["RTINSECONDS"]) if "RTINSECONDS" in meta else None,
                    }
                    spectra.append({"meta": meta_data, "peaks": peaks})
                in_block = False
                continue

            if not in_block:
                continue

            # metadata line: KEY=VALUE
            if "=" in line and not line[0].isdigit():
                k, v = line.split("=", 1)
                # CHARGE=2+ -> meta["CHARGE"] = "2+"
                meta[k.strip().upper()] = v.strip()
                continue

            # peak line: "mz intensity"
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mz = float(parts[0])
                    inten = float(parts[1])
                    peaks.append((mz, inten))
                except ValueError:
                    # ignore malformed peak line
                    pass

    return spectra





import os
import numpy as np

MGF_PATH = "data/example.mgf"
OUT_NPZ  = "output/spectra_raw.npz"

def save_spectra_npz(spectra, out_path: str):
    """
    Save spectra as NPZ:
      - meta_pepmass
      - meta_charge
      - meta_rt
      - peaks_mz
      - peaks_int
    """
    S = len(spectra)
    meta_pepmass = np.full((S,), np.nan, dtype=float)
    meta_charge  = np.full((S,), -1, dtype=int)
    meta_rt      = np.full((S,), np.nan, dtype=float)

    peaks_mz  = np.empty((S,), dtype=object)
    peaks_int = np.empty((S,), dtype=object)

    for i, sp in enumerate(spectra):
        meta = sp.get("meta", {})
        meta_pepmass[i] = meta.get("PEPMASS") if meta.get("PEPMASS") is not None else np.nan
        meta_charge[i]  = meta.get("CHARGE")  if meta.get("CHARGE")  is not None else -1
        meta_rt[i]      = meta.get("RTINSECONDS") if meta.get("RTINSECONDS") is not None else np.nan

        p = np.asarray(sp.get("peaks", []), dtype=float)
        if p.size == 0:
            peaks_mz[i]  = np.zeros((0,), dtype=float)
            peaks_int[i] = np.zeros((0,), dtype=float)
        else:
            peaks_mz[i]  = p[:, 0].astype(float)
            peaks_int[i] = p[:, 1].astype(float)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        meta_pepmass=meta_pepmass,
        meta_charge=meta_charge,
        meta_rt=meta_rt,
        peaks_mz=peaks_mz,
        peaks_int=peaks_int,
    )

def load_spectra_npz(npz_path: str):
    """
    Load NPZ and reconstruct spectra list in the same format:
    Spectrum = {"meta": {...}, "peaks": [(mz,inten), ...]}
    """
    d = np.load(npz_path, allow_pickle=True)
    meta_pepmass = d["meta_pepmass"]
    meta_charge  = d["meta_charge"]
    meta_rt      = d["meta_rt"]
    peaks_mz     = d["peaks_mz"]
    peaks_int    = d["peaks_int"]

    spectra = []
    for i in range(len(meta_pepmass)):
        pep = None if np.isnan(meta_pepmass[i]) else float(meta_pepmass[i])
        ch  = None if meta_charge[i] < 0 else int(meta_charge[i])
        rt  = None if np.isnan(meta_rt[i]) else float(meta_rt[i])

        mz_arr  = peaks_mz[i]
        int_arr = peaks_int[i]
        peaks = list(zip(mz_arr.tolist(), int_arr.tolist()))

        spectra.append({"meta": {"PEPMASS": pep, "CHARGE": ch, "RTINSECONDS": rt},
                        "peaks": peaks})
    return spectra

def main():
    spectra = parse_mgf("C:/Users/Lenovo/Desktop/576dataset/09062023_Mehta_GR10000524_DDRC_Sample4_561_cirrhotic_output.mgf")

    print(f"[mgf_parse] parsed spectra: {len(spectra)}")

    if len(spectra) > 0:
        print("[mgf_parse] spectrum[0].meta =", spectra[0]["meta"])
        print("[mgf_parse] spectrum[0].num_peaks =", len(spectra[0]["peaks"]))

    save_spectra_npz(spectra, OUT_NPZ)
    print(f"[mgf_parse] saved -> {OUT_NPZ}")

    s2 = load_spectra_npz(OUT_NPZ)
    print(f"[mgf_parse] reload check: {len(s2)} spectra, first num_peaks={len(s2[0]['peaks']) if len(s2)>0 else 0}")

if __name__ == "__main__":
    main()
