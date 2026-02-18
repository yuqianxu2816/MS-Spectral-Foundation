"""
Module 2: Peak filtering and preprocessing
"""

from __future__ import annotations

from typing import Callable, Optional, Literal
import numpy as np

Unit = Literal["Da", "ppm"]
ScaleMethod = Optional[Literal["root", "log", "rank", "none"]]


def _to_array(peaks) -> np.ndarray:
    # Convert peaks to float array of shape (N,2)，pairs of (mz, intensity)
    if peaks is None:
        return np.zeros((0, 2), dtype=float)
    arr = np.asarray(peaks, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("peaks must have shape (N,2): [[mz,intensity], ...]")
    return arr

def set_mz_range(min_mz: Optional[float] = None, max_mz: Optional[float] = None) -> Callable[[np.ndarray], np.ndarray]:
    # Restrict m/z values to [min_mz, max_mz] (inclusive).
    # if both None: no-op
    # if one side None: infer from data
    # if max < min: swap
    def _fn(peaks: np.ndarray) -> np.ndarray:
        p = _to_array(peaks)
        if p.shape[0] == 0:
            return p

        mz = p[:, 0]
        lo = float(np.min(mz)) if min_mz is None else float(min_mz)
        hi = float(np.max(mz)) if max_mz is None else float(max_mz)
        if hi < lo:
            lo, hi = hi, lo

        keep = (mz >= lo) & (mz <= hi)   # inclusive
        return p[keep]

    return _fn


def remove_precursor_peak(tol: float, unit: Unit = "Da") -> Callable[[np.ndarray, Optional[float]], np.ndarray]:
    # Remove peaks around precursor_mz within tolerance.
    
    if unit not in ("Da", "ppm"):
        raise ValueError("unit must be 'Da' or 'ppm'")

    def _fn(peaks: np.ndarray, precursor_mz: Optional[float] = None) -> np.ndarray:
        p = _to_array(peaks)
        if p.shape[0] == 0:
            return p

        # If precursor_mz is missing -> keep all peaks
        if precursor_mz is None or (isinstance(precursor_mz, float) and np.isnan(precursor_mz)):
            return p 

        precursor_mz = float(precursor_mz)
        mz = p[:, 0]

        # unit="Da": |mz - precursor| <= tol
        # unit="ppm": |mz - precursor| <= precursor * tol * 1e-6
        if unit == "Da":
            thr = tol
        else:
            thr = precursor_mz * float(tol) * 1e-6
        keep = np.abs(mz - precursor_mz) > float(thr)

        return p[keep]

    return _fn


def scale_intensity(
    scaling: ScaleMethod = None,
    max_intensity: Optional[float] = None,
    *, # keyword-only parameter： degree, base, max_rank
    degree: int = 2,
    base: int = 2,
    max_rank: Optional[int] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Normalize intensities relative to the most intense peak to a specified maximum value (optional).
    Scale peak intensities using various methods: root, log, or rank transformation.
    """
    def _fn(peaks: np.ndarray) -> np.ndarray:
        p = _to_array(peaks).copy()
        if p.shape[0] == 0:
            return p

        inten = p[:, 1].astype(float)
        inten = np.maximum(inten, 0.0) # negative intensities are set to 0

        # scale relative to most intense peak (optional)
        if max_intensity is not None:
            mx = float(np.max(inten)) if inten.size > 0 else 0.0
            if inten.size > 0:
                mx = float(np.max(inten))
            else:
                mx = 0.0

            if mx > 0:
                # ex: mx=100, max_intensity=10 
                # -> scale factor = 10/100 = 0.1 -> all intensities are scaled down by 0.1
                inten = inten / mx * float(max_intensity)

        if scaling is None or scaling == "none":
            p[:, 1] = inten
            return p

        if scaling == "root":
            if int(degree) > 0:
                d = int(degree)
            else:
                d = 2
            p[:, 1] = np.power(inten, 1.0 / d)
            return p

        if scaling == "log":
            if float(base) > 1:
                b = float(base)
            else:
                b = 2.0
            p[:, 1] = np.log(inten + 1.0) / np.log(b)
            return p

        if scaling == "rank":
            # rank-transform: highest intensity gets max_rank
            n = inten.size
            if n == 0: # no peaks, return unchanged
                p[:, 1] = inten
                return p
            if max_rank is None:
                mr = n # max_rank unspecified: max_rank = num peaks
            else:
                mr = int(max_rank) # max_rank specified
            if mr < n:
                raise ValueError("max_rank should be >= number of peaks")
            """
            Ex: inten = [50, 200, 100], n=3, max_rank=None -> mr=3

            order = [0, 2, 1] 
            base_r = [1, 2, 3] 

            ranks[0] = 1 * (3/3) = 1 
            ranks[2] = 2 * (3/3) = 2
            ranks[1] = 3 * (3/3) = 3

            p = [1, 3, 2]
            """
            order = np.argsort(inten)  # sort low to high, smallest -> 1, largest -> mr
            ranks = np.empty_like(inten, dtype=float)
            base_r = np.arange(1, n + 1, dtype=float) # map positions to [1, 2, 3, ..., n]
            ranks[order] = base_r * (float(mr) / float(n)) # then scale to [1...mr]
            p[:, 1] = ranks
            return p

        raise ValueError("scaling must be one of: None, 'none', 'root', 'log', 'rank'")

    return _fn


def filter_intensity(min_intensity: float = 0.0, max_peaks: Optional[int] = None) -> Callable[[np.ndarray], np.ndarray]:
    """
    Filter out low-intensity peaks using a relative threshold (min_intensity * max_intensity).
    Keep only the relative intense peaks and top max_peaks, then sort by m/z
    """
    def _fn(peaks: np.ndarray) -> np.ndarray:
        p = _to_array(peaks)
        if p.shape[0] == 0:
            return p

        # remove zero/negative
        p = p[p[:, 1] > 0.0]
        if p.shape[0] == 0:
            return p

        inten = p[:, 1]
        mx = float(np.max(inten))
        thr = float(min_intensity) * mx  # set relative threshold

        # remove peaks with intensity <= thr
        p = p[inten > thr]
        if p.shape[0] == 0:
            return p

        # keep top max_peaks
        if max_peaks is not None and p.shape[0] > int(max_peaks):
            idx = np.argsort(p[:, 1])[::-1][: int(max_peaks)]
            p = p[idx]

        # sort by mz
        p = p[np.argsort(p[:, 0])]
        return p

    return _fn


def discard_low_quality(min_peaks: int) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    """
    If remaining peaks < min_peaks, discard the spectrum
    """
    def _fn(peaks: np.ndarray) -> Optional[np.ndarray]:
        p = _to_array(peaks)
        if p.shape[0] < int(min_peaks):
            return None
        return p
    return _fn

def _scale_to_unit_norm(peaks: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize intensity vector to unit norm.
    If norm=0 -> return unchanged (all zero intensities).

    Input: peaks = [[100, 3], [200, 4]]

    1. inten = [3, 4]
    2. norm = √(3² + 4²) = √(9 + 16) = √25 = 5
    3. [3/5, 4/5] = [0.6, 0.8]

    return: [[100, 0.6], [200, 0.8]]
    """
    p = _to_array(peaks).copy() # avoid modifying original array
    if p.shape[0] == 0:
        return p
    inten = np.maximum(p[:, 1], 0.0)
    norm = np.sqrt(np.sum(inten * inten))
    if norm <= eps:
        p[:, 1] = inten # all zero intensities, return unchanged
        return p
    p[:, 1] = inten / norm
    return p

def apply_preprocessing_pipeline(
    peaks: np.ndarray,
    pipeline: list[Callable],
    precursor_mz: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Apply the list of preprocessing functions in order.
    """
    x: Optional[np.ndarray] = peaks
    for fn in pipeline:
        if x is None:
            return None

        # Support functions like remove_precursor_peak that require precursor_mz
        try:
            x = fn(x, precursor_mz) 
        except TypeError: # function does not accept precursor_mz, call with just peaks
            x = fn(x) 
    return x




import os
import numpy as np
from .mgf_parse import load_spectra_npz, save_spectra_npz

IN_NPZ  = "output/spectra_raw.npz"
OUT_NPZ = "output/spectra_filtered.npz"


MIN_MZ = 100.0
MAX_MZ = 2000.0
MIN_INTENSITY = 1e-12    
MAX_PEAKS = 200
MIN_PEAKS = 5
REMOVE_PRECURSOR_TOL = 1.5
PRECURSOR_UNIT = "Da"  

def main():

    spectra = load_spectra_npz(IN_NPZ)
    print(f"[peak_filter] loaded spectra: {len(spectra)} from {IN_NPZ}")

    # Output the first 100 lines to a txt file
    out_txt = "output/spectra_first100.txt"
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        for i, sp in enumerate(spectra[:100]):
            f.write(f"# Spectrum {i+1}\n")
            f.write(str(sp) + "\n\n")
    print(f"[peak_filter] wrote first 100 spectra to {out_txt}")


    # Automatically calculate parameters
    all_mz = np.concatenate([np.array(sp["peaks"], dtype=float)[:,0] for sp in spectra if len(sp["peaks"]) > 0])
    all_intensity = np.concatenate([np.array(sp["peaks"], dtype=float)[:,1] for sp in spectra if len(sp["peaks"]) > 0])
    peak_counts = np.array([len(sp["peaks"]) for sp in spectra])

    MIN_MZ = float(np.min(all_mz))
    MAX_MZ = float(np.max(all_mz))
    # MIN_INTENSITY = float(np.min(all_intensity[all_intensity > 0])) if np.any(all_intensity > 0) else 1e-12
    MIN_INTENSITY = 0.01   # Retain peaks with intensity > 1% of max peak
    MAX_PEAKS = int(np.median(peak_counts))
    MIN_PEAKS = int(np.min(peak_counts))
    REMOVE_PRECURSOR_TOL = 1.5  # Still use the default
    PRECURSOR_UNIT = "Da"       # Still use the default

    print(f"[peak_filter] auto params: MIN_MZ={MIN_MZ}, MAX_MZ={MAX_MZ}, MIN_INTENSITY={MIN_INTENSITY}, MAX_PEAKS={MAX_PEAKS}, MIN_PEAKS={MIN_PEAKS}")

    pipeline = [
        set_mz_range(MIN_MZ, MAX_MZ),
        remove_precursor_peak(REMOVE_PRECURSOR_TOL, PRECURSOR_UNIT),
        scale_intensity("root", 1),
        filter_intensity(MIN_INTENSITY, MAX_PEAKS),
        discard_low_quality(MIN_PEAKS),
        _scale_to_unit_norm,
    ]

    filtered = []
    dropped = 0

    for sp in spectra:
        precursor_mz = sp["meta"].get("PEPMASS", None)
        peaks = np.array(sp["peaks"], dtype=float)

        out = apply_preprocessing_pipeline(peaks, pipeline, precursor_mz=precursor_mz)
        if out is None:
            dropped += 1
            continue

        sp2 = {
            "meta": sp["meta"],
            "peaks": list(map(tuple, out.tolist())),  # list[(mz,inten)]
        }
        filtered.append(sp2)


    print(f"[peak_filter] kept={len(filtered)} dropped={dropped}")

    # Output the spectrograms after the first 100 filters to a txt file
    out_txt2 = "output/spectra_filtered_first100.txt"
    os.makedirs(os.path.dirname(out_txt2) or ".", exist_ok=True)
    with open(out_txt2, "w", encoding="utf-8") as f:
        for i, sp in enumerate(filtered[:100]):
            f.write(f"# Filtered Spectrum {i+1}\n")
            f.write(str(sp) + "\n\n")
    print(f"[peak_filter] wrote first 100 filtered spectra to {out_txt2}")

    save_spectra_npz(filtered, OUT_NPZ)
    print(f"[peak_filter] saved -> {OUT_NPZ}")

    s2 = load_spectra_npz(OUT_NPZ)
    print(f"[peak_filter] reload check: {len(s2)} spectra")
    if len(s2) > 0:
        print("[peak_filter] spectrum[0].meta =", s2[0]["meta"])
        print("[peak_filter] spectrum[0].num_peaks =", len(s2[0]["peaks"]))

if __name__ == "__main__":
    main()
