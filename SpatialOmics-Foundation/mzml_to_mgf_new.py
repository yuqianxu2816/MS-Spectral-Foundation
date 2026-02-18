"""
Module 0: Convert mzML to MGF
"""

from pyteomics import mzml, mgf

spectra = []
with mzml.read("09062023_Mehta_GR10000524_DDRC_Sample4_561_cirrhotic.mzML") as reader:
    for sp in reader:
        if sp.get("ms level") != 2:
            continue
        ion = sp["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]
        pepmass = ion.get("selected ion m/z")
        charge = ion.get("charge state")

        params = {}
        if pepmass is not None:
            params["pepmass"] = pepmass
        if charge is not None:
            params["charge"] = f"{int(charge)}+"

        spectra.append({"m/z array": sp["m/z array"], "intensity array": sp["intensity array"], "params": params})

mgf.write(spectra, "09062023_Mehta_GR10000524_DDRC_Sample4_561_cirrhotic_output.mgf")
