"""
Precompute the numerical sky-averaged antenna power of the LISA TDI channels.

The tables are saved to ``src/fast_lisa_subtraction/simulation/data/`` and are
read (and interpolated) by ``fast_lisa_subtraction.simulation.response.response``
when ``method="numerical"`` (the default). Rerun this script if the response
computation in ``response.py`` changes.

Usage:
    python scripts/generate_response_tables.py
"""

import os
import numpy as np

from tqdm import tqdm

from fast_lisa_subtraction.simulation.response import sky_averaged_antenna_power

CHANNELS = ["X", "Y", "Z", "A", "E", "T"]
OUTFILE = os.path.join(os.path.dirname(__file__), "..", "src", "fast_lisa_subtraction",
                       "simulation", "data", "sky_averaged_antenna_power.npz")

#log-spaced at low frequency (where the antenna power is smooth) and
#linearly spaced above 10 mHz to resolve the ~c/(4L) = 30 mHz oscillations
f_low = np.logspace(-8, -2, 200, endpoint=False)
f_high = np.arange(1e-2, 2.0, 4e-4)
freqs = np.concatenate([f_low, f_high])
print(f"Calculating sky-averaged antenna power for {len(freqs)} frequencies from {freqs[0]:.1e} Hz to {freqs[-1]:.1e} Hz")

if __name__ == "__main__":
    tables = {"f": freqs}
    chunk = 200  # frequencies per call: each chunk auto-sizes its quadrature grid to its own fmax
    with tqdm(total=len(CHANNELS)*len(freqs), desc="Generating response tables", unit="freq") as pbar:
        for ch in CHANNELS:
            pbar.set_description(f"Computing response for channel {ch}")
            values = []
            for i in range(0, len(freqs), chunk):
                values.append(sky_averaged_antenna_power(freqs[i:i+chunk], channel=ch, integrator="leggauss"))
                pbar.update(len(values[-1]))
            tables[ch] = np.concatenate(values)

    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    np.savez_compressed(OUTFILE, **tables)
    print(f"saved {os.path.abspath(OUTFILE)}")
