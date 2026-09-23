# Installation

Install the package with pip from the [GitHub repository](https://github.com/fdesanti/fast-lisa-subtraction),
in a **new** environment with Python 3.10–3.13
(e.g. `python3 -m venv ~/envs/lisa && source ~/envs/lisa/bin/activate`).

```{important}
**Install the right PyTorch build first.**

The default `pip install torch` from PyPI is built for CUDA 13. The GPU backends of GBGPU and
LISA Analysis Tools only exist for CUDA 12 and are incompatible with CUDA 13. Install PyTorch
**before** the package, with the command for your setup below.

Refer to the official [PyTorch](https://pytorch.org/get-started/locally/) page for more details.
```

## Quick installation

**CPU** only

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "fast-lisa-subtraction @ git+https://github.com/fdesanti/fast-lisa-subtraction.git"
```

**GPU** `[cuda12]`

```bash
pip install "torch==2.14.0+cu126" --index-url https://download.pytorch.org/whl/cu126
pip install "fast-lisa-subtraction[cuda12] @ git+https://github.com/fdesanti/fast-lisa-subtraction.git"
```

The `[cuda12]` extra adds the GPU backends: `cupy-cuda12x`, `gbgpu-cuda12x` and
`lisaanalysistools-cuda12x`.

*Note: The GPU installation works with the CPU as well.*

To install from a local clone instead, install PyTorch as above and then run:

```bash
git clone https://github.com/fdesanti/fast-lisa-subtraction.git
cd fast-lisa-subtraction
pip install .              # CPU only
pip install ".[cuda12]"    # GPU with CUDA 12
```

## Checking your CUDA version

For the GPU install, run `nvidia-smi` and check the CUDA version in its header (labelled
`CUDA Version`, or `CUDA UMD Version` on recent drivers). It must be 12.0 or higher; 13.x is fine,
since newer drivers also run CUDA 12 code. This corresponds to an NVIDIA driver 525.60.13 or newer.
The GPU must have compute capability 7.0–9.0 (Volta to Hopper, e.g. V100, RTX 20xx–40xx, A100,
H100): the CUDA 12 builds contain no GPU code for Blackwell (RTX 50xx, B100/B200).

## Checking the installation

```python
from fast_lisa_subtraction import GalacticBinaryPopulation, SourceCatalog

device = "cuda"   # or "cpu"
df  = GalacticBinaryPopulation(device=device).sample(10).dataframe()
cat = SourceCatalog(catalog_df=df, use_gpu=(device == "cuda"))
print(cat.GB.backend.name, cat.GB.orbits.backend.name)
```

This prints `gbgpu_cuda12x lisatools_cuda12x` with the GPU install and `gbgpu_cpu lisatools_cpu` on
the CPU. If a GPU install prints `gbgpu_cpu`, the GPU backend could not be loaded: the `[WARNING]`
printed just before says why.

## Prerequisites

- Linux x86_64.
- A C/C++ compiler and the GSL and FFTW3 libraries: the `lisa-data-challenge` dependency is compiled
  from source during the install. For example `sudo dnf install gcc-c++ gsl-devel fftw-devel`,
  `sudo apt install g++ libgsl-dev libfftw3-dev`, or `conda install -c conda-forge gsl fftw` in a
  conda environment.
- For the GPU install, an NVIDIA GPU and driver as described above. The CUDA toolkit is **not**
  needed: the CUDA 12 libraries come as pip wheels with PyTorch.

The other dependencies, including [GBGPU](https://github.com/mikekatz04/GBGPU),
[LISA Analysis Tools](https://github.com/mikekatz04/LISAanalysistools) and, with `[cuda12]`,
[CuPy](https://cupy.dev/), are installed automatically.

*Notes:*
- `lisaanalysistools` 1.2.8 crashes with `Illegal instruction` on CPUs without AVX-512 (e.g. Intel
  Core 12th–14th generation), so it is excluded from the requirements.
- To reproduce the tested versions exactly, add
  `"gbgpu-cuda12x==1.2.4" "lisaanalysistools-cuda12x==1.2.7" "gpubackendtools==0.1.1"` to the
  `pip install` of the package.
