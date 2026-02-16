# Installation

We recommend to create first a conda environment

```
conda create -n lisa_env python=3.11 -y
```

Then activate it

```
conda activate lisa_env
```

and install the `GSL` and `FFTW3` libraries

```
conda install -c conda-forge gsl fftw -y
```

You can then clone this repository and install it

```
git clone https://github.com/fdesanti/fast-lisa-subtraction.git
cd fast_lisa_subtraction
pip install .
```


## Prerequisites:
This package relies on:

- [GBGPU](https://github.com/mikekatz04/GBGPU) to generate GB waveforms (both on CPU/GPU)
- [CuPy](https://cupy.dev/) for the GPU acceleration

which are *NOT* automatically installed. Please refer to their official documentations to install and match your own hardware. 

*Note:* CuPy is not mandatory as the package is meant to work also on CPU-only machines, exploiting NumPy.
