"""
Some utility functions to compute the characteristic strain and Omega_GW related quantities for LISA TDI channels.
We follow Eqs. in Sec 2.3 of arXiv:2009.11845
"""

import torch
import numpy as np

#define constants
L = 2.5*1e9 #m  LISA arm length
c = 3*1e8   #m/s
pm = 1e-12  #m
fm = 1e-15  #m
pi = np.pi
f_star = c/(2*pi*L) # Hz


def Sn(f, Nx, channel):
    """Add the LISA response to a noise power spectral density.

    Parameters
    ----------
    f : numpy.ndarray or torch.Tensor
        Frequency array in Hz.
    Nx : float or array-like
        Noise PSD in ``m^2/Hz`` for the specified channel.
    channel : str
        TDI channel, one of ``"A"``, ``"E"``, or ``"T"``.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Noise PSD including the LISA response.
    """

    omega = 2*pi*f*L/c
    sin_omega = torch.sin(omega) if isinstance(omega, torch.Tensor) else np.sin(omega)

    #compute the response
    if channel =="T":
        R_tilde = 9/20 * (omega)**6 / (1.8*1e3+0.7*(omega)**8)

    elif channel in ["A", "E"]:
        R_tilde = 9/20 * 1 / (1 + 0.7 * (omega)**2)

    return Nx / (16 * sin_omega**2*(omega)**2 * R_tilde)

def characteristic_strain(f, Nx, channel):
    r"""Compute the characteristic strain for a TDI channel.

    Parameters
    ----------
    f : numpy.ndarray or torch.Tensor
        Frequency array in Hz.
    Nx : float or array-like
        Noise PSD in ``m^2/Hz`` for the specified channel.
    channel : str
        TDI channel, one of ``"A"``, ``"E"``, or ``"T"``.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Characteristic strain in ``m/sqrt(Hz)`` for the specified channel.

    Notes
    -----
    The characteristic strain is computed as

    .. math::
        h_c(f) = \sqrt{f\,S_n(f)},

    where :math:`S_n(f)` includes the LISA response.
    """

    #compute the noise power spectral density with the LISA response
    Sn_ch = Sn(f, Nx, channel)
    
    if isinstance(Sn_ch, torch.Tensor):
        return torch.sqrt(f * Sn_ch)
    else:
        return np.sqrt(f * Sn_ch)


def psd_2_omega_gw(f, Sn):
    r"""
    Convert a PSD to :math:`\Omega_{\rm GW}`.

    Parameters
    ----------
    f : numpy.ndarray or torch.Tensor
        Frequency array in Hz.
    Sn : numpy.ndarray or torch.Tensor
        Noise power spectral density in ``1/Hz``.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Dimensionless :math:`\Omega_{\rm GW}`.

    Notes
    -----
    The conversion uses

    .. math::
        \Omega_{\rm GW}(f) = \frac{4\pi^2 f^3 S_n(f)}{3 H_0^2},

    with :math:`H_0 = 3.24\times 10^{-18}\,\mathrm{s^{-1}}`.
    """

    return 4*pi**2 * (f**3) * Sn / (3*(3.24*10**(-18))**2)

def characteristic_strain_2_omega_gw(f, Sn):
    r"""
    Convert characteristic strain to :math:`\Omega_{\rm GW}`.

    Parameters
    ----------
    f : numpy.ndarray or torch.Tensor
        Frequency array in Hz.
    Sn : numpy.ndarray or torch.Tensor
        Characteristic strain in ``1/sqrt(Hz)``.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Dimensionless :math:`\Omega_{\rm GW}`.

    Notes
    -----
    The conversion uses

    .. math::
        h_c(f) = \sqrt{f\,S_n(f)} \Rightarrow S_n(f) = \frac{h_c^2(f)}{f},

    and therefore

    .. math::
        \Omega_{\rm GW}(f) = \frac{4\pi^2 f^3 S_n(f)}{3 H_0^2},

    with :math:`H_0 = 3.24\times 10^{-18}\,\mathrm{s^{-1}}`.
    """
    Sn = Sn**2 / f  # Convert characteristic strain to noise power spectral density
    return 4*pi**2 * (f**3) * Sn / (3*(3.24e-18)**2)


def coarse(f, Si):
    """Compute a coarse-grained point and uncertainty for one bin.

    Parameters
    ----------
    f : numpy.ndarray
        Frequency values in the bin.
    Si : numpy.ndarray
        Spectrum values in the bin.

    Returns
    -------
    tuple
        ``(fc, Sc, sigma)`` where ``fc`` is the weighted frequency,
        ``Sc`` is the weighted spectrum value, and ``sigma`` is the
        standard deviation of ``Si``.
    """
    sigma_i = 1/np.var(Si) * np.ones_like(Si)
    wi = sigma_i / np.sum(sigma_i)
    fc = np.sum(wi * f)
    Sc = np.sum(wi * Si)
    return fc, Sc, np.std(Si)


def coarse_grain_data(f, data, n=100):
    r"""
    Apply coarse-graining to spectrum data.

    Parameters
    ----------
    f : numpy.ndarray
        Frequency array.
    data : numpy.ndarray
        Spectrum values to be coarse-grained.
    n : int, optional
        Number of logarithmic bins.

    Returns
    -------
    tuple of numpy.ndarray
        ``(f_coarse, data_coarse, sigma_coarse)`` corresponding to the
        coarse-grained frequency, spectrum, and standard deviation.

    Notes
    -----
    The coarse-grained value in each bin is

    .. math::
        S_k = \frac{\sum_i \sigma_i S_i}{\sum_i \sigma_i},

    where :math:`S_i` are the PSD values and :math:`\sigma_i` their
    associated uncertainties.
    """
    fk = np.logspace(np.log10(f.min()), np.log10(f.max()), n+1)
    #fk_centers = 0.5 * (fk[:-1] + fk[1:])

    coarsed = np.zeros(n)
    f_coarse = np.zeros(n)
    sigma_coarsed = np.zeros(n)
    
    for i in range(n):
        mask = (f >= fk[i]) & (f < fk[i+1])

        f_coarse[i], coarsed[i], sigma_coarsed[i] = coarse(f[mask], data[mask])
    
    return f_coarse, coarsed, sigma_coarsed
