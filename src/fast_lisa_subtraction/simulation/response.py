"""
LISA TDI response functions.

The analytical fits follow Sec 2.3 of arXiv:2009.11845, while the numerical
(semi-analytic) sky-averaged response follows arXiv:2108.01167.
"""

import os
import torch
import numpy as np
from scipy import interpolate
from lisaconstants import SPEED_OF_LIGHT
from .catalog import YEAR

#define constants
#NOTE: c must match the one used by the LDC noise models (lisaconstants), otherwise
#the TDI transfer-function nulls of noise PSD and response misalign and their ratio
#(e.g. in Sn) develops spurious spikes at multiples of c/(4L) ~ 0.03 Hz
L = 2.5*1e9 #m  LISA arm length
c = SPEED_OF_LIGHT #m/s (=299792458.0)
pm = 1e-12  #m
fm = 1e-15  #m
pi = np.pi
f_star = c/(2*pi*L) # Hz

#YEAR = 31558149.763545603  # sidereal YEAR in s
AU_s = 499.00478383615643  # astronomical unit in s

#oriented links entering the Michelson combinations
_LINKS = {"X": ("13", "12"), "Y": ("21", "23"), "Z": ("32", "31")}

#channel combinations in terms of the Michelson variables
_CHANNEL_COMBINATIONS = {"X": {"X": 1.0},
                         "Y": {"Y": 1.0},
                         "Z": {"Z": 1.0},
                         "A": {"Z": 1/np.sqrt(2), "X": -1/np.sqrt(2)},
                         "E": {"X": 1/np.sqrt(6), "Y": -2/np.sqrt(6), "Z": 1/np.sqrt(6)},
                         "T": {"X": 1/np.sqrt(3), "Y": 1/np.sqrt(3), "Z": 1/np.sqrt(3)}}


def _lisa_geometry(t=0.0):
    r"""LISA constellation geometry for Keplerian orbits at time ``t``.

    Returns the oriented link unit vectors
    :math:`\hat{n}_{rs} = (\vec{R}_r - \vec{R}_s)/L` and the spacecraft
    positions :math:`\vec{R}_i` of the vertices of the X, Y, Z Michelson
    combinations, in units of the arm length ``L`` (see Fig. 2 of
    arXiv:2108.01167).
    """
    lamb, kappa = 0.0, 0.0
    L_s = L / c  # arm length in seconds
    a = AU_s
    e = L_s / (2.0 * np.sqrt(3.0) * a)

    nn = np.arange(1, 4)
    Beta = (nn - 1) * 2.0 * pi / 3.0 + lamb
    alpha = 2.0 * pi * t / YEAR + kappa

    sa, ca = np.sin(alpha), np.cos(alpha)
    x = a*ca + a*e*(sa*ca*np.sin(Beta) - (1.0 + sa**2)*np.cos(Beta))
    y = a*sa + a*e*(sa*ca*np.cos(Beta) - (1.0 + ca**2)*np.sin(Beta))
    z = -np.sqrt(3.0)*a*e*np.cos(alpha - Beta)
    pos = np.stack([x, y, z]) / L_s  # (coordinate, spacecraft), in units of L

    n23 = pos[:, 1] - pos[:, 2]
    n31 = pos[:, 2] - pos[:, 0]
    n12 = pos[:, 0] - pos[:, 1]

    links = {"13": -n31, "12": n12,
             "21": -n12, "23": n23,
             "32": -n23, "31": n31}

    #vertex spacecraft of each Michelson, relative to the constellation center
    #(only relative positions matter, see the phase factor in Eq. (32))
    center = pos.mean(axis=1, keepdims=True)
    pos = pos - center
    vertices = {"X": pos[:, 0], "Y": pos[:, 1], "Z": pos[:, 2]}

    return links, vertices


def _arm_transfer(x, kn):
    r"""Single-arm (round-trip) transfer function :math:`\Upsilon_{rs}`,
    Eq. (35) of arXiv:2108.01167.

    Parameters
    ----------
    x : float or numpy.ndarray
        Dimensionless frequency :math:`\omega L/c`.
    kn : numpy.ndarray
        Dot product :math:`\hat{k}\cdot\hat{n}_{rs}` between the GW
        propagation direction and the link unit vector.
    """
    gam_rs = np.sinc(0.5*x*(1.0 - kn)/pi) * np.exp(-0.5j*x*(1.0 - kn))
    gam_sr = np.sinc(0.5*x*(1.0 + kn)/pi) * np.exp(-0.5j*x*(1.0 + kn))
    return gam_rs + gam_sr * np.exp(-1.0j*x*(1.0 - kn))


def _channel_PC(x, lam, bet, geometry, channel):
    r"""Plus/cross pattern amplitudes of a TDI channel at zero polarization angle.

    Implements the sky-dependent part of the TDI X response of Eq. (32) of
    arXiv:2108.01167 (and its cyclic permutations Y, Z), with the antenna
    pattern functions of Eq. (34), the polarization basis of Eqs. (25)-(27)
    and the vertex phase factor :math:`e^{-i\omega \hat{k}\cdot\vec{R}_1}`,
    which encodes the relative position of the X, Y, Z vertices and matters
    when combining them into A, E, T.

    Returns ``(P, C)`` such that :math:`F^{+}_{\rm ch}(\psi) = P\cos\psi + C\sin\psi`
    and :math:`F^{\times}_{\rm ch}(\psi) = -P\sin\psi + C\cos\psi`.
    ``lam`` and ``bet`` can be scalars or broadcastable arrays.
    """
    links, vertices = geometry
    lam, bet = np.broadcast_arrays(np.asarray(lam, dtype=float), np.asarray(bet, dtype=float))
    cl, sl = np.cos(lam), np.sin(lam)
    cb, sb = np.cos(bet), np.sin(bet)

    #GW propagation direction and polarization basis vectors, Eqs. (25)-(27)
    k = -np.stack([cb*cl, cb*sl, sb])
    u = np.stack([sl, -cl, np.zeros_like(sl)])
    v = np.stack([-sb*cl, -sb*sl, cb])

    P_mich, C_mich = {}, {}
    needed = _CHANNEL_COMBINATIONS[channel].keys()
    for mich in needed:
        arm_a, arm_b = _LINKS[mich]
        P_mich[mich], C_mich[mich] = 0.0, 0.0
        for arm, sign in [(arm_a, 1.0), (arm_b, -1.0)]:
            n = links[arm]
            kn = np.einsum("i,i...->...", n, k)
            nu = np.einsum("i,i...->...", n, u)
            nv = np.einsum("i,i...->...", n, v)
            psi_arm = _arm_transfer(x, kn)
            P_mich[mich] = P_mich[mich] + sign * (nu*nu - nv*nv) * psi_arm
            C_mich[mich] = C_mich[mich] + sign * (2.0*nu*nv) * psi_arm

        #vertex phase of Eq. (32) (position in units of L, hence the factor x)
        phase = np.exp(-1j * x * np.einsum("i,i...->...", vertices[mich], k))
        P_mich[mich] = P_mich[mich] * phase
        C_mich[mich] = C_mich[mich] * phase

    P = sum(coeff * P_mich[mich] for mich, coeff in _CHANNEL_COMBINATIONS[channel].items())
    C = sum(coeff * C_mich[mich] for mich, coeff in _CHANNEL_COMBINATIONS[channel].items())
    return P, C


def sky_averaged_antenna_power(f, channel="A", t=0.0, integrator="leggauss",
                               n_beta=None, n_lambda=None):
    r"""Numerical sky- and polarization-averaged antenna power :math:`\langle |F_{\rm ch}|^2 \rangle`.

    Semi-analytic computation following Sec. 5.2 of arXiv:2108.01167: the
    Michelson responses are built from the single-arm transfer functions and
    the antenna pattern functions (Eqs. (32)-(36)), combined into the requested
    channel, and averaged over sky location and polarization angle,

    .. math::
        \langle |F_{\rm ch}|^2 \rangle = \frac{1}{8\pi^2}
        \int_0^{2\pi} d\psi \int_0^{2\pi} d\lambda
        \int_{-\pi/2}^{\pi/2} d\beta \, \cos\beta \, |F^{+}_{\rm ch}(\psi, \lambda, \beta)|^2 .

    Parameters
    ----------
    f : numpy.ndarray or float
        Frequency array in Hz.
    channel : str
        TDI channel, one of ``"X"``, ``"Y"``, ``"Z"``, ``"A"``, ``"E"``, ``"T"``.
    t : float, optional
        Time (in s) at which the constellation geometry is evaluated. The
        sky-averaged result is essentially time independent.
    integrator : str, optional
        ``"leggauss"`` (default) performs the :math:`\psi` average analytically
        (:math:`\langle |F|^2\rangle_\psi = (|P|^2 + |C|^2)/2`) and the sky
        average with Gauss-Legendre quadrature in :math:`\sin\beta` and a
        trapezoidal rule in :math:`\lambda` (fast, vectorized).
        ``"scipy"`` uses ``scipy.integrate.tplquad`` over
        :math:`(\psi, \lambda, \beta)` (accurate but slow; useful for
        cross-checks).
    n_beta, n_lambda : int, optional
        Number of quadrature nodes in ecliptic latitude/longitude for the
        ``"leggauss"`` integrator. Defaults scale with the maximum frequency
        to resolve the oscillatory transfer functions.

    Returns
    -------
    numpy.ndarray
        Dimensionless :math:`\langle |F_{\rm ch}|^2 \rangle`, with the low
        frequency limits :math:`16 \times 3/20 = 2.4` for X, Y, Z and
        :math:`16 \times 9/40 = 3.6` for A, E.
    """
    if channel not in _CHANNEL_COMBINATIONS:
        raise ValueError(f"Unknown channel '{channel}', must be one of {list(_CHANNEL_COMBINATIONS)}")

    f = np.atleast_1d(np.asarray(f, dtype=float))
    x_all = 2*pi*f*L/c
    geometry = _lisa_geometry(t)
    result = np.empty_like(f)

    if integrator == "leggauss":
        #quadrature resolution must grow with omega*L/c to resolve the oscillations
        if n_beta is None:
            n_beta = int(max(64, 1.5*x_all.max() + 32))
        if n_lambda is None:
            n_lambda = 2*n_beta

        #Gauss-Legendre in sin(beta) (absorbs the cos(beta) Jacobian), uniform in lambda
        sin_beta, w_beta = np.polynomial.legendre.leggauss(n_beta)
        bet = np.arcsin(sin_beta)[:, None]
        lam = (2*pi*np.arange(n_lambda)/n_lambda)[None, :]
        weights = w_beta[:, None] * (2*pi/n_lambda) / (4*pi)

        for i, x in enumerate(x_all):
            P, C = _channel_PC(x, lam, bet, geometry, channel)
            result[i] = np.sum(weights * 0.5*(np.abs(P)**2 + np.abs(C)**2))

    elif integrator in ("scipy", "tplquad"):
        from scipy import integrate

        for i, x in enumerate(x_all):
            def integrand(psi, lam, bet):
                P, C = _channel_PC(x, lam, bet, geometry, channel)
                Fp = np.cos(psi)*P + np.sin(psi)*C
                return np.cos(bet) * np.abs(Fp)**2

            res = integrate.tplquad(integrand, -0.5*pi, 0.5*pi,
                                    lambda bet: 0.0, lambda bet: 2*pi,
                                    lambda bet, lam: 0.0, lambda bet, lam: 2*pi)
            result[i] = res[0] / (8*pi**2)

    else:
        raise ValueError(f"Unknown integrator '{integrator}', must be 'leggauss' or 'scipy'")

    return result


#precomputed antenna power tables (see scripts/generate_response_tables.py)
_TABLE_FILE = os.path.join(os.path.dirname(__file__), "data", "sky_averaged_antenna_power.npz")
_table_interpolants = {}


def _antenna_power_from_table(f, channel):
    r"""Interpolate the precomputed :math:`\langle |F_{\rm ch}|^2 \rangle` tables.

    The tables (generated by ``scripts/generate_response_tables.py``) are
    interpolated with a cubic spline in log-log space. Frequencies outside the
    tabulated range are clamped to its boundary values. If a channel is not
    stored, X, Y, Z are reconstructed as
    :math:`(\langle|F_A|^2\rangle + \langle|F_E|^2\rangle + \langle|F_T|^2\rangle)/3`
    (exact by unitarity of the A, E, T combination and the three-fold symmetry
    of the constellation).
    """
    if channel not in _table_interpolants:
        with np.load(_TABLE_FILE) as data:
            log_f = np.log10(data["f"])
            if channel in data.files:
                power = data[channel]
            elif channel in ("X", "Y", "Z"):
                power = (data["A"] + data["E"] + data["T"]) / 3
            else:
                raise KeyError(f"channel '{channel}' not found in {_TABLE_FILE}")
            _table_interpolants[channel] = interpolate.CubicSpline(log_f, np.log10(power))
            _table_interpolants[channel].log_f_range = (log_f[0], log_f[-1])

    spline = _table_interpolants[channel]
    log_f = np.clip(np.log10(f), *spline.log_f_range)
    return 10**spline(log_f)


def response(f, channel, tdi2=True, method="numerical", **kwargs):
    r"""Compute the LISA TDI response :math:`\mathcal{R}(f)` for a channel.

    Parameters
    ----------
    f : numpy.ndarray or torch.Tensor
        Frequency array in Hz.
    channel : str
        TDI channel, one of ``"X"``, ``"Y"``, ``"Z"``, ``"A"``, ``"E"``, ``"T"``.
    tdi2 : bool, optional
        If ``True``, use TDI 2.0 response; if ``False``, TDI 1.5. Default is ``True``.
    method : str, optional
        ``"numerical"`` (default) uses the exact sky- and polarization-averaged
        response computed semi-analytically as in arXiv:2108.01167: the
        precomputed tables shipped with the package are read and interpolated
        with a cubic spline (see :func:`_antenna_power_from_table`); pass
        ``from_file=False`` to recompute on the fly with
        :func:`sky_averaged_antenna_power` (slow).
        ``"analytic"`` uses the analytical fits of Sec 2.3 of
        arXiv:2009.11845 for A, E, T (and the fit
        :math:`16 \times 3/20/(1+0.6\omega^2)` of arXiv:2108.01167 for X, Y, Z).

        .. warning::
            The two methods follow different polarization conventions for
            A and E: the analytic fit of arXiv:2009.11845 is a factor of 2
            larger than the simulation-validated convention of
            arXiv:2108.01167, :math:`\langle |F_{A,E}|^2 \rangle \to 3.6`
            at low frequency.

    **kwargs
        For ``method="numerical"``: ``from_file`` (default ``True``) selects
        the precomputed tables; when ``from_file=False`` the remaining options
        (``t``, ``integrator``, ``n_beta``, ``n_lambda``) are forwarded to
        :func:`sky_averaged_antenna_power`.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Dimensionless response :math:`\sin^2\omega \, \omega^2 \,
        \langle |F_{\rm ch}|^2 \rangle` with :math:`\omega = 2\pi f L/c`,
        where for the analytic method
        :math:`\langle |F_{\rm ch}|^2 \rangle = 16\,\tilde{R}(\omega)`.
    """

    is_torch = isinstance(f, torch.Tensor)
    omega = 2*pi*f*L/c
    sin_omega = torch.sin(omega) if is_torch else np.sin(omega)

    if tdi2:
        sin_2omega = torch.sin(2*omega) if is_torch else np.sin(2*omega)
        tdi_factor = 4 * sin_2omega**2
    else:
        tdi_factor = 1.0

    if method == "analytic":
        if channel == "T":
            R_tilde = 9/20 * (omega)**6 / (1.8*1e3+0.7*(omega)**8)

        elif channel in ["A", "E"]:
            R_tilde = 9/20 * 1 / (1 + 0.7 * (omega)**2)

        elif channel in ["X", "Y", "Z"]:
            R_tilde = 3/20 * 1 / (1 + 0.6 * (omega)**2)

        else:
            raise ValueError(f"Unknown channel '{channel}', must be one of {list(_CHANNEL_COMBINATIONS)}")

        antenna_power = 16 * R_tilde

    elif method == "numerical":
        if channel not in _CHANNEL_COMBINATIONS:
            raise ValueError(f"Unknown channel '{channel}', must be one of {list(_CHANNEL_COMBINATIONS)}")

        f_np = f.detach().cpu().numpy() if is_torch else np.asarray(f)
        if kwargs.pop("from_file", True):
            antenna_power = _antenna_power_from_table(f_np, channel)
        else:
            antenna_power = sky_averaged_antenna_power(f_np, channel=channel, **kwargs)
        if is_torch:
            antenna_power = torch.as_tensor(antenna_power, dtype=f.dtype, device=f.device)

    else:
        raise ValueError(f"Unknown method '{method}', must be 'analytic' or 'numerical'")

    return sin_omega**2 * (omega)**2 * antenna_power * tdi_factor
