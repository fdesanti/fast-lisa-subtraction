"""
Some standard population priors. See Abbott et al 2021 (arxiv.org/pdf/2010.14533)
"""

import torch
import numpy as np

from ..utils import interp1d
from .analytical import Prior

class PowerLaw(Prior):
    r"""
    Power-law prior distribution.

    .. math::

        p(\theta) =
        \begin{cases}
            \theta^{\alpha}, & \theta_{\min} \le \theta \le \theta_{\max} \\
            0,               & \text{otherwise}
        \end{cases}

    where :math:`\theta_{\min}` and :math:`\theta_{\max}` are the minimum
    and maximum values of the prior, respectively.

    Parameters
    ----------
    alpha : float
        Power-law index :math:`\alpha`.
    minimum : float
        Minimum value of the prior :math:`\theta_{\min}`.
    maximum : float
        Maximum value of the prior :math:`\theta_{\max}`.
    name : str, optional
        Name of the prior parameter.
    device : str, optional
        Torch device used for sampling.
    """
    def __init__(self, alpha, minimum, maximum, name=None, device='cpu'):
        """Initialize a power-law prior.

        Parameters
        ----------
        alpha : float
            Power-law index.
        minimum : float
            Minimum value of the support.
        maximum : float
            Maximum value of the support.
        name : str, optional
            Name of the prior parameter.
        device : str, optional
            Torch device used for sampling.
        """
        self.alpha  = alpha
        super().__init__(minimum, maximum, name, device)
        
        #compute the pdf and cdf
        self._compute_pdf()
        self._compute_cdf()

    @property
    def pdf(self):
        """Probability density function evaluated on the internal grid.

        Returns
        -------
        torch.Tensor
            Discretized PDF values over ``self.x``.
        """
        return self._pdf

    @property
    def pdf_max(self):
        """Maximum value of the discretized PDF.

        Returns
        -------
        torch.Tensor
            Maximum PDF value.
        """
        return self._pdf.max()

    @property
    def cdf(self):
        """Cumulative distribution function evaluated on the internal grid.

        Returns
        -------
        torch.Tensor
            Discretized CDF values over ``self.x``.
        """
        return self._cdf
    
    def _compute_pdf(self):
        """Compute a discretized power-law PDF on a fixed grid.

        Returns
        -------
        None
        """
        self.x = torch.linspace(self.minimum, self.maximum, int(1e4), device=self.device)
        pdf = self.x.pow(self.alpha)
        pdf = pdf / torch.trapz(pdf, self.x)
        self._pdf = pdf

    def _compute_cdf(self):
        """Compute a discretized CDF from the cached PDF.

        Returns
        -------
        None
        """
        cdf = torch.cumsum(self.pdf, dim=-1)
        self._cdf = cdf / cdf[-1]

    def log_prob(self, x, standardize=False):
        """Evaluate the log-probability of the power-law prior.

        Parameters
        ----------
        x : array-like or torch.Tensor
            Points at which to evaluate ``log p(x)``.
        standardize : bool, optional
            If True, interpret ``x`` as standardized and de-standardize
            before evaluating.

        Returns
        -------
        torch.Tensor
            Log-probability values.
        """
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)

        if standardize:
            x = self.destandardize(x)

        xmin  = torch.as_tensor(self.minimum, device=self.device, dtype=x.dtype)
        xmax  = torch.as_tensor(self.maximum, device=self.device, dtype=x.dtype)
        alpha = torch.as_tensor(self.alpha, device=self.device, dtype=x.dtype)

        inside = (x >= xmin) & (x <= xmax)

        logp = torch.full_like(x, -float("inf"))

        # alpha = -1 case
        if torch.isclose(alpha, torch.tensor(-1.0, device=self.device)):
            norm = torch.log(xmax) - torch.log(xmin)
            logp_inside = -torch.log(x) - torch.log(norm)

        else:
            norm = (xmax.pow(alpha + 1) - xmin.pow(alpha + 1)) / (alpha + 1)
            logp_inside = alpha * torch.log(x) - torch.log(norm)

        logp[inside] = logp_inside[inside]
        return logp

    def sample(self, num_samples, standardize=False):
        """Sample from the power-law prior.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw.
        standardize : bool, optional
            If True, return standardized samples.

        Returns
        -------
        torch.Tensor
            Samples drawn from the prior.
        """
        u = torch.rand(num_samples, device=self.device)
        samples = interp1d(self._cdf, self.x, u).flatten()

        if standardize:
            samples = self.standardize(samples)
        return samples


class BrokenPowerLaw(Prior):
    r"""Broken power-law prior distribution.

    Notes
    -----
    The density is piecewise:

    .. math::
        p(\theta) \propto \theta^{\alpha}, \quad
        \theta_{\min} \le \theta < \theta_{\mathrm{break}},

    .. math::
        p(\theta) \propto \theta^{\beta}\,
        \theta_{\mathrm{break}}^{\alpha-\beta}, \quad
        \theta_{\mathrm{break}} \le \theta \le \theta_{\max}.

    The break point is determined by ``b``. If ``0 <= b <= 1``, then
    :math:`\theta_{\mathrm{break}} = \theta_{\min} + b(\theta_{\max}-\theta_{\min})`;
    otherwise ``b`` is treated as an absolute break value within the support.

    Optional low-end smoothing :math:`S(\theta; \delta)` can be applied in
    the interval :math:`[\theta_{\min}, \theta_{\min}+\delta]`.

    Parameters
    ----------
    alpha : float
        Power-law index for :math:`\theta < \theta_{\mathrm{break}}`.
    beta : float
        Power-law index for :math:`\theta \ge \theta_{\mathrm{break}}`.
    b : float
        Break parameter (fraction or absolute).
    minimum : float
        Lower bound of the support.
    maximum : float
        Upper bound of the support.
    delta_p : float or None, optional
        Optional smoothing width :math:`\delta`. If None or 0, smoothing
        is disabled.
    name : str or None, optional
        Parameter name.
    device : str, optional
        Torch device used for sampling.
    """
    def __init__(self, alpha, beta, b, minimum, maximum, delta_p=None, name=None, device='cpu'):
        r"""Initialize a broken power-law prior.

        Parameters
        ----------
        alpha : float
            Power-law index for :math:`\theta < \theta_{\mathrm{break}}`.
        beta : float
            Power-law index for :math:`\theta \ge \theta_{\mathrm{break}}`.
        b : float
            Break parameter (fraction or absolute).
        minimum : float
            Lower bound of the support.
        maximum : float
            Upper bound of the support.
        delta_p : float or None, optional
            Optional smoothing width :math:`\delta`. If None or 0, smoothing
            is disabled.
        name : str or None, optional
            Parameter name.
        device : str, optional
            Torch device used for sampling.

        Raises
        ------
        ValueError
            If ``b`` is provided as an absolute value outside the support.
        """
        self.alpha   = float(alpha)
        self.beta    = float(beta)
        self.device  = device

        super().__init__(minimum, maximum, name, device)

        # Resolve break: allow fraction in [0,1] or absolute within [min,max]
        if 0.0 <= b <= 1.0:
            self.b = float(b)
            self.break_point = self.minimum + self.b * (self.maximum - self.minimum)
        else:
            if not (self.minimum < b < self.maximum):
                raise ValueError(f"`b` must be in [0,1] (fraction) or within ({self.minimum},{self.maximum}) as absolute.")
            self.break_point = float(b)
            self.b = (self.break_point - self.minimum) / (self.maximum - self.minimum)

        # Optional smoothing width (delta); treat None/0 as disabled
        self.delta_p = None if (delta_p is None or float(delta_p) <= 0.0) else float(delta_p)

        # Precompute PDF/CDF grids
        self._compute_pdf()
        self._compute_cdf()

    @property
    def pdf(self):
        """Probability density function evaluated on the internal grid.

        Returns
        -------
        torch.Tensor
            Discretized PDF values over ``self.x``.
        """
        return self._pdf

    @property
    def pdf_max(self):
        """Maximum value of the discretized PDF.

        Returns
        -------
        torch.Tensor
            Maximum PDF value.
        """
        return self._pdf.max()

    @property
    def cdf(self):
        """Cumulative distribution function evaluated on the internal grid.

        Returns
        -------
        torch.Tensor
            Discretized CDF values over ``self.x``.
        """
        return self._cdf

    def _compute_pdf(self):
        """Compute a discretized broken power-law PDF on a fixed grid.

        Returns
        -------
        None
        """
        # Discretization grid (same density as PowerLaw)
        self.x = torch.linspace(self.minimum, self.maximum, int(1e4), device=self.device)

        pdf = torch.zeros_like(self.x)

        # continuity factor to match values at theta_break
        fix_factor = (self.break_point ** (self.alpha - self.beta))

        mask_lo = self.x < self.break_point
        mask_hi = ~mask_lo  # includes theta == break

        # piecewise power law
        if mask_lo.any():
            pdf[mask_lo] = self.x[mask_lo].pow(self.alpha)
        if mask_hi.any():
            pdf[mask_hi] = self.x[mask_hi].pow(self.beta) * fix_factor

        # optional low-end smoothing near theta_min
        if self.delta_p is not None:
            pdf = pdf * self._smoothing(self.x)

        # normalize by integral over theta
        norm = torch.trapz(pdf, self.x)
        if not torch.isfinite(norm) or norm <= 0:
            raise RuntimeError("BrokenPowerLaw PDF normalization failed (non-positive or non-finite integral).")
        self._pdf = pdf / norm

    def _compute_cdf(self):
        """Compute a discretized CDF from the cached PDF.

        Returns
        -------
        None
        """
        # cum-trapezoid so that CDF[0] = 0 and CDF[-1] = 1
        x = self.x
        p = self.pdf
        cdf = torch.zeros_like(p)
        # trapezoid increments
        dx = x[1:] - x[:-1]
        incr = 0.5 * (p[1:] + p[:-1]) * dx
        cdf[1:] = torch.cumsum(incr, dim=-1)
        # normalize
        cdf = cdf / cdf[-1]
        self._cdf = cdf

    def _f(self, p: torch.Tensor) -> torch.Tensor:
        r"""Compute the smoothing helper function.

        .. math::
            f(\theta) = \exp\!\left(\frac{\delta}{\theta-\theta_{\min}}
            + \frac{\delta}{(\theta-\theta_{\min})-\delta}\right).

        This is only evaluated where denominators are positive; callers
        should mask inputs appropriately.

        Parameters
        ----------
        p : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            ``f(p)`` values.
        """
        p1 = p - self.minimum
        eps = torch.finfo(p.dtype).eps
        d1 = torch.clamp(p1, min=eps)
        d2 = torch.clamp(p1 - self.delta_p, min=eps)
        return torch.exp(self.delta_p / d1 + self.delta_p / d2)

    def _smoothing(self, samples: torch.Tensor) -> torch.Tensor:
        r"""Compute the low-end smoothing factor.

        .. math::
            S(\theta) =
            \begin{cases}
              0, & \theta < \theta_{\min}, \\
              \frac{1}{f(\theta)+1}, & \theta_{\min} \le \theta < \theta_{\min}+\delta, \\
              1, & \theta \ge \theta_{\min}+\delta.
            \end{cases}

        Parameters
        ----------
        samples : torch.Tensor
            Input samples.

        Returns
        -------
        torch.Tensor
            Smoothing factors for each sample.
        """
        S = torch.zeros_like(samples)
        p_thr = self.minimum + self.delta_p

        mask_mid = (samples >= self.minimum) & (samples < p_thr)
        if mask_mid.any():
            S[mask_mid] = 1.0 / (self._f(samples[mask_mid]) + 1.0)

        S[samples >= p_thr] = 1.0
        # (samples < minimum) stay at 0.0, but those are out of support anyway
        return S

    def sample(self, num_samples: int, standardize: bool = False):
        """Sample from the broken power-law prior.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw.
        standardize : bool, optional
            If True, return standardized samples.

        Returns
        -------
        torch.Tensor
            Samples drawn from the prior.
        """
        # explicit endpoints ensure no extrapolation outside [min,max]
        cdf_aug = torch.cat([torch.tensor([0.0], device=self.device), self._cdf])
        x_aug   = torch.cat([torch.tensor([self.minimum], device=self.device), self.x])

        u = torch.rand(num_samples, device=self.device)  # in [0,1)
        samples = interp1d(cdf_aug, x_aug, u).flatten()

        # final tiny safety (handles any numerical wiggles)
        samples = torch.clamp(samples, min=self.minimum, max=self.maximum)

        if standardize:
            samples = self.standardize(samples)
        return samples
    
class PowerLawPlusPeak(Prior):
    """Power-law plus peak prior distribution.

    Parameters
    ----------
    alpha : float
        Power-law index for the background.
    beta : float
        Power-law index for the peak component.
    minimum : float
        Lower bound of the support.
    maximum : float
        Upper bound of the support.
    peak : float
        Peak location or scale parameter (implementation-specific).
    name : str, optional
        Name of the prior parameter.
    """
    def __init__(self, alpha, beta, minimum, maximum, peak, name='Parameter'):
        """Initialize a power-law plus peak prior.

        Parameters
        ----------
        alpha : float
            Power-law index for the background.
        beta : float
            Power-law index for the peak component.
        minimum : float
            Lower bound of the support.
        maximum : float
            Upper bound of the support.
        peak : float
            Peak location or scale parameter (implementation-specific).
        name : str, optional
            Name of the prior parameter.
        """
        self.alpha = alpha
        self.beta = beta
        self.peak = peak
        super().__init__(minimum, maximum, name)
    
    def sample(self, N):
        """Sample from the power-law plus peak prior.

        Parameters
        ----------
        N : int
            Number of samples to draw.

        Raises
        ------
        NotImplementedError
            Sampling for this prior is not implemented.
        """
        raise NotImplementedError("Sampling from PowerLawPlusPeak is not implemented")
