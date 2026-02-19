"""Some standard distributions for sampling the priors."""

import torch
import numpy as np

from scipy.special import gammaincinv
from torch.distributions import Gamma as torchGamma
from torch.distributions import LogNormal as torchLogNormal

from ..utils import TensorSamples

N_ = int(1e7)

class Prior:
    """Base class for analytical priors.

    Parameters
    ----------
    minimum : float or None, optional
        Lower bound of the prior support. If None, the bound is estimated
        from sampled values.
    maximum : float or None, optional
        Upper bound of the prior support. If None, the bound is estimated
        from sampled values.
    name : str, optional
        Parameter name.
    device : str, optional
        Torch device used for sampling and computations.
    """
    def __init__(self, minimum=None, maximum=None, name='Parameter', device='cpu'):
        """Initialize a prior instance.

        Parameters
        ----------
        minimum : float or None, optional
            Lower bound of the support. If None, estimated from samples.
        maximum : float or None, optional
            Upper bound of the support. If None, estimated from samples.
        name : str, optional
            Parameter name.
        device : str, optional
            Torch device used for sampling and computations.
        """
        
        if minimum is not None and maximum is not None:
            assert minimum < maximum, "Minimum must be less than maximum"
        
        self._name   = name
        self.device  = device
        self.minimum = minimum
        self.maximum = maximum
        
    @property
    def name(self):
        """Parameter name.

        Returns
        -------
        str
            Name of the parameter.
        """
        return self._name

    @property
    def minimum(self):
        """Lower bound of the prior support.

        Returns
        -------
        float or torch.Tensor
            Minimum value of the support.
        """
        return self._minimum
    @minimum.setter
    def minimum(self, value):
        """Set the lower bound of the prior support.

        Parameters
        ----------
        value : float or None
            Lower bound. If None, estimated from samples.
        """
        if value is not None:
            self._minimum = value
        else:
            self._minimum = self.sample(N_).min()
            #self._minimum = torch.tensor(self._minimum).to(self.device)
    
    @property
    def maximum(self):
        """Upper bound of the prior support.

        Returns
        -------
        float or torch.Tensor
            Maximum value of the support.
        """
        return self._maximum
    @maximum.setter
    def maximum(self, value):
        """Set the upper bound of the prior support.

        Parameters
        ----------
        value : float or None
            Upper bound. If None, estimated from samples.
        """
        if value is not None:
            self._maximum = value
        else:
            self._maximum = self.sample(N_).max()
            #self._maximum = torch.tensor(self._maximum).to(self.device)
    
    @property
    def mean(self):
        """Mean of the prior.

        Returns
        -------
        torch.Tensor
            Mean estimated from sampled values.
        """
        if not hasattr(self, '_mean'):
            self._mean = self.sample(N_).mean()
        return self._mean
    
    @property
    def std(self):
        """Standard deviation of the prior.

        Returns
        -------
        torch.Tensor
            Standard deviation estimated from sampled values.
        """
        if not hasattr(self, '_std'):
            self._std = self.sample(N_).std()
        return self._std
    
    def clip(self, samples):
        """Filter samples to the prior support.

        Parameters
        ----------
        samples : torch.Tensor
            Samples to clip.

        Returns
        -------
        torch.Tensor
            Samples within ``[minimum, maximum]``.
        """
        mask = (samples >= self.minimum) & (samples <= self.maximum)
        return samples[mask]
    
    def standardize(self, samples):
        """Standardize samples using the prior mean and standard deviation.

        Parameters
        ----------
        samples : torch.Tensor
            Samples to standardize.

        Returns
        -------
        torch.Tensor
            Standardized samples.
        """
        return (samples - self.mean) / self.std
    
    def destandardize(self, samples):
        """Reverse standardization to the original scale.

        Parameters
        ----------
        samples : torch.Tensor
            Standardized samples.

        Returns
        -------
        torch.Tensor
            Samples in the original scale.
        """
        return samples * self.std + self.mean

    def sample(self, num_samples, standardize=False):
        """Draw samples from the prior.

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

        Raises
        ------
        NotImplementedError
            If the subclass does not implement sampling.
        """
        raise NotImplementedError
    

class Uniform(Prior):
    """Uniform prior distribution.

    Parameters
    ----------
    minimum : float
        Minimum value of the prior.
    maximum : float
        Maximum value of the prior.
    name : str, optional
        Name of the prior parameter.
    device : str, optional
        Torch device used for sampling.
    """
    def __init__(self, minimum, maximum, name=None, device='cpu'):
        """Initialize a uniform prior.

        Parameters
        ----------
        minimum : float
            Minimum value of the prior.
        maximum : float
            Maximum value of the prior.
        name : str, optional
            Name of the prior parameter.
        device : str, optional
            Torch device used for sampling.
        """
        super().__init__(minimum, maximum, name, device)

        self._mean = (self.minimum + self.maximum) / 2
        self._std  = (self.maximum - self.minimum) / np.sqrt(12)
    
    def sample(self, num_samples, standardize=False):
        """Sample from the uniform prior.

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
        #samples = np.random.uniform(self.minimum, self.maximum, num_samples)
        samples = torch.empty(int(num_samples), device=self.device).uniform_(self.minimum, self.maximum)
        if standardize:
            return self.standardize(samples)
        return samples
    

class Gaussian(Prior):
    """Gaussian prior distribution.

    Parameters
    ----------
    mean : float
        Mean of the Gaussian prior.
    std_dev : float
        Standard deviation of the Gaussian prior.
    name : str, optional
        Name of the prior parameter.
    device : str, optional
        Torch device used for sampling.
    """
    def __init__(self, mean, std_dev, name=None, device='cpu'):
        """Initialize a Gaussian prior.

        Parameters
        ----------
        mean : float
            Mean of the Gaussian prior.
        std_dev : float
            Standard deviation of the Gaussian prior.
        name : str, optional
            Name of the prior parameter.
        device : str, optional
            Torch device used for sampling.
        """
        self._mean = mean
        self._std  = std_dev
        super().__init__(name=name, device=device)
    
    def sample(self, num_samples, standardize=False):
        """Sample from the Gaussian prior.

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
        samples = torch.normal(self.mean, self.std, size=(int(num_samples),), device=self.device)
        if standardize:
            return self.standardize(samples)
        return samples
    
class Cauchy(Prior):
    """Cauchy prior distribution.

    Parameters
    ----------
    loc : float
        Location parameter of the Cauchy prior.
    scale : float
        Scale parameter of the Cauchy prior.
    minimum : float or None, optional
        Lower bound of the support.
    maximum : float or None, optional
        Upper bound of the support.
    name : str, optional
        Name of the prior parameter.
    device : str, optional
        Torch device used for sampling.
    """
    def __init__(self, loc, scale, minimum=None, maximum=None, name=None, device='cpu'):
        """Initialize a Cauchy prior.

        Parameters
        ----------
        location : float
            Location parameter of the Cauchy prior.
        scale : float
            Scale parameter of the Cauchy prior.
        name : str, optional
            Name of the prior parameter.
        device : str, optional
            Torch device used for sampling.
        """
        self._mean = loc
        self._std  = scale  # Note: Cauchy does not have a defined mean or std, but we use these for standardization
        self._minimum = minimum
        self._maximum = maximum
        super().__init__(name=name, device=device, minimum=minimum, maximum=maximum)
    
    def sample(self, num_samples, standardize=False):
        """Sample from the Cauchy prior.

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
        samples = torch.empty(int(num_samples), device=self.device).cauchy_(self.mean, self.std)

        # Clip the samples 
        if self._minimum is not None and self._maximum is not None:
            samples = (samples - self._minimum) % (self._maximum - self._minimum) + self._minimum

        if standardize:
            return self.standardize(samples)
        return samples

class Gamma(Prior):
    """Gamma prior distribution with an optional location shift.

    Parameters
    ----------
    alpha : float
        Shape parameter of the Gamma prior.
    beta : float
        Rate parameter of the Gamma prior (inverse scale).
    offset : float, optional
        Additive offset applied to samples.
    minimum : float or None, optional
        Lower bound of the support.
    maximum : float or None, optional
        Upper bound of the support.
    name : str, optional
        Name of the prior parameter.
    device : str, optional
        Torch device used for sampling.

    Notes
    -----
    The unshifted Gamma density is

    .. math::
        p(\\theta) = \\frac{\\beta^{\\alpha}}{\\Gamma(\\alpha)}
        \\theta^{\\alpha - 1} e^{-\\beta \\theta}.

    Samples are drawn from ``torch.distributions.Gamma(alpha, beta)``
    and then shifted by a median-based centering ``m`` and the provided
    ``offset``:

    .. math::
        x = z - m + \\mathrm{offset},

    where ``z`` is Gamma-distributed and ``m`` is computed from the
    inverse incomplete gamma function.
    """
    def __init__(self, alpha, beta, offset=0, minimum=None, maximum=None, name=None, device='cpu'):
        """Initialize a shifted Gamma prior.

        Parameters
        ----------
        alpha : float
            Shape parameter.
        beta : float
            Rate parameter (inverse scale).
        offset : float, optional
            Additive offset applied to samples.
        minimum : float or None, optional
            Lower bound of the support.
        maximum : float or None, optional
            Upper bound of the support.
        name : str, optional
            Name of the prior parameter.
        device : str, optional
            Torch device used for sampling.
        """
        self.alpha  = torch.tensor(float(alpha), device=device)
        self.beta   = torch.tensor(float(beta), device=device)
        self.offset = offset
        self.med    = torch.tensor(gammaincinv(alpha, 0.5) / beta, device=device)

        self.gamma  = torchGamma(self.alpha, self.beta, validate_args=False)

        super().__init__(minimum, maximum, name, device)

        #self._mean = self.alpha * self.beta + self.offset
        #self._std  = np.sqrt(self.alpha * self.beta**2)


    def log_prob(self, x, standardize=False):
        """Evaluate the log-probability of the shifted Gamma prior.

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

        Notes
        -----
        The evaluation uses the shifted variable

        .. math::
            z = x - \\mathrm{offset} + m,

        and returns ``-inf`` where ``z <= 0``.
        """
        x = torch.as_tensor(x, device=self.device, dtype=self.alpha.dtype)

        if standardize:
            x = self.destandardize(x)

        # Invert the shift:
        z = x - self.offset + self.med

        # Support: z > 0 for Gamma
        logp = self.gamma.log_prob(z)
        logp = torch.where(z > 0, logp, torch.full_like(logp, -float("inf")))

        return logp
        

    def sample(self, num_samples, standardize=False):
        """Sample from the shifted Gamma prior.

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
        samples = self.gamma.sample((int(num_samples),)) 
        samples -= self.med
        samples += self.offset
        if standardize:
            return self.standardize(samples)
        return samples

class LogNormal(Prior):
    """Shifted log-normal prior distribution.

    Parameters
    ----------
    mu : float
        Location parameter used as an additive shift.
    sigma : float
        Log-space standard deviation.
    minimum : float or None, optional
        Lower bound of the support.
    maximum : float or None, optional
        Upper bound of the support.
    name : str, optional
        Name of the prior parameter.
    device : str, optional
        Torch device used for sampling.

    Notes
    -----
    Sampling draws ``y`` from a log-normal distribution and returns a
    shifted value:

    .. math::
        x = y + \\mu, \\quad y \\sim \\mathrm{LogNormal}(0, \\sigma).
    """
    def __init__(self, mu, sigma, minimum=None, maximum=None, name=None, device='cpu'):
        """Initialize a shifted log-normal prior.

        Parameters
        ----------
        mu : float
            Location parameter used as an additive shift.
        sigma : float
            Log-space standard deviation.
        minimum : float or None, optional
            Lower bound of the support.
        maximum : float or None, optional
            Upper bound of the support.
        name : str, optional
            Name of the prior parameter.
        device : str, optional
            Torch device used for sampling.
        """
        self.mu    = torch.tensor(float(mu), device=device)
        self.sigma = torch.tensor(float(sigma), device=device)
        super().__init__(minimum, maximum, name, device)

    def log_prob(self, x, standardize=False):
        """Evaluate the log-probability of the shifted log-normal prior.

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

        Notes
        -----
        The evaluation uses the shifted variable

        .. math::
            y = x - \\mu,

        and applies a log-normal density with parameters ``(mu, sigma)``
        to ``y``.
        """
        x = torch.as_tensor(x, device=self.device, dtype=self.sigma.dtype)

        if standardize:
            x = self.destandardize(x)

        # Invert the shift: y = x - offset
        y = x - self.mu 
        logp = torchLogNormal(self.mu, self.sigma, validate_args=False).log_prob(y)
        logp = torch.where(y > 0, logp, torch.full_like(logp, -float("inf")))

        return logp

    def sample(self, num_samples, standardize=False):
        """Sample from the shifted log-normal prior.

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
        
        samples = torch.empty(int(num_samples), device=self.device).log_normal_(0, self.sigma)
        samples += self.mu

        if standardize:
            return self.standardize(samples)
        return samples

class UniformCosine(Prior):
    """Angle prior uniform in the cosine of the angle.

    This is commonly used for inclination angles.

    Parameters
    ----------
    minimum : float, optional
        Minimum angle in radians.
    maximum : float, optional
        Maximum angle in radians.
    name : str, optional
        Name of the prior parameter.
    device : str, optional
        Torch device used for sampling.

    Notes
    -----
    Uniformity in :math:`\\cos(\\theta)` implies a density proportional to
    :math:`\\sin(\\theta)` over the support.
    """
    def __init__(self, minimum=0, maximum=np.pi, name=None, device='cpu'):
        """Initialize a cosine-uniform angle prior.

        Parameters
        ----------
        minimum : float, optional
            Minimum angle in radians.
        maximum : float, optional
            Maximum angle in radians.
        name : str, optional
            Name of the prior parameter.
        device : str, optional
            Torch device used for sampling.
        """
        super().__init__(minimum, maximum, name, device)
    
    def sample(self, num_samples, standardize=False):
        """Sample from a cosine-uniform angle prior.

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
        samples = torch.tensor([], device=self.device)
        nsamp_ = num_samples
        while len(samples) < num_samples:
            samp_   = torch.empty(nsamp_, device=self.device).uniform_(-1, 1)
            samp_   = self.clip(torch.arccos(samp_))
            samples = torch.cat([samples, samp_])

            nsamp_  = num_samples - len(samples)  // 2 
        
        samples = samples[:num_samples]
        
        if standardize:
            return self.standardize(samples)
        return samples
    
class UniformSine(Prior):
    """Angle prior uniform in the sine of the angle.

    This is commonly used for ecliptic latitude angles.

    Parameters
    ----------
    minimum : float, optional
        Minimum angle in radians.
    maximum : float, optional
        Maximum angle in radians.
    name : str, optional
        Name of the prior parameter.
    device : str, optional
        Torch device used for sampling.

    Notes
    -----
    Uniformity in :math:`\\sin(\\theta)` implies a density proportional to
    :math:`\\cos(\\theta)` over the support.
    """
    def __init__(self, minimum=-np.pi/2, maximum=np.pi/2, name=None, device='cpu'):
        """Initialize a sine-uniform angle prior.

        Parameters
        ----------
        minimum : float, optional
            Minimum angle in radians.
        maximum : float, optional
            Maximum angle in radians.
        name : str, optional
            Name of the prior parameter.
        device : str, optional
            Torch device used for sampling.
        """
        super().__init__(minimum, maximum, name, device)
    
    def sample(self, num_samples, standardize=False):
        """Sample from a sine-uniform angle prior.

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
        samples = torch.tensor([], device=self.device)
        nsamp_ = num_samples
        while len(samples) < num_samples:
            samp_   = torch.empty(nsamp_, device=self.device).uniform_(-1, 1)
            samp_   = self.clip(torch.arcsin(samp_))
            samples = torch.cat([samples, samp_])

            nsamp_  = num_samples - len(samples)  // 2 
        
        samples = samples[:num_samples]
        
        if standardize:
            return self.standardize(samples)
        return samples


class MultivariatePrior():
    """Container for a multivariate distribution built from 1D priors.

    Parameters
    ----------
    priors : dict or list
        Collection of :class:`Prior` instances. If a dict is provided,
        the values are used in the order of the keys.
    """

    def __init__(self, priors):
        """Initialize a multivariate prior container.

        Parameters
        ----------
        priors : dict or list
            Collection of :class:`Prior` instances.

        Raises
        ------
        AssertionError
            If ``priors`` is not a dict or list, or contains non-``Prior``
            instances.
        """
        #check assertion
        assert isinstance(priors, dict) or isinstance(priors, list), "Priors must be a dictionary or list"
        assert all(isinstance(prior, Prior) for prior in priors.values()), "All priors must be instances of the Prior class"

        #assign prior
        self.priors = priors if isinstance(priors, list) else [priors[key] for key in priors]

    @property
    def names(self):
        """Names of the component priors.

        Returns
        -------
        list of str
            Names of each prior in order.
        """
        return [prior.name for prior in self.priors]
    
    @property
    def means(self):
        """Means of the component priors.

        Returns
        -------
        dict
            Mapping from parameter name to mean.
        """
        return {prior.name: prior.mean for prior in self.priors}
    
    @property
    def stds(self):
        """Standard deviations of the component priors.

        Returns
        -------
        dict
            Mapping from parameter name to standard deviation.
        """
        return {prior.name: prior.std for prior in self.priors}
    
    @property
    def minimums(self):
        """Minimum bounds of the component priors.

        Returns
        -------
        dict
            Mapping from parameter name to minimum value.
        """
        return {prior.name: prior.minimum for prior in self.priors}
    
    @property
    def maximums(self):
        """Maximum bounds of the component priors.

        Returns
        -------
        dict
            Mapping from parameter name to maximum value.
        """
        return {prior.name: prior.maximum for prior in self.priors}

    @property
    def metadata(self):
        """Metadata associated with the multivariate prior.

        Returns
        -------
        dict
            Dictionary containing:

            ``means`` : dict
                Mapping from parameter name to mean.
            ``stds`` : dict
                Mapping from parameter name to standard deviation.
            ``bounds`` : dict
                Mapping from parameter name to ``(min, max)``.
            ``inference_parameters`` : list of str
                Parameter names in order.
        """
        if not hasattr(self, '_metadata'):
            self._metadata = self._get_prior_metadata()
        return self._metadata

    def _get_prior_metadata(self):
        """Build a metadata dictionary for the component priors.

        Returns
        -------
        dict
            Metadata for the component priors.
        """
        prior_metadata = dict()
        #prior_metadata['priors'] = self.priors
        prior_metadata['means']  = {prior.name: prior.mean for prior in self.priors}
        prior_metadata['stds']   = {prior.name: prior.std  for prior in self.priors}
        prior_metadata['bounds'] = {prior.name: (prior.minimum, prior.maximum) for prior in self.priors}
        prior_metadata['inference_parameters'] = self.names
        return prior_metadata

    def standardize(self, samples):
        """Standardize samples using component prior statistics.

        Parameters
        ----------
        samples : dict
            Mapping from parameter name to samples.

        Returns
        -------
        dict
            Standardized samples.
        """
        return {par: self.priors[i].standardize(samples[par]) for i, par in enumerate(self.names)}
    
    def destandardize(self, samples):
        """Reverse standardization using component prior statistics.

        Parameters
        ----------
        samples : dict
            Mapping from parameter name to standardized samples.

        Returns
        -------
        dict
            De-standardized samples.
        """
        return {par: self.priors[i].destandardize(samples[par]) for i, par in enumerate(self.names)}
   
    def to_array(self, samples):
        """Convert a dictionary of samples to a 2D NumPy array.

        Parameters
        ----------
        samples : dict
            Mapping from parameter name to samples (torch tensors).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(N, D)`` where ``D`` is the number of priors.
        """
        return np.array([samples[par].cpu().numpy() for par in self.names]).T

    def to_dict(self, samples):
        """Convert a 2D array of samples to a dictionary.

        Parameters
        ----------
        samples : numpy.ndarray
            Array of shape ``(N, D)``.

        Returns
        -------
        dict
            Mapping from parameter name to column arrays.
        """
        return {par: samples[:, i] for i, par in enumerate(self.names)}
    
    def sample(self, num_samples, standardize=False):
        """Sample from each component prior.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw.
        standardize : bool, optional
            If True, return standardized samples.

        Returns
        -------
        TensorSamples
            Samples drawn from the prior.
        """
        samples = {prior.name: prior.sample(num_samples, standardize) for prior in self.priors}
        return TensorSamples.from_dict(samples)
    
    def sample_array(self, num_samples, standardize=False):
        """Sample from the prior distribution and return a NumPy array.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw.
        standardize : bool, optional
            If True, return standardized samples.

        Returns
        -------
        numpy.ndarray
            Samples drawn from the prior as an array of shape ``(N, D)``.
        """
        return self.sample(num_samples, standardize).numpy()
