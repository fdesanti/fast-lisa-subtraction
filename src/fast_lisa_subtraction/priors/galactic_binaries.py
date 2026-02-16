
import torch
import numpy as np

from ..utils import read_catalog
from .copulas import Copula
from .analytical import *
from .population import *

class RandomFromCatalog(Prior):
    """Sample a single parameter from a catalogue.

    Parameters
    ----------
    catalogue_path : str or os.PathLike
        Path to the catalogue HDF5 file.
    name : str
        Column name to sample.
    minimum : float or None, optional
        Lower bound of the support.
    maximum : float or None, optional
        Upper bound of the support.
    device : str, optional
        Torch device used for sampling.
    """

    def __init__(self, catalog_path, name, minimum=None, maximum=None, device='cpu'):
        """Initialize a catalogue-backed prior.

        Parameters
        ----------
        catalogue_path : str or os.PathLike
            Path to the catalogue HDF5 file.
        name : str
            Column name to sample.
        minimum : float or None, optional
            Lower bound of the support.
        maximum : float or None, optional
            Upper bound of the support.
        device : str, optional
            Torch device used for sampling.
        """
        self.catalogue_df = read_catalog(catalog_path, verbose=False)
        parameter = self.catalogue_df[name].values
        self.parameter = torch.tensor(parameter, device=device)
        super().__init__(minimum, maximum, name, device)

    def sample(self, num_samples, standardize=False):
        """Sample from the catalogue.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw.
        standardize : bool, optional
            If True, return standardized samples.

        Returns
        -------
        torch.Tensor
            Samples drawn from the catalogue.
        """
        #use torch multinomial
        indices = torch.multinomial(torch.ones(len(self.parameter), device=self.device), 
                                    int(num_samples), 
                                    replacement=True)
        samples = self.parameter[indices] 
        #samples += 1e-2 * torch.rand_like(samples)  # Add a small noise to avoid exact duplicates

        if standardize:
            return self.standardize(samples)
        return samples


class GalacticBinaryPopulation(MultivariatePrior):
    """Multivariate prior for monochromatic Galactic binaries.

    This class defines a multivariate prior over intrinsic and extrinsic
    parameters and provides sampling utilities, including optional copula
    correlation between frequency and frequency derivative.

    Parameters
    ----------
    priors : dict or list, optional
        Dictionary or list of dictionaries specifying priors for each
        parameter. If None, defaults are used.
    device : str, optional
        Torch device used for sampling.

    References
    ----------
    [1] `A. Lamberts et al. (2019) <https://academic.oup.com/mnras/article/490/4/5888/5585418>`_
    
    [2] `F. De Santi et al. (2026) <https://arxiv.org/abs/XXXX.XXXXX>`_
    """
    def __init__(self, priors=None, device='cpu'):
        """Initialize the Galactic binary population prior.

        Parameters
        ----------
        priors : dict or list, optional
            Dictionary or list of dictionaries specifying priors for each
            parameter. If None, defaults are used.
        device : str, optional
            Torch device used for sampling.
        """
        
        self.device = device
        #default prior
        _prior = dict(
            Frequency           = PowerLaw(alpha=-2.97, minimum=1e-4, maximum=1e-1, name='Frequency', device=device),
            FrequencyDerivative = Gamma(alpha=1.96, beta=2.6, offset=-21.1, name='FrequencyDerivative', device=device),
            Amplitude           = LogNormal(mu=-22.11, sigma=0.28, minimum=-24.5, maximum=-20.5, name='Amplitude', device=device),
            InitialPhase        = Uniform(0, 2*np.pi, name='InitialPhase', device=device),
            Inclination         = UniformCosine(0, np.pi, name='Inclination', device=device),
            Polarization        = Uniform(0, 2*np.pi, name='Polarization', device=device),
            EclipticLatitude    = Cauchy(loc=0, scale=0.08, name='EclipticLatitude', device=device, minimum=-np.pi/2, maximum=np.pi/2),
            EclipticLongitude   = Cauchy(loc=-np.pi, scale=0.18, name='EclipticLongitude', device=device, minimum=-2*np.pi, maximum=0),
        )
        
        if priors is not None:
            if isinstance(priors, dict):
                _prior.update(priors)
            
            elif isinstance(priors, list):
                for prior in priors:
                    _prior.update(prior)
            else:
                raise ValueError("Priors must be a dictionary or list")
        
        super().__init__(_prior)

    def sample(self, num_samples, standardize=False, copula=False, **copula_kwargs):
        """Sample from the prior distribution.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw.
        standardize : bool, optional
            Whether to standardize the samples.
        copula : bool, optional
            If True, draw correlated samples for frequency and frequency
            derivative using a copula.
        **copula_kwargs : dict
            Additional keyword arguments for the copula function (for
            example, correlation coefficient).

        Returns
        -------
        hyperion.core.TensorSamples
            Samples drawn from the prior distribution.
        """
        num_samples = int(num_samples)
        samples = super().sample(num_samples, standardize=standardize)

        # Get the samples for frequency and frequency derivative
        amp   = samples['Amplitude']
        f     = samples['Frequency']
        f_dot = samples['FrequencyDerivative']  
    
        if copula:
            # Apply the Gaussian copula to the samples
            f_dot, f = Copula(f_dot, f, **copula_kwargs)

            # Update the samples with the new values
            samples['Frequency'] = f
            samples['FrequencyDerivative'] = 10**f_dot

        samples['Amplitude'] = 10**amp*f**(2/3)
        
        return samples
    
