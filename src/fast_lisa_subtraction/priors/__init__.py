from .copulas import *
from .analytical import *
from .population import *
from .galactic_binaries import *

prior_dict = {'uniform': Uniform, 'PowerLaw': PowerLaw, 'LogNormal': LogNormal, 'Gamma': Gamma}

__all__ = ['Uniform', 'LogNormal', 'Gamma', 'PowerLaw', 'BrokenPowerLaw', 'UniformCosine', 'UniformSine', 'Gaussian', 'Cauchy', 'GalacticBinaryPopulation']