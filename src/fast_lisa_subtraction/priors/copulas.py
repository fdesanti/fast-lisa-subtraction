import torch

from .analytical import Gamma
from torch.distributions import Normal, MultivariateNormal, StudentT, Exponential, Chi2

__all__ = ['Copula']

class Copula_:
    """Apply copulas to samples from marginal distributions."""
    @staticmethod
    def quantile(x, u):
        """Compute empirical quantiles for 1D tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.
        u : torch.Tensor
            Quantile probabilities in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Quantile values corresponding to ``u``.
        """
        # Sort the input tensor
        sorted_x, _ = torch.sort(x, dim=0) 
        N = sorted_x.numel()                   
        
        # Compute float positions in [0, N-1] for linear interpolation
        pos = u * (N - 1)
        lo = torch.floor(pos).long().clamp(0, N - 1)
        hi = torch.ceil( pos).long().clamp(0, N - 1)
        w  = (pos - lo)            

        # Interpolate
        x_lo = sorted_x[lo]
        x_hi = sorted_x[hi]
        return x_lo * (1 - w) + x_hi * w

    @staticmethod
    def empirical_cdf(x):
        """Compute the empirical CDF of a 1D tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input samples.

        Returns
        -------
        tuple of torch.Tensor
            ``(sorted_x, cdf)`` where ``cdf`` is the empirical cumulative
            distribution at each sorted value.
        """
        # Sort the input tensor
        sorted_x, _ = torch.sort(x, dim=0) 
        N = sorted_x.numel()                   
        
        # Compute the empirical CDF
        cdf = torch.arange(1, N + 1, dtype=torch.float32, device=x.device) / N
        return sorted_x, cdf

    @classmethod
    def gaussian_copula(cls, x, y, rho): 
        """Apply a Gaussian copula to two marginal samples.

        Parameters
        ----------
        x : torch.Tensor
            Samples from the first marginal distribution.
        y : torch.Tensor
            Samples from the second marginal distribution.
        rho : float
            Correlation coefficient between the marginals.

        Returns
        -------
        tuple of torch.Tensor
            Transformed samples with Gaussian-copula dependence.
        """
        # Get some properties of the input tensors
        device = x.device
        num_samples = x.size(0) 

        # Compute the covariance matrix
        cov = torch.tensor([[1.0, rho], [rho, 1.0]], device=device)

        # Instantiate a multivariate normal distribution
        multivariate_norm = MultivariateNormal(loc=torch.zeros(2, device=device),
                                            covariance_matrix=cov)

        # Sample from the multivariate normal distribution
        z = multivariate_norm.sample((num_samples,))

        # Convert to uniform via inverse CDF
        norm = Normal(loc=torch.zeros(1, device=device), scale=torch.ones(1, device=device))
        u = norm.cdf(z[:, 0])
        v = norm.cdf(z[:, 1])

        # Transform the uniform samples to the marginal distributions
        x_transformed = cls.quantile(x, u)
        y_transformed = cls.quantile(y, v)

        return x_transformed, y_transformed
    
    @classmethod
    def student_t_copula(cls, x, y, rho, df, eps=1e-7):
        """Apply a Student's t copula to two marginal samples.

        Parameters
        ----------
        x : torch.Tensor
            Samples from the first marginal distribution.
        y : torch.Tensor
            Samples from the second marginal distribution.
        rho : float
            Correlation coefficient between the marginals.
        df : float
            Degrees of freedom for the t distribution.
        eps : float, optional
            Clamp value to keep uniforms within ``(0, 1)``.

        Returns
        -------
        tuple of torch.Tensor
            Transformed samples with Student's t dependence.
        """
        device = x.device
        num_samples = x.size(0)

        cov = torch.tensor([[1.0, rho], [rho, 1.0]], device=device)
        mvn = MultivariateNormal(loc=torch.zeros(2, device=device), covariance_matrix=cov)
        z = mvn.sample((num_samples,))

        chi2_samples = Chi2(df).sample((num_samples,)).to(device)
        w = chi2_samples / df
        t_samples = z / torch.sqrt(w).unsqueeze(1)

        tdist = StudentT(df=df)
        u = tdist.cdf(t_samples[:, 0]).to(device)
        v = tdist.cdf(t_samples[:, 1]).to(device)

        # keep strictly inside (0,1) for stability in quantiles
        u = u.clamp(eps, 1 - eps)
        v = v.clamp(eps, 1 - eps)

        x_transformed = cls.quantile(x, u)
        y_transformed = cls.quantile(y, v)
        return x_transformed, y_transformed

    @classmethod
    def clayton_copula(cls, x, y, theta, eps=1e-12):
        """Apply a Clayton copula to two marginal samples.

        Parameters
        ----------
        x : torch.Tensor
            Samples from the first marginal distribution.
        y : torch.Tensor
            Samples from the second marginal distribution.
        theta : float
            Copula parameter (must be positive).
        eps : float, optional
            Clamp value for numerical stability.

        Returns
        -------
        tuple of torch.Tensor
            Transformed samples with Clayton dependence.
        """
        assert theta > 0, "Theta must be positive for Clayton copula"
        device = x.device
        num_samples = x.size(0)

        # W ~ Gamma(1/theta, 1)  (confirm your Gamma uses beta as rate=1)
        w = Gamma(alpha=1.0/theta, beta=1.0, device=device).sample(num_samples)
        w = w.clamp_min(eps)

        e = Exponential(torch.ones(1, device=device)).sample((num_samples, 2))
        e1, e2 = e[:, 0], e[:, 1]

        u = (1.0 + e1 / w).pow(-1.0 / theta)
        v = (1.0 + e2 / w).pow(-1.0 / theta)

        u = u.clamp(1e-7, 1 - 1e-7)
        v = v.clamp(1e-7, 1 - 1e-7)

        x_transformed = cls.quantile(x, u)
        y_transformed = cls.quantile(y, v)

        return x_transformed, y_transformed
    
    @classmethod
    def frank_copula(cls, x, y, k):
        """Apply a Frank copula to two marginal samples.

        Parameters
        ----------
        x : torch.Tensor
            Samples from the first marginal distribution.
        y : torch.Tensor
            Samples from the second marginal distribution.
        k : float
            Copula correlation parameter.

        Returns
        -------
        tuple of torch.Tensor
            Transformed samples with Frank dependence.
        """
        theta = k
        assert theta != 0, "Theta must be non-zero for Frank copula"
        device      = x.device
        num_samples = x.size(0)
        theta = torch.tensor(theta, device=device)

        # 1) draw U and W ~ Uniform(0,1)
        u0 = torch.rand(num_samples, device=device)
        w  = torch.rand(num_samples, device=device)

        # 2) compute V via the closed-form inverse of the conditional CDF
        exp_neg_t = torch.exp(-theta)
        exp_neg_tu = torch.exp(-theta * u0)
        numerator   = w * (exp_neg_t - 1.0)
        denominator = (1.0 - w) * exp_neg_tu + w
        v0 = - (1.0/theta) * torch.log1p(numerator / denominator)

        # 3) map uniforms (u0, v0) to your empirical marginals
        x_transformed = cls.quantile(x, u0)
        y_transformed = cls.quantile(y, v0)

        return x_transformed, y_transformed
        


    def __call__(self, x, y, kind='gaussian', **copula_kwargs):
        """Dispatch to a specific copula implementation.

        Parameters
        ----------
        x : torch.Tensor
            Samples from the first marginal distribution.
        y : torch.Tensor
            Samples from the second marginal distribution.
        kind : str, optional
            Copula type. Supported values are ``'gaussian'``, ``'clayton'``,
            ``'student_t'``, and ``'frank'``.
        **copula_kwargs : dict
            Additional keyword arguments forwarded to the copula method.

        Returns
        -------
        tuple of torch.Tensor
            Transformed samples with the requested dependence structure.

        Raises
        ------
        ValueError
            If ``kind`` is not recognized.
        """
        if kind == 'gaussian':
            return self.gaussian_copula(x, y, **copula_kwargs)
        elif kind == 'clayton':
            return self.clayton_copula(x, y, **copula_kwargs)
        elif kind == 'student_t':
            return self.student_t_copula(x, y, **copula_kwargs)
        elif kind == 'frank':
            return self.frank_copula(x, y, **copula_kwargs)
        else:
            raise ValueError(f"Unknown copula type: {kind}")
        
Copula = Copula_()
