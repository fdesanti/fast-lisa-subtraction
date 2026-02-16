import os
import time
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..utils import latexify
from ..utils import log as logger
from .catalog import SourceCatalog

from tqdm import tqdm
from ldc.lisa.noise import get_noise_model

def convergence(S0, S1, tol):
    """Check convergence between two PSD estimates.

    Parameters
    ----------
    S0 : dict
        Previous PSD estimate keyed by channel (``"A"``, ``"E"``, ``"T"``).
    S1 : dict
        Current PSD estimate keyed by channel (``"A"``, ``"E"``, ``"T"``).
    tol : float
        Relative tolerance threshold for convergence.

    Returns
    -------
    bool
        True if all channels satisfy the relative tolerance.
    """
    C = False
    for k in ["A", "E", "T"]:
        diag = np.absolute(S0[k] - S1[k])/S0[k] # relative diff
        if (diag<=tol).all():
            C = True
    return C

class SubtractionAlgorithm(SourceCatalog):
    """Local subtraction algorithm for Galactic binary catalogues.

    Parameters
    ----------
    catalog_path : str or os.PathLike
        Path to the source catalogue.
    tdi_data_path : str or os.PathLike
        Path to the AET data catalogue.
    use_gpu : bool, optional
        If True, attempt to use GPU acceleration.
    verbose : bool, optional
        If True, enable progress and status logging.
    **gbgpu_kwargs : dict
        Additional keyword arguments forwarded to the GBGPU backend.
    """

    def __init__(self, catalog_path, tdi_data_path, use_gpu=True, verbose=True, **gbgpu_kwargs):
        """Initialize the subtraction algorithm.

        Parameters
        ----------
        catalog_path : str or os.PathLike
            Path to the source catalogue.
        tdi_data_path : str or os.PathLike
            Path to the AET data catalogue.
        use_gpu : bool, optional
            If True, attempt to use GPU acceleration.
        verbose : bool, optional
            If True, enable progress and status logging.
        **gbgpu_kwargs : dict
            Additional keyword arguments forwarded to the GBGPU backend.
        """
        super().__init__(catalog_path, use_gpu=use_gpu, verbose=verbose, **gbgpu_kwargs)
        self.verbose = verbose
        global xp, median_filter, gaussian_filter1d, savgol_filter, gaussian_window, convolve 
        if self.use_gpu:
            import cupy as xp
            from cupyx.scipy.signal import savgol_filter, convolve
            from cupyx.scipy.ndimage import median_filter, gaussian_filter1d
            from cupyx.scipy.signal.windows import gaussian as gaussian_window
            
        else:
            import numpy as xp
            from scipy.signal import savgol_filter, convolve
            from scipy.ndimage import median_filter, gaussian_filter1d
            from scipy.signal.windows import gaussian as gaussian_window

        self.catalogue_path = catalog_path
        self.AET_path  = tdi_data_path

        #read the data
        self.AET = self.read_aet_data(tdi_data_path)

    @property
    def metadata(self):
        """Return the current AET metadata dictionary.

        Returns
        -------
        dict
            AET data and metadata.
        """
        return self.AET

    @staticmethod
    def read_aet_data(tdi_data_path):
        """Read an AET waveform catalogue from HDF5.

        Parameters
        ----------
        tdi_data_path : str or os.PathLike
            Path to the AET HDF5 file.

        Returns
        -------
        dict
            Dictionary of AET arrays keyed by channel.
        """
        AET_data = dict()
        with h5py.File(tdi_data_path, 'r') as file:
            for ch in file.keys():
                AET_data[ch] = xp.array(file[ch])
        return AET_data
    
    @latexify
    def plot_catalogues(self):
        """Plot resolved and unresolved catalogues.

        Returns
        -------
        list of matplotlib.figure.Figure
            Figures generated for each plotted parameter.
        """

        if not hasattr(self, "resolved_cat") or not hasattr(self, "unresolved_cat"):
            logger.error("No resolved and/or unresolved catalogue found. Please run the subtraction algorithm first.")
            return

        res_cat = self.resolved_cat
        unres_cat = self.unresolved_cat
        original_cat = self.cat_df
   
        xlabel = {'Frequency'        : r'$f$ [Hz]',
                'FrequencyDerivative': r'$\dot{f}$ [Hz/s]',
                'Amplitude'          : r'$\mathcal{A}$',
        }

        figs, axs = plt.subplots(1, 3, figsize=(18, 5))
        for i, key in enumerate(['Frequency', 'FrequencyDerivative', 'Amplitude']):
            # Get data
            orig_raw  = original_cat[key].to_numpy()
            unres_raw = unres_cat[key].to_numpy()
            res_raw   = res_cat[key].to_numpy()

            orig_pos  = orig_raw[np.isfinite(orig_raw) & (orig_raw > 0)]
            unres_pos = unres_raw[np.isfinite(unres_raw) & (unres_raw > 0)]
            res_pos   = res_raw[np.isfinite(res_raw) & (res_raw > 0)]

            # Common bin edges 
            all_pos = np.concatenate([orig_pos, unres_pos, res_pos]) if (orig_pos.size + unres_pos.size + res_pos.size) else np.array([])

            if all_pos.size == 0:
                logger.error(f"No positive finite values found in '{key}' to plot.")
                return

            bin_edges = np.logspace(np.log10(all_pos.min()), np.log10(all_pos.max()), 100)

            # Plot histograms 
            axs[i].hist(orig_pos,  bins=bin_edges, density=False, alpha=0.5, label='All')
            axs[i].hist(res_pos,   bins=bin_edges, density=False, alpha=0.5, label='Resolved')
            axs[i].hist(unres_pos, bins=bin_edges, density=False, alpha=0.5, label='Unresolved')

            axs[i].set_yscale('log')   # log y-axis
            axs[i].set_xscale('log')   # log x-axis
            axs[i].set_xlabel(xlabel[key])
            axs[i].set_ylabel('Count')
            axs[i].minorticks_on()
            axs[i].legend()
        
        figs.tight_layout()
        figs.savefig(f'catalog_plots.pdf', bbox_inches='tight')
        plt.close(figs)
        return figs


    def psd_smooth_moving_average(self, PSD, Nsegments=300, methoduse="median",  extra_smooth="convolution", **kwargs):
        """Smooth the PSD using a moving average or median.

        Parameters
        ----------
        PSD : dict
            Instrumental noise PSD per channel.
        Nsegments : int, optional
            Segment length used for smoothing.
        methoduse : str, optional
            Smoothing method: ``"mean"`` or ``"median"``.
        extra_smooth : str, optional
            Additional smoothing method: ``"none"``, ``"convolution"``,
            ``"whittaker"``, ``"savgol"``, or ``"gaussian_kernel"``.
        **kwargs : dict
            Additional parameters forwarded to smoothing routines.

        Returns
        -------
        dict
            Smoothed PSD per channel.
        """
        # Compute absolute value of the data channels
        # the 2df factor accounts the FFT normalization
        AET2 = dict([(k,2*self.df*xp.absolute(self.AET[k])**2) for k in ["A", "E", "T"]])

        # Compute PSD with methoduse
        S = dict()
        for k in ["A", "E", "T"]:
            #print(k)
            if methoduse == "mean":
                Sk = xp.convolve(AET2[k], xp.ones(Nsegments), "same") / Nsegments
            
            elif methoduse == "median":
                # If Q = N1**2 + N2**2 + ... + Nk**2, where Nk independent random normal variables, then
                # Q ~ chi2(k) distribution. For chi2 distributions, mean=k, and median ~= k*(1-2/(9*k))**3.
                # This is why we have defined "norm", a normalization factor by setting k=2 as
                # 
                # norm = mean/median= 1/(1-2/(9*2))**(-3) = 1/0.7023319615912207
                norm = kwargs.get("norm", 1/0.7023319615912207)
                Sk = median_filter(AET2[k], size=Nsegments) * norm
            

            # Extra smoothing ----------------------------- 
            # No extra smoothing
            if extra_smooth.lower() == "none":
                Sk_extra = Sk

            # Convolve on the running "methoduse" of the data
            elif extra_smooth.lower() == "convolution":
                sigma = kwargs.get("sigma", 5)
                window = gaussian_window(len(Sk), std=sigma)
                Sk_extra = convolve(Sk, window, mode='same') / window.sum() # Normalize the result
            
            # Gaussian kernel smoothing
            elif "gaussian" in extra_smooth.lower():
                #Sk_extra = doKernelSmoothing(AET.f.squeeze()[1:], Sk.squeeze()[1:], n=order)
                #Sk_extra = xp.insert(Sk_extra, 0, 0) # First element is usually a nan, that"s why we add 0 by hand
                Sk_extra = gaussian_filter1d(Sk.squeeze(), sigma=kwargs.get("sigma", 1))
            
            # Whittaker smoother
            elif extra_smooth.lower() == "whittaker":
                #whittaker_smoother = WhittakerSmoother(lmbda=int(order), order=1, data_length=len(Sk.squeeze()))
                #Sk_extra = 10**xp.array(whittaker_smoother.smooth( xp.log10( Sk.squeeze()) ))
                raise NotImplementedError("Whittaker smoother is not implemented yet. Please use another smoothing method.")
            
            # Savitzky-Golay filter
            elif extra_smooth.lower() == "savgol":
                order = kwargs.get("order", 2)
                Sk_extra = 10**savgol_filter( xp.log10( Sk.squeeze() ), int(Sk.squeeze().shape[0]/100), int(order))
                #Sk_extra = xp.insert(Sk_extra, 0, 0) # First element is usually a nan, that"s why we add 0 by hand 

            else:
                raise ValueError(f"Unknown smoothing method: {extra_smooth}")
            
            S[k] = AET2[k].copy()*0
            S[k] += Sk_extra 
            
            # Add the instrumental noise
            S[k] += PSD[k]

        return S
    
    @latexify
    def local_subtraction(self, PSD):
        """Run a single local subtraction pass.

        Parameters
        ----------
        PSD : dict
            Instrumental+confusion PSD per channel.

        Returns
        -------
        int
            Number of sources subtracted in this pass.
        """
        # Batch the sources
        num_sources = len(self.cat)
        if num_sources <= self.batch_size: 
            sources_ids = [np.arange(num_sources)]
        else:
            sources_ids = np.array_split(np.arange(num_sources), num_sources/self.batch_size) # Split the number of sources


        # Initialize the subtracted sources mask
        subtracted = np.zeros((len(self.cat)), dtype=bool)
        if self.doplot: A = xp.zeros((len(self.f)), dtype=xp.complex128)

         #run the subtraction loop
        for j_batch, srcs_ids_jbatch  in enumerate(tqdm(sources_ids, ncols=80, ascii=' =', disable=False if self.verbose else True)):

            # Get the parameters for the batch
            params = self.get_batch_params(self.cat, index=srcs_ids_jbatch)#Nbatch=self.batch_size, it=j_batch)
 
            # Compute the waveforms
            batch_f, batch_AET = self.generate_template(params, dt=self.dt, channels=["A", "E", "T"], Tobs=self.Tobs, tdi2=self.tdi2, oversample=self.oversample) 
            B, F = batch_f.shape
            
            # Compute the SNR
            snr_tot = self.compute_total_snr(batch_AET, PSD, wvf_freqs=batch_f, f=self.f)

            # Identify the "loud" sources that pass the SNR threshold
            loud_mask = xp.where(snr_tot >= self.snr_thresh)[0]  # indices of loud sources
            
            #indices   = xp.searchsorted(self.f, batch_f)[loud_mask]
            # Compute the indices for the frequencies in the batch
            i0 = xp.searchsorted(self.f, batch_f[:, 0]).flatten() # index of the first frequency
            indices = xp.tile(xp.arange(F), B).reshape(B, F) + i0[:, None]# indices for all frequencies in the batch
            indices = indices[loud_mask] # only the loud sources
            
            if isinstance(indices, np.ndarray):              # Clip the indices 
                indices = np.clip(indices, 0, len(self.f)-1) # handle different numpy implementation

            # Update the subtracted sources mask
            if isinstance(loud_mask, np.ndarray):
                subtracted[srcs_ids_jbatch[loud_mask]] = True
            else:
                subtracted[srcs_ids_jbatch[loud_mask.get()]] = True

            # Subtract the sources from the data channels
            if indices.shape[0] > 0:
                for ch in ["A", "E", "T"]:
                    xp.add.at(self.AET[ch].real, indices, -batch_AET[ch][loud_mask].real)
                    xp.add.at(self.AET[ch].imag, indices, -batch_AET[ch][loud_mask].imag)
    
                if self.doplot:
                    xp.add.at(A.real, indices, batch_AET["A"][loud_mask].real)
                    xp.add.at(A.imag, indices, batch_AET["A"][loud_mask].imag)
            
        # Plot the subtracted sources
        if self.doplot:
            if not hasattr(self, "fig_sub"):
                self.fig_sub = plt.figure(figsize=(12,10))
                self.fig_sub.add_subplot(111)
                self.iter = 1
            else:
                self.iter += 1

            ff = self.f.get() if self.use_gpu else self.f
            aa = (2 * self.df * xp.absolute(A)).get() if self.use_gpu else 2 * self.df * xp.absolute(A)
            self.fig_sub.axes[0].loglog(ff, aa, label=f"it = {self.iter}", alpha=0.5)
            self.fig_sub.axes[0].set_xlabel("$f$ [Hz]")
            self.fig_sub.axes[0].set_ylabel(r"$2\,\Delta f\,|\tilde{A}(f)|$")
            self.fig_sub.axes[0].set_xlim(1e-5, 1e-1)
            self.fig_sub.axes[0].set_title(f"Subtracted sources")
            self.fig_sub.axes[0].legend(loc="upper left")
            self.fig_sub.savefig("sub_sources.pdf", bbox_inches="tight")

        if self.verbose:
            logger.info(f"Subtracted sources: {subtracted.sum()}")

        # Update the catalogue of resolved sources
        self.resolved_cat = pd.concat([self.resolved_cat, self.cat[subtracted]], ignore_index=True)

        # Keep only the non-subtracted sources in the catalogue
        self.cat = self.cat[~subtracted]
        self.cat = self.cat.reset_index(drop=True)
    
        #return the number of subtracted sources
        return subtracted.sum()

    @latexify
    def icloop(self, batch_size=10_000, lisa_noise='SciRDv1', maxiter=10, snr_threshold=7, kappa=.15, tol=1e-3,
               doplot=False, verbose=True, **psd_kwargs):
        """Iteratively subtract resolved sources from the data.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for subtraction.
        lisa_noise : str, optional
            Noise model name.
        maxiter : int, optional
            Maximum number of iterations.
        snr_threshold : float, optional
            SNR threshold for subtraction.
        kappa : float, optional
            Safety factor for initial over-threshold selection.
        tol : float, optional
            Convergence tolerance.
        doplot : bool, optional
            If True, generate diagnostic plots.
        verbose : bool, optional
            If True, enable progress and status logging.
        **psd_kwargs : dict
            Additional parameters for PSD smoothing.

        Returns
        -------
        tuple
            ``(AET, PSD)`` where ``AET`` is the updated metadata dictionary
            and ``PSD`` is the final smoothed PSD.
        """
        self.verbose = verbose 
        
        start = time.time()
       
        if self.verbose:
            logger.info("Starting the local subtraction algorithm")
        
        # Initialize some attributes
        self.doplot     = doplot
        self.batch_size = batch_size
        self.snr_thresh = snr_threshold
        
        # Get the data
        self.df   = self.AET["df"]
        self.f    = self.AET["f"]
        self.Tobs = self.AET["Tobs"].get() if self.use_gpu else self.AET["Tobs"]
        self.dt   = self.AET["dt"].get()   if self.use_gpu else self.AET["dt"]
        self.tdi2 = self.AET["tdi2"].get() if self.use_gpu else self.AET["tdi2"]
        self.oversample = self.AET["oversample"].get() if self.use_gpu else self.AET["oversample"]
        for ch in ["A", "E", "T"]:
            self.AET[f"{ch}_original"] = self.AET[ch].get() if self.use_gpu else self.AET[ch]

        # Get the noise for all frequencies in a dictionary
        lisa_noise = get_noise_model(lisa_noise, self.AET["f"])
        self.lisa_noise = lisa_noise

        # Get the noise for all frequencies in a dictionary
        noise = {ch: lisa_noise.psd(self.AET["f"], option=ch, tdi2=self.tdi2) for ch in ["A", "E", "T"]}
        self.initial_noise = noise

        # Discard very under-threshold sources
        sel = self.AET["snr"] > kappa*snr_threshold
        sel = sel.get() if self.use_gpu else sel

        if self.verbose:
            logger.info(f"Minimum SNR = {self.AET['snr'].min():.5f}, maximum SNR = {self.AET['snr'].max():.2f}")
            logger.info(f"Selecting sources with SNR > {self.AET['snr'][sel].min():.2f}")
            logger.info(f"Selecting {sel.sum()}/{len(self.cat_df)} sources for subtraction")

        self.cat = self.cat_df.copy() # restrict the loop
        self.cat = self.cat[sel]
        self.cat = self.cat.reset_index(drop=True)

        # Initialize the catalogue of unresolved sources (before subtraction)
        self.unresolved_cat = self.cat_df[~sel].copy()
        self.unresolved_cat = self.unresolved_cat.reset_index(drop=True)

        # Initialize the catalogue of resolved sources (empty at the beginning)
        self.resolved_cat = self.cat.iloc[0:0].copy()

        if sel.sum() == 0:
            if self.verbose: logger.info("No sources above the SNR threshold. Exiting...")
            return self.AET, noise
        
        # Compute the initial PSD
        S0 = self.psd_smooth_moving_average(PSD=noise, **psd_kwargs)
        if self.verbose: logger.info("Initial PSD computed")

        if doplot:
            if self.verbose: logger.info(f"Making initial plot")
            fig = plt.figure(figsize=(12,10))
            fig.add_subplot(111)
            fplot =  self.f.get() if self.use_gpu else self.f 
            Sn = np.absolute(noise["A"].get()) if self.use_gpu else xp.absolute(noise["A"])
            Sn_smooth = np.absolute(S0["A"].get()) if self.use_gpu else xp.absolute(S0["A"])
            fig.axes[0].loglog(fplot, Sn, "k--", label="noise")
            fig.axes[0].loglog(fplot, Sn_smooth, label=r"$S_n$ (it=0)")
            fig.axes[0].legend(loc="upper left")
            fig.axes[0].set_xlabel("$f$ [Hz]")
            fig.axes[0].set_xlim(1e-5, fplot.max())
            fig.axes[0].set_ylim(1e-46, 1e-36)
            fig.axes[0].set_ylabel("PSD A")

            if not hasattr(self, "fig_sum"):
                self.fig_sum = plt.figure(figsize=(12,10))
                self.fig_sum.add_subplot(111)
            aa = (2 * self.df * xp.absolute(self.AET["A"])).get() if self.use_gpu else 2 * self.df * xp.absolute(self.AET["A"])
            self.fig_sum.axes[0].loglog(fplot, aa, label=f"Initial Sources", alpha=0.5)
            self.fig_sum.axes[0].set_xlabel("$f$ [Hz]")
            self.fig_sum.axes[0].set_xlim(1e-5, 1e-1)
            self.fig_sum.axes[0].legend(loc="upper left")
            self.fig_sum.savefig("sum_sources.pdf", bbox_inches="tight")

        # Run the subtraction loop
        Num_subtracted = 0
        for it in tqdm(range(1, maxiter + 1), disable=not self.verbose):
            if self.verbose: logger.info(f"Starting iteration {it}")

            # Run the local subtraction
            num_subtracted_iter = self.local_subtraction(PSD=S0)
            Num_subtracted += num_subtracted_iter
            if self.verbose: logger.info(f"{num_subtracted_iter} source subtracted at iter {it}.")
            
            # Compute the new PSD
            S1 = self.psd_smooth_moving_average(PSD=noise, **psd_kwargs)
            
            if self.verbose: logger.info("New PSD computed")

            if doplot:
                if self.verbose: logger.info(f"Making plot for iteration {it}")
                # PSD plot
                S1plot = xp.absolute(S1["A"]).get() if self.use_gpu else xp.absolute(S1["A"])
                fig.axes[0].loglog(fplot, S1plot, label=rf"$S_n$ (it=${it}$)")#, color=pp[0].get_color())
                fig.axes[0].legend(loc="upper left")
                fig.savefig(f"total_psd.pdf", bbox_inches="tight")
                 
                # Sum sources plot
                aa = (2 * self.df * xp.absolute(self.AET["A"])).get() if self.use_gpu else 2 * self.df * xp.absolute(self.AET["A"])
                self.fig_sum.axes[0].loglog(fplot, aa, label=f"it = {it}", alpha=0.5)
                self.fig_sum.axes[0].legend(loc="upper left")
                self.fig_sum.axes[0].set_xlim(1e-5, 1e-1)
                self.fig_sum.axes[0].set_xlabel("$f$ [Hz]")
                self.fig_sum.axes[0].set_ylabel(r"$2\,\Delta f\,|\tilde{A}(f)|$")
                self.fig_sum.axes[0].set_title(f"Total sources")

                self.fig_sum.savefig(f"sum_sources.pdf", bbox_inches="tight")

            # Check for convergence
            if((convergence(S0, S1, tol)) or
                (num_subtracted_iter == 0)) and it>=3:
                if self.verbose:
                    logger.info("convergence reached, or all sources subtracted")
                break
            else:
                # Replace latest noise+background estimate for new iteration
                S0 = S1.copy()
        self.Sconf = S1
        
        # End of the subtraction loop
        end = time.time()
        self.runtime = end - start
        self.iterations = it
        if self.verbose:
            logger.info
            logger.info(f"Subtraction algorithm took {self.runtime: .1f} seconds and {it} iterations")
            logger.info(f"Total subtracted sources: {Num_subtracted}")
        
        
        # Update the unresolved catalogue with the remaining sources
        self.unresolved_cat = pd.concat([self.unresolved_cat, self.cat], ignore_index=True)

        #update the metadata
        self.update_metadata()

        if doplot:
            cat_figs = self.plot_catalogues()
            
            return self.AET, S1, cat_figs
        
        return self.AET, S1
    
    def update_metadata(self):
        """Update the AET metadata after subtraction.

        Returns
        -------
        None
        """
        #self.AET["original_cat"]   = self.cat_df
        self.AET["resolved_cat"]   = self.resolved_cat
        self.AET["unresolved_cat"] = self.unresolved_cat
        self.AET["num_sources"]    = len(self.cat_df)
        self.AET["num_resolved"]   = len(self.resolved_cat)
        self.AET["num_unresolved"] = len(self.unresolved_cat)
        self.AET["runtime"]        = self.runtime
        self.AET["iterations"]     = self.iterations
        self.AET["f"]              = self.f.get() if self.use_gpu else self.f
        self.AET["Sconf"]          = {}
        for ch in ["A", "E", "T"]:
            self.AET["Sconf"][ch] = self.Sconf[ch].get() if self.use_gpu else self.Sconf[ch]
    

    def run(self, **run_kwargs):
        """Run the local subtraction algorithm (alias for ``icloop``).

        Parameters
        ----------
        **run_kwargs : dict
            Keyword arguments forwarded to :meth:`icloop`.

        Returns
        -------
        tuple
            ``(AET, PSD)`` after subtraction.
        """
        return self.icloop(**run_kwargs)
