import os
import h5py
import numpy as np
import pandas as pd

from tqdm import tqdm
from lisaconstants import SIDEREALYEAR_J2000DAY
from ldc.lisa.noise import get_noise_model

from ..utils import read_catalog, log

YEAR = SIDEREALYEAR_J2000DAY*24*60*60

try:
    from gbgpu.gbgpu import GBGPU
except ImportError:
    log.warning("GBGPU is not installed. Please install gbgpu to use SourceCatalog.")

class SourceCatalog:
    """Generate GW waveforms from a Galactic binary catalogue.

    Parameters
    ----------
    catalog_path : str or os.PathLike, optional
        Path to the catalogue file. Used if ``catalog_df`` is not provided.
    catalog_df : pandas.DataFrame, optional
        In-memory catalogue. If provided, ``catalog_path`` is ignored.
    use_gpu : bool, optional
        If True, attempt to use CuPy for GPU acceleration.
    verbose : bool, optional
        If True, enable progress and status logging.
    **gbgpu_kwargs : dict
        Additional keyword arguments forwarded to :class:`gbgpu.gbgpu.GBGPU`.
    """
    def __init__(self, catalog_path=None, catalog_df=None, use_gpu=True, verbose=True, **gbgpu_kwargs):
        """Initialize the catalogue handler.

        Parameters
        ----------
        catalog_path : str or os.PathLike, optional
            Path to the catalogue file. Used if ``catalog_df`` is not provided.
        catalog_df : pandas.DataFrame, optional
            In-memory catalogue. If provided, ``catalog_path`` is ignored.
        use_gpu : bool, optional
            If True, attempt to use CuPy for GPU acceleration.
        verbose : bool, optional
            If True, enable progress and status logging.
        **gbgpu_kwargs : dict
            Additional arguments forwarded to :class:`gbgpu.gbgpu.GBGPU`.
        """
        self.verbose = verbose 
        #setting device for the simulation
        if use_gpu:
            global xp
            try:
                import cupy as xp
                if verbose: log.info("Cupy is available: using the GPU")
            except ImportError:
                import numpy as xp
                use_gpu = False
                log.warning("Cupy NOT available: using the CPU")
        else:
            import numpy as xp
            use_gpu = False
            log.info("Using the CPU")

        self.use_gpu = use_gpu
            
        #read the catalogue
        if catalog_df is None:
            self.cat_path = catalog_path
            self.cat_dir, self.cat_name = catalog_path.rsplit("/", 1)
            self.cat_name = self.cat_name.split(".")[0]
            self.cat_df   = read_catalog(catalog_path, verbose)
        else:
            assert isinstance(catalog_df, pd.DataFrame), "catalog_df should be a pandas DataFrame"
            self.cat_df = catalog_df
            self.cat_name = 'GB_catalogue'
       
        #initialize the GBGPU class
        self.GB = GBGPU(use_gpu=self.use_gpu, **gbgpu_kwargs)

    @property
    def Nbinaries(self):
        """Number of binaries in the catalogue.

        Returns
        -------
        int
            Number of rows in the catalogue.
        """
        return len(self.cat_df)

    @staticmethod
    def get_batch_params(cat_df, index):
        """Extract parameter arrays for a batch of binaries.

        Parameters
        ----------
        cat_df : pandas.DataFrame
            Catalogue containing binary parameters.
        index : array-like
            Indices of the rows to extract.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(9, N)`` with parameters ordered as
            ``[amp, f0, fdot, fddot, phi0, iota, psi, lam, beta_sky]``.
        """
       
        data_batch = cat_df.iloc[index]
       
        #parameters
        amp      = data_batch["Amplitude"].values           # amplitude
        f0       = data_batch["Frequency"].values           # f0
        fdot     = data_batch["FrequencyDerivative"].values # fdot
        fddot    = np.full_like(fdot, 0.0)                  # fddot
        phi0     = data_batch["InitialPhase"].values        # initial phase
        iota     = data_batch["Inclination"].values         # inclination
        psi      = data_batch["Polarization"].values        # polarization angle
        lam      = data_batch['EclipticLongitude'].values   # ecliptic longitude
        beta_sky = data_batch['EclipticLatitude'].values    # ecliptic latitude

        return np.array([amp, f0, fdot, fddot, phi0, iota, psi, lam, beta_sky])
    
    @staticmethod
    def get_frequency_array(Tobs, dt, min_freq=None, max_freq=None):
        """Construct a frequency array for a given observation time and cadence.

        Parameters
        ----------
        Tobs : float
            Observation time in seconds.
        dt : float
            Sampling cadence in seconds.
        min_freq : float or None, optional
            Minimum frequency to include.
        max_freq : float or None, optional
            Maximum frequency to include.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            Frequency array spanning the requested range.
        """

        df = 1/Tobs                     # Define the frequency resolution

        # Define the frequency vector depending on the duration
        ndata = int(Tobs/dt)
        if (ndata % 2)==0:              # Get the number of requencies
            nfft = int((ndata/2)+1)
        else:
            nfft = int((ndata+1)/2)

        F    = df*nfft                 # make the positive frequency vector
        fvec = xp.arange(0, F, df)

        # Choose some appropriate bounds in frequency
        min_freq = min_freq if min_freq is not None else fvec.min() # [Hz]
        max_freq = max_freq if max_freq is not None else fvec.max() # [Hz]
        ind  = xp.where(xp.logical_and(fvec>=min_freq, fvec<=max_freq))
        fvec = fvec[ind]

        return fvec
    
    
    @staticmethod
    def compute_total_snr(AET, PSD, f, wvf_freqs):
        r"""Compute the network SNR in A/E/T channels.

        Parameters
        ----------
        AET : dict
            Dictionary containing waveforms in the A/E/T channels.
        PSD : dict
            Dictionary containing noise PSDs for the A/E/T channels.
        f : numpy.ndarray or cupy.ndarray
            Frequency array for the noise PSD.
        wvf_freqs : numpy.ndarray or cupy.ndarray
            Batched frequency arrays for the waveforms.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            Total network SNR for each waveform.

        Notes
        -----
        The per-channel SNR is computed as

        .. math::
            \mathrm{SNR}_X = 4\,\Delta f \sum_k
            \frac{|h_X(f_k)|^2}{S_X(f_k)},

        and the network SNR is

        .. math::
            \mathrm{SNR} = \sqrt{\mathrm{SNR}_A + \mathrm{SNR}_E + \mathrm{SNR}_T}.
        """
        #get the delta f 
        df = xp.abs(f[1]-f[0])

        #look for the indices of the waveform frequencies in the noise PSD
        idx = xp.searchsorted(f, wvf_freqs)
        idx = xp.clip(idx, 1, len(f)-1)
        
        #compute the SNR in each channel and return the total network SNR
        SNR_A = 4*df * xp.sum((AET["A"] * AET["A"].conj() / PSD["A"][idx]).real, axis=-1)
        SNR_E = 4*df * xp.sum((AET["E"] * AET["E"].conj() / PSD["E"][idx]).real, axis=-1)
        SNR_T = 4*df * xp.sum((AET["T"] * AET["T"].conj() / PSD["T"][idx]).real, axis=-1)

#        print(xp.any(xp.isnan((AET["A"] * AET["A"].conj() / PSD["A"][idx]))), xp.any(xp.isnan(SNR_E)), xp.any(xp.isnan(SNR_T)))
        return xp.sqrt((SNR_A + SNR_E + SNR_T))
        

    def save_catalogue(self, outdir, cat, sum_wvf, Nbinaries):
        """Save generated waveform and source catalogues to HDF5.

        Parameters
        ----------
        outdir : str or os.PathLike or None
            Output directory. If None, the catalogue directory is used.
        cat : pandas.DataFrame
            Source catalogue to save.
        sum_wvf : dict
            Summed waveform dictionary, including metadata entries.
        Nbinaries : int
            Total number of binaries used in the simulation.

        Returns
        -------
        None
        """

        if outdir is None:
            outdir = self.cat_dir
        else:
            os.makedirs(outdir, exist_ok=True)
        
        tdi_outfilepath = f"{outdir}/tdi_cat_{self.cat_name}_{int(Nbinaries)}_binaries.h5"
        cat_outfilepath = f"{outdir}/{self.cat_name}_{int(Nbinaries)}_binaries.h5"

        #saving tdi catalogue to h5 file
        with h5py.File(tdi_outfilepath, 'w') as file:
            for ch, data in sum_wvf.items():
                file.create_dataset(ch, data=data)
        if self.verbose:
            log.info(f"TDI catalogue saved in {tdi_outfilepath}")

        #saving the catalogue to h5 file
        with h5py.File(cat_outfilepath, 'w') as f:
            for key in cat.keys():
                f.create_dataset(key, data=cat[key])
        if self.verbose:
            log.info(f"Catalogue saved in {cat_outfilepath}")

    def generate_template(self, params, dt, Tobs, channels, **gbgpu_kwargs):
        """Generate waveforms for a batch of binaries.

        Parameters
        ----------
        params : numpy.ndarray or cupy.ndarray
            Array containing the parameters of the binaries.
        dt : float
            Data cadence in seconds.
        Tobs : float
            Observation time in seconds.
        channels : list of str
            TDI channels to generate.
        **gbgpu_kwargs : dict
            Additional arguments forwarded to ``GBGPU.run_wave``.

        Returns
        -------
        tuple
            ``(batch_freqs, ch_wvfs)`` where ``batch_freqs`` is the
            frequency array for each waveform and ``ch_wvfs`` is a
            dictionary of channel waveforms.
        """
        #run the wave simulation
        self.GB.run_wave(*params, N=int(32*Tobs/YEAR), dt=dt, T=Tobs, **gbgpu_kwargs)

        #get batch frequencies and waveforms
        batch_freqs = self.GB.freqs
        ch_wvfs = {ch: getattr(self.GB, ch) for ch in channels}

        return batch_freqs, ch_wvfs


    def generate_catalogue(self, Nbatch=int(1e4), Nmax_binaries=None, Tobs=1, AET=True, outdir=None, save=True, snr_threshold=None, noise_model='SciRDv1', duty_cycle=1, **gbgpu_kwargs):
        """Generate waveforms for the full catalogue.

        Parameters
        ----------
        Nbatch : int, optional
            Number of binaries per batch.
        Nmax_binaries : int or None, optional
            Maximum number of binaries to generate. If None, the full
            catalogue is used; otherwise a random subset is sampled.
        Tobs : float, optional
            Observation time in years.
        AET : bool, optional
            If True, generate A/E/T channels; otherwise X/Y/Z.
        outdir : str or os.PathLike or None, optional
            Output directory. If None, uses the catalogue directory.
        save : bool, optional
            If True, save waveforms and catalogue to HDF5.
        snr_threshold : float or None, optional
            Threshold SNR for detectability cuts. If None, no cut is applied.
        noise_model : str, optional
            Noise model name passed to ``get_noise_model``.
        duty_cycle : float, optional
            Detector duty cycle.
        **gbgpu_kwargs : dict
            Additional arguments forwarded to ``GBGPU.run_wave``.

        Returns
        -------
        dict
            Dictionary containing the sum of generated waveforms for each
            TDI channel along with metadata.
        """
        #define the TDI channels
        channels        = ["A", "E", "T"] if AET else ["X", "Y", "Z"]

        #define some simulation constants
        dt   = self.cat_df["Cadence"].values[0] if "Cadence" in self.cat_df else 15
        Tobs *= YEAR

        frequencies = self.get_frequency_array(Tobs, dt)

        # Get the noise model interpolant
        tdi2 = gbgpu_kwargs.get("tdi2", True)
        noise_model = get_noise_model(noise_model, frequencies)
        noise_PSD   = {ch: noise_model.psd(option=ch, freq=frequencies, tdi2=tdi2) for ch in channels}
        
        #number of simulations to run
        if Nmax_binaries is None:
            #use the whole catalogue
            Nbinaries = len(self.cat_df)
            cat = self.cat_df

        else:
            #use random subset of the catalogue
            Nbinaries = Nmax_binaries
            cat = self.cat_df.sample(Nbinaries)

        #Adjust the amplitude of the binaries to account for the duty cycle
        cat["Amplitude"] = cat["Amplitude"].values * np.sqrt(duty_cycle)
        
        #Loop over the batches to generate the data
        Nbatch = int(Nbatch)

        #define the summed TDI channel
        sum_wvf         = {ch: xp.zeros_like(frequencies, dtype=xp.complex128) for ch in channels}
        sum_wvf["f"]    = frequencies.get() if self.use_gpu else frequencies
        sum_wvf["df"]   = 1/Tobs
        sum_wvf["dt"]   = dt
        sum_wvf["Tobs"] = Tobs
        sum_wvf["tdi2"] = tdi2
        sum_wvf["duty_cycle"] = duty_cycle
        sum_wvf["oversample"] = gbgpu_kwargs.get("oversample", 1)
    
        for kwarg in gbgpu_kwargs:
            sum_wvf[kwarg] = gbgpu_kwargs[kwarg]

        SNR  = xp.zeros(len(cat)) #to store the SNR of each binary
        keep = xp.ones(len(cat), dtype=bool) #to be used to mask out the failed and/or undetectable binaries
        
        #generate batched indices
        sources_idxs = np.array_split(np.arange(Nbinaries), Nbinaries/Nbatch) # Split the number of sources

        #loop over the batches
        for idxs in tqdm(sources_idxs, desc="Generating waveforms", total=len(sources_idxs), disable=not self.verbose):
            #get parameters for the batch
            params = self.get_batch_params(cat, index=idxs)#Nbatch=Nbatch, it=it)

            #run the wave simulation
            batch_freqs, ch_wvfs = self.generate_template(params, dt=dt, Tobs=Tobs, channels=channels, **gbgpu_kwargs)

            #compute the SNR of the batch
            snr_batch = self.compute_total_snr(AET=ch_wvfs, PSD=noise_PSD, f=frequencies, wvf_freqs=batch_freqs)
            SNR[idxs] = snr_batch
       
            # Compute insertion indices for the entire batch
            #indices = xp.searchsorted(frequencies, batch_freqs)
            #indices = xp.clip(indices, 1, len(frequencies)-1)
            i0 = xp.searchsorted(frequencies, batch_freqs[:, 0]).flatten() # first index
            B, F = batch_freqs.shape
            indices = xp.tile(xp.arange(F), B).reshape(B, F) + i0[:, None]# indices for all frequencies in the batch
            
            if isinstance(indices, np.ndarray):              # Clip the indices 
                indices = np.clip(indices, 0, len(frequencies)-1) # handle different numpy implementation
            
            # For each channel, add the contributions separately for real and imaginary parts
            for ch in channels:
                xp.add.at(sum_wvf[ch].real, indices, ch_wvfs[ch].real) #real part
                xp.add.at(sum_wvf[ch].imag, indices, ch_wvfs[ch].imag) #imaginary part
        
        #store the SNR in the summary dictionary
        sum_wvf["snr"] = SNR
        
        #cut out the undetectable binaries
        if snr_threshold:
            keep[SNR > snr_threshold] = False

        if save:
            keep = keep.get() if self.use_gpu else keep
            cat = cat[keep]
            for key in sum_wvf.keys():
                try:
                    sum_wvf[key] = sum_wvf[key].get() if self.use_gpu else sum_wvf[key]
                except:
                    pass
            if self.verbose:
                log.info(f"Saving the catalogue with {len(cat)} binaries")
            self.save_catalogue(outdir, cat, sum_wvf, Nbinaries)
            
        return sum_wvf
