import h5py 
import numpy as np
import pandas as pd

from .logger import log

def read_catalog(filepath, verbose=True):
    """Read a gravitational-wave source catalogue from an HDF5 file.

    The function targets LISA LDC Radler-style catalogues, where galactic
    binaries are stored under ``/H5LISA/GWSources/GalBinaries``. If that
    path is not present, the file root is treated as the catalogue group.

    Parameters
    ----------
    filepath : str or os.PathLike
        Path to the input HDF5 file.
    verbose : bool, optional
        If True, log basic information about the read and the number of
        sources found.

    Returns
    -------
    pandas.DataFrame
        DataFrame where each column corresponds to a dataset (field) in
        the catalogue group and each row corresponds to a source.

    Notes
    -----
    All datasets within the selected group are assumed to be 1D and of
    equal length so they can be combined into a tabular structure.
    """

    with h5py.File(filepath, 'r') as f:
        #LDC Radler dataset
        try: GB = f['H5LISA']['GWSources']['GalBinaries']
        except: GB = f

        if verbose:
            log.info(f'Reading catalogue data from {filepath}')
       
        data = pd.DataFrame({key: np.array(GB[key]) for key in GB.keys()})
        if verbose:
            log.info(f"Catalogue contains {len(data)} sources")
    return data
