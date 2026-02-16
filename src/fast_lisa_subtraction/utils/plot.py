import shutil
import matplotlib.pyplot as plt

def latexify(plot_func):
    """Apply LaTeX-style rcParams to a matplotlib plotting function.

    Parameters
    ----------
    plot_func : callable
        Plotting function to wrap.

    Returns
    -------
    callable
        Wrapped plotting function with LaTeX-related rcParams applied.

    Notes
    -----
    If a ``latex`` executable is available, text rendering is switched to
    LaTeX and the ``amsmath`` package is enabled.
    """
    def wrapper_plot(*args, **kwargs):
        """Execute the wrapped plotting function with LaTeX rcParams.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to ``plot_func``.
        **kwargs : dict
            Keyword arguments forwarded to ``plot_func``.

        Returns
        -------
        object
            Return value of ``plot_func``.
        """
        plt.rcParams["font.size"] = 18

        #use latex if available on the machine
        if shutil.which("latex"): 
            plt.rcParams.update({"text.usetex": True, 
                                 "font.family": "serif", 
                                 "text.latex.preamble": r"\usepackage{amsmath}"})
            
        return plot_func(*args, **kwargs)
    return wrapper_plot
