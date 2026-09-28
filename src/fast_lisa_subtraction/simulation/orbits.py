"""
LISA orbits on the same device as GBGPU.

GBGPU passes its own arrays (CuPy on the GPU, NumPy on the CPU) to ``Orbits.get_pos``,
which raises a ``ValueError`` if the orbits live on the other device. By default the
orbits are created on the first available backend, which is not necessarily GBGPU's.

:func:`get_orbits` creates the orbits on the requested device. When that is not possible
(e.g. a GPU run without ``lisaanalysistools-cuda12x``), the returned orbits copy the data
between CPU and GPU. This is exact and costs less than 1% of the runtime.
"""

from functools import lru_cache

from ..utils import log, repair_cupy_runtime

#lisatools picks a backend when it is imported, so fix CuPy first
repair_cupy_runtime()

try:
    from lisatools.detector import EqualArmlengthOrbits
except ImportError:
    EqualArmlengthOrbits = None
    log.warning("LISA Analysis Tools is not installed. Please install lisaanalysistools to use SourceCatalog.")

#backends to try, in order
GPU_BACKENDS = ("cuda12x", "cuda11x")
CPU_BACKENDS = ("cpu",)


def _to_host(array):
    """Copy an array to the CPU (NumPy arrays are returned unchanged)."""
    get = getattr(array, "get", None)
    return get() if callable(get) else array


def _to_device(array):
    """Copy an array to the GPU."""
    import cupy as cp
    return cp.asarray(array)


def _uses_gpu(orbits):
    """Return True if ``orbits`` (or any object with an ``xp`` attribute) works with CuPy."""
    xp = getattr(orbits, "xp", None)
    return getattr(xp, "__name__", "") == "cupy"


class BackendSafeMixin:
    """Make ``get_pos`` of a LISA orbits class accept arrays from either device.

    The inputs are copied to the device of the orbits, and the result is copied back
    to the device of the inputs.
    """

    def get_pos(self, t, sc):
        """Compute the spacecraft positions.

        Parameters
        ----------
        t : float or numpy.ndarray or cupy.ndarray
            Time in seconds.
        sc : int or numpy.ndarray or cupy.ndarray
            Spacecraft indices.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            Spacecraft positions, on the same device as ``t``.
        """
        caller_on_gpu = hasattr(t, "__cuda_array_interface__")
        orbits_on_gpu = _uses_gpu(self)

        if caller_on_gpu == orbits_on_gpu:
            #same device: nothing to copy
            return super().get_pos(t, sc)

        if orbits_on_gpu:
            #orbits on the GPU, caller on the CPU
            return _to_host(super().get_pos(_to_device(t), _to_device(sc)))

        #orbits on the CPU, caller on the GPU
        return _to_device(super().get_pos(_to_host(t), _to_host(sc)))


@lru_cache(maxsize=None)
def backend_safe(orbits_class):
    """Add :class:`BackendSafeMixin` to an orbits class.

    Parameters
    ----------
    orbits_class : type
        A :class:`lisatools.detector.Orbits` subclass.

    Returns
    -------
    type
        A subclass of ``orbits_class`` whose ``get_pos`` accepts arrays from either device
        (``orbits_class`` itself if it already has the mixin).
    """
    if issubclass(orbits_class, BackendSafeMixin):
        return orbits_class
    return type(f"BackendSafe{orbits_class.__name__}", (BackendSafeMixin, orbits_class), {})


if EqualArmlengthOrbits is not None:

    class BackendSafeOrbits(BackendSafeMixin, EqualArmlengthOrbits):
        """Equal-armlength orbits that accept arrays from either device."""

else:
    BackendSafeOrbits = None


def _instantiate(orbits_class, backends):
    """Create the orbits on the first available backend in ``backends`` (None if none works)."""
    for backend in backends:
        try:
            return orbits_class(force_backend=backend)
        except TypeError:
            #lisaanalysistools < 1.2 has no force_backend
            return None
        except Exception:
            continue
    return None


def get_orbits(use_gpu, orbits_class=None, backend=None):
    """Create LISA orbits on the requested device.

    Parameters
    ----------
    use_gpu : bool
        If True, create orbits that work with CuPy arrays (for GBGPU on the GPU).
    orbits_class : type or None, optional
        Orbits class to use. Defaults to :class:`lisatools.detector.EqualArmlengthOrbits`.
    backend : str or None, optional
        Backend used by GBGPU (e.g. ``'cuda12x'``), tried first when ``use_gpu`` is True.

    Returns
    -------
    lisatools.detector.Orbits or None
        The orbits, or None if ``lisaanalysistools`` is not installed.
    """
    if EqualArmlengthOrbits is None:
        return None

    base = EqualArmlengthOrbits if orbits_class is None else orbits_class
    safe = BackendSafeOrbits if orbits_class is None else backend_safe(orbits_class)

    if not use_gpu:
        #force the CPU: by default lisatools would pick CUDA when it is available
        orbits = _instantiate(safe, CPU_BACKENDS)
        return orbits if orbits is not None else _legacy_orbits(base, use_gpu=False)

    gpu_backends = GPU_BACKENDS
    if backend in GPU_BACKENDS:
        gpu_backends = (backend,) + tuple(b for b in GPU_BACKENDS if b != backend)

    orbits = _instantiate(safe, gpu_backends)
    if orbits is not None:
        return orbits

    #no GPU backend for lisatools: keep the orbits on the CPU, get_pos copies the data
    orbits = _instantiate(safe, CPU_BACKENDS)
    if orbits is not None:
        log.warning(
            "The GPU backend of LISA Analysis Tools is not available: evaluating the LISA "
            "orbits on the CPU. Install 'lisaanalysistools-cuda12x' (or 'lisaanalysistools-cuda11x') "
            "to run them natively on the GPU."
        )
        return orbits

    #lisaanalysistools < 1.2: use_gpu works, so no copies are needed
    return _legacy_orbits(base, use_gpu=True)


def _legacy_orbits(orbits_class, use_gpu):
    """Create the orbits with the ``use_gpu`` argument of lisaanalysistools < 1.2 (None if it fails)."""
    for kwargs in ({"use_gpu": use_gpu}, {}):
        try:
            return orbits_class(**kwargs)
        except Exception:
            continue
    return None
